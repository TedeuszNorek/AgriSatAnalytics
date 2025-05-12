"""
Data Ingest - Import and manage field data
"""
import os
import json
import logging
import datetime
import traceback
import uuid
import random
from typing import Dict, List, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
from folium.plugins import Draw
from shapely.geometry import Polygon, shape
from streamlit.components.v1 import html

from database import get_db, Field
from utils.data_access import (
    get_sentinel_hub_config, 
    parse_geojson, 
    get_bbox_from_polygon,
    get_country_boundary
)
from utils.mock_data import (
    generate_mock_field_data,
    get_mock_field_boundary,
    save_mock_data
)
from config import (
    SENTINEL_HUB_CLIENT_ID,
    SENTINEL_HUB_CLIENT_SECRET,
    CROP_TYPES,
    APP_NAME,
    APP_DESCRIPTION
)

# Configure page
st.set_page_config(
    page_title=f"{APP_NAME} - Data Ingest",
    page_icon="🛰️",
    layout="wide"
)

# Configure logging
logger = logging.getLogger(__name__)

# Header
st.markdown("# Data Ingest")
st.markdown("Import and manage agricultural field data")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## Data Options")
    data_option = st.radio(
        "Choose Data Source",
        ["Upload GeoJSON", "Draw on Map", "Import from Database", "Generate Test Data"]
    )
    
    st.markdown("## Field Management")
    manage_option = st.radio(
        "Manage Fields",
        ["Add New Field", "Edit Existing Field", "Delete Field"]
    )
    
    # Dodaj przycisk czyszczenia cache, jeśli mamy dane w sesji
    if "drawn_features" in st.session_state or "generated_field" in st.session_state:
        if st.button("Clear Cache Data", type="secondary"):
            if "drawn_features" in st.session_state:
                del st.session_state["drawn_features"]
            if "generated_field" in st.session_state:
                del st.session_state["generated_field"]
            st.success("Cache data cleared!")
            st.rerun()

# Main content
if data_option == "Upload GeoJSON" and manage_option == "Add New Field":
    st.markdown("## Upload Field Boundary")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload GeoJSON file", type=["geojson", "json"])
    
    if uploaded_file is not None:
        try:
            # Read file content
            geojson_content = uploaded_file.read().decode()
            geojson_data = json.loads(geojson_content)
            
            # Parse GeoJSON and extract polygon
            polygon, crs = parse_geojson(geojson_data)
            bbox = get_bbox_from_polygon(polygon)
            
            # Calculate center and area
            centroid = polygon.centroid
            center_lat, center_lon = centroid.y, centroid.x
            
            # Convert area to hectares (assuming coordinates are in degrees)
            # This is a rough approximation, more accurate calculations would require reprojection
            area_m2 = polygon.area * 111000 * 111000  # Approximate conversion from degrees to meters
            area_hectares = area_m2 / 10000  # Convert m² to hectares
            
            # Display map with the field boundary
            st.markdown("### Field Boundary")
            
            # Create map centered on the field
            m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
            
            # Add GeoJSON to map
            folium.GeoJson(
                geojson_data,
                name="Field Boundary",
                style_function=lambda x: {
                    'fillColor': '#28a745',
                    'color': '#28a745',
                    'weight': 2,
                    'fillOpacity': 0.4
                }
            ).add_to(m)
            
            # Add marker at center
            folium.Marker(
                [center_lat, center_lon],
                popup="Field Center",
                icon=folium.Icon(color="green", icon="leaf")
            ).add_to(m)
            
            # Display map
            folium_static(m, width=800, height=500)
            
            # Field details form
            st.markdown("### Field Details")
            
            with st.form("field_details_form"):
                field_name = st.text_input("Field Name", value=f"Field {datetime.datetime.now().strftime('%Y%m%d')}")
                crop_type = st.selectbox("Crop Type", options=[""] + CROP_TYPES)
                
                # Display calculated area and coordinates
                st.markdown(f"**Calculated Area:** {area_hectares:.2f} hectares")
                st.markdown(f"**Center Coordinates:** Lat: {center_lat:.6f}, Lon: {center_lon:.6f}")
                
                # Additional notes
                notes = st.text_area("Notes", placeholder="Enter any additional information about this field")
                
                # Submit button
                submit_button = st.form_submit_button("Save Field")
                
                if submit_button:
                    try:
                        # Create database session
                        db = next(get_db())
                        
                        # Check if field name already exists
                        existing_field = db.query(Field).filter(Field.name == field_name).first()
                        
                        if existing_field:
                            st.error(f"Field name '{field_name}' already exists. Please choose a different name.")
                        else:
                            # Create new field
                            new_field = Field(
                                name=field_name,
                                geojson=json.dumps(geojson_data),
                                center_lat=float(center_lat),
                                center_lon=float(center_lon),
                                area_hectares=float(area_hectares),
                                crop_type=crop_type if crop_type else None
                            )
                            
                            # Add and commit to database
                            db.add(new_field)
                            db.commit()
                            
                            st.success(f"Field '{field_name}' has been saved successfully!")
                            st.balloons()
                            
                    except Exception as e:
                        st.error(f"Error saving field: {str(e)}")
                        logger.error(f"Error saving field: {traceback.format_exc()}")
                        
        except Exception as e:
            st.error(f"Error processing GeoJSON file: {str(e)}")
            logger.error(f"Error processing GeoJSON file: {traceback.format_exc()}")

elif data_option == "Draw on Map" and manage_option == "Add New Field":
    st.markdown("## Wybierz obszar do analizy")
    
    # Inicjalizacja stanu sesji
    if "drawn_features" not in st.session_state:
        st.session_state.drawn_features = None
    
    if 'drawn_polygon' not in st.session_state:
        st.session_state.drawn_polygon = None
    
    # Dodaj wyszukiwarkę lokalizacji
    from geopy.geocoders import Nominatim
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        location_name = st.text_input("Wyszukaj lokalizację (np. 'Brazil', 'Rio de Janeiro', 'Sao Paulo')", 
                                      value="Brazil" if "location_name" not in st.session_state else st.session_state.get("location_name", ""))
    
    with col2:
        search_button = st.button("Wyszukaj", use_container_width=True)
    
    if search_button or ("location_coords" not in st.session_state and location_name):
        try:
            # Geocoding
            geolocator = Nominatim(user_agent="agro-insight")
            location = geolocator.geocode(location_name)
            
            if location:
                lat, lon = location.latitude, location.longitude
                st.session_state.location_coords = (lat, lon)
                st.session_state.location_name = location_name
                st.session_state.location_info = location
                st.success(f"Znaleziono lokalizację: {location}")
            else:
                st.error(f"Nie znaleziono lokalizacji '{location_name}'")
                # Domyślne współrzędne (Brazylia)
                lat, lon = -10.3333, -53.2000
                st.session_state.location_coords = (lat, lon)
        except Exception as e:
            st.error(f"Błąd podczas wyszukiwania lokalizacji: {str(e)}")
            # Domyślne współrzędne (Brazylia)
            lat, lon = -10.3333, -53.2000
            st.session_state.location_coords = (lat, lon)
    elif "location_coords" in st.session_state:
        lat, lon = st.session_state.location_coords
    else:
        # Domyślne współrzędne (Brazylia)
        lat, lon = -10.3333, -53.2000
        st.session_state.location_coords = (lat, lon)
    
    # Wybór zoomu mapy
    zoom_level = st.slider("Poziom przybliżenia mapy", min_value=3, max_value=18, value=5)
    
    # Tworzenie mapy z narzędziami do rysowania
    m = folium.Map(location=[lat, lon], zoom_start=zoom_level)
    
    # Dodaj kontrolkę rysowania
    draw = Draw(
        export=True,
        position='topleft',
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={
            'featureGroup': None
        }
    )
    draw.add_to(m)
    
    # Dodaj informacje o regionie jeśli są dostępne
    if "location_info" in st.session_state and hasattr(st.session_state.location_info, "raw"):
        if "boundingbox" in st.session_state.location_info.raw:
            bbox = st.session_state.location_info.raw["boundingbox"]
            # Format [min_lat, max_lat, min_lon, max_lon]
            if len(bbox) == 4:
                south, north, west, east = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                
                # Rysowanie granic regionu
                folium.Rectangle(
                    bounds=[[south, west], [north, east]],
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.1,
                    weight=2
                ).add_to(m)
    
    # Dodaj instrukcje
    st.markdown("""
    ### Instrukcje:
    1. Wyszukaj interesującą lokalizację (np. "Brazil", "Amazonia", "Rio Grande do Sul")
    2. Dostosuj poziom przybliżenia mapy przy użyciu suwaka
    3. Możesz narysować konkretny obszar używając narzędzi rysowania po lewej stronie mapy (opcjonalnie)
    4. Kontynuuj, aby zapisać wybrany obszar i rozpocząć analizę
    """)
    
    # Wyświetl mapę
    folium_static(m, width=800, height=500)
    
    # Uproszczone wybieranie typu pola bez konieczności rysowania
    st.markdown("## Zapisz wybrany obszar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        field_name = st.text_input("Nazwa obszaru", value=location_name if "location_name" in st.session_state else "Nowy obszar")
    
    with col2:
        crop_type = st.selectbox("Typ uprawy", options=[""] + CROP_TYPES)
    
    # Przycisk do zapisania bez konieczności rysowania kształtu
    save_region_button = st.button("Zapisz obszar i rozpocznij analizę", use_container_width=True)
    
    if save_region_button:
        # Tworzymy polygon na podstawie granic wyszukanego regionu
        if "location_info" in st.session_state and hasattr(st.session_state.location_info, "raw"):
            if "boundingbox" in st.session_state.location_info.raw:
                try:
                    # Format [min_lat, max_lat, min_lon, max_lon]
                    bbox = st.session_state.location_info.raw["boundingbox"]
                    south, north, west, east = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    
                    # Tworzymy prostokątny polygon z granic
                    coords = [
                        [west, south],
                        [east, south],
                        [east, north],
                        [west, north],
                        [west, south]
                    ]
                    
                    # Tworzenie GeoJSON
                    geojson_data = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coords]
                        },
                        "properties": {}
                    }
                    
                    # Zapisz do stanu sesji
                    st.session_state.drawn_polygon = geojson_data
                    st.session_state.drawn_features = geojson_data
                    
                    # Oblicz środek i powierzchnię
                    center_lat = (south + north) / 2
                    center_lon = (west + east) / 2
                    
                    # Przybliżona powierzchnia (w stopniach kwadratowych)
                    from shapely.geometry import Polygon
                    import pyproj
                    
                    polygon = Polygon(coords)
                    geod = pyproj.Geod(ellps="WGS84")
                    area_m2 = abs(geod.geometry_area_perimeter(polygon)[0])
                    area_hectares = area_m2 / 10000  # Konwersja na hektary
                    
                    # Zapis do bazy danych
                    try:
                        # Create database session
                        db = next(get_db())
                        
                        # Sprawdźmy, czy pole o tej nazwie już istnieje
                        existing_field = db.query(Field).filter(Field.name == field_name).first()
                        
                        if existing_field:
                            # Aktualizuj istniejące pole
                            existing_field.geojson = geojson_data
                            existing_field.center_lat = center_lat
                            existing_field.center_lon = center_lon
                            existing_field.area_hectares = area_hectares
                            existing_field.crop_type = crop_type if crop_type else None
                            db.commit()
                            st.success(f"Zaktualizowano obszar '{field_name}'! Możesz teraz przejść do analizy.")
                        else:
                            # Utwórz nowe pole
                            new_field = Field(
                                name=field_name,
                                geojson=geojson_data,
                                center_lat=center_lat,
                                center_lon=center_lon,
                                area_hectares=area_hectares,
                                crop_type=crop_type if crop_type else None
                            )
                            
                            # Dodaj i zatwierdź w bazie danych
                            db.add(new_field)
                            db.commit()
                            
                            st.success(f"Obszar '{field_name}' został zapisany! Przejdź do Field Analysis, aby zobaczyć dane.")
                            st.balloons()
                    except Exception as e:
                        st.error(f"Błąd podczas zapisywania obszaru: {str(e)}")
                        logger.error(f"Błąd podczas zapisywania obszaru: {traceback.format_exc()}")
                except Exception as e:
                    st.error(f"Błąd podczas przetwarzania granic regionu: {str(e)}")
            else:
                st.warning("Nie udało się określić granic regionu. Spróbuj wyszukać bardziej precyzyjną lokalizację.")
        else:
            st.warning("Brak informacji o regionie. Wyszukaj lokalizację przed zapisaniem.")
            
    # Wyjaśnienie dla użytkownika
    with st.expander("Dlaczego nie muszę rysować obszaru?"):
        st.markdown("""
        Aplikacja automatycznie używa granic geograficznych wyszukanego regionu (np. Brazylii, stanu, miasta) 
        jako obszaru analizy. Jeśli potrzebujesz dokładniejszego obszaru, możesz użyć narzędzi rysowania 
        na mapie przed zapisaniem.
        """)
            
    # STARA IMPLEMENTACJA RYSOWANIA - Schowana w expander
    with st.expander("Zaawansowane opcje rysowania (opcjonalne)"):
        # Dostępne akcje rysowania
        draw_actions = ["polygon", "rectangle", "circle", "marker", "clear"]
        selected_action = st.selectbox("Wybierz narzędzie rysowania", options=draw_actions)
        
        draw_button = st.button("Symuluj narysowanie pola")
        
        # Przetwarzanie narysowanych obiektów
        if draw_button:
            # W tej demonstracji generujemy przykładowy obiekt GeoJSON jako zastępstwo
            # ponieważ rzeczywiste przechwytywanie rysowania wymaga integracji JS z Streamlit
            
            # Użyj środka wyszukanego regionu jako podstawy dla narysowanego kształtu
            lat, lon = st.session_state.location_coords
            
            if selected_action in ["polygon", "rectangle"]:
                # Tworzymy prostokątny polygon wokół centrum
                delta = 0.05  # Około 5km
                st.session_state.drawn_polygon = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [lon - delta, lat - delta],
                            [lon + delta, lat - delta],
                            [lon + delta, lat + delta],
                            [lon - delta, lat + delta],
                            [lon - delta, lat - delta]
                        ]]
                    },
                    "properties": {}
                }
                st.success("Pole zostało narysowane! Możesz kontynuować poniżej.")
            elif selected_action == "clear":
                st.session_state.drawn_polygon = None
                st.info("Usunięto narysowane pole.")
            else:
                st.warning(f"Akcja '{selected_action}' nie jest obecnie obsługiwana w tym demo.")
    
    # Sprawdź czy mamy narysowany polygon
    if st.session_state.drawn_polygon:
        # Generuj przykładowy GeoJSON dla narysowanego kształtu
        # W prawdziwej implementacji dane byłyby przekazane z mapy
        
        # Wykorzystaj współrzędne z narysowanego poligonu
        coords = st.session_state.drawn_polygon["geometry"]["coordinates"][0]
        
        # Oblicz środek poligonu (średnia współrzędnych)
        center_lon = sum(coord[0] for coord in coords) / len(coords)
        center_lat = sum(coord[1] for coord in coords) / len(coords)
        
        # Oblicz przybliżoną powierzchnię
        from shapely.geometry import Polygon
        import pyproj
        from functools import partial
        from shapely.ops import transform
        
        polygon = Polygon(coords)
        geod = pyproj.Geod(ellps="WGS84")
        area_m2 = abs(geod.geometry_area_perimeter(polygon)[0])
        area_hectares = area_m2 / 10000  # Konwersja na hektary
        # Tworzymy GeoJSON na podstawie narysowanego poligonu
        geojson_data = st.session_state.drawn_polygon
            
        # Zapisz do stanu sesji
        st.session_state.drawn_features = geojson_data
        st.success("Field boundary drawn successfully! Fill in the details below to save.")
    
    # Wyświetl formularz tylko wtedy, gdy użytkownik narysował kształt
    if st.session_state.drawn_features:
        # Ekstrakcja danych z narysowanego obiektu
        try:
            geojson_data = st.session_state.drawn_features
            
            # Parse GeoJSON and extract polygon
            polygon, crs = parse_geojson(geojson_data)
            bbox = get_bbox_from_polygon(polygon)
            
            # Calculate center and area
            centroid = polygon.centroid
            center_lat, center_lon = centroid.y, centroid.x
            
            # Convert area to hectares (assuming coordinates are in degrees)
            area_m2 = polygon.area * 111000 * 111000  # Approximate conversion from degrees to meters
            area_hectares = area_m2 / 10000  # Convert m² to hectares
            
            # Field details form
            st.markdown("### Field Details")
            
            with st.form("drawn_field_details_form"):
                field_name = st.text_input("Field Name", value=f"Field {datetime.datetime.now().strftime('%Y%m%d')}")
                crop_type = st.selectbox("Crop Type", options=[""] + CROP_TYPES)
                
                # Display calculated area and coordinates
                st.markdown(f"**Calculated Area:** {area_hectares:.2f} hectares")
                st.markdown(f"**Center Coordinates:** Lat: {center_lat:.6f}, Lon: {center_lon:.6f}")
                
                # Additional notes
                notes = st.text_area("Notes", placeholder="Enter any additional information about this field")
                
                # Submit button
                submit_button = st.form_submit_button("Save Field")
                
                if submit_button:
                    try:
                        # Create database session
                        db = next(get_db())
                        
                        # Check if field name already exists
                        existing_field = db.query(Field).filter(Field.name == field_name).first()
                        
                        if existing_field:
                            st.error(f"Field name '{field_name}' already exists. Please choose a different name.")
                        else:
                            # Create new field
                            new_field = Field(
                                name=field_name,
                                geojson=json.dumps(geojson_data),
                                center_lat=float(center_lat),
                                center_lon=float(center_lon),
                                area_hectares=float(area_hectares),
                                crop_type=crop_type if crop_type else None
                            )
                            
                            # Add and commit to database
                            db.add(new_field)
                            db.commit()
                            
                            # Clear the session state
                            st.session_state.drawn_features = None
                            
                            st.success(f"Field '{field_name}' has been saved successfully!")
                            st.balloons()
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Error saving field: {str(e)}")
                        logger.error(f"Error saving field: {traceback.format_exc()}")
                        
        except Exception as e:
            st.error(f"Error processing drawn boundary: {str(e)}")
            logger.error(f"Error processing drawn boundary: {traceback.format_exc()}")
    else:
        st.info("Draw a field boundary on the map using the drawing tools to continue.")
        
elif data_option == "Generate Test Data" and manage_option == "Add New Field":
    st.markdown("## Generate Test Field Data")
    st.info("This option generates mock field data for testing and demonstration purposes.")
    
    # Inicjalizacja stanu sesji dla wygenerowanych danych
    if "generated_field" not in st.session_state:
        st.session_state.generated_field = None
    
    # Formularz do generowania danych testowych
    with st.form("generate_test_data_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            field_name = st.text_input("Field Name", value=f"Test Field {datetime.datetime.now().strftime('%Y%m%d')}")
            crop_type = st.selectbox("Crop Type", options=CROP_TYPES)
            area_hectares = st.slider("Area (hectares)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
        
        with col2:
            lat = st.number_input("Center Latitude", value=50.0611, format="%.6f")
            lon = st.number_input("Center Longitude", value=19.9383, format="%.6f")
            include_imagery = st.checkbox("Include mock satellite imagery", value=True)
        
        generate_button = st.form_submit_button("Generate Test Field")
        
        if generate_button:
            try:
                # Generowanie granic pola
                geojson_data = get_mock_field_boundary(lat, lon, area_hectares)
                
                # Generowanie danych testowych
                field_data = generate_mock_field_data()
                
                # Dodanie geojson do danych
                field_data["geojson"] = geojson_data
                
                # Zapisz do sesji
                st.session_state.generated_field = {
                    "name": field_name,
                    "crop_type": crop_type,
                    "center_lat": lat,
                    "center_lon": lon,
                    "area_hectares": area_hectares,
                    "geojson": geojson_data,
                    "data": field_data
                }
                
                st.success("Test field data generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating test data: {str(e)}")
                logger.error(f"Error generating test data: {traceback.format_exc()}")
    
    # Wyświetl wygenerowane dane, jeśli dostępne
    if st.session_state.generated_field:
        field = st.session_state.generated_field
        
        st.markdown("### Generated Field Preview")
        
        # Pokaż mapę z granicami pola
        m = folium.Map(location=[field["center_lat"], field["center_lon"]], zoom_start=14)
        
        # Dodaj GeoJSON do mapy
        folium.GeoJson(
            field["geojson"],
            name="Field Boundary",
            style_function=lambda x: {
                'fillColor': '#28a745',
                'color': '#28a745',
                'weight': 2,
                'fillOpacity': 0.4
            }
        ).add_to(m)
        
        # Dodaj marker w środku
        folium.Marker(
            [field["center_lat"], field["center_lon"]],
            popup=field["name"],
            icon=folium.Icon(color="green", icon="leaf")
        ).add_to(m)
        
        # Wyświetl mapę
        folium_static(m, width=700, height=400)
        
        # Przyciski akcji
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save to Database", type="primary"):
                try:
                    # Create database session
                    db = next(get_db())
                    
                    # Check if field name already exists
                    existing_field = db.query(Field).filter(Field.name == field["name"]).first()
                    
                    if existing_field:
                        st.error(f"Field name '{field['name']}' already exists. Please choose a different name.")
                    else:
                        # Create new field
                        new_field = Field(
                            name=field["name"],
                            geojson=json.dumps(field["geojson"]),
                            center_lat=float(field["center_lat"]),
                            center_lon=float(field["center_lon"]),
                            area_hectares=float(field["area_hectares"]),
                            crop_type=field["crop_type"]
                        )
                        
                        # Add and commit to database
                        db.add(new_field)
                        db.commit()
                        
                        # Zapisz dane do pliku
                        field_dir = os.path.join("data", "mock")
                        os.makedirs(field_dir, exist_ok=True)
                        
                        with open(os.path.join(field_dir, f"{field['name'].replace(' ', '_').lower()}.json"), "w") as f:
                            json.dump(field["data"], f, indent=2)
                        
                        # Clear the session state
                        st.session_state.generated_field = None
                        
                        st.success(f"Field '{field['name']}' has been saved to database and mock data stored!")
                        st.balloons()
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error saving field: {str(e)}")
                    logger.error(f"Error saving field: {traceback.format_exc()}")
        
        with col2:
            if st.button("Regenerate Data", type="secondary"):
                # Zachowaj te same parametry, ale wygeneruj nowe dane
                field_data = generate_mock_field_data()
                geojson_data = get_mock_field_boundary(field["center_lat"], field["center_lon"], field["area_hectares"])
                
                # Aktualizuj sesję
                st.session_state.generated_field["data"] = field_data
                st.session_state.generated_field["geojson"] = geojson_data
                
                st.success("Test data regenerated successfully!")
                st.rerun()
                
        # Pokaż przykładowy wykres wygenerowanych danych
        if "ndvi_time_series" in field["data"]:
            st.markdown("### Sample NDVI Time Series")
            
            # Przekształć dane do wykresu
            ndvi_data = field["data"]["ndvi_time_series"]
            dates = list(ndvi_data.keys())
            values = list(ndvi_data.values())
            
            # Stwórz wykres
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(dates, values, marker='o', linestyle='-', color='green')
            ax.set_title(f"NDVI Time Series for {field['name']}")
            ax.set_xlabel("Date")
            ax.set_ylabel("NDVI")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)

elif data_option == "Import from Database" or manage_option in ["Edit Existing Field", "Delete Field"]:
    st.markdown("## Manage Existing Fields")
    
    # Get fields from database
    fields = []
    try:
        db = next(get_db())
        fields = db.query(Field).all()
    except Exception as e:
        st.error(f"Error fetching fields from database: {str(e)}")
    
    if fields:
        # Create a dataframe for display
        field_data = []
        for field in fields:
            field_data.append({
                "ID": field.id,
                "Name": field.name,
                "Crop Type": field.crop_type or "Not specified",
                "Area (ha)": round(field.area_hectares, 2),
                "Created": field.created_at.strftime('%Y-%m-%d')
            })
        
        st.dataframe(pd.DataFrame(field_data), use_container_width=True)
        
        # Field selection
        field_names = [field.name for field in fields]
        selected_field_name = st.selectbox("Select Field", options=field_names)
        
        # Get the selected field
        selected_field = next((field for field in fields if field.name == selected_field_name), None)
        
        if selected_field:
            # Display field details
            st.markdown("### Field Details")
            
            # Create two columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display map with field boundary
                try:
                    # Parse GeoJSON
                    geojson_data = json.loads(selected_field.geojson) if isinstance(selected_field.geojson, str) else selected_field.geojson
                    
                    # Create map centered on the field
                    m = folium.Map(location=[selected_field.center_lat, selected_field.center_lon], zoom_start=13)
                    
                    # Add GeoJSON to map
                    folium.GeoJson(
                        geojson_data,
                        name="Field Boundary",
                        style_function=lambda x: {
                            'fillColor': '#28a745',
                            'color': '#28a745',
                            'weight': 2,
                            'fillOpacity': 0.4
                        }
                    ).add_to(m)
                    
                    # Add marker at center
                    folium.Marker(
                        [selected_field.center_lat, selected_field.center_lon],
                        popup=selected_field.name,
                        icon=folium.Icon(color="green", icon="leaf")
                    ).add_to(m)
                    
                    # Display map
                    folium_static(m, width=600, height=400)
                    
                except Exception as e:
                    st.error(f"Error displaying field boundary: {str(e)}")
            
            with col2:
                # Display field information
                st.markdown(f"**Name:** {selected_field.name}")
                st.markdown(f"**Crop Type:** {selected_field.crop_type or 'Not specified'}")
                st.markdown(f"**Area:** {selected_field.area_hectares:.2f} hectares")
                st.markdown(f"**Center:** Lat: {selected_field.center_lat:.6f}, Lon: {selected_field.center_lon:.6f}")
                st.markdown(f"**Created:** {selected_field.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Actions based on manage_option
                if manage_option == "Edit Existing Field":
                    with st.form("edit_field_form"):
                        st.markdown("### Edit Field")
                        
                        new_name = st.text_input("Field Name", value=selected_field.name)
                        new_crop_type = st.selectbox(
                            "Crop Type", 
                            options=[""] + CROP_TYPES, 
                            index=0 if not selected_field.crop_type else CROP_TYPES.index(selected_field.crop_type) + 1
                        )
                        
                        update_button = st.form_submit_button("Update Field")
                        
                        if update_button:
                            try:
                                # Update field in database
                                db = next(get_db())
                                field_to_update = db.query(Field).filter(Field.id == selected_field.id).first()
                                
                                if field_to_update:
                                    # Check if the new name already exists for another field
                                    if new_name != selected_field.name:
                                        existing_field = db.query(Field).filter(Field.name == new_name).first()
                                        if existing_field:
                                            st.error(f"Field name '{new_name}' already exists. Please choose a different name.")
                                            st.stop()
                                    
                                    # Update field properties
                                    field_to_update.name = new_name
                                    field_to_update.crop_type = new_crop_type if new_crop_type else None
                                    
                                    # Commit changes
                                    db.commit()
                                    st.success(f"Field '{new_name}' has been updated successfully!")
                                    st.rerun()
                                else:
                                    st.error("Field not found in database.")
                                    
                            except Exception as e:
                                st.error(f"Error updating field: {str(e)}")
                                logger.error(f"Error updating field: {traceback.format_exc()}")
                
                elif manage_option == "Delete Field":
                    st.markdown("### Delete Field")
                    st.warning(f"Are you sure you want to delete the field '{selected_field.name}'? This action cannot be undone.")
                    
                    if st.button("Delete Field", type="primary"):
                        try:
                            # Delete field from database
                            db = next(get_db())
                            field_to_delete = db.query(Field).filter(Field.id == selected_field.id).first()
                            
                            if field_to_delete:
                                db.delete(field_to_delete)
                                db.commit()
                                st.success(f"Field '{selected_field.name}' has been deleted.")
                                st.rerun()
                            else:
                                st.error("Field not found in database.")
                                
                        except Exception as e:
                            st.error(f"Error deleting field: {str(e)}")
                            logger.error(f"Error deleting field: {traceback.format_exc()}")
    else:
        st.info("No fields found in the database. Use the 'Add New Field' option to create fields.")

# Help information
with st.expander("Help & Information"):
    st.markdown("""
    ## How to add a field
    
    ### Option 1: Upload GeoJSON
    1. Prepare a GeoJSON file with your field boundary
    2. Select "Upload GeoJSON" from the sidebar
    3. Upload your file and fill in the field details
    4. Click "Save Field" to store it in the database
    
    ### Option 2: Draw on Map (Coming Soon)
    In a future update, you'll be able to draw field boundaries directly on the map.
    
    ## GeoJSON Format
    GeoJSON files should contain a polygon or multipolygon geometry representing your field boundary.
    
    Example:
    ```json
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [longitude1, latitude1],
            [longitude2, latitude2],
            ...
            [longitude1, latitude1]
          ]
        ]
      }
    }
    ```
    
    ## Managing Fields
    - **Edit Field**: Change a field's name or crop type
    - **Delete Field**: Permanently remove a field from the database
    """)