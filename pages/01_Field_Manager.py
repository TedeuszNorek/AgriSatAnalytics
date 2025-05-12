"""
Field Manager - Wyszukaj, dodaj i analizuj pola rolne
"""
import os
import json
import logging
import datetime
import traceback
from typing import Dict, List, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
from matplotlib.figure import Figure
from folium.plugins import Draw
from geopy.geocoders import Nominatim
import pyproj
from shapely.geometry import Polygon

from database import get_db, Field, SatelliteImage, TimeSeries
from utils.data_access import (
    get_sentinel_hub_config, 
    fetch_sentinel_data,
    get_bbox_from_polygon,
    parse_geojson,
    save_to_geotiff,
    save_stac_metadata
)
from utils.processing import (
    calculate_ndvi,
    calculate_evi,
    calculate_zonal_statistics,
    extract_time_series,
    detect_anomalies,
    apply_cloud_mask,
    save_processed_data
)
from utils.visualization import (
    create_index_map,
    create_multi_temporal_figure,
    create_anomaly_figure,
    create_histogram_figure,
    fig_to_base64
)
from config import (
    SENTINEL_HUB_CLIENT_ID,
    SENTINEL_HUB_CLIENT_SECRET,
    SATELLITE_INDICES,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DEFAULT_MAX_CLOUD_COVERAGE,
    CROP_TYPES
)

# Konfiguracja loggera
logger = logging.getLogger(__name__)

# Tytuł strony
st.set_page_config(page_title="Agro Insight - Field Manager", layout="wide")
st.title("Field Manager")
st.markdown("Wyszukaj, dodaj i analizuj obszary rolne")

# Główne zakładki
tab1, tab2, tab3 = st.tabs(["Wyszukaj i dodaj obszar", "Zarządzaj obszarami", "Analiza danych satelitarnych"])

# Zakładka 1: Wyszukiwanie i dodawanie obszaru
with tab1:
    st.header("Wyszukaj i zapisz nowy obszar")
    
    # Inicjalizacja stanu sesji
    if "drawn_features" not in st.session_state:
        st.session_state.drawn_features = None
    
    if 'drawn_polygon' not in st.session_state:
        st.session_state.drawn_polygon = None
    
    # Wyszukiwarka lokalizacji
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
    
    # Szczegóły obszaru
    st.subheader("Zapisz wybrany obszar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        field_name = st.text_input("Nazwa obszaru", value=location_name if "location_name" in st.session_state else "Nowy obszar")
    
    with col2:
        crop_type = st.selectbox("Typ uprawy", options=[""] + CROP_TYPES)
    
    # Przycisk do zapisania bez konieczności rysowania kształtu
    save_region_button = st.button("Zapisz obszar", use_container_width=True)
    
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
                            st.success(f"Zaktualizowano obszar '{field_name}'!")
                            
                            # Zachowaj ID pola do analizy
                            st.session_state.selected_field_id = existing_field.id
                            
                            # Automatyczne przełączenie na zakładkę analizy
                            st.session_state.active_tab = 2
                            st.rerun()
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
                            
                            # Zachowaj ID pola do analizy
                            st.session_state.selected_field_id = new_field.id
                            
                            st.success(f"Obszar '{field_name}' został zapisany!")
                            st.balloons()
                            
                            # Automatyczne przełączenie na zakładkę analizy
                            st.session_state.active_tab = 2
                            st.rerun()
                    except Exception as e:
                        st.error(f"Błąd podczas zapisywania obszaru: {str(e)}")
                        logger.error(f"Błąd podczas zapisywania obszaru: {traceback.format_exc()}")
                except Exception as e:
                    st.error(f"Błąd podczas przetwarzania granic regionu: {str(e)}")
            else:
                st.warning("Nie udało się określić granic regionu. Spróbuj wyszukać bardziej precyzyjną lokalizację.")
        else:
            st.warning("Brak informacji o regionie. Wyszukaj lokalizację przed zapisaniem.")

# Zakładka 2: Zarządzanie obszarami
with tab2:
    st.header("Zarządzaj zapisanymi obszarami")
    
    # Pobierz wszystkie pola z bazy danych
    try:
        db = next(get_db())
        all_fields = db.query(Field).all()
        
        if not all_fields:
            st.info("Brak zapisanych obszarów. Przejdź do zakładki 'Wyszukaj i dodaj obszar', aby dodać nowy.")
        else:
            # Wyświetl tabelę z polami
            field_data = []
            for field in all_fields:
                field_data.append({
                    "ID": field.id,
                    "Nazwa": field.name,
                    "Typ uprawy": field.crop_type or "Nie określono",
                    "Powierzchnia (ha)": f"{field.area_hectares:.2f}",
                    "Współrzędne": f"Lat: {field.center_lat:.4f}, Lon: {field.center_lon:.4f}",
                    "Data utworzenia": field.created_at.strftime("%Y-%m-%d")
                })
            
            field_df = pd.DataFrame(field_data)
            st.dataframe(field_df, use_container_width=True)
            
            # Wybór pola do zarządzania
            col1, col2 = st.columns(2)
            
            with col1:
                selected_field_id = st.selectbox(
                    "Wybierz obszar do zarządzania",
                    options=[field.id for field in all_fields],
                    format_func=lambda x: next((field.name for field in all_fields if field.id == x), "")
                )
            
            with col2:
                action = st.selectbox(
                    "Wybierz akcję",
                    options=["Wyświetl na mapie", "Analizuj", "Usuń"]
                )
            
            # Wykonaj wybraną akcję
            if st.button("Wykonaj akcję", use_container_width=True):
                selected_field = db.query(Field).filter(Field.id == selected_field_id).first()
                
                if action == "Wyświetl na mapie":
                    # Wyświetl mapę z wybranym polem
                    try:
                        geojson_data = json.loads(selected_field.geojson) if isinstance(selected_field.geojson, str) else selected_field.geojson
                        
                        m = folium.Map(location=[selected_field.center_lat, selected_field.center_lon], zoom_start=8)
                        
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
                        
                        folium_static(m, width=800, height=500)
                        
                    except Exception as e:
                        st.error(f"Błąd podczas wyświetlania obszaru: {str(e)}")
                
                elif action == "Analizuj":
                    # Zapisz ID pola do analizy i przejdź do zakładki analizy
                    st.session_state.selected_field_id = selected_field_id
                    st.session_state.active_tab = 2
                    st.rerun()
                
                elif action == "Usuń":
                    # Potwierdzenie usunięcia
                    if st.checkbox(f"Potwierdź usunięcie obszaru '{selected_field.name}'"):
                        try:
                            # Usuń powiązane dane
                            db.query(SatelliteImage).filter(SatelliteImage.field_id == selected_field.id).delete()
                            db.query(TimeSeries).filter(TimeSeries.field_id == selected_field.id).delete()
                            
                            # Usuń pole
                            db.delete(selected_field)
                            db.commit()
                            
                            st.success(f"Obszar '{selected_field.name}' został usunięty.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Błąd podczas usuwania obszaru: {str(e)}")
                            db.rollback()
    
    except Exception as e:
        st.error(f"Błąd podczas pobierania danych: {str(e)}")

# Zakładka 3: Analiza danych satelitarnych
with tab3:
    st.header("Analiza danych satelitarnych")
    
    # Pobierz wszystkie pola
    try:
        db = next(get_db())
        all_fields = db.query(Field).all()
        
        if not all_fields:
            st.info("Brak zapisanych obszarów do analizy. Przejdź do zakładki 'Wyszukaj i dodaj obszar', aby dodać nowy.")
        else:
            # Wybór pola do analizy
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Używaj ID zapisanego w session_state, jeśli jest dostępne
                default_field_id = st.session_state.get('selected_field_id', all_fields[0].id if all_fields else None)
                
                selected_field_id = st.selectbox(
                    "Wybierz obszar do analizy",
                    options=[field.id for field in all_fields],
                    index=[field.id for field in all_fields].index(default_field_id) if default_field_id in [field.id for field in all_fields] else 0,
                    format_func=lambda x: next((field.name for field in all_fields if field.id == x), "")
                )
            
            # Pobierz wybrane pole
            selected_field = db.query(Field).filter(Field.id == selected_field_id).first()
            
            # Parametry analizy
            st.markdown("### Parametry analizy")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Data początkowa
                start_date = st.date_input(
                    "Data początkowa",
                    value=datetime.datetime.strptime(DEFAULT_START_DATE, "%Y-%m-%d").date(),
                    min_value=datetime.datetime(2015, 1, 1).date(),
                    max_value=datetime.datetime.now().date()
                )
                
            with col2:
                # Data końcowa
                end_date = st.date_input(
                    "Data końcowa",
                    value=datetime.datetime.strptime(DEFAULT_END_DATE, "%Y-%m-%d").date(),
                    min_value=datetime.datetime(2015, 1, 1).date(),
                    max_value=datetime.datetime.now().date()
                )
                
            with col3:
                # Maksymalne zachmurzenie
                max_cloud_coverage = st.slider(
                    "Maksymalne zachmurzenie (%)",
                    min_value=0,
                    max_value=100,
                    value=DEFAULT_MAX_CLOUD_COVERAGE,
                    step=5
                )
            
            # Wybór indeksu satelitarnego
            satellite_index = st.selectbox(
                "Indeks satelitarny",
                options=list(SATELLITE_INDICES.keys()),
                format_func=lambda x: SATELLITE_INDICES[x]
            )
            
            # Przycisk do uruchomienia analizy
            run_analysis = st.button("Uruchom analizę", use_container_width=True)
            
            # Informacja o parametrach
            with st.expander("Więcej informacji o parametrach"):
                st.markdown("""
                ### Parametry analizy
                - **Data początkowa/końcowa**: Zakres dat dla danych satelitarnych
                - **Maks. zachmurzenie**: Filtruje obrazy z większym zachmurzeniem
                - **NDVI**: Normalized Difference Vegetation Index (indeks wegetacji)
                - **EVI**: Enhanced Vegetation Index (ulepszona wersja NDVI)
                """)
            
            # Wyświetl informacje o polu
            if selected_field:
                st.markdown(f"## Obszar: {selected_field.name}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**Typ uprawy:** {selected_field.crop_type or 'Nie określono'}")
                
                with col2:
                    st.markdown(f"**Powierzchnia:** {selected_field.area_hectares:.2f} hektarów")
                
                with col3:
                    st.markdown(f"**Lokalizacja:** Lat: {selected_field.center_lat:.6f}, Lon: {selected_field.center_lon:.6f}")
                
                # Wyświetl mapę
                try:
                    # Parse GeoJSON
                    geojson_data = json.loads(selected_field.geojson) if isinstance(selected_field.geojson, str) else selected_field.geojson
                    
                    # Create map centered on the field
                    m = folium.Map(location=[selected_field.center_lat, selected_field.center_lon], zoom_start=8)
                    
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
                    
                    # Display map
                    folium_static(m, width=800, height=400)
                    
                except Exception as e:
                    st.error(f"Błąd podczas wyświetlania granic obszaru: {str(e)}")
                
                # Uruchom analizę, jeśli przycisk został naciśnięty
                if run_analysis:
                    # Wyświetl komunikat o ładowaniu
                    with st.spinner(f"Analizuję dane satelitarne dla {selected_field.name}..."):
                        try:
                            # Get field boundary
                            geojson_data = json.loads(selected_field.geojson) if isinstance(selected_field.geojson, str) else selected_field.geojson
                            
                            # Parse GeoJSON
                            polygon, crs = parse_geojson(geojson_data)
                            
                            # Get bounding box
                            bbox = get_bbox_from_polygon(polygon)
                            
                            # Set up Sentinel Hub config
                            config = get_sentinel_hub_config(
                                client_id=SENTINEL_HUB_CLIENT_ID,
                                client_secret=SENTINEL_HUB_CLIENT_SECRET
                            )
                            
                            # Fetch satellite data
                            time_interval = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                            
                            satellite_data, metadata = fetch_sentinel_data(
                                bbox=bbox,
                                time_interval=time_interval,
                                config=config,
                                max_cloud_coverage=max_cloud_coverage / 100.0
                            )
                            
                            if satellite_data is None:
                                st.error("Nie udało się pobrać danych satelitarnych. Spróbuj zmienić parametry.")
                                st.warning("Typowe przyczyny błędu: zbyt mało zdjęć (za krótki okres) lub zbyt duże zachmurzenie.")
                            else:
                                # Process data
                                st.success("Dane satelitarne pobrane pomyślnie!")
                                
                                # Display preview of satellite images
                                st.subheader("Podgląd obrazów satelitarnych")
                                
                                # Create tabs for different visualizations
                                viz_tab1, viz_tab2, viz_tab3 = st.tabs(["RGB", "NDVI", "Analiza czasowa"])
                                
                                with viz_tab1:
                                    st.markdown("### Zdjęcie w naturalnych kolorach (RGB)")
                                    # Display RGB image
                                    if 'RGB' in satellite_data:
                                        rgb_img = satellite_data['RGB']
                                        
                                        # Normalize to 0-1 range for display
                                        rgb_normalized = rgb_img / np.max(rgb_img)
                                        
                                        # Create figure
                                        fig, ax = plt.subplots(figsize=(10, 8))
                                        ax.imshow(rgb_normalized)
                                        ax.set_title("RGB Image")
                                        ax.axis('off')
                                        
                                        st.pyplot(fig)
                                    else:
                                        st.info("Brak danych RGB.")
                                
                                with viz_tab2:
                                    st.markdown("### Indeks wegetacji (NDVI)")
                                    
                                    # Calculate NDVI if not already available
                                    if 'NDVI' in satellite_data:
                                        ndvi = satellite_data['NDVI']
                                    else:
                                        st.info("Obliczanie NDVI...")
                                        if 'B04' in satellite_data and 'B08' in satellite_data:
                                            red = satellite_data['B04']
                                            nir = satellite_data['B08']
                                            ndvi = calculate_ndvi(red, nir)
                                        else:
                                            st.warning("Brak danych do obliczenia NDVI.")
                                            ndvi = None
                                    
                                    # Display NDVI image
                                    if ndvi is not None:
                                        # Create NDVI visualization
                                        fig = create_index_map(
                                            ndvi,
                                            title="NDVI",
                                            colormap="RdYlGn",
                                            vmin=-1,
                                            vmax=1
                                        )
                                        
                                        st.pyplot(fig)
                                        
                                        # Show statistics
                                        stats = {
                                            "Min": float(np.nanmin(ndvi)),
                                            "Max": float(np.nanmax(ndvi)),
                                            "Mean": float(np.nanmean(ndvi)),
                                            "Median": float(np.nanmedian(ndvi))
                                        }
                                        
                                        st.markdown("**Statystyki NDVI:**")
                                        st.json(stats)
                                    else:
                                        st.info("Brak danych NDVI.")
                                
                                with viz_tab3:
                                    st.markdown("### Analiza czasowa")
                                    
                                    # Check if we have time series data in the database
                                    time_series_data = db.query(TimeSeries).filter(
                                        TimeSeries.field_id == selected_field.id,
                                        TimeSeries.series_type == satellite_index
                                    ).first()
                                    
                                    if time_series_data:
                                        # Display existing time series
                                        st.info("Wyświetlanie istniejących danych czasowych")
                                        
                                        # Parse JSON data
                                        ts_data = json.loads(time_series_data.data) if isinstance(time_series_data.data, str) else time_series_data.data
                                        
                                        # Convert to DataFrame
                                        dates = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in ts_data["dates"]]
                                        values = ts_data["values"]
                                        
                                        ts_df = pd.DataFrame({
                                            "Date": dates,
                                            satellite_index: values
                                        })
                                        
                                        # Plot time series
                                        fig, ax = plt.subplots(figsize=(12, 6))
                                        ax.plot(ts_df["Date"], ts_df[satellite_index], 'o-', linewidth=2)
                                        ax.set_title(f"{SATELLITE_INDICES[satellite_index]} Time Series")
                                        ax.set_xlabel("Date")
                                        ax.set_ylabel(satellite_index)
                                        ax.grid(True)
                                        fig.autofmt_xdate()
                                        
                                        st.pyplot(fig)
                                        
                                        # Display statistics
                                        st.markdown("**Statystyki czasowe:**")
                                        stats = {
                                            "Count": len(values),
                                            "Min": min(values),
                                            "Max": max(values),
                                            "Mean": sum(values) / len(values) if values else 0,
                                            "Start Date": dates[0].strftime("%Y-%m-%d") if dates else "N/A",
                                            "End Date": dates[-1].strftime("%Y-%m-%d") if dates else "N/A"
                                        }
                                        
                                        st.json(stats)
                                        
                                        # Show anomalies if available
                                        if time_series_data.anomalies:
                                            anomalies = json.loads(time_series_data.anomalies) if isinstance(time_series_data.anomalies, str) else time_series_data.anomalies
                                            
                                            if anomalies and "indices" in anomalies:
                                                st.markdown("**Wykryte anomalie:**")
                                                
                                                # Convert anomaly indices to dates
                                                anomaly_dates = [dates[i].strftime("%Y-%m-%d") for i in anomalies["indices"]]
                                                
                                                # Display anomalies
                                                for i, date in enumerate(anomaly_dates):
                                                    st.markdown(f"{i+1}. Anomalia wykryta: {date}")
                                    else:
                                        st.info("Brak danych czasowych. Uruchom analizę z dłuższym zakresem dat, aby wygenerować serie czasowe.")
                        
                        except Exception as e:
                            st.error(f"Błąd podczas analizy: {str(e)}")
                            logger.error(f"Error in analysis: {traceback.format_exc()}")
            else:
                st.warning("Wybierz obszar do analizy.")
    
    except Exception as e:
        st.error(f"Błąd podczas pobierania danych: {str(e)}")

# Automatyczne przełączanie między zakładkami
if 'active_tab' in st.session_state:
    active_tab_index = st.session_state.active_tab
    
    # Usunięcie klucza, aby nie zostać "uwięzionym" w jednej zakładce
    del st.session_state.active_tab
    
    st.rerun()