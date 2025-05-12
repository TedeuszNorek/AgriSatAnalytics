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
tab1, tab2, tab3, tab4 = st.tabs(["Wyszukaj i dodaj obszar", "Zarządzaj obszarami", "Analiza danych satelitarnych", "Prognoza plonów"])

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

# Zakładka 4: Prognoza plonów
with tab4:
    st.header("Prognoza plonów")
    
    # Import from database.py
    from database import get_db, Field, YieldForecast, TimeSeries
    
    # Funkcja do tworzenia przykładowych danych pól
    def create_sample_fields():
        """Tworzy przykładowe dane pól do demonstracji"""
        field_attrs = dir(Field)
        
        sample_fields = []
        
        # Pole 1: Przykładowe pole pszenicy w Polsce
        field1 = Field()
        field1.id = 1
        field1.name = "Pole pszenicy - Mazowsze"
        field1.center_lat = 52.2297
        field1.center_lon = 21.0122
        field1.area_hectares = 15.5
        field1.crop_type = "Wheat"
        if 'geojson' in field_attrs:
            # Przykładowy GeoJSON dla prostokątnego pola
            field1.geojson = {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [21.0100, 52.2280],
                        [21.0145, 52.2280],
                        [21.0145, 52.2310],
                        [21.0100, 52.2310],
                        [21.0100, 52.2280]
                    ]]
                }
            }
        if 'created_at' in field_attrs:
            field1.created_at = datetime.datetime.now()
        
        # Pole 2: Przykładowe pole kukurydzy
        field2 = Field()
        field2.id = 2
        field2.name = "Pole kukurydzy - Wielkopolska"
        field2.center_lat = 52.4083
        field2.center_lon = 16.9335
        field2.area_hectares = 22.8
        field2.crop_type = "Corn"
        if 'geojson' in field_attrs:
            # Przykładowy GeoJSON dla prostokątnego pola
            field2.geojson = {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [16.9300, 52.4060],
                        [16.9370, 52.4060],
                        [16.9370, 52.4105],
                        [16.9300, 52.4105],
                        [16.9300, 52.4060]
                    ]]
                }
            }
        if 'created_at' in field_attrs:
            field2.created_at = datetime.datetime.now()
        
        sample_fields.append(field1)
        sample_fields.append(field2)
        
        return sample_fields
    
    # Pobierz pola z bazy danych
    fields = []
    try:
        db = next(get_db())
        fields = db.query(Field).all()
    except Exception as e:
        st.error(f"Błąd podczas pobierania pól z bazy danych: {str(e)}")
    
    if not fields:
        st.warning("Nie znaleziono pól w bazie danych. Używam przykładowych pól do demonstracji.")
        fields = create_sample_fields()
    
    # Główna zawartość - podzielona na kolumny dla opcji i wyników
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Opcje prognozy")
        
        # Wybór pola
        field_names = [field.name for field in fields]
        selected_field_name = st.selectbox("Wybierz pole", options=field_names, key="yield_forecast_field")
        
        # Pobierz wybrane pole
        selected_field = next((field for field in fields if field.name == selected_field_name), None)
        
        if selected_field:
            # Typ uprawy (użyj tego z pola, jeśli dostępny)
            default_crop_type = ""
            if selected_field.crop_type and selected_field.crop_type in CROP_TYPES:
                default_crop_type = selected_field.crop_type
                crop_index = CROP_TYPES.index(default_crop_type)
            else:
                crop_index = 0
            
            crop_type = st.selectbox(
                "Typ uprawy",
                options=[""] + CROP_TYPES,
                index=0 if not default_crop_type else crop_index + 1,
                key="yield_forecast_crop"
            )
            
            # Rodzaj prognozy
            forecast_type = st.radio(
                "Rodzaj prognozy",
                ["Plon końcowy sezonu", "Prognoza w czasie"],
                key="yield_forecast_type"
            )

            # Dodawanie określenia czasu - określ okres prognozy
            current_date = datetime.datetime.now().date()
            current_year = current_date.year
            
            forecast_period = st.radio(
                "Okres prognozy",
                [f"Krótkoterminowa (do {current_date + datetime.timedelta(days=30)})",
                 f"Średnioterminowa (do {current_date + datetime.timedelta(days=90)})",
                 f"Długoterminowa (do {current_date.replace(year=current_year+1)})"],
                key="yield_forecast_period"
            )
            
            # Pobierz liczbę dni na podstawie wybranego okresu
            if "Krótkoterminowa" in forecast_period:
                forecast_horizon = 30
            elif "Średnioterminowa" in forecast_period:
                forecast_horizon = 90
            else:
                # Dni do końca roku i dodatkowe 30 dni
                days_to_next_year = (datetime.date(current_year+1, 1, 1) - current_date).days
                forecast_horizon = days_to_next_year + 30
            
            # Opcje danych
            include_weather = st.checkbox("Uwzględnij dane pogodowe", value=True, key="yield_forecast_weather")
            include_market = st.checkbox("Uwzględnij dane rynkowe", value=True, key="yield_forecast_market")
            
            # Przycisk do uruchomienia prognozy
            run_forecast = st.button("Uruchom prognozę", type="primary", use_container_width=True, key="yield_forecast_run")
            
            st.markdown("### Pomoc")
            st.markdown("""
            - **Plon końcowy sezonu**: Przewiduje końcowy plon przy zbiorach
            - **Prognoza w czasie**: Przewiduje rozwój plonu w czasie
            - **Dane pogodowe**: Uwzględnia temperaturę, opady itp.
            - **Dane rynkowe**: Uwzględnia dane kontraktów terminowych na surowce
            """)
    
    with col2:
        # Wyświetl informacje o polu
        if selected_field:
            # Wyświetl informacje o polu
            st.markdown(f"### Pole: {selected_field.name}")
            
            info_cols = st.columns(3)
            
            with info_cols[0]:
                st.markdown(f"**Typ uprawy:** {selected_field.crop_type or 'Nie określono'}")
            
            with info_cols[1]:
                st.markdown(f"**Powierzchnia:** {selected_field.area_hectares:.2f} hektarów")
            
            with info_cols[2]:
                st.markdown(f"**Lokalizacja:** Lat: {selected_field.center_lat:.6f}, Lon: {selected_field.center_lon:.6f}")
            
            # Sprawdź, czy prognoza powinna zostać uruchomiona
            if run_forecast:
                # Zweryfikuj typ uprawy
                if not crop_type:
                    st.error("Wybierz typ uprawy, aby wygenerować prognozę plonu.")
                    st.stop()
                
                # Wyświetl komunikat o ładowaniu
                with st.spinner(f"Generowanie {forecast_type} dla {selected_field.name}..."):
                    try:
                        # Tymczasowe użycie danych przykładowych
                        
                        # Informacja o przykładowych danych
                        st.info(f"To jest demonstracja funkcji prognozy plonów z wykorzystaniem przykładowych danych. Okres prognozy: {forecast_horizon} dni od {current_date} do {current_date + datetime.timedelta(days=forecast_horizon)}.")
                        
                        # Utwórz wizualizację prognozy
                        st.markdown("## Wyniki prognozy plonów")
                        
                        if forecast_type == "Plon końcowy sezonu":
                            # Utwórz kolumny dla wyników prognozy
                            result_cols = st.columns([2, 1])
                            
                            with result_cols[0]:
                                # Utwórz przykładową prognozę do wizualizacji
                                
                                # Przykładowa prognoza plonu z przedziałem ufności
                                predicted_yield = 4.5 + np.random.normal(0, 0.2)  # Średni plon 4.5 t/ha z pewnym szumem
                                lower_bound = predicted_yield - 0.5
                                upper_bound = predicted_yield + 0.5
                                
                                # Wyświetl metryki
                                st.metric(
                                    "Przewidywany plon",
                                    f"{predicted_yield:.2f} t/ha",
                                    delta=f"{predicted_yield - 4.0:.2f} vs. zeszły rok"  # Zakładając, że zeszły rok to 4.0 t/ha
                                )
                                
                                st.markdown(f"**Przedział ufności:** {lower_bound:.2f} - {upper_bound:.2f} t/ha")
                                st.markdown(f"**Całkowity zbiór (szacowany):** {predicted_yield * selected_field.area_hectares:.2f} ton")
                                
                                # Utwórz prostą wizualizację prognozy
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                # Dane z poprzednich lat (przykładowe)
                                years = [2022, 2023, 2024, 2025]
                                yields = [3.8, 4.0, 4.2, predicted_yield]
                                
                                # Wykres historycznych i przewidywanych plonów
                                ax.bar(years[:-1], yields[:-1], color='blue', alpha=0.7, label='Historyczny plon')
                                ax.bar(years[-1], yields[-1], color='green', alpha=0.7, label='Przewidywany plon')
                                
                                # Dodaj słupki błędów dla prognozy
                                ax.errorbar(years[-1], yields[-1], yerr=[[predicted_yield - lower_bound], [upper_bound - predicted_yield]], 
                                           fmt='o', color='darkgreen', ecolor='darkgreen', capsize=5)
                                
                                # Dodaj etykiety i tytuł
                                ax.set_xlabel('Rok')
                                ax.set_ylabel('Plon (ton/hektar)')
                                ax.set_title(f'Prognoza plonu dla {selected_field.name} - {crop_type}')
                                
                                # Dodaj siatkę
                                ax.grid(True, linestyle='--', alpha=0.7)
                                
                                # Dodaj legendę
                                ax.legend()
                                
                                st.pyplot(fig)
                            
                            with result_cols[1]:
                                st.markdown("### Czynniki wpływające na plon")
                                
                                # Utwórz przykładowe znaczenie cech
                                features = ['NDVI (Lipiec)', 'Temperatura (Czerwiec)', 'Opady (Maj-Lipiec)', 
                                           'Wilgotność gleby (Czerwiec)', 'Poprzedni plon']
                                importance = [0.35, 0.25, 0.20, 0.15, 0.05]
                                
                                # Utwórz wykres znaczenia cech
                                fig, ax = plt.subplots(figsize=(8, 6))
                                
                                y_pos = np.arange(len(features))
                                ax.barh(y_pos, importance, align='center', color='green', alpha=0.7)
                                ax.set_yticks(y_pos)
                                ax.set_yticklabels(features)
                                ax.invert_yaxis()  # Etykiety czytane od góry do dołu
                                ax.set_xlabel('Względne znaczenie')
                                ax.set_title('Znaczenie cech')
                                
                                st.pyplot(fig)
                                
                                # Warunki pogodowe
                                st.markdown("### Warunki pogodowe")
                                st.markdown(f"Temperatura: **Powyżej średniej** (+1.2°C)")
                                st.markdown(f"Opady: **Poniżej średniej** (-15%)")
                                st.markdown(f"Stopniodni wzrostu: **720** (Normalne: 650)")
                                
                                if include_market:
                                    # Warunki rynkowe
                                    st.markdown("### Perspektywa rynkowa")
                                    st.markdown(f"Aktualna cena {crop_type}: **185 EUR/t**")
                                    st.markdown(f"Prognoza ceny (zbiory): **195 EUR/t**")
                                    st.markdown(f"Rekomendacja przechowywania: **Wstrzymaj**")
                        elif forecast_type == "Prognoza w czasie":
                            # Utwórz przykładową prognozę szeregów czasowych
                            today = datetime.datetime.now()
                            forecast_dates = [(today + datetime.timedelta(days=i*10)) for i in range(forecast_horizon // 10)]
                            
                            # Wygeneruj przykładowe dane prognozy z wzorcem sezonowym i rosnącą niepewnością
                            forecast_data = {}
                            
                            # Zacznij od bieżącego stanu (zakładając fazę wzrostu)
                            current_yield = 2.0  # t/ha
                            
                            # Wygeneruj prognozę z rosnącą niepewnością
                            mean_yields = []
                            lower_bounds = []
                            upper_bounds = []
                            
                            for i, date in enumerate(forecast_dates):
                                # Proste modelowanie wzrostu z pewnym szumem
                                days_from_start = i * 10
                                progress = min(1.0, days_from_start / 120)  # Zakładając 120 dni do pełnej dojrzałości
                                
                                # Logistyczny wzrost do finalnego plonu
                                max_yield = 4.5  # t/ha
                                mean_yield = current_yield + (max_yield - current_yield) * (1 / (1 + np.exp(-10 * (progress - 0.5))))
                                
                                # Zwiększająca się niepewność im dalej w przyszłość
                                uncertainty = 0.1 + 0.5 * (i / len(forecast_dates))
                                
                                mean_yields.append(mean_yield)
                                lower_bounds.append(mean_yield - uncertainty)
                                upper_bounds.append(mean_yield + uncertainty)
                            
                            # Utwórz wizualizację prognozy
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Przekonwertuj daty na formaty do wyświetlenia
                            date_strs = [date.strftime('%Y-%m-%d') for date in forecast_dates]
                            
                            # Wykres średniej prognozy
                            ax.plot(date_strs, mean_yields, 'o-', color='green', linewidth=2, label='Prognozowany plon')
                            
                            # Dodaj przedział ufności
                            ax.fill_between(date_strs, lower_bounds, upper_bounds, color='green', alpha=0.2, label='Przedział ufności')
                            
                            # Dodaj etykiety i tytuł
                            ax.set_xlabel('Data')
                            ax.set_ylabel('Prognozowany plon (t/ha)')
                            ax.set_title(f'Prognoza plonu w czasie dla {selected_field.name} - {crop_type}')
                            
                            # Obróć etykiety osi X dla lepszej czytelności
                            plt.xticks(rotation=45)
                            
                            # Dodaj siatkę
                            ax.grid(True, linestyle='--', alpha=0.7)
                            
                            # Dodaj legendę
                            ax.legend()
                            
                            # Dopasuj układ
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                            
                            # Dodaj kluczowe daty i fazy wzrostu
                            st.markdown("### Kluczowe daty i fazy wzrostu")
                            
                            # Przykładowe daty faz wzrostu
                            phase_data = []
                            
                            # Oblicz daty na podstawie bieżącej daty
                            emergence_date = today + datetime.timedelta(days=5)
                            heading_date = today + datetime.timedelta(days=45)
                            flowering_date = today + datetime.timedelta(days=70)
                            maturity_date = today + datetime.timedelta(days=110)
                            harvest_date = today + datetime.timedelta(days=130)
                            
                            phase_data.append({"Faza": "Wschody", "Data": emergence_date.strftime('%Y-%m-%d'), "Prognozowany plon": "N/A"})
                            phase_data.append({"Faza": "Kłoszenie", "Data": heading_date.strftime('%Y-%m-%d'), "Prognozowany plon": "1.2 t/ha"})
                            phase_data.append({"Faza": "Kwitnienie", "Data": flowering_date.strftime('%Y-%m-%d'), "Prognozowany plon": "2.5 t/ha"})
                            phase_data.append({"Faza": "Dojrzałość", "Data": maturity_date.strftime('%Y-%m-%d'), "Prognozowany plon": "4.3 t/ha"})
                            phase_data.append({"Faza": "Zbiór", "Data": harvest_date.strftime('%Y-%m-%d'), "Prognozowany plon": "4.5 t/ha"})
                            
                            # Wyświetl tabelę faz wzrostu
                            st.table(phase_data)
                            
                            # Dodaj rekomendacje
                            st.markdown("### Rekomendacje")
                            st.markdown(f"Optymalny okres zbiorów: **{maturity_date.strftime('%Y-%m-%d')} - {harvest_date.strftime('%Y-%m-%d')}**")
                            
                            if include_weather:
                                st.markdown("### Prognoza pogody")
                                st.markdown(f"Najbliższe 10 dni: **Umiarkowane temperatury, niskie opady**")
                                st.markdown(f"Długoterminowa (30 dni): **Temperatury powyżej średniej, opady w normie**")
                    
                    except Exception as e:
                        st.error(f"Błąd podczas generowania prognozy: {str(e)}")
                        logger.error(f"Error in yield forecast: {traceback.format_exc()}")
            else:
                # Wyświetl instrukcje, gdy nie wykonano prognozy
                st.markdown("## Prognoza plonów")
                st.markdown("""
                Wybierz parametry po lewej stronie i kliknij 'Uruchom prognozę', aby wygenerować prognozę plonów dla wybranego pola.
                
                Prognoza wykorzystuje:
                - Dane satelitarne (NDVI, EVI)
                - Dane pogodowe (temperatura, opady)
                - Historyczne plony
                - Dane rynkowe (opcjonalnie)
                
                Możesz wybrać prognozę końcową (plon przy zbiorach) lub prognozę w czasie (rozwój plonu).
                """)
                
                # Wyświetl przykładowy obraz NDVI, jeśli dostępny
                try:
                    # Sprawdź, czy istnieją dane satelitarne dla wybranego pola
                    satellite_images = db.query(SatelliteImage).filter(
                        SatelliteImage.field_id == selected_field.id,
                        SatelliteImage.image_type == "NDVI"
                    ).order_by(SatelliteImage.acquisition_date.desc()).first()
                    
                    if satellite_images:
                        st.markdown("### Ostatni obraz NDVI")
                        st.markdown(f"Data: {satellite_images.acquisition_date}")
                        
                        # Tutaj możemy dodać wyświetlanie obrazu, jeśli mamy sposób na jego pobranie
                except Exception as e:
                    pass  # Ciche obsłużenie błędu, jeśli nie można pobrać obrazów

# Automatyczne przełączanie między zakładkami
if 'active_tab' in st.session_state:
    active_tab_index = st.session_state.active_tab
    
    # Usunięcie klucza, aby nie zostać "uwięzionym" w jednej zakładce
    del st.session_state.active_tab
    
    st.rerun()