import os
import streamlit as st
import pandas as pd
import json
import datetime
import logging
import asyncio
import uuid
import time
from shapely.geometry import shape, Polygon
from typing import Dict, List, Tuple, Optional
import folium
from streamlit_folium import folium_static
import plotly.express as px

from utils.data_access import (
    parse_geojson, 
    get_bbox_from_polygon, 
    get_available_scenes,
    fetch_sentinel_data,
    get_country_boundary,
    save_to_geotiff,
    save_stac_metadata
)
from utils.processing import (
    calculate_vegetation_indices,
    calculate_zonal_statistics
)
from utils.visualization import (
    plot_ndvi_map,
    create_folium_map,
    add_geojson_to_map
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Data Ingest - Agro Insight",
    page_icon="📥",
    layout="wide"
)

# Initialize session state for this page
if "field_geojson" not in st.session_state:
    st.session_state.field_geojson = None
if "field_polygon" not in st.session_state:
    st.session_state.field_polygon = None
if "field_name" not in st.session_state:
    st.session_state.field_name = ""
if "ingest_status" not in st.session_state:
    st.session_state.ingest_status = {}
if "available_fields" not in st.session_state:
    st.session_state.available_fields = []

# Header
st.title("📥 Data Ingest")
st.markdown("""
Import field boundaries and satellite data for analysis. You can:
1. Draw a polygon on the map
2. Upload a GeoJSON file
3. Enter a country code for country-level analysis
""")

# Main layout
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Field Selection")
    
    # Tab layout for different input methods
    tab1, tab2, tab3 = st.tabs(["Draw Polygon", "Upload GeoJSON", "Country Code"])
    
    with tab1:
        st.markdown("Draw a polygon on the map to define your field boundary.")
        
        # Create a Folium map for drawing
        m = folium.Map(location=[50.0, 10.0], zoom_start=4)
        
        # Add drawing controls
        folium.plugins.Draw(
            export=True,
            position='topleft',
            draw_options={
                'polyline': False,
                'rectangle': True,
                'circle': False,
                'circlemarker': False,
                'marker': False
            }
        ).add_to(m)
        
        # Display the map
        folium_static(m)
        
        # Button to capture drawn polygon
        if st.button("Use Drawn Polygon"):
            st.warning("This functionality requires JavaScript integration to capture the drawn polygon. Please use GeoJSON upload instead.")
    
    with tab2:
        uploaded_file = st.file_uploader("Upload GeoJSON file", type=["json", "geojson"])
        
        if uploaded_file:
            try:
                geojson_data = json.load(uploaded_file)
                field_polygon, _ = parse_geojson(geojson_data)
                
                # Display the polygon on a map
                bbox = field_polygon.bounds
                center_lat = (bbox[1] + bbox[3]) / 2
                center_lon = (bbox[0] + bbox[2]) / 2
                
                # Create a Folium map centered on the polygon
                m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                
                # Convert the polygon to GeoJSON format
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": field_polygon.__geo_interface__,
                            "properties": {"name": "Uploaded Field"}
                        }
                    ]
                }
                
                # Add the polygon to the map
                folium.GeoJson(
                    geojson_data,
                    style_function=lambda x: {
                        'fillColor': '#28a745',
                        'color': '#28a745',
                        'fillOpacity': 0.3,
                        'weight': 2
                    },
                    tooltip="Uploaded Field"
                ).add_to(m)
                
                # Display the map
                folium_static(m)
                
                # Save to session state
                st.session_state.field_geojson = geojson_data
                st.session_state.field_polygon = field_polygon
                
                # Input field for the field name
                field_name = st.text_input("Field Name", "Field_" + str(uuid.uuid4())[:8])
                st.session_state.field_name = field_name
                
                st.success("GeoJSON file loaded successfully.")
                
            except Exception as e:
                st.error(f"Error loading GeoJSON file: {str(e)}")
    
    with tab3:
        country_code = st.text_input("Country ISO Code (e.g., PL for Poland)")
        
        if st.button("Get Country Boundary"):
            if country_code:
                with st.spinner(f"Fetching boundary for {country_code}..."):
                    country_polygon = get_country_boundary(country_code)
                    
                    if country_polygon:
                        # Display the country polygon on a map
                        bbox = country_polygon.bounds
                        center_lat = (bbox[1] + bbox[3]) / 2
                        center_lon = (bbox[0] + bbox[2]) / 2
                        
                        # Create a Folium map centered on the country
                        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
                        
                        # Convert the polygon to GeoJSON format
                        geojson_data = {
                            "type": "FeatureCollection",
                            "features": [
                                {
                                    "type": "Feature",
                                    "geometry": country_polygon.__geo_interface__,
                                    "properties": {"name": f"Country: {country_code.upper()}"}
                                }
                            ]
                        }
                        
                        # Add the polygon to the map
                        folium.GeoJson(
                            geojson_data,
                            style_function=lambda x: {
                                'fillColor': '#3388ff',
                                'color': '#3388ff',
                                'fillOpacity': 0.2,
                                'weight': 2
                            },
                            tooltip=f"Country: {country_code.upper()}"
                        ).add_to(m)
                        
                        # Display the map
                        folium_static(m)
                        
                        # Save to session state
                        st.session_state.field_geojson = geojson_data
                        st.session_state.field_polygon = country_polygon
                        st.session_state.field_name = f"Country_{country_code.upper()}"
                        
                        st.success(f"Successfully loaded boundary for {country_code.upper()}")
                    else:
                        st.error(f"Could not find boundary for country code: {country_code}")
            else:
                st.error("Please enter a country ISO code.")

with col2:
    st.subheader("Data Acquisition Options")
    
    if st.session_state.field_polygon:
        # Time range selection
        st.markdown("##### Time Range")
        col_start, col_end = st.columns(2)
        
        with col_start:
            start_date = st.date_input(
                "Start Date",
                value=datetime.date.today() - datetime.timedelta(days=365),
                max_value=datetime.date.today()
            )
        
        with col_end:
            end_date = st.date_input(
                "End Date",
                value=datetime.date.today(),
                min_value=start_date,
                max_value=datetime.date.today()
            )
        
        # Cloud coverage filter
        cloud_coverage = st.slider(
            "Maximum Cloud Coverage (%)",
            min_value=0,
            max_value=100,
            value=20,
            step=5,
            help="Filter Sentinel-2 scenes by maximum cloud coverage percentage."
        )
        
        # Data selection
        st.markdown("##### Data Selection")
        
        indices_to_calculate = st.multiselect(
            "Vegetation Indices to Calculate",
            options=["NDVI", "EVI", "NDWI"],
            default=["NDVI", "EVI", "NDWI"],
            help="Select which vegetation indices to calculate from the satellite data."
        )
        
        # Start ingestion button
        if st.button("Start Data Ingestion"):
            if not st.session_state.field_polygon:
                st.error("Please select a field boundary first.")
            else:
                try:
                    # Convert polygon to Sentinel Hub BBox
                    bbox = get_bbox_from_polygon(st.session_state.field_polygon)
                    
                    # Get available scenes
                    with st.spinner("Finding available Sentinel-2 scenes..."):
                        scenes = get_available_scenes(
                            bbox,
                            datetime.datetime.combine(start_date, datetime.time.min),
                            datetime.datetime.combine(end_date, datetime.time.max),
                            max_cloud_coverage=float(cloud_coverage)
                        )
                    
                    if scenes:
                        st.success(f"Found {len(scenes)} scenes matching your criteria.")
                        
                        # Display sample scene IDs
                        sample_ids = [scene["id"] for scene in scenes[:3]]
                        st.markdown(f"Sample scene IDs: {', '.join(sample_ids)}")
                        
                        # Store ingestion information in session state
                        ingest_id = str(uuid.uuid4())
                        st.session_state.ingest_status[ingest_id] = {
                            "field_name": st.session_state.field_name,
                            "polygon": st.session_state.field_polygon,
                            "bbox": bbox,
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat(),
                            "cloud_coverage": cloud_coverage,
                            "indices": indices_to_calculate,
                            "scenes": scenes,
                            "status": "queued",
                            "processed_scenes": 0,
                            "total_scenes": len(scenes),
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        
                        # Update available fields list if this is a new field
                        if st.session_state.field_name not in st.session_state.available_fields:
                            st.session_state.available_fields.append(st.session_state.field_name)
                        
                        st.markdown(f"Ingestion ID: `{ingest_id}`")
                        st.info("Data ingestion has been queued. The process will start in the background.")
                        
                        # In a real application, this would be handled by a background task
                        # For this demo, we'll start it immediately and show progress
                        with st.spinner(f"Processing scene 1/{len(scenes)}..."):
                            # Fetch the first scene as a demo
                            if len(scenes) > 0:
                                first_scene = scenes[0]
                                time_interval = (
                                    first_scene.get("properties", {}).get("datetime", ""),
                                    first_scene.get("properties", {}).get("datetime", "")
                                )
                                
                                # Fetch data for the first scene
                                image, metadata = fetch_sentinel_data(
                                    bbox,
                                    time_interval,
                                    resolution=10
                                )
                                
                                # Calculate vegetation indices
                                indices = calculate_vegetation_indices(image)
                                
                                # Save data for the first scene
                                if "NDVI" in indices_to_calculate:
                                    ndvi_filename = f"{st.session_state.field_name}_NDVI_{first_scene['id']}"
                                    save_to_geotiff(indices["ndvi"], metadata, ndvi_filename)
                                    save_stac_metadata(metadata, ndvi_filename)
                                
                                # Update status
                                st.session_state.ingest_status[ingest_id]["processed_scenes"] = 1
                                st.session_state.ingest_status[ingest_id]["status"] = "processing"
                                
                                # Calculate some field statistics for demonstration
                                if "ndvi" in indices:
                                    ndvi_stats = calculate_zonal_statistics(
                                        indices["ndvi"],
                                        st.session_state.field_polygon
                                    )
                                    
                                    st.markdown("##### Sample NDVI Statistics")
                                    st.json(ndvi_stats)
                                    
                                    # Create a sample visualization
                                    fig = plot_ndvi_map(indices["ndvi"], f"NDVI Map - {st.session_state.field_name}")
                                    st.pyplot(fig)
                        
                        st.success(f"Processed 1/{len(scenes)} scenes. The remaining scenes would be processed in the background.")
                        st.session_state.ingest_status[ingest_id]["status"] = "partially_complete"
                    else:
                        st.warning("No scenes found matching your criteria. Try adjusting the date range or cloud coverage settings.")
                
                except Exception as e:
                    st.error(f"Error in data ingestion: {str(e)}")
                    logger.exception("Data ingestion error")
    else:
        st.info("Please select a field boundary using one of the input methods on the left.")

# Ingestion Status
st.markdown("---")
st.subheader("Ingestion Status")

if st.session_state.ingest_status:
    status_data = []
    
    for ingest_id, status in st.session_state.ingest_status.items():
        status_data.append({
            "Ingestion ID": ingest_id[:8] + "...",
            "Field Name": status["field_name"],
            "Scenes": f"{status['processed_scenes']}/{status['total_scenes']}",
            "Status": status["status"].replace("_", " ").title(),
            "Timestamp": status["timestamp"]
        })
    
    status_df = pd.DataFrame(status_data)
    st.dataframe(status_df, use_container_width=True)
else:
    st.info("No data ingestion tasks have been created yet.")

# Available Fields
st.markdown("---")
st.subheader("Available Fields")

if st.session_state.available_fields:
    for field_name in st.session_state.available_fields:
        st.markdown(f"- {field_name}")
else:
    st.info("No fields have been processed yet. Start by ingesting data for a field.")
