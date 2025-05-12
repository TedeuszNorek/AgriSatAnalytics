"""
Field Analysis - Analyze satellite data for fields
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
    PROCESSED_DATA_DIR,
    APP_NAME
)

# Configure page
st.set_page_config(
    page_title=f"{APP_NAME} - Field Analysis",
    page_icon="🛰️",
    layout="wide"
)

# Configure logging
logger = logging.getLogger(__name__)

# Header
st.markdown("# Field Analysis")
st.markdown("Analyze satellite data for agricultural fields")
st.markdown("---")

# Get fields from database
fields = []
try:
    db = next(get_db())
    fields = db.query(Field).all()
except Exception as e:
    st.error(f"Error fetching fields from database: {str(e)}")

if not fields:
    st.warning("No fields found in the database. Please add fields in the Data Ingest page.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## Analysis Options")
    
    # Field selection
    field_names = [field.name for field in fields]
    selected_field_name = st.selectbox("Select Field", options=field_names)
    
    # Get the selected field
    selected_field = next((field for field in fields if field.name == selected_field_name), None)
    
    if selected_field:
        # Analysis type
        analysis_type = st.radio(
            "Analysis Type",
            ["NDVI Analysis", "EVI Analysis", "Time Series Analysis", "Anomaly Detection"]
        )
        
        # Date range selection
        st.markdown("## Date Range")
        
        # Set default dates (past year)
        default_start_date = DEFAULT_START_DATE
        default_end_date = DEFAULT_END_DATE
        
        start_date = st.date_input(
            "Start Date",
            value=default_start_date,
            min_value=datetime.datetime(2015, 6, 23),  # Sentinel-2 launch date
            max_value=datetime.datetime.now()
        )
        
        end_date = st.date_input(
            "End Date",
            value=default_end_date,
            min_value=start_date,
            max_value=datetime.datetime.now()
        )
        
        # Cloud coverage option
        max_cloud_cover = st.slider(
            "Max Cloud Coverage (%)",
            min_value=0,
            max_value=100,
            value=int(DEFAULT_MAX_CLOUD_COVERAGE),
            step=5
        )
        
        # Run analysis button
        run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)
    
    st.markdown("## Help")
    st.markdown("""
    - **NDVI**: Normalized Difference Vegetation Index
    - **EVI**: Enhanced Vegetation Index
    - **Cloud Coverage**: Higher values include more cloudy images
    """)

# Main content
if selected_field:
    # Display field info
    st.markdown(f"## Field: {selected_field.name}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Crop Type:** {selected_field.crop_type or 'Not specified'}")
    
    with col2:
        st.markdown(f"**Area:** {selected_field.area_hectares:.2f} hectares")
    
    with col3:
        st.markdown(f"**Location:** Lat: {selected_field.center_lat:.6f}, Lon: {selected_field.center_lon:.6f}")
    
    # Display map
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
        
        # Display map
        folium_static(m, width=800, height=400)
        
    except Exception as e:
        st.error(f"Error displaying field boundary: {str(e)}")
    
    # Check if analysis should be run
    if run_analysis:
        # Display loading message
        with st.spinner(f"Running {analysis_type} for {selected_field.name}..."):
            try:
                # Check if Sentinel Hub credentials are valid
                if not SENTINEL_HUB_CLIENT_ID or not SENTINEL_HUB_CLIENT_SECRET:
                    st.error("Sentinel Hub credentials are not set. Please set them in the environment variables.")
                    st.stop()
                
                # Convert dates to datetime
                start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
                end_datetime = datetime.datetime.combine(end_date, datetime.time.max)
                
                # Format time interval for Sentinel Hub
                time_interval = (start_datetime.strftime("%Y-%m-%dT00:00:00Z"), end_datetime.strftime("%Y-%m-%dT23:59:59Z"))
                
                # Get bounding box from field GeoJSON
                polygon, crs = parse_geojson(selected_field.geojson if isinstance(selected_field.geojson, str) else json.dumps(selected_field.geojson))
                bbox = get_bbox_from_polygon(polygon)
                
                # Fetch satellite data
                st.info("Fetching satellite data from Sentinel Hub API. This may take a moment...")
                
                try:
                    # Fetch data from Sentinel Hub
                    sat_data, metadata = fetch_sentinel_data(
                        bbox=bbox,
                        time_interval=time_interval,
                        resolution=10  # 10m resolution
                    )
                    
                    if sat_data is None:
                        st.error("Failed to fetch satellite data. Check logs for details.")
                        st.stop()
                    
                    # Process based on analysis type
                    if analysis_type == "NDVI Analysis":
                        # Extract NDVI from satellite data
                        ndvi_image = sat_data.get("ndvi")
                        rgb_image = sat_data.get("rgb")
                        scl_image = sat_data.get("scl")
                        
                        if ndvi_image is None:
                            st.error("No NDVI data found in the satellite imagery.")
                            st.stop()
                        
                        # Apply cloud masking
                        if scl_image is not None:
                            ndvi_masked = apply_cloud_mask(ndvi_image, scl_image)
                        else:
                            ndvi_masked = ndvi_image
                        
                        # Calculate statistics
                        ndvi_stats = calculate_zonal_statistics(ndvi_masked)
                        
                        # Save processed data
                        save_processed_data(
                            field_name=selected_field.name,
                            data_type="ndvi",
                            data=ndvi_masked,
                            metadata={
                                "stats": ndvi_stats,
                                "acquisition_date": metadata.get("timestamp"),
                                "cloud_coverage": metadata.get("cloud_coverage", 0)
                            }
                        )
                        
                        # Store in database
                        db = next(get_db())
                        
                        # Convert ndvi to geotiff and save
                        image_path = os.path.join(
                            PROCESSED_DATA_DIR, 
                            selected_field.name, 
                            f"ndvi_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tiff"
                        )
                        metadata_path = os.path.join(
                            PROCESSED_DATA_DIR, 
                            selected_field.name, 
                            f"ndvi_metadata_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        )
                        
                        save_to_geotiff(ndvi_masked, metadata, image_path)
                        save_stac_metadata(metadata, metadata_path)
                        
                        # Create satellite image record
                        satellite_image = SatelliteImage(
                            field_id=selected_field.id,
                            image_type="NDVI",
                            image_path=image_path,
                            metadata_path=metadata_path,
                            acquisition_date=datetime.datetime.fromisoformat(metadata.get("timestamp").split("+")[0]),
                            cloud_cover_percentage=metadata.get("cloud_coverage", 0),
                            scene_id=metadata.get("scene_id", "unknown"),
                            stats=ndvi_stats
                        )
                        
                        db.add(satellite_image)
                        db.commit()
                        
                        # Display results
                        st.markdown("## NDVI Analysis Results")
                        
                        # Create NDVI visualization
                        fig = create_index_map(
                            ndvi_masked,
                            index_type="NDVI",
                            title=f"NDVI - {selected_field.name} - {datetime.datetime.now().strftime('%Y-%m-%d')}"
                        )
                        
                        # Display map
                        st.pyplot(fig)
                        
                        # Display statistics
                        st.markdown("### NDVI Statistics")
                        
                        # Create columns for statistics
                        stat_cols = st.columns(5)
                        
                        with stat_cols[0]:
                            st.metric("Minimum", f"{ndvi_stats.get('min', 0):.3f}")
                        
                        with stat_cols[1]:
                            st.metric("Maximum", f"{ndvi_stats.get('max', 0):.3f}")
                        
                        with stat_cols[2]:
                            st.metric("Mean", f"{ndvi_stats.get('mean', 0):.3f}")
                        
                        with stat_cols[3]:
                            st.metric("Median", f"{ndvi_stats.get('median', 0):.3f}")
                        
                        with stat_cols[4]:
                            st.metric("Std Dev", f"{ndvi_stats.get('std', 0):.3f}")
                        
                        # Display histogram
                        hist_fig = create_histogram_figure(
                            ndvi_masked,
                            index_type="NDVI",
                            title=f"NDVI Distribution - {selected_field.name}"
                        )
                        
                        st.pyplot(hist_fig)
                        
                        # Mark analysis as complete
                        st.success("NDVI analysis completed successfully.")
                        st.balloons()
                    
                    elif analysis_type == "EVI Analysis":
                        # For now, just show a placeholder
                        st.markdown("## EVI Analysis Results")
                        st.info("EVI analysis functionality will be available in a future update.")
                    
                    elif analysis_type == "Time Series Analysis":
                        st.markdown("## Time Series Analysis")
                        st.info("Time series analysis functionality will be available in a future update.")
                        
                        # Placeholder for time series visualization
                        st.markdown("### Example Time Series Visualization")
                        
                        # Create a dummy time series for visualization
                        today = datetime.datetime.now()
                        dates = [(today - datetime.timedelta(days=i*30)) for i in range(12)]
                        
                        # Generate some dummy NDVI data (seasonal pattern with noise)
                        time_series_data = {}
                        for date in dates:
                            # Seasonal pattern (higher in summer, lower in winter)
                            # Assuming Northern Hemisphere
                            month = date.month
                            base_value = 0.4 + 0.3 * np.sin((month - 3) * np.pi / 6)  # Peak in July
                            
                            # Add some noise
                            noise = np.random.normal(0, 0.05)
                            time_series_data[date.isoformat()] = base_value + noise
                        
                        # Create visualization
                        ts_fig = create_multi_temporal_figure(
                            time_series_data,
                            title=f"NDVI Time Series - {selected_field.name}",
                            y_label="NDVI"
                        )
                        
                        st.pyplot(ts_fig)
                        
                    elif analysis_type == "Anomaly Detection":
                        st.markdown("## Anomaly Detection")
                        st.info("Anomaly detection functionality will be available in a future update.")
                        
                        # Placeholder for anomaly detection visualization
                        st.markdown("### Example Anomaly Detection Visualization")
                        
                        # Create a dummy time series with anomalies for visualization
                        today = datetime.datetime.now()
                        dates = [(today - datetime.timedelta(days=i*30)) for i in range(12)]
                        
                        # Generate some dummy NDVI data (seasonal pattern with noise)
                        time_series_data = {}
                        for date in dates:
                            # Seasonal pattern (higher in summer, lower in winter)
                            # Assuming Northern Hemisphere
                            month = date.month
                            base_value = 0.4 + 0.3 * np.sin((month - 3) * np.pi / 6)  # Peak in July
                            
                            # Add some noise
                            noise = np.random.normal(0, 0.05)
                            time_series_data[date.isoformat()] = base_value + noise
                        
                        # Add artificial anomalies
                        anomaly_dates = [dates[2], dates[7]]
                        for date in anomaly_dates:
                            if date.month >= 6 and date.month <= 8:  # Summer months
                                time_series_data[date.isoformat()] -= 0.3  # Low anomaly in summer
                            else:
                                time_series_data[date.isoformat()] += 0.3  # High anomaly in winter
                        
                        # Run anomaly detection
                        anomalies = detect_anomalies(time_series_data, method="zscore", threshold=2.0)
                        
                        # Create visualization
                        anomaly_fig = create_anomaly_figure(
                            time_series_data,
                            anomalies,
                            title=f"NDVI Anomalies - {selected_field.name}",
                            y_label="NDVI"
                        )
                        
                        st.pyplot(anomaly_fig)
                
                except Exception as e:
                    st.error(f"Error fetching or processing satellite data: {str(e)}")
                    logger.error(f"Error in satellite data processing: {traceback.format_exc()}")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                logger.error(f"Error during analysis: {traceback.format_exc()}")
    
    else:
        # Display instructions
        st.info("Select analysis options in the sidebar and click 'Run Analysis' to start.")
        
        # Display previous analysis if available
        st.markdown("## Previous Analysis")
        
        # Check if there are saved satellite images for this field
        try:
            db = next(get_db())
            satellite_images = db.query(SatelliteImage).filter(SatelliteImage.field_id == selected_field.id).order_by(SatelliteImage.acquisition_date.desc()).limit(5).all()
            
            if satellite_images:
                st.markdown(f"Found {len(satellite_images)} previous analyses for this field.")
                
                # Create tabs for each previous analysis
                tabs = st.tabs([f"{img.image_type} - {img.acquisition_date.strftime('%Y-%m-%d')}" for img in satellite_images])
                
                for i, (tab, img) in enumerate(zip(tabs, satellite_images)):
                    with tab:
                        st.markdown(f"### {img.image_type} Analysis - {img.acquisition_date.strftime('%Y-%m-%d')}")
                        
                        # Display statistics if available
                        if img.stats:
                            st.markdown("#### Statistics")
                            
                            stat_cols = st.columns(5)
                            
                            stats = img.stats
                            with stat_cols[0]:
                                st.metric("Minimum", f"{stats.get('min', 0):.3f}")
                            
                            with stat_cols[1]:
                                st.metric("Maximum", f"{stats.get('max', 0):.3f}")
                            
                            with stat_cols[2]:
                                st.metric("Mean", f"{stats.get('mean', 0):.3f}")
                            
                            with stat_cols[3]:
                                st.metric("Median", f"{stats.get('median', 0):.3f}")
                            
                            with stat_cols[4]:
                                st.metric("Std Dev", f"{stats.get('std', 0):.3f}")
                        
                        # Display image if available
                        if os.path.exists(img.image_path):
                            st.markdown("#### Visualization")
                            st.markdown(f"Image path: {img.image_path}")
                            
                            # For now, just indicate that the image is available
                            st.info("Image visualization will be available in a future update.")
                        else:
                            st.warning(f"Image file not found at {img.image_path}")
            else:
                st.info("No previous analyses found for this field.")
        
        except Exception as e:
            st.error(f"Error fetching previous analyses: {str(e)}")
            logger.error(f"Error fetching previous analyses: {traceback.format_exc()}")

else:
    st.warning("Please select a field from the sidebar.")

# Help information
with st.expander("Help & Information"):
    st.markdown("""
    ## Field Analysis Tools
    
    This page provides tools to analyze satellite data for your agricultural fields:
    
    - **NDVI Analysis**: Calculate vegetation health using the Normalized Difference Vegetation Index
    - **EVI Analysis**: Calculate Enhanced Vegetation Index, which is more sensitive to variations in dense vegetation
    - **Time Series Analysis**: Track changes in vegetation indices over time
    - **Anomaly Detection**: Identify unusual patterns or outliers in vegetation data
    
    ## Data Sources
    
    Analyses use Sentinel-2 satellite imagery, which provides:
    - High-resolution (10m/pixel) multispectral data
    - Regular revisit times (approximately every 5 days)
    - Free and open data access
    
    ## Interpreting Results
    
    - **NDVI Range**: -1.0 to 1.0, with healthy vegetation typically between 0.4 and 0.8
    - **Field Statistics**: Provides overall health metrics for your field
    - **Time Series**: Shows patterns and trends of vegetation health over time
    """)