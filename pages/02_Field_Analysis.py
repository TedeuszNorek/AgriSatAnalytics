import os
import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
import logging
import asyncio
import uuid
import glob
from pathlib import Path
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import shape, Polygon, box
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_bounds

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
    calculate_zonal_statistics,
    detect_anomalies,
    detect_drought,
    calculate_rolling_variance,
    perform_pca_clustering
)
from utils.visualization import (
    plot_ndvi_map,
    plot_interactive_ndvi_map,
    plot_time_series,
    plot_drought_detection,
    create_folium_map,
    add_geojson_to_map,
    overlay_raster_on_map
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Field Analysis - Agro Insight",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state variables if not already set
if "selected_field" not in st.session_state:
    st.session_state.selected_field = None
if "available_fields" not in st.session_state:
    st.session_state.available_fields = []
if "field_geojson" not in st.session_state:
    st.session_state.field_geojson = None
if "field_statistics" not in st.session_state:
    st.session_state.field_statistics = {}
if "ndvi_time_series" not in st.session_state:
    st.session_state.ndvi_time_series = {}

# Helper function to load available fields
def load_available_fields():
    """Load available fields from processed data directory"""
    data_dir = Path("./data/geotiff")
    if not data_dir.exists():
        return []
    
    # Get unique field names from filenames
    field_names = set()
    for file in data_dir.glob("*.tif"):
        # Extract field name from filename (format: fieldname_index_sceneid.tif)
        parts = file.stem.split('_')
        if len(parts) >= 2:
            field_name = parts[0]
            field_names.add(field_name)
    
    return list(field_names)

# Helper function to load field data
def load_field_data(field_name):
    """Load all data for a specific field"""
    data_dir = Path("./data/geotiff")
    
    # Find all files for this field
    field_files = list(data_dir.glob(f"{field_name}_*.tif"))
    
    if not field_files:
        return None
    
    # Process each file to extract data and metadata
    field_data = []
    
    for file in field_files:
        try:
            # Extract index type and date from filename
            filename_parts = file.stem.split('_')
            if len(filename_parts) >= 3:
                index_type = filename_parts[1]  # e.g., NDVI, EVI
                scene_id = filename_parts[2]
                
                # Read raster data
                with rasterio.open(file) as src:
                    # Read first band
                    raster_data = src.read(1)
                    
                    # Get metadata
                    metadata = src.tags()
                    
                    # Try to extract date from metadata
                    date_str = None
                    if 'time_interval' in metadata:
                        try:
                            time_interval = json.loads(metadata['time_interval'].replace("'", '"'))
                            date_str = time_interval[0].split('T')[0]  # Extract just the date part
                        except:
                            # If parsing fails, use the scene ID or a default date
                            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
                    
                    # Calculate basic statistics
                    valid_data = raster_data[~np.isnan(raster_data)]
                    if len(valid_data) > 0:
                        stats = {
                            "min": float(np.nanmin(valid_data)),
                            "max": float(np.nanmax(valid_data)),
                            "mean": float(np.nanmean(valid_data)),
                            "std": float(np.nanstd(valid_data))
                        }
                    else:
                        stats = {"min": 0, "max": 0, "mean": 0, "std": 0}
                    
                    # Add to field data
                    field_data.append({
                        "filename": str(file),
                        "index_type": index_type,
                        "scene_id": scene_id,
                        "date": date_str,
                        "raster_data": raster_data,
                        "metadata": metadata,
                        "statistics": stats,
                        "bounds": src.bounds,
                        "transform": src.transform
                    })
        except Exception as e:
            logger.error(f"Error loading field data from {file}: {e}")
    
    # Sort by date
    field_data.sort(key=lambda x: x.get("date", ""))
    
    return field_data

def extract_time_series(field_data, index_type="NDVI"):
    """Extract time series data for a specific index type"""
    time_series = {}
    
    for data in field_data:
        if data["index_type"] == index_type and data["date"]:
            # Calculate mean value for the entire raster
            mean_value = data["statistics"]["mean"]
            time_series[data["date"]] = mean_value
    
    return time_series

# Header
st.title("🔍 Field Analysis")
st.markdown("""
Analyze satellite data for agricultural fields. Monitor vegetation indices, detect anomalies, and track field health over time.
""")

# Load available fields
available_fields = load_available_fields()
if not available_fields and "available_fields" in st.session_state:
    available_fields = st.session_state.available_fields

# Field selection
st.sidebar.header("Field Selection")
selected_field = st.sidebar.selectbox(
    "Select Field", 
    options=available_fields,
    index=0 if available_fields else None,
    help="Choose a field to analyze"
)

if selected_field:
    st.session_state.selected_field = selected_field
    
    # Load field data
    with st.spinner(f"Loading data for {selected_field}..."):
        field_data = load_field_data(selected_field)
    
    if field_data:
        # Extract time series data for different indices
        ndvi_time_series = extract_time_series(field_data, "NDVI")
        evi_time_series = extract_time_series(field_data, "EVI")
        ndwi_time_series = extract_time_series(field_data, "NDWI")
        
        # Save to session state for use in other pages
        st.session_state.ndvi_time_series = ndvi_time_series
        
        # Display field overview
        st.header(f"Field Overview: {selected_field}")
        
        # Create tabs for different analysis views
        tab1, tab2, tab3, tab4 = st.tabs(["Vegetation Indices", "Time Series Analysis", "Anomaly Detection", "Advanced Analytics"])
        
        with tab1:
            st.subheader("Latest Vegetation Indices")
            
            # Find the latest data for each index
            latest_ndvi = next((data for data in reversed(field_data) if data["index_type"] == "NDVI"), None)
            latest_evi = next((data for data in reversed(field_data) if data["index_type"] == "EVI"), None)
            latest_ndwi = next((data for data in reversed(field_data) if data["index_type"] == "NDWI"), None)
            
            # Display in three columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### NDVI")
                st.markdown("*Normalized Difference Vegetation Index*")
                if latest_ndvi:
                    # Display date
                    st.markdown(f"**Date:** {latest_ndvi['date']}")
                    
                    # Display statistics
                    stats = latest_ndvi["statistics"]
                    st.markdown(f"**Mean:** {stats['mean']:.3f}")
                    st.markdown(f"**Min:** {stats['min']:.3f}")
                    st.markdown(f"**Max:** {stats['max']:.3f}")
                    
                    # Display map
                    fig = plot_ndvi_map(latest_ndvi["raster_data"], "NDVI Map")
                    st.pyplot(fig)
            
            with col2:
                st.markdown("### EVI")
                st.markdown("*Enhanced Vegetation Index*")
                if latest_evi:
                    # Display date
                    st.markdown(f"**Date:** {latest_evi['date']}")
                    
                    # Display statistics
                    stats = latest_evi["statistics"]
                    st.markdown(f"**Mean:** {stats['mean']:.3f}")
                    st.markdown(f"**Min:** {stats['min']:.3f}")
                    st.markdown(f"**Max:** {stats['max']:.3f}")
                    
                    # Display map
                    fig = plot_ndvi_map(latest_evi["raster_data"], "EVI Map")
                    st.pyplot(fig)
            
            with col3:
                st.markdown("### NDWI")
                st.markdown("*Normalized Difference Water Index*")
                if latest_ndwi:
                    # Display date
                    st.markdown(f"**Date:** {latest_ndwi['date']}")
                    
                    # Display statistics
                    stats = latest_ndwi["statistics"]
                    st.markdown(f"**Mean:** {stats['mean']:.3f}")
                    st.markdown(f"**Min:** {stats['min']:.3f}")
                    st.markdown(f"**Max:** {stats['max']:.3f}")
                    
                    # Display map
                    fig = plot_ndvi_map(latest_ndwi["raster_data"], "NDWI Map")
                    st.pyplot(fig)
            
            # Display interactive map with latest data
            st.subheader("Interactive Field Map")
            
            if latest_ndvi:
                # Get bounds
                bounds = latest_ndvi["bounds"]
                bounds_list = [bounds.left, bounds.bottom, bounds.right, bounds.top]
                
                # Create base map
                center_lat = (bounds.bottom + bounds.top) / 2
                center_lon = (bounds.left + bounds.right) / 2
                m = create_folium_map(center_lat, center_lon, zoom=10)
                
                # Add NDVI overlay
                m = overlay_raster_on_map(
                    m, 
                    latest_ndvi["raster_data"], 
                    [bounds.left, bounds.bottom, bounds.right, bounds.top],
                    colormap="RdYlGn",
                    name="NDVI"
                )
                
                # Display the map
                folium_static(m)
                
                # Add download option for raw data
                st.download_button(
                    label="Download Latest NDVI Data (CSV)",
                    data=pd.DataFrame({
                        "date": latest_ndvi["date"],
                        "mean_ndvi": latest_ndvi["statistics"]["mean"],
                        "min_ndvi": latest_ndvi["statistics"]["min"],
                        "max_ndvi": latest_ndvi["statistics"]["max"]
                    }, index=[0]).to_csv(index=False),
                    file_name=f"{selected_field}_latest_ndvi.csv",
                    mime="text/csv"
                )
        
        with tab2:
            st.subheader("Vegetation Indices Time Series")
            
            # Time series plots
            if ndvi_time_series:
                # Convert time series to lists
                ndvi_dates = list(ndvi_time_series.keys())
                ndvi_values = list(ndvi_time_series.values())
                
                # Create time series plot
                fig = plot_time_series(
                    ndvi_dates,
                    ndvi_values,
                    title="NDVI Time Series",
                    y_label="NDVI Value"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate NDVI statistics
                if len(ndvi_values) > 0:
                    ndvi_stats = {
                        "mean": np.mean(ndvi_values),
                        "max": np.max(ndvi_values),
                        "min": np.min(ndvi_values),
                        "std": np.std(ndvi_values)
                    }
                    
                    st.markdown("### NDVI Statistics Over Time")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean NDVI", f"{ndvi_stats['mean']:.3f}")
                    col2.metric("Max NDVI", f"{ndvi_stats['max']:.3f}")
                    col3.metric("Min NDVI", f"{ndvi_stats['min']:.3f}")
                    col4.metric("NDVI Variability", f"{ndvi_stats['std']:.3f}")
            
            # Additional time series for EVI and NDWI if available
            col1, col2 = st.columns(2)
            
            with col1:
                if evi_time_series:
                    evi_dates = list(evi_time_series.keys())
                    evi_values = list(evi_time_series.values())
                    
                    fig = plot_time_series(
                        evi_dates,
                        evi_values,
                        title="EVI Time Series",
                        y_label="EVI Value"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if ndwi_time_series:
                    ndwi_dates = list(ndwi_time_series.keys())
                    ndwi_values = list(ndwi_time_series.values())
                    
                    fig = plot_time_series(
                        ndwi_dates,
                        ndwi_values,
                        title="NDWI Time Series",
                        y_label="NDWI Value"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Download option for time series data
            if ndvi_time_series:
                time_series_df = pd.DataFrame({
                    "date": ndvi_dates,
                    "ndvi": ndvi_values
                })
                
                if evi_time_series:
                    evi_df = pd.DataFrame({
                        "date": list(evi_time_series.keys()),
                        "evi": list(evi_time_series.values())
                    })
                    time_series_df = pd.merge(time_series_df, evi_df, on="date", how="outer")
                
                if ndwi_time_series:
                    ndwi_df = pd.DataFrame({
                        "date": list(ndwi_time_series.keys()),
                        "ndwi": list(ndwi_time_series.values())
                    })
                    time_series_df = pd.merge(time_series_df, ndwi_df, on="date", how="outer")
                
                st.download_button(
                    label="Download Time Series Data (CSV)",
                    data=time_series_df.to_csv(index=False),
                    file_name=f"{selected_field}_time_series.csv",
                    mime="text/csv"
                )
        
        with tab3:
            st.subheader("Anomaly Detection")
            
            # NDVI Anomaly Analysis
            if ndvi_time_series and len(ndvi_time_series) > 3:
                ndvi_dates = list(ndvi_time_series.keys())
                ndvi_values = list(ndvi_time_series.values())
                
                # Detect anomalies
                anomaly_threshold = st.slider(
                    "Anomaly Detection Threshold (σ)",
                    min_value=1.0,
                    max_value=3.0,
                    value=2.0,
                    step=0.1,
                    help="Z-score threshold for detecting anomalies in NDVI values"
                )
                
                anomalies = detect_anomalies(ndvi_values, threshold=anomaly_threshold)
                
                # Display anomalies
                if anomalies:
                    st.markdown(f"**Detected {len(anomalies)} anomalies in the NDVI time series.**")
                    
                    # Plot with anomalies
                    fig = plot_time_series(
                        ndvi_dates,
                        ndvi_values,
                        title="NDVI Time Series with Anomalies",
                        y_label="NDVI Value",
                        anomalies=anomalies
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # List anomalies
                    anomaly_data = []
                    for idx in anomalies:
                        anomaly_data.append({
                            "Date": ndvi_dates[idx],
                            "NDVI Value": ndvi_values[idx],
                            "Z-Score": abs((ndvi_values[idx] - np.mean(ndvi_values)) / np.std(ndvi_values))
                        })
                    
                    st.markdown("### Detected Anomalies")
                    st.dataframe(pd.DataFrame(anomaly_data))
                else:
                    st.success("No anomalies detected in the NDVI time series based on the current threshold.")
                
                # Drought Detection
                st.subheader("Drought Detection")
                st.markdown("""
                Drought detection analyzes consecutive NDVI drops of 0.15 or more, which can indicate water stress in crops.
                """)
                
                drought_events = detect_drought(ndvi_values, ndvi_dates)
                
                if drought_events:
                    st.warning(f"**Detected {len(drought_events)} potential drought periods!**")
                    
                    # Plot with drought periods
                    fig = plot_drought_detection(ndvi_dates, ndvi_values, drought_events)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # List drought events
                    drought_data = []
                    for event in drought_events:
                        drought_data.append({
                            "Start Date": event["start_date"],
                            "End Date": event["end_date"],
                            "Duration (periods)": event["duration"],
                            "Severity (NDVI drop)": event["severity"]
                        })
                    
                    st.markdown("### Detected Drought Events")
                    st.dataframe(pd.DataFrame(drought_data))
                else:
                    st.success("No drought periods detected in the NDVI time series.")
                    
                    # Show regular time series
                    fig = plot_drought_detection(ndvi_dates, ndvi_values, [])
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for anomaly detection. Please select a field with a longer time series.")
        
        with tab4:
            st.subheader("Advanced Analytics")
            
            # Rolling Variance Analysis
            if ndvi_time_series and len(ndvi_time_series) > 3:
                st.markdown("### Field Heterogeneity Analysis")
                st.markdown("""
                This analysis calculates rolling variance of NDVI values over time to detect potential issues with field consistency, 
                such as uneven germination, patchy growth, or irrigation problems.
                """)
                
                ndvi_dates = list(ndvi_time_series.keys())
                ndvi_values = list(ndvi_time_series.values())
                
                # Calculate rolling variance
                window_size = min(3, len(ndvi_values))
                rolling_var = calculate_rolling_variance(ndvi_values, window_size=window_size)
                
                # Plot rolling variance
                fig = go.Figure()
                
                # Add NDVI line
                fig.add_trace(go.Scatter(
                    x=ndvi_dates,
                    y=ndvi_values,
                    mode='lines+markers',
                    name='NDVI',
                    line=dict(color='green', width=2)
                ))
                
                # Add rolling variance line with secondary y-axis
                fig.add_trace(go.Scatter(
                    x=ndvi_dates,
                    y=rolling_var,
                    mode='lines+markers',
                    name='Rolling Variance',
                    line=dict(color='red', width=2),
                    yaxis="y2"
                ))
                
                # Set up layout with two y-axes
                fig.update_layout(
                    title='NDVI and Rolling Variance',
                    xaxis_title='Date',
                    yaxis=dict(
                        title='NDVI Value',
                        titlefont=dict(color='green'),
                        tickfont=dict(color='green')
                    ),
                    yaxis2=dict(
                        title='Variance',
                        titlefont=dict(color='red'),
                        tickfont=dict(color='red'),
                        anchor='x',
                        overlaying='y',
                        side='right'
                    ),
                    height=500,
                    width=800,
                    template='plotly_white',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                avg_variance = np.mean(rolling_var)
                latest_variance = rolling_var[-1] if rolling_var else 0
                
                st.markdown(f"**Average Field Variance:** {avg_variance:.4f}")
                st.markdown(f"**Latest Field Variance:** {latest_variance:.4f}")
                
                if latest_variance > avg_variance * 1.5:
                    st.warning("⚠️ Field shows increased heterogeneity. This may indicate uneven growth or other field issues.")
                elif latest_variance < avg_variance * 0.5:
                    st.success("✅ Field shows more uniform conditions than average.")
                else:
                    st.info("ℹ️ Field variance is within normal range for this field's history.")
                
                # Multi-temporal PCA analysis
                st.markdown("### Multi-Temporal Analysis")
                
                # Get data for PCA
                if len(field_data) >= 3:
                    st.markdown("""
                    This analysis uses Principal Component Analysis (PCA) and clustering to identify patterns in the field's
                    multi-temporal satellite data. This can help identify zones with similar growth patterns.
                    """)
                    
                    # Collect raster data
                    rasters = []
                    dates = []
                    
                    for data in field_data:
                        if data["index_type"] == "NDVI":
                            rasters.append(data["raster_data"])
                            dates.append(data["date"])
                    
                    if len(rasters) >= 3:
                        # Stack rasters
                        try:
                            # Reshape rasters for PCA
                            stacked_data = []
                            for raster in rasters:
                                flat_raster = raster.flatten()
                                stacked_data.append(flat_raster)
                            
                            # Convert to pixel-based matrix
                            stacked_data = np.array(stacked_data).T  # Shape: (pixels, timesteps)
                            
                            # Run PCA and clustering
                            n_clusters = st.selectbox("Number of Clusters", options=[2, 3, 4, 5], index=1)
                            
                            st.text("Computing PCA and clustering...")
                            pca_results, explained_variance, cluster_labels = perform_pca_clustering(
                                stacked_data, n_clusters=n_clusters
                            )
                            
                            if pca_results is not None:
                                # Reshape cluster results back to image
                                if cluster_labels is not None:
                                    cluster_image = cluster_labels.reshape(rasters[0].shape)
                                    
                                    # Plot cluster image
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    cmap = plt.cm.tab10
                                    im = ax.imshow(cluster_image, cmap=cmap, interpolation='nearest')
                                    cbar = plt.colorbar(im, ax=ax)
                                    cbar.set_label('Cluster')
                                    plt.title(f'Field Zones (PCA + {n_clusters} Clusters)')
                                    plt.axis('off')
                                    st.pyplot(fig)
                                    
                                    # Plot explained variance
                                    st.markdown(f"**Explained Variance:** PC1 ({explained_variance[0]:.1%}), PC2 ({explained_variance[1]:.1%})")
                                    
                                    # Create a DataFrame for downloading
                                    cluster_df = pd.DataFrame({
                                        'pixel_id': range(len(cluster_labels)),
                                        'cluster': cluster_labels
                                    })
                                    
                                    st.download_button(
                                        label="Download Cluster Data (CSV)",
                                        data=cluster_df.to_csv(index=False),
                                        file_name=f"{selected_field}_clusters.csv",
                                        mime="text/csv"
                                    )
                                    
                                    # Interpretations of clusters
                                    st.markdown("### Cluster Interpretation")
                                    st.markdown("""
                                    - Clusters represent areas in the field with similar growing patterns over time
                                    - Large differences between clusters may indicate soil variability or management zones
                                    - Areas with consistently low NDVI across time may need attention
                                    """)
                                    
                                    # Show interpretation based on cluster count
                                    unique_clusters = np.unique(cluster_labels)
                                    st.markdown(f"**Number of zones detected:** {len(unique_clusters)}")
                                    
                                    if len(unique_clusters) >= 3:
                                        st.warning("Multiple distinct zones detected. This field may benefit from zone-based management.")
                                    else:
                                        st.success("Field appears relatively uniform across growing seasons.")
                        except Exception as e:
                            st.error(f"Error in multi-temporal analysis: {str(e)}")
                            logger.exception("Multi-temporal analysis error")
                    else:
                        st.info("Not enough NDVI data for multi-temporal analysis. At least 3 dates are required.")
                else:
                    st.info("Not enough data for multi-temporal analysis. Please select a field with more images.")
            else:
                st.info("Not enough data for advanced analytics. Please select a field with a longer time series.")
    else:
        st.error(f"No data found for field '{selected_field}'. Please process data for this field in the Data Ingest section.")
else:
    st.info("""
    No fields available for analysis. Please go to the Data Ingest section to process field data first.
    
    You can:
    1. Draw a field boundary on the map
    2. Upload a GeoJSON file with field boundaries
    3. Select a country for country-level analysis
    """)
    
    # Display sample agricultural field image
    st.image("https://pixabay.com/get/g879b44be88f7b57d084cc1720ca16fd7c256f93e15c6d03affe5cf8e36c009d93abf1f23dd42863852eb153851fdb35af3cfc66d04d2878ab81d1192661dd77a_1280.jpg", 
             caption="Analyze agricultural fields with satellite data")

# Bottom-page links
st.markdown("---")
st.markdown("""
👈 Go to **Data Ingest** to process more fields

👉 Continue to **Yield Forecast** to predict crop production
""")
