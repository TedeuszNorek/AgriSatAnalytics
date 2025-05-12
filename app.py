"""
Agro Insight - Satellite data analytics for agriculture
"""
import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import matplotlib.pyplot as plt

from database import init_db, get_db, Field
from utils.data_access import get_sentinel_hub_config, check_sentinel_hub_credentials
from config import (
    SENTINEL_HUB_CLIENT_ID, 
    SENTINEL_HUB_CLIENT_SECRET,
    SATELLITE_INDICES,
    CROP_TYPES,
    APP_NAME,
    APP_VERSION,
    APP_DESCRIPTION
)

# Initialize database
init_db()

# Configure page
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logger = logging.getLogger(__name__)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"<h1 class='main-header'>{APP_NAME}</h1>", unsafe_allow_html=True)
st.markdown(f"<p>{APP_DESCRIPTION}</p>", unsafe_allow_html=True)
st.markdown(f"<p><small>Version {APP_VERSION}</small></p>", unsafe_allow_html=True)

# Check Sentinel Hub credentials
credentials_valid = False
if SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET:
    credentials_valid = check_sentinel_hub_credentials(
        SENTINEL_HUB_CLIENT_ID, 
        SENTINEL_HUB_CLIENT_SECRET
    )

if credentials_valid:
    st.markdown(
        "<div class='success-box'>✅ Sentinel Hub credentials are valid.</div>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<div class='warning-box'>⚠️ Sentinel Hub credentials are not set or invalid.</div>",
        unsafe_allow_html=True
    )
    
    st.markdown("""
    To use the full functionality of Agro Insight, you need to set up your Sentinel Hub credentials:
    
    1. Create a free account at [Sentinel Hub](https://www.sentinel-hub.com/)
    2. Generate OAuth credentials in your dashboard
    3. Add the credentials to the application's environment variables
    """)

# Main content
st.markdown("<h2 class='sub-header'>Dashboard Overview</h2>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Field Summary", "Recent Activity", "Quick Actions"])

with tab1:
    st.markdown("### Field Summary")
    
    # Get fields from database
    fields = []
    try:
        db = next(get_db())
        fields = db.query(Field).all()
    except Exception as e:
        st.error(f"Error fetching fields from database: {str(e)}")
    
    if fields:
        # Create a summary table
        field_data = []
        for field in fields:
            field_data.append({
                "Name": field.name,
                "Crop Type": field.crop_type or "Not specified",
                "Area (ha)": field.area_hectares,
                "Created": field.created_at.strftime('%Y-%m-%d')
            })
        
        st.dataframe(pd.DataFrame(field_data), use_container_width=True)
        
        # Create a map showing all fields
        st.markdown("### Field Locations")
        
        if fields:
            # Calculate average center for initial map view
            avg_lat = sum(f.center_lat for f in fields) / len(fields)
            avg_lon = sum(f.center_lon for f in fields) / len(fields)
            
            # Create map
            m = folium.Map(location=[avg_lat, avg_lon], zoom_start=10)
            
            # Add field markers
            for field in fields:
                folium.Marker(
                    [field.center_lat, field.center_lon],
                    popup=field.name,
                    tooltip=field.name
                ).add_to(m)
                
                # Add field boundary if available
                if field.geojson:
                    geojson_data = json.loads(field.geojson) if isinstance(field.geojson, str) else field.geojson
                    folium.GeoJson(
                        geojson_data,
                        name=field.name,
                        style_function=lambda x: {
                            'fillColor': '#28a745',
                            'color': '#28a745',
                            'weight': 2,
                            'fillOpacity': 0.2
                        }
                    ).add_to(m)
            
            # Display map
            folium_static(m, width=800, height=500)
    else:
        st.info("No fields found in the database. Add fields in the Data Ingest page.")

with tab2:
    st.markdown("### Recent Activity")
    
    # Placeholder for recent activity
    activity_data = [
        {"date": "2025-05-12", "type": "Field Added", "description": "New field 'Test Field 1' added"},
        {"date": "2025-05-11", "type": "Analysis Run", "description": "NDVI analysis for 'Test Field 1'"},
        {"date": "2025-05-10", "type": "Forecast Created", "description": "Yield forecast for 'Test Field 1'"}
    ]
    
    st.dataframe(pd.DataFrame(activity_data), use_container_width=True)

with tab3:
    st.markdown("### Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Add New Field")
        if st.button("Add Field", use_container_width=True):
            st.switch_page("pages/01_Data_Ingest.py")
    
    with col2:
        st.markdown("#### Run Analysis")
        if st.button("Analyze Field", use_container_width=True):
            st.switch_page("pages/02_Field_Analysis.py")
    
    with col3:
        st.markdown("#### Generate Report")
        if st.button("Create Report", use_container_width=True):
            st.switch_page("pages/05_Reports.py")

# Footer
st.markdown("---")
st.markdown(
    "<p><small>Powered by Sentinel Hub | Developed by Vortex Analytics</small></p>",
    unsafe_allow_html=True
)