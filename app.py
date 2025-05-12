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
    /* Status section styling */
    .success-box h3, .error-box h3 {
        margin-top: 0;
        margin-bottom: 5px;
        text-align: center;
    }
    .success-box p, .error-box p {
        margin-bottom: 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"<h1 class='main-header'>{APP_NAME}</h1>", unsafe_allow_html=True)
st.markdown(f"<p>{APP_DESCRIPTION}</p>", unsafe_allow_html=True)
st.markdown(f"<p><small>Version {APP_VERSION}</small></p>", unsafe_allow_html=True)

# Check satellite data sources connections
from utils.data_access import check_sentinel_hub_credentials, check_planet_api_connection
from config import SENTINEL_HUB_CLIENT_ID, SENTINEL_HUB_CLIENT_SECRET, PLANET_API_KEY

# Get connection statuses
sentinel_status = check_sentinel_hub_credentials(SENTINEL_HUB_CLIENT_ID, SENTINEL_HUB_CLIENT_SECRET)
planet_status = check_planet_api_connection(PLANET_API_KEY)

# Create satellite connection status section
st.markdown("<h2 class='sub-header'>Satellite Connection Status</h2>", unsafe_allow_html=True)

# Create a tabbed interface for the satellite connections
sentinel_tab, planet_tab = st.tabs(["Sentinel Hub (ESA)", "Planet API"])

# Sentinel Hub tab
with sentinel_tab:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sentinel data source information
        st.markdown("### Sentinel Satellite Data")
        
        # Sentinel-2 information
        st.markdown(f"""
        **{sentinel_status['service_name']}**
        - Provider: {sentinel_status['data_provider']}
        - Resolution: {sentinel_status['resolution']}
        - Data Products: {', '.join(sentinel_status['data_products'])}
        - Frequency: Images refreshed every **{sentinel_status['refresh_rate']} days**
        - Cloud Detection: Automatic cloud masking applied
        """)
        
        # Sentinel-1 information if included
        st.markdown("""
        **Sentinel-1 Radar Satellite:**
        - Resolution: 10m per pixel
        - Data Products: Surface backscatter, soil moisture estimates
        - Frequency: Images refreshed every **6 days** (on average)
        - All-weather capability: Operates through clouds
        """)

    with col2:
        # Connection status
        st.markdown("### Connection Status")
        
        if sentinel_status['valid']:
            st.markdown(
                "<div class='success-box'><h3>✅ CONNECTED</h3><p>Sentinel data access is <strong>LIVE</strong></p></div>",
                unsafe_allow_html=True
            )
            
            # Add a recent update timestamp
            st.markdown(f"**Last Status Check:** {sentinel_status['last_check']}")
            
            # Add data refresh information
            next_refresh = (datetime.datetime.now() + datetime.timedelta(days=sentinel_status['refresh_rate'])).strftime('%Y-%m-%d')
            st.markdown(f"**Next Data Refresh:** ~{next_refresh}")
            
        else:
            st.markdown(
                "<div class='error-box'><h3>❌ DISCONNECTED</h3><p>Sentinel data access is <strong>OFFLINE</strong></p></div>",
                unsafe_allow_html=True
            )
            
            # Show error message
            st.markdown(f"**Error:** {sentinel_status['message']}")
            
            st.markdown("""
            To enable live satellite data:
            1. Create an account at [Sentinel Hub](https://www.sentinel-hub.com/)
            2. Generate OAuth credentials in your dashboard
            3. Add credentials to environment variables
            """)
            
            # Add option to use test data
            st.warning("⚠️ Using mock data for demonstration purpose.")

# Planet tab
with planet_tab:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Planet data source information
        st.markdown("### Planet Satellite Data")
        
        st.markdown(f"""
        **{planet_status['service_name']}**
        - Provider: {planet_status['data_provider']}
        - Resolution: {planet_status['resolution']}
        - Data Products: {', '.join(planet_status['data_products'])}
        - Frequency: Images refreshed **daily**
        - Advanced Platform: Commercial high-resolution satellite imagery
        """)
        
        # Add more information about Planet's capabilities
        st.markdown("""
        **Planet Platform Features:**
        - Global daily imaging
        - Archive access back to 2009
        - On-demand tasking available
        - API access for automated processing
        """)

    with col2:
        # Connection status
        st.markdown("### Connection Status")
        
        if planet_status['valid']:
            st.markdown(
                "<div class='success-box'><h3>✅ CONNECTED</h3><p>Planet data access is <strong>LIVE</strong></p></div>",
                unsafe_allow_html=True
            )
            
            # Add a recent update timestamp
            st.markdown(f"**Last Status Check:** {planet_status['last_check']}")
            
            # Add data refresh information
            next_refresh = (datetime.datetime.now() + datetime.timedelta(days=planet_status['refresh_rate'])).strftime('%Y-%m-%d')
            st.markdown(f"**Next Data Refresh:** ~{next_refresh}")
            
        else:
            st.markdown(
                "<div class='error-box'><h3>❌ DISCONNECTED</h3><p>Planet data access is <strong>OFFLINE</strong></p></div>",
                unsafe_allow_html=True
            )
            
            # Show error message
            st.markdown(f"**Error:** {planet_status['message']}")
            
            st.markdown("""
            To enable Planet API access:
            1. Create an account at [Planet](https://www.planet.com/)
            2. Generate an API key in your account settings
            3. Add the API key to environment variables
            """)
            
            # Add link to Planet API documentation
            st.markdown("[Learn more about Planet API](https://developers.planet.com/docs/apis/)")

# Overall data source status
st.markdown("---")
if sentinel_status['valid'] or planet_status['valid']:
    st.success("✅ At least one satellite data source is available. You can proceed with the application.")
else:
    st.error("❌ No satellite data sources are available. Please configure at least one data source to use the full functionality.")
    st.warning("Demo mode is enabled with mock data for demonstration purposes.")

# Display visual separator
st.markdown("---")

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

# Sidebar debug options
with st.sidebar:
    st.markdown("---")
    if st.checkbox("Show Debug Options"):
        st.markdown("### Debug & Diagnostics")
        
        # Debug buttons
        if st.button("Test Sentinel Hub Connection", use_container_width=True):
            with st.spinner("Testing Sentinel Hub connection..."):
                try:
                    import traceback
                    
                    # Check if credentials exist
                    if not SENTINEL_HUB_CLIENT_ID or not SENTINEL_HUB_CLIENT_SECRET:
                        st.error("❌ Sentinel Hub credentials are not set properly.")
                        st.markdown("""
                        ### Missing Credentials
                        You need to set both SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET 
                        environment variables.
                        
                        1. Create an account at [Sentinel Hub](https://www.sentinel-hub.com/)
                        2. Create OAuth credentials in your dashboard
                        3. Add these credentials to the application's environment variables
                        """)
                        st.stop()
                    
                    # Log the credentials (without revealing them)
                    st.info(f"Using Sentinel Hub credentials with client ID starting with '{SENTINEL_HUB_CLIENT_ID[:4]}...'")
                    
                    # Test the connection
                    is_valid = check_sentinel_hub_credentials(
                        SENTINEL_HUB_CLIENT_ID, 
                        SENTINEL_HUB_CLIENT_SECRET
                    )
                    
                    if is_valid:
                        st.success("✅ Sentinel Hub connection successful! Credentials are valid.")
                    else:
                        st.error("❌ Could not authenticate with Sentinel Hub. Please check your credentials.")
                        st.markdown("""
                        ### Troubleshooting Steps:
                        1. Ensure you've created an account at [Sentinel Hub](https://www.sentinel-hub.com/)
                        2. Verify your OAuth credentials in your dashboard
                        3. Update the credentials in the environment variables
                        
                        Current error: 401 Unauthorized - This means your credentials are incorrect or expired.
                        """)
                except Exception as e:
                    st.error(f"Error during Sentinel Hub connection test: {str(e)}")
                    st.code(traceback.format_exc(), language="python")
        
        # Help with Sentinel Hub credentials
        if st.button("How to get Sentinel Hub Credentials", use_container_width=True):
            st.markdown("""
            ## Jak uzyskać poprawne dane uwierzytelniające dla Sentinel Hub
            
            Sentinel Hub używa **uwierzytelniania OAuth2**, które jest inne niż zwykły klucz API. Musisz wykonać następujące kroki:
            
            1. Utwórz konto na [Sentinel Hub](https://www.sentinel-hub.com/), jeśli jeszcze go nie masz
            2. Zaloguj się i przejdź do panelu konta
            3. Przejdź do sekcji "OAuth clients" w ustawieniach
            4. Utwórz nowego klienta OAuth, klikając "Create New OAuth Client"
            5. Nadaj nazwę (np. "AgroInsight") i zapisz
            6. Otrzymasz dwie wartości:
               - **Client ID** - to NIE jest Twój adres email
               - **Client Secret** - to NIE jest Twój kod API
            7. Używasz tych wartości jako zmiennych środowiskowych:
               - `SENTINEL_HUB_CLIENT_ID` - Wartość Client ID
               - `SENTINEL_HUB_CLIENT_SECRET` - Wartość Client Secret
            
            Zwróć uwagę, że te dane są inne niż zwykłe dane dostępowe do konta!
            """)
        
        # Add test button that doesn't require actual API access
        if st.button("Test UI Functions (No API)", use_container_width=True):
            with st.spinner("Testing UI functions without API calls..."):
                try:
                    # Create a simple test map
                    import folium
                    from streamlit_folium import folium_static
                    
                    st.markdown("### Test Map Display")
                    m = folium.Map(location=[50.0, 19.0], zoom_start=6)
                    folium.Marker([50.0, 19.0], popup="Test Marker").add_to(m)
                    folium_static(m, width=600, height=400)
                    
                    # Create a sample chart
                    import matplotlib.pyplot as plt
                    import numpy as np
                    
                    st.markdown("### Test Chart Display")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    x = np.linspace(0, 10, 100)
                    y = np.sin(x)
                    ax.plot(x, y)
                    ax.set_title("Test Sine Wave")
                    ax.grid(True)
                    st.pyplot(fig)
                    
                    st.success("✅ UI functions testing successful. The application can display maps and charts correctly.")
                except Exception as e:
                    st.error(f"Error during UI functions test: {str(e)}")
                    st.code(traceback.format_exc(), language="python")
        
        # Database connection test
        if st.button("Test Database Connection", use_container_width=True):
            with st.spinner("Testing database connection..."):
                try:
                    import traceback
                    # Try to get a session and execute a simple query
                    db = next(get_db())
                    result = db.execute("SELECT 1").fetchone()
                    
                    if result and result[0] == 1:
                        st.success("✅ Database connection successful!")
                    else:
                        st.error("❌ Database connection test failed.")
                except Exception as e:
                    st.error(f"Error during database connection test: {str(e)}")
                    st.code(traceback.format_exc(), language="python")

# Footer
st.markdown("---")
st.markdown(
    "<p><small>Powered by Sentinel Hub | Developed by Vortex Analytics</small></p>",
    unsafe_allow_html=True
)