"""
Data Sources & Satellite Connections - Information about connected satellites and available data
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

from utils.data_access import get_data_access_manager

# Page configuration
st.set_page_config(
    page_title="Data Sources - Agro Insight",
    page_icon="🛰️",
    layout="wide"
)

# Title and description
st.title("🛰️ Data Sources & Satellite Connections")
st.markdown("""
**Complete overview** of all satellite data sources, API connections, and available datasets in the Agro Insight platform.
This page shows which satellites we're connected to and what agricultural data each provides.
""")

# Get data access manager
data_manager = get_data_access_manager()

# Sidebar with connection controls
st.sidebar.header("🔗 Connection Status")

# Check API connections
st.sidebar.subheader("API Status")

# Check Sentinel Hub connection
if st.sidebar.button("🔍 Test Sentinel Hub"):
    with st.spinner("Testing Sentinel Hub connection..."):
        try:
            # Try to get an OAuth token
            token_info = data_manager.get_oauth_token()
            if token_info and 'access_token' in token_info:
                st.sidebar.success("✅ Sentinel Hub: Connected")
                expires_in = token_info.get('expires_in', 0)
                st.sidebar.info(f"Token expires in: {expires_in} seconds")
            else:
                st.sidebar.error("❌ Sentinel Hub: Authentication failed")
        except Exception as e:
            st.sidebar.error(f"❌ Sentinel Hub: Error - {str(e)}")

# Check Planet API connection
if st.sidebar.button("🌍 Test Planet API"):
    with st.spinner("Testing Planet API connection..."):
        try:
            auth_result = data_manager.authenticate_planet()
            if auth_result:
                st.sidebar.success("✅ Planet API: Connected")
            else:
                st.sidebar.error("❌ Planet API: Authentication failed")
        except Exception as e:
            st.sidebar.error(f"❌ Planet API: Error - {str(e)}")

# Environment variables status
st.sidebar.subheader("🔑 API Keys Status")

# Check for required environment variables
required_keys = [
    "SENTINEL_HUB_CLIENT_ID",
    "SENTINEL_HUB_CLIENT_SECRET", 
    "PLANET_API_KEY"
]

for key in required_keys:
    if os.environ.get(key):
        st.sidebar.success(f"✅ {key}: Configured")
    else:
        st.sidebar.error(f"❌ {key}: Missing")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🛰️ Sentinel Satellites",
    "🌍 Planet Labs",
    "📊 Available Data Types",
    "⚙️ Configuration"
])

with tab1:
    st.header("Sentinel Satellites (ESA Copernicus Program)")
    st.markdown("""
    We're connected to the **European Space Agency's Copernicus program** through Sentinel Hub API.
    This gives us access to multiple Sentinel satellites with different capabilities.
    """)
    
    # Sentinel satellites information
    sentinel_data = [
        {
            "Satellite": "Sentinel-1A/1B",
            "Type": "SAR (Radar)",
            "Resolution": "10-40m",
            "Revisit Time": "6 days",
            "Key Data": "All-weather imaging, soil moisture, crop monitoring",
            "Best For": "Weather-independent monitoring, flood detection",
            "Bands": "C-band radar (5.405 GHz)",
            "Status": "✅ Active"
        },
        {
            "Satellite": "Sentinel-2A/2B", 
            "Type": "Optical",
            "Resolution": "10-60m",
            "Revisit Time": "5 days",
            "Key Data": "NDVI, EVI, land cover, vegetation health",
            "Best For": "Crop health monitoring, harvest prediction",
            "Bands": "13 spectral bands (443-2190 nm)",
            "Status": "✅ Active"
        },
        {
            "Satellite": "Sentinel-3A/3B",
            "Type": "Ocean/Land",
            "Resolution": "300m-1.2km",
            "Revisit Time": "2-3 days",
            "Key Data": "Sea/land surface temperature, ocean color",
            "Best For": "Climate monitoring, water quality",
            "Bands": "21 spectral bands",
            "Status": "✅ Active"
        },
        {
            "Satellite": "Sentinel-5P",
            "Type": "Atmospheric",
            "Resolution": "3.5-7km",
            "Revisit Time": "Daily",
            "Key Data": "Air quality, greenhouse gases (CO, NO2, CH4)",
            "Best For": "Environmental monitoring, climate data",
            "Bands": "UV-VIS-NIR-SWIR",
            "Status": "✅ Active"
        }
    ]
    
    # Display as interactive table
    df_sentinel = pd.DataFrame(sentinel_data)
    st.dataframe(df_sentinel, use_container_width=True)
    
    # Detailed breakdown of Sentinel-2 (our primary source)
    st.subheader("🎯 Sentinel-2 Detailed Capabilities")
    st.markdown("""
    **Sentinel-2** is our primary data source for agricultural monitoring. Here's what each band provides:
    """)
    
    sentinel2_bands = [
        {"Band": "B1 - Coastal", "Wavelength": "443 nm", "Resolution": "60m", "Agricultural Use": "Coastal and aerosol studies"},
        {"Band": "B2 - Blue", "Wavelength": "490 nm", "Resolution": "10m", "Agricultural Use": "Soil/vegetation discrimination"},
        {"Band": "B3 - Green", "Wavelength": "560 nm", "Resolution": "10m", "Agricultural Use": "Green vegetation peak"},
        {"Band": "B4 - Red", "Wavelength": "665 nm", "Resolution": "10m", "Agricultural Use": "Chlorophyll absorption, NDVI calculation"},
        {"Band": "B5 - Red Edge 1", "Wavelength": "705 nm", "Resolution": "20m", "Agricultural Use": "Vegetation stress detection"},
        {"Band": "B6 - Red Edge 2", "Wavelength": "740 nm", "Resolution": "20m", "Agricultural Use": "Chlorophyll content"},
        {"Band": "B7 - Red Edge 3", "Wavelength": "783 nm", "Resolution": "20m", "Agricultural Use": "Vegetation health monitoring"},
        {"Band": "B8 - NIR", "Wavelength": "842 nm", "Resolution": "10m", "Agricultural Use": "Biomass estimation, NDVI calculation"},
        {"Band": "B8A - Narrow NIR", "Wavelength": "865 nm", "Resolution": "20m", "Agricultural Use": "Water vapor absorption"},
        {"Band": "B9 - Water Vapor", "Wavelength": "945 nm", "Resolution": "60m", "Agricultural Use": "Atmospheric correction"},
        {"Band": "B10 - Cirrus", "Wavelength": "1375 nm", "Resolution": "60m", "Agricultural Use": "Cloud detection"},
        {"Band": "B11 - SWIR 1", "Wavelength": "1610 nm", "Resolution": "20m", "Agricultural Use": "Moisture content, crop type classification"},
        {"Band": "B12 - SWIR 2", "Wavelength": "2190 nm", "Resolution": "20m", "Agricultural Use": "Drought stress, soil discrimination"}
    ]
    
    df_bands = pd.DataFrame(sentinel2_bands)
    st.dataframe(df_bands, use_container_width=True)

with tab2:
    st.header("Planet Labs Constellation")
    st.markdown("""
    We're connected to **Planet Labs** through their API, providing access to the world's largest 
    constellation of Earth observation satellites for high-resolution daily imaging.
    """)
    
    # Planet satellites information
    planet_data = [
        {
            "Constellation": "PlanetScope",
            "Satellites": "150+ Doves",
            "Resolution": "3-4m",
            "Revisit Time": "Daily",
            "Key Data": "RGB + NIR, daily field monitoring",
            "Best For": "Change detection, crop emergence tracking",
            "Coverage": "Global",
            "Status": "✅ Active"
        },
        {
            "Constellation": "RapidEye",
            "Satellites": "5 satellites",
            "Resolution": "5m",
            "Revisit Time": "Daily",
            "Key Data": "5-band multispectral (RGB + Red Edge + NIR)",
            "Best For": "Precision agriculture, detailed crop analysis",
            "Coverage": "Global",
            "Status": "🔄 Archived (historical data)"
        },
        {
            "Constellation": "SkySat",
            "Satellites": "21 satellites",
            "Resolution": "50cm-1m",
            "Revisit Time": "Multiple daily",
            "Key Data": "High-resolution RGB, video capability",
            "Best For": "Detailed field inspection, infrastructure monitoring",
            "Coverage": "Selected areas",
            "Status": "✅ Active"
        }
    ]
    
    df_planet = pd.DataFrame(planet_data)
    st.dataframe(df_planet, use_container_width=True)
    
    # Planet data products
    st.subheader("🎯 Planet Data Products Available")
    
    planet_products = [
        {"Product": "4-Band PlanetScope", "Description": "RGB + NIR imagery", "Resolution": "3m", "Use Case": "Basic vegetation monitoring"},
        {"Product": "8-Band PlanetScope", "Description": "Coastal, RGB, Red Edge, NIR bands", "Resolution": "3m", "Use Case": "Advanced crop analysis"},
        {"Product": "Analytic Ortho Tile", "Description": "Radiometrically calibrated", "Resolution": "3-5m", "Use Case": "Scientific analysis, NDVI calculation"},
        {"Product": "Visual Ortho Tile", "Description": "Color-corrected RGB", "Resolution": "3-5m", "Use Case": "Visual inspection, mapping"},
        {"Product": "Surface Reflectance", "Description": "Atmospherically corrected", "Resolution": "3m", "Use Case": "Time-series analysis, monitoring"}
    ]
    
    df_planet_products = pd.DataFrame(planet_products)
    st.dataframe(df_planet_products, use_container_width=True)

with tab3:
    st.header("📊 Available Agricultural Data Types")
    st.markdown("""
    Here's a comprehensive overview of all agricultural data types we can access and generate
    from our satellite connections.
    """)
    
    # Create expandable sections for different data categories
    with st.expander("🌿 Vegetation Indices", expanded=True):
        vegetation_indices = [
            {"Index": "NDVI", "Formula": "(NIR - Red) / (NIR + Red)", "Range": "-1 to +1", "Indicates": "Vegetation health and density"},
            {"Index": "EVI", "Formula": "2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)", "Range": "-1 to +1", "Indicates": "Enhanced vegetation, reduces atmospheric effects"},
            {"Index": "SAVI", "Formula": "(NIR - Red) / (NIR + Red + L) * (1 + L)", "Range": "-1 to +1", "Indicates": "Soil-adjusted vegetation (L=0.5)"},
            {"Index": "NDWI", "Formula": "(Green - NIR) / (Green + NIR)", "Range": "-1 to +1", "Indicates": "Water content in vegetation"},
            {"Index": "LAI", "Formula": "Derived from NDVI/EVI", "Range": "0 to 8+", "Indicates": "Leaf area per unit ground area"},
            {"Index": "FAPAR", "Formula": "Fraction of absorbed radiation", "Range": "0 to 1", "Indicates": "Photosynthetic activity"}
        ]
        
        df_veg = pd.DataFrame(vegetation_indices)
        st.dataframe(df_veg, use_container_width=True)
    
    with st.expander("🌾 Crop Monitoring Parameters"):
        crop_parameters = [
            {"Parameter": "Crop Type Classification", "Source": "Multi-band analysis", "Accuracy": "85-95%", "Update Frequency": "Weekly"},
            {"Parameter": "Growth Stage Detection", "Source": "Time-series NDVI", "Accuracy": "80-90%", "Update Frequency": "Weekly"},
            {"Parameter": "Biomass Estimation", "Source": "NIR + Red Edge bands", "Accuracy": "70-85%", "Update Frequency": "Bi-weekly"},
            {"Parameter": "Yield Prediction", "Source": "ML models + indices", "Accuracy": "75-90%", "Update Frequency": "Monthly"},
            {"Parameter": "Harvest Readiness", "Source": "NDVI trends + weather", "Accuracy": "80-90%", "Update Frequency": "Weekly"},
            {"Parameter": "Stress Detection", "Source": "Red Edge + thermal", "Accuracy": "70-85%", "Update Frequency": "Daily"}
        ]
        
        df_crop = pd.DataFrame(crop_parameters)
        st.dataframe(df_crop, use_container_width=True)
    
    with st.expander("🌡️ Environmental Monitoring"):
        environmental_data = [
            {"Data Type": "Soil Moisture", "Source": "Sentinel-1 SAR", "Resolution": "10m", "Coverage": "All weather conditions"},
            {"Data Type": "Surface Temperature", "Source": "Sentinel-3, thermal bands", "Resolution": "300m", "Coverage": "Day/night"},
            {"Data Type": "Precipitation", "Source": "Weather APIs + radar", "Resolution": "1km", "Coverage": "Real-time"},
            {"Data Type": "Drought Stress", "Source": "NDWI + temperature", "Resolution": "10m", "Coverage": "Weekly updates"},
            {"Data Type": "Flood Detection", "Source": "Sentinel-1 + optical", "Resolution": "10m", "Coverage": "Real-time alerts"},
            {"Data Type": "Air Quality", "Source": "Sentinel-5P", "Resolution": "7km", "Coverage": "Daily global"}
        ]
        
        df_env = pd.DataFrame(environmental_data)
        st.dataframe(df_env, use_container_width=True)
    
    with st.expander("💹 Market Intelligence Data"):
        market_data = [
            {"Data Source": "Commodity Futures", "Provider": "Yahoo Finance", "Coverage": "Global markets", "Update": "Real-time"},
            {"Data Source": "Weather Forecasts", "Provider": "Multiple APIs", "Coverage": "Global", "Update": "Hourly"},
            {"Data Source": "Supply Chain", "Provider": "Satellite analysis", "Coverage": "Regional", "Update": "Weekly"},
            {"Data Source": "Energy Prices", "Provider": "Financial APIs", "Coverage": "Global", "Update": "Real-time"},
            {"Data Source": "Trade Flows", "Provider": "Economic APIs", "Coverage": "International", "Update": "Daily"},
            {"Data Source": "Storage Levels", "Provider": "Satellite + reports", "Coverage": "Regional", "Update": "Weekly"}
        ]
        
        df_market = pd.DataFrame(market_data)
        st.dataframe(df_market, use_container_width=True)

with tab4:
    st.header("⚙️ Configuration & Setup")
    st.markdown("""
    **API Configuration and Data Access Settings**
    """)
    
    # Configuration information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔑 Required API Keys")
        
        api_config = [
            {"Service": "Sentinel Hub", "Variables": ["SENTINEL_HUB_CLIENT_ID", "SENTINEL_HUB_CLIENT_SECRET"], "Status": "Required"},
            {"Service": "Planet Labs", "Variables": ["PLANET_API_KEY"], "Status": "Required"},
            {"Service": "Weather APIs", "Variables": ["OPENWEATHER_API_KEY"], "Status": "Optional"},
            {"Service": "Financial APIs", "Variables": ["ALPHA_VANTAGE_KEY"], "Status": "Optional"}
        ]
        
        for config in api_config:
            with st.expander(f"{config['Service']} Configuration"):
                st.write(f"**Status:** {config['Status']}")
                st.write("**Required Variables:**")
                for var in config['Variables']:
                    if os.environ.get(var):
                        st.success(f"✅ {var}: Configured")
                    else:
                        st.error(f"❌ {var}: Missing")
                        st.code(f"export {var}=your_api_key_here")
    
    with col2:
        st.subheader("📊 Data Processing Settings")
        
        # Current settings (would be configurable in full implementation)
        settings_data = [
            {"Setting": "Default Cloud Cover Limit", "Value": "20%", "Description": "Maximum cloud coverage for image selection"},
            {"Setting": "Time Series Length", "Value": "12 months", "Description": "Historical data period for analysis"},
            {"Setting": "Update Frequency", "Value": "Weekly", "Description": "How often to check for new satellite data"},
            {"Setting": "Spatial Resolution", "Value": "10m", "Description": "Target pixel size for analysis"},
            {"Setting": "Processing Priority", "Value": "Real-time", "Description": "Speed vs accuracy trade-off"},
            {"Setting": "Data Storage", "Value": "30 days", "Description": "Local cache retention period"}
        ]
        
        df_settings = pd.DataFrame(settings_data)
        st.dataframe(df_settings, use_container_width=True)
    
    # Data flow diagram
    st.subheader("🔄 Data Flow Architecture")
    
    # Create a simple flow diagram using text
    st.markdown("""
    ```
    🛰️ Satellites (Sentinel-1/2/3/5P, Planet) 
           ↓
    🌐 API Gateways (Sentinel Hub, Planet API)
           ↓
    🔄 Data Processing (NDVI, EVI, Classifications)
           ↓
    🧠 AI Analysis (LLM Agents, Predictions)
           ↓
    📊 User Interface (Streamlit Dashboard)
           ↓
    💾 Results Storage (Local Cache + Database)
    ```
    """)
    
    # Performance metrics
    st.subheader("📈 System Performance")
    
    # Sample performance data (in real implementation, this would be live)
    perf_data = [
        {"Metric": "Data Latency", "Value": "2-6 hours", "Description": "Time from satellite acquisition to availability"},
        {"Metric": "Processing Speed", "Value": "~50 fields/hour", "Description": "Analysis throughput"},
        {"Metric": "Accuracy Rate", "Value": "85-95%", "Description": "Crop classification accuracy"},
        {"Metric": "Uptime", "Value": "99.5%", "Description": "API availability"},
        {"Metric": "Cache Hit Rate", "Value": "78%", "Description": "Percentage of requests served from cache"},
        {"Metric": "Storage Usage", "Value": "2.3 GB", "Description": "Current local data storage"}
    ]
    
    df_perf = pd.DataFrame(perf_data)
    st.dataframe(df_perf, use_container_width=True)

# Footer with additional information
st.markdown("---")
st.markdown("""
### 📚 Additional Resources

- **Sentinel Hub Documentation**: [docs.sentinel-hub.com](https://docs.sentinel-hub.com)
- **Planet API Documentation**: [developers.planet.com](https://developers.planet.com)
- **Copernicus Program**: [copernicus.eu](https://copernicus.eu)
- **ESA Earth Online**: [earth.esa.int](https://earth.esa.int)

### 🆘 Support

If you need help configuring API keys or accessing specific data sources, please refer to the documentation 
or contact support. All satellite data is subject to the respective providers' terms of service.
""")