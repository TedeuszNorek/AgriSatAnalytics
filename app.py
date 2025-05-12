import os
import streamlit as st
from utils.data_access import check_sentinel_hub_credentials
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

# Initialize session state variables if not already set
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "selected_field" not in st.session_state:
    st.session_state.selected_field = None
if "last_analysis_results" not in st.session_state:
    st.session_state.last_analysis_results = None

# Page configuration
st.set_page_config(
    page_title="Agro Insight - Vortex Analytics",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🛰️ Agro Insight")
st.markdown("### Satellite-Powered Agricultural Analytics")

# Dashboard header with satellite imagery
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    Welcome to Agro Insight, a satellite data analytics module for Vortex Analytics. 
    Monitor agricultural fields, track vegetation indices, predict yields, and analyze market trends using 
    free Copernicus/Sentinel data.
    """)
    
    st.markdown("""
    ### Key Features
    - 🛰️ Real-time satellite data analysis from Sentinel-2
    - 🌱 Vegetation indices monitoring (NDVI, EVI, NDWI)
    - 📈 Yield forecasting and market signal detection
    - 🗺️ Field boundary mapping and visualization
    - 📊 Custom reports and analytics
    """)
    
with col2:
    # Display a satellite image from the provided stock photos
    st.image("https://pixabay.com/get/g4efcf32e7e1316c10c5547bc49ca7c6fe56ddd2b0787c5f35e822bf6392641e24ddb0209cb35a405581d490e27e587e7_1280.jpg", 
             caption="Sentinel-2 Satellite Imagery")

# Check Sentinel Hub credentials
client_id = os.getenv("SENTINEL_HUB_CLIENT_ID", "")
client_secret = os.getenv("SENTINEL_HUB_CLIENT_SECRET", "")

credentials_valid = False
if client_id and client_secret:
    with st.spinner("Checking Sentinel Hub credentials..."):
        credentials_valid = check_sentinel_hub_credentials(client_id, client_secret)

if not credentials_valid:
    st.warning("""
    ⚠️ Sentinel Hub credentials are not set or invalid.
    
    To use all features of Agro Insight, you need valid Sentinel Hub credentials:
    1. Create a free account at [Sentinel Hub](https://www.sentinel-hub.com/)
    2. Get your Client ID and Client Secret from the dashboard
    3. Set them as environment variables:
       - SENTINEL_HUB_CLIENT_ID
       - SENTINEL_HUB_CLIENT_SECRET
    """)
else:
    st.success("✅ Connected to Sentinel Hub API")

# Quick access cards
st.subheader("Quick Access")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🗺️ Field Monitoring")
    st.markdown("View NDVI, vegetation health, and soil moisture for your fields.")
    if st.button("Open Field Monitoring", key="field_monitoring_btn"):
        st.switch_page("pages/02_Field_Analysis.py")

with col2:
    st.markdown("### 📊 Yield Forecast")
    st.markdown("ML-powered crop yield predictions based on historical data.")
    if st.button("View Yield Forecasts", key="yield_forecast_btn"):
        st.switch_page("pages/03_Yield_Forecast.py")

with col3:
    st.markdown("### 💹 Market Signals")
    st.markdown("Detect market opportunities based on satellite data analytics.")
    if st.button("Analyze Market Signals", key="market_signals_btn"):
        st.switch_page("pages/04_Market_Signals.py")

# Recent analysis and activity
st.subheader("Recent Activity")
if st.session_state.last_analysis_results:
    # Display recent analysis results if available
    st.json(st.session_state.last_analysis_results)
else:
    # Display placeholder with sample agricultural field image
    st.image("https://pixabay.com/get/g879b44be88f7b57d084cc1720ca16fd7c256f93e15c6d03affe5cf8e36c009d93abf1f23dd42863852eb153851fdb35af3cfc66d04d2878ab81d1192661dd77a_1280.jpg", 
             caption="Agricultural field monitoring with satellite data")
    st.info("No recent analysis available. Start by adding a field in the Data Ingest section.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Agro Insight - Powered by Copernicus/Sentinel Open Data | Using free tier resources only</p>
    <p>Data refreshed every 5 days | Resolution: 10-20m/pixel</p>
</div>
""", unsafe_allow_html=True)
