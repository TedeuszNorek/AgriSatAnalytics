"""
Configuration file for Agro Insight - Satellite data analytics for agriculture.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)

# Add console handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger('').addHandler(console)

# Application metadata
APP_NAME = "Agro Insight"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = "Satellite data analytics for agriculture"
APP_AUTHOR = "Vortex Analytics"

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
for directory in [DATA_DIR, CACHE_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Nested data directories
SATELLITE_DATA_DIR = os.path.join(DATA_DIR, "satellite")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")

# Create nested directories
for directory in [SATELLITE_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Sentinel Hub configuration
SENTINEL_HUB_CLIENT_ID = os.getenv("SENTINEL_HUB_CLIENT_ID")
SENTINEL_HUB_CLIENT_SECRET = os.getenv("SENTINEL_HUB_CLIENT_SECRET")

# Default study parameters
DEFAULT_START_DATE = datetime.utcnow() - timedelta(days=365)  # 1 year ago
DEFAULT_END_DATE = datetime.utcnow()
DEFAULT_MAX_CLOUD_COVERAGE = 20.0  # percentage

# Satellite data parameters
SATELLITE_INDICES = {
    "NDVI": "Normalized Difference Vegetation Index",
    "EVI": "Enhanced Vegetation Index",
    "NDWI": "Normalized Difference Water Index",
    "NDMI": "Normalized Difference Moisture Index",
    "NBR": "Normalized Burn Ratio"
}

# Crop types
CROP_TYPES = [
    "Wheat",
    "Corn",
    "Soybean",
    "Barley",
    "Canola",
    "Rice",
    "Cotton",
    "Sunflower",
    "Other"
]

# Market data configuration
COMMODITIES = {
    "ZW=F": "Wheat Futures",
    "ZC=F": "Corn Futures",
    "ZS=F": "Soybean Futures",
    "ZR=F": "Rice Futures",
    "ZO=F": "Oats Futures",
    "ZL=F": "Soybean Oil Futures"
}

# Cache configuration
CACHE_EXPIRATION = 86400  # 24 hours in seconds

# Application settings
SETTINGS = {
    "debug_mode": os.getenv("DEBUG", "False").lower() in ["true", "1", "yes"],
    "cache_enabled": True,
    "max_threads": 4,
    "request_timeout": 30,  # seconds
    "rate_limits": {
        "sentinel_hub": 10,  # requests per minute
        "weather_api": 60,   # requests per minute
        "market_data": 100   # requests per minute
    }
}

# Function to get all configurations as a dictionary (for debugging)
def get_config_dict() -> Dict[str, Any]:
    """Return all configurations as a dictionary."""
    return {
        "app_info": {
            "name": APP_NAME,
            "version": APP_VERSION,
            "description": APP_DESCRIPTION
        },
        "paths": {
            "base_dir": BASE_DIR,
            "data_dir": DATA_DIR,
            "cache_dir": CACHE_DIR,
            "models_dir": MODELS_DIR,
            "satellite_data_dir": SATELLITE_DATA_DIR,
            "processed_data_dir": PROCESSED_DATA_DIR,
            "reports_dir": REPORTS_DIR
        },
        "sentinel_hub": {
            "client_id_configured": bool(SENTINEL_HUB_CLIENT_ID),
            "client_secret_configured": bool(SENTINEL_HUB_CLIENT_SECRET)
        },
        "study_params": {
            "default_start_date": DEFAULT_START_DATE.isoformat(),
            "default_end_date": DEFAULT_END_DATE.isoformat(),
            "default_max_cloud_coverage": DEFAULT_MAX_CLOUD_COVERAGE
        },
        "satellite_indices": SATELLITE_INDICES,
        "crop_types": CROP_TYPES,
        "commodities": COMMODITIES,
        "cache": {
            "expiration": CACHE_EXPIRATION
        },
        "settings": SETTINGS
    }