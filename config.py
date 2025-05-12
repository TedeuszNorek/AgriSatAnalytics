"""
Configuration file for Agro Insight - Satellite data analytics for agriculture.
"""
import os
from pathlib import Path

# Application settings
APP_NAME = "Agro Insight"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Satellite data analytics for agricultural insights and market predictions"

# API Rate limits (free tier)
SENTINEL_HUB_REQUESTS_PER_MINUTE = 30
SENTINEL_HUB_PROCESSING_UNITS_PER_MINUTE = 10

# Directories
ROOT_DIR = Path(__file__).parent.absolute()
DATA_DIR = os.getenv("DATA_DIR", ROOT_DIR / "data")
CACHE_DIR = os.getenv("CACHE_DIR", ROOT_DIR / "cache")
LOG_DIR = os.getenv("LOG_DIR", ROOT_DIR)

# Ensure directories exist
Path(DATA_DIR).mkdir(exist_ok=True, parents=True)
Path(CACHE_DIR).mkdir(exist_ok=True, parents=True)
Path(DATA_DIR / "geotiff").mkdir(exist_ok=True, parents=True)
Path(DATA_DIR / "metadata").mkdir(exist_ok=True, parents=True) 
Path(DATA_DIR / "market").mkdir(exist_ok=True, parents=True)

# API credentials
SENTINEL_HUB_CLIENT_ID = os.getenv("SENTINEL_HUB_CLIENT_ID", "")
SENTINEL_HUB_CLIENT_SECRET = os.getenv("SENTINEL_HUB_CLIENT_SECRET", "")

# Sentinel Hub configuration
SENTINEL_HUB_BASE_URL = "https://services.sentinel-hub.com"
SENTINEL_HUB_OAUTH_URL = f"{SENTINEL_HUB_BASE_URL}/oauth/token"
SENTINEL_HUB_API_URL = f"{SENTINEL_HUB_BASE_URL}/api/v1"

# Weather API configuration
OPEN_METEO_API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Vegetation indices thresholds
NDVI_GOOD_THRESHOLD = 0.7
NDVI_MODERATE_THRESHOLD = 0.5
NDVI_POOR_THRESHOLD = 0.3

DROUGHT_NDVI_CHANGE_THRESHOLD = -0.15
DROUGHT_MIN_PERIODS = 2

# Market signal thresholds
MARKET_SIGNAL_CONFIDENCE_THRESHOLD = 0.7
ANOMALY_ZSCORE_THRESHOLD = 2.0

# Default image resolution in meters
DEFAULT_RESOLUTION = 10

# Maximum cloud coverage percentage for Sentinel-2 scenes
DEFAULT_MAX_CLOUD_COVERAGE = 20.0

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Model settings
MODEL_SAVE_DIR = os.path.join(DATA_DIR, "models")
Path(MODEL_SAVE_DIR).mkdir(exist_ok=True, parents=True)

# Report settings
REPORT_SAVE_DIR = os.path.join(DATA_DIR, "reports")
Path(REPORT_SAVE_DIR).mkdir(exist_ok=True, parents=True)
