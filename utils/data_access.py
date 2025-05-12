import os
import json
import datetime
import logging
import asyncio
import diskcache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
    DownloadFailedException,
)
import geopandas as gpd
from shapely.geometry import Polygon, shape

# Configure logger
logger = logging.getLogger(__name__)

# Configure cache
cache_dir = Path("./cache")
cache_dir.mkdir(exist_ok=True)
cache = diskcache.Cache(str(cache_dir))

# Configure data directory
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

# Global rate limiter for Sentinel Hub API (30 req/min for free tier)
MAX_REQUESTS_PER_MINUTE = 30
request_semaphore = asyncio.Semaphore(8)  # Limit parallel requests

# Configure Sentinel Hub
def get_sentinel_hub_config() -> SHConfig:
    """Get Sentinel Hub configuration from environment variables."""
    config = SHConfig()
    config.sh_client_id = os.getenv("SENTINEL_HUB_CLIENT_ID", "")
    config.sh_client_secret = os.getenv("SENTINEL_HUB_CLIENT_SECRET", "")
    
    if not config.sh_client_id or not config.sh_client_secret:
        logger.warning("Sentinel Hub credentials not found in environment variables")
    
    return config

@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def check_sentinel_hub_credentials(client_id: str, client_secret: str) -> bool:
    """Check if Sentinel Hub credentials are valid."""
    try:
        url = "https://services.sentinel-hub.com/oauth/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        }
        
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        
        # If we get here, credentials are valid
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to validate Sentinel Hub credentials: {e}")
        return False

async def fetch_with_rate_limit(function, *args, **kwargs):
    """Execute function with rate limiting."""
    async with request_semaphore:
        # Add exponential backoff for API requests
        @retry(
            retry=retry_if_exception_type((DownloadFailedException, requests.exceptions.HTTPError)),
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=4, max=60)
        )
        def execute_with_retry():
            return function(*args, **kwargs)
        
        return execute_with_retry()

def parse_geojson(geojson_data: Union[str, Dict]) -> Tuple[Polygon, CRS]:
    """Parse GeoJSON data and return Shapely polygon and CRS."""
    if isinstance(geojson_data, str):
        geojson_data = json.loads(geojson_data)
    
    # Extract the first feature if it's a FeatureCollection
    if geojson_data.get("type") == "FeatureCollection":
        feature = geojson_data["features"][0]
    else:
        feature = geojson_data
    
    geometry = feature.get("geometry", {})
    field_polygon = shape(geometry)
    
    # Get CRS or default to WGS84
    crs_info = geojson_data.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    crs = CRS.WGS84
    
    return field_polygon, crs

def get_bbox_from_polygon(polygon: Polygon) -> BBox:
    """Convert Shapely polygon to Sentinel Hub BBox."""
    minx, miny, maxx, maxy = polygon.bounds
    return BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)

@cache.memoize(expire=86400)  # Cache for 24 hours
def get_available_scenes(
    bbox: BBox, 
    start_date: datetime.datetime, 
    end_date: datetime.datetime,
    max_cloud_coverage: float = 20.0
) -> List[Dict]:
    """Get available Sentinel-2 scenes for the given bbox and time range."""
    config = get_sentinel_hub_config()
    
    catalog_url = "https://services.sentinel-hub.com/api/v1/catalog/search"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.oauth_token}"
    }
    
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox.get_lower_left() + bbox.get_upper_right(),
        "timeFrom": start_date.isoformat(),
        "timeTo": end_date.isoformat(),
        "query": {
            "cloudCoverPercentage": {
                "lt": max_cloud_coverage
            }
        },
        "limit": 50
    }
    
    response = requests.post(catalog_url, headers=headers, json=payload)
    response.raise_for_status()
    
    results = response.json().get("features", [])
    return results

@retry(
    retry=retry_if_exception_type((DownloadFailedException, requests.exceptions.HTTPError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def fetch_sentinel_data(
    bbox: BBox, 
    time_interval: Tuple[str, str],
    resolution: int = 10,
    data_collection: DataCollection = DataCollection.SENTINEL2_L2A,
) -> Tuple[np.ndarray, Dict]:
    """Fetch Sentinel-2 data for the specified bounding box and time interval."""
    config = get_sentinel_hub_config()
    
    # Define request for true color image
    size = bbox_to_dimensions(bbox, resolution=resolution)
    
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
                units: "DN"
            }],
            output: {
                bands: 10,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B11, sample.B12];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=data_collection,
                time_interval=time_interval,
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", MimeType.TIFF)
        ],
        bbox=bbox,
        size=size,
        config=config
    )
    
    response = request.get_data()
    
    if not response:
        raise DownloadFailedException("Empty response received from Sentinel Hub")
    
    image = response[0]
    
    # Get metadata
    metadata = {
        "bbox": bbox.get_polygon(),
        "time_interval": time_interval,
        "resolution": resolution,
        "size": size,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    return image, metadata

def get_country_boundary(country_code: str) -> Optional[Polygon]:
    """Get country boundary polygon from ISO country code."""
    try:
        url = f"https://nominatim.openstreetmap.org/search?country={country_code}&format=geojson"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        if data.get("features"):
            return shape(data["features"][0]["geometry"])
        return None
    except Exception as e:
        logger.error(f"Error fetching country boundary: {e}")
        return None

def save_to_geotiff(image: np.ndarray, metadata: Dict, filename: str) -> str:
    """Save image and metadata to GeoTIFF file."""
    import rasterio
    from rasterio.transform import from_bounds
    
    # Create output directory if it doesn't exist
    output_dir = data_dir / "geotiff"
    output_dir.mkdir(exist_ok=True)
    
    # Generate filename if not provided
    if not filename.endswith(".tif"):
        filename = f"{filename}.tif"
    
    output_path = output_dir / filename
    
    # Get bounds from metadata
    bbox = metadata["bbox"]
    minx, miny, maxx, maxy = bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]
    
    # Create transform
    height, width = image.shape[0], image.shape[1]
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Get number of bands
    count = image.shape[2] if len(image.shape) > 2 else 1
    
    # Write the GeoTIFF
    with rasterio.open(
        str(output_path),
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=image.dtype,
        crs='+proj=latlong',
        transform=transform,
    ) as dst:
        if count == 1:
            dst.write(image, 1)
        else:
            for i in range(count):
                dst.write(image[:, :, i], i + 1)
        
        # Write metadata
        dst.update_tags(**{k: str(v) for k, v in metadata.items() if isinstance(v, (str, int, float))})
    
    return str(output_path)

def save_stac_metadata(metadata: Dict, filename: str) -> str:
    """Save STAC metadata to JSON file."""
    # Create output directory if it doesn't exist
    output_dir = data_dir / "metadata"
    output_dir.mkdir(exist_ok=True)
    
    # Generate filename if not provided
    if not filename.endswith(".json"):
        filename = f"{filename}.json"
    
    output_path = output_dir / filename
    
    # Create STAC-like metadata
    stac_metadata = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": filename.replace(".json", ""),
        "properties": {
            "datetime": metadata.get("timestamp"),
            "start_datetime": metadata.get("time_interval")[0],
            "end_datetime": metadata.get("time_interval")[1],
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [metadata.get("bbox")]
        },
        "bbox": [
            metadata.get("bbox")[0][0],  # min x
            metadata.get("bbox")[0][1],  # min y
            metadata.get("bbox")[2][0],  # max x
            metadata.get("bbox")[2][1]   # max y
        ],
        "assets": {},
        "links": []
    }
    
    # Write metadata
    with open(output_path, 'w') as f:
        json.dump(stac_metadata, f, indent=2)
    
    return str(output_path)
