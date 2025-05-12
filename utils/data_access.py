"""
Module for accessing satellite data from Sentinel Hub and other sources.
"""
import os
import json
import logging
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import asyncio
import time
from functools import wraps

import numpy as np
from shapely.geometry import Polygon, shape
from sentinelhub import (
    SHConfig, 
    CRS, 
    BBox, 
    DataCollection, 
    MimeType, 
    SentinelHubRequest, 
    bbox_to_dimensions,
    SentinelHubDownloadClient,
    DownloadRequest
)
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    SENTINEL_HUB_CLIENT_ID,
    SENTINEL_HUB_CLIENT_SECRET,
    CACHE_DIR,
    SATELLITE_DATA_DIR,
    SETTINGS
)

# Set up logging
logger = logging.getLogger(__name__)

def get_sentinel_hub_config() -> SHConfig:
    """Get Sentinel Hub configuration from environment variables."""
    config = SHConfig()
    
    if SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET:
        config.sh_client_id = SENTINEL_HUB_CLIENT_ID
        config.sh_client_secret = SENTINEL_HUB_CLIENT_SECRET
        logger.info("Sentinel Hub credentials loaded from environment variables")
    else:
        logger.warning("Sentinel Hub credentials not found in environment variables")
    
    return config

def check_sentinel_hub_credentials(client_id: str, client_secret: str) -> bool:
    """Check if Sentinel Hub credentials are valid."""
    try:
        # Create a configuration with the provided credentials
        config = SHConfig()
        config.sh_client_id = client_id
        config.sh_client_secret = client_secret
        
        # Try to get an OAuth token, which will verify credentials
        config.sh_base_url = "https://services.sentinel-hub.com"
        
        # If token retrieval succeeds, credentials are valid
        token_info = config.get_token()
        return bool(token_info)
    except Exception as e:
        logger.error(f"Failed to validate Sentinel Hub credentials: {str(e)}")
        return False

async def fetch_with_rate_limit(function, *args, **kwargs):
    """Execute function with rate limiting."""
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def execute_with_retry():
        return function(*args, **kwargs)
    
    # Implement rate limiting based on settings
    await asyncio.sleep(1 / SETTINGS["rate_limits"]["sentinel_hub"])
    return execute_with_retry()

def parse_geojson(geojson_data: Union[str, Dict]) -> Tuple[Polygon, CRS]:
    """Parse GeoJSON data and return Shapely polygon and CRS."""
    if isinstance(geojson_data, str):
        geojson_data = json.loads(geojson_data)
    
    # Get the geometry from the GeoJSON
    if "geometry" in geojson_data:
        geometry = geojson_data["geometry"]
    elif "features" in geojson_data and len(geojson_data["features"]) > 0:
        geometry = geojson_data["features"][0]["geometry"]
    else:
        geometry = geojson_data
    
    # Create Shapely polygon from the geometry
    polygon = shape(geometry)
    
    # Get CRS from the GeoJSON or use default WGS84
    crs_string = geojson_data.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    crs = CRS.from_string(crs_string.split(":")[-1] if ":" in crs_string else "4326")
    
    return polygon, crs

def get_bbox_from_polygon(polygon: Polygon) -> BBox:
    """Convert Shapely polygon to Sentinel Hub BBox."""
    minx, miny, maxx, maxy = polygon.bounds
    return BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)

def get_available_scenes(
    bbox: BBox, 
    start_date: datetime.datetime, 
    end_date: datetime.datetime,
    max_cloud_coverage: float = 20.0
) -> List[Dict]:
    """Get available Sentinel-2 scenes for the given bbox and time range."""
    config = get_sentinel_hub_config()
    
    # Format dates for the API
    start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")
    
    # Create download client
    download_client = SentinelHubDownloadClient(config=config)
    
    # Create search request
    search_request = {
        "dataFilter": {
            "timeRange": {
                "from": start_date_str,
                "to": end_date_str
            },
            "maxCloudCoverage": max_cloud_coverage
        },
        "collections": [
            {"id": "sentinel-2-l2a"}
        ],
        "bbox": bbox.get_lower_left() + bbox.get_upper_right(),
        "limit": 50
    }
    
    try:
        # Execute search
        search_results = download_client.search(search_request)
        
        # Process and return results
        scenes = []
        for tile in search_results:
            scenes.append({
                "id": tile["id"],
                "date": tile["properties"]["datetime"],
                "cloud_cover": tile["properties"].get("eo:cloud_cover", 0),
                "metadata": tile
            })
        
        logger.info(f"Found {len(scenes)} scenes for the given bbox and time range")
        return scenes
    except Exception as e:
        logger.error(f"Error fetching available scenes: {str(e)}")
        return []

def fetch_sentinel_data(
    bbox: BBox, 
    time_interval: Tuple[str, str],
    resolution: int = 10,
    data_collection: DataCollection = DataCollection.SENTINEL2_L2A,
) -> Tuple[np.ndarray, Dict]:
    """Fetch Sentinel-2 data for the specified bounding box and time interval."""
    config = get_sentinel_hub_config()
    
    # Calculate image dimensions based on the bounding box and requested resolution
    width, height = bbox_to_dimensions(bbox, resolution=resolution)
    
    # Define the evaluation script for NDVI calculation
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: ["B02", "B03", "B04", "B08", "dataMask", "SCL"],
            output: [
                { id: "RGB", bands: 3 },
                { id: "NDVI", bands: 1 },
                { id: "SCL", bands: 1 }
            ]
        };
    }

    function evaluatePixel(sample) {
        let ndvi = index(sample.B08, sample.B04);
        
        return {
            RGB: [sample.B04, sample.B03, sample.B02],
            NDVI: [ndvi],
            SCL: [sample.SCL]
        };
    }
    """
    
    try:
        # Create SentinelHub request
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=data_collection,
                    time_interval=time_interval,
                    mosaicking_order='leastCC'
                )
            ],
            responses=[
                SentinelHubRequest.output_response('RGB', MimeType.PNG),
                SentinelHubRequest.output_response('NDVI', MimeType.TIFF),
                SentinelHubRequest.output_response('SCL', MimeType.TIFF)
            ],
            bbox=bbox,
            size=(width, height),
            config=config
        )
        
        # Get the data from SentinelHub
        data = request.get_data()
        
        # Extract the RGB, NDVI, and SCL from the response
        rgb_image = data[0]  # RGB image as PNG
        ndvi_image = data[1]  # NDVI as TIFF
        scl_image = data[2]  # Scene Classification Layer as TIFF
        
        # Create metadata dictionary
        metadata = {
            "bbox": bbox.get_lower_left() + bbox.get_upper_right(),
            "crs": bbox.crs.ogc_string(),
            "time_interval": time_interval,
            "resolution": resolution,
            "width": width,
            "height": height,
            "data_collection": str(data_collection),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Return the data and metadata as a tuple of arrays
        return {
            "rgb": rgb_image,
            "ndvi": ndvi_image,
            "scl": scl_image
        }, metadata
    
    except Exception as e:
        logger.error(f"Error fetching Sentinel data: {str(e)}")
        return None, {"error": str(e)}

def get_country_boundary(country_code: str) -> Optional[Polygon]:
    """Get country boundary polygon from ISO country code."""
    # This is a placeholder. In a real implementation, this would fetch
    # country boundaries from a geospatial database or service.
    logger.warning("get_country_boundary is a placeholder and does not actually retrieve boundaries")
    return None

def save_to_geotiff(image: np.ndarray, metadata: Dict, filename: str) -> str:
    """Save image and metadata to GeoTIFF file."""
    import rasterio
    from rasterio.transform import from_bounds
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create transform from metadata
    left, bottom, right, top = metadata.get('bbox', [0, 0, 1, 1])
    transform = from_bounds(left, bottom, right, top, 
                           metadata.get('width', image.shape[1]), 
                           metadata.get('height', image.shape[0]))
    
    # Get CRS from metadata
    crs = metadata.get('crs', 'EPSG:4326')
    
    # Determine number of bands
    count = 1
    if len(image.shape) > 2:
        count = image.shape[2]
    
    # Write the image to a GeoTIFF file
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=image.shape[0],
        width=image.shape[1],
        count=count,
        dtype=image.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        if count == 1:
            dst.write(image, 1)
        else:
            for i in range(count):
                dst.write(image[:, :, i], i + 1)
        
        # Write metadata
        dst.update_tags(**{k: str(v) for k, v in metadata.items() if isinstance(v, (str, int, float))})
    
    logger.info(f"Image saved to {filename}")
    return filename

def save_stac_metadata(metadata: Dict, filename: str) -> str:
    """Save STAC metadata to JSON file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert any non-serializable objects to strings
    def convert_to_serializable(obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        elif hasattr(obj, '__str__'):
            return str(obj)
        else:
            return obj
    
    serializable_metadata = {k: convert_to_serializable(v) for k, v in metadata.items()}
    
    # Write metadata to file
    with open(filename, 'w') as f:
        json.dump(serializable_metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {filename}")
    return filename