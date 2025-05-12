"""
Module for processing satellite data and extracting agricultural metrics.
"""
import os
import json
import logging
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import math

import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt

from config import (
    PROCESSED_DATA_DIR,
    SATELLITE_INDICES
)

# Set up logging
logger = logging.getLogger(__name__)

# Mask values for various satellite data products
MASK_VALUES = {
    'SCL': {
        'NO_DATA': 0,
        'SATURATED_DEFECTIVE': 1,
        'DARK_AREA_PIXELS': 2,
        'CLOUD_SHADOWS': 3,
        'VEGETATION': 4,
        'BARE_SOIL': 5,
        'WATER': 6,
        'CLOUDS_LOW_PROBA': 7,
        'CLOUDS_MED_PROBA': 8,
        'CLOUDS_HIGH_PROBA': 9,
        'CIRRUS': 10,
        'SNOW_ICE': 11
    }
}

# Valid ranges for various indices
INDEX_RANGES = {
    'NDVI': (-1.0, 1.0),
    'EVI': (-1.0, 1.0),
    'NDWI': (-1.0, 1.0),
    'NDMI': (-1.0, 1.0),
    'NBR': (-1.0, 1.0)
}

def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Calculate the Normalized Difference Vegetation Index.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        nir: Near-infrared band (B8 for Sentinel-2)
        red: Red band (B4 for Sentinel-2)
        
    Returns:
        NDVI array with values between -1.0 and 1.0
    """
    # Avoid division by zero
    denominator = nir + red
    valid_mask = denominator > 0
    
    # Initialize output array with NaN values
    ndvi = np.full_like(nir, np.nan, dtype=np.float32)
    
    # Calculate NDVI only for valid pixels
    ndvi[valid_mask] = (nir[valid_mask] - red[valid_mask]) / denominator[valid_mask]
    
    # Clip values to valid range
    ndvi = np.clip(ndvi, -1.0, 1.0)
    
    return ndvi

def calculate_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """
    Calculate the Enhanced Vegetation Index.
    
    EVI = G * ((NIR - Red) / (NIR + C1 * Red - C2 * Blue + L))
    
    Args:
        nir: Near-infrared band (B8 for Sentinel-2)
        red: Red band (B4 for Sentinel-2)
        blue: Blue band (B2 for Sentinel-2)
        
    Returns:
        EVI array with values between -1.0 and 1.0
    """
    # EVI constants
    G = 2.5
    C1 = 6.0
    C2 = 7.5
    L = 1.0
    
    # Avoid division by zero
    denominator = nir + C1 * red - C2 * blue + L
    valid_mask = denominator > 0
    
    # Initialize output array with NaN values
    evi = np.full_like(nir, np.nan, dtype=np.float32)
    
    # Calculate EVI only for valid pixels
    evi[valid_mask] = G * (nir[valid_mask] - red[valid_mask]) / denominator[valid_mask]
    
    # Clip values to valid range
    evi = np.clip(evi, -1.0, 1.0)
    
    return evi

def calculate_ndwi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """
    Calculate the Normalized Difference Water Index.
    
    NDWI = (NIR - SWIR) / (NIR + SWIR)
    
    Args:
        nir: Near-infrared band (B8 for Sentinel-2)
        swir: Short-wave infrared band (B11 for Sentinel-2)
        
    Returns:
        NDWI array with values between -1.0 and 1.0
    """
    # Avoid division by zero
    denominator = nir + swir
    valid_mask = denominator > 0
    
    # Initialize output array with NaN values
    ndwi = np.full_like(nir, np.nan, dtype=np.float32)
    
    # Calculate NDWI only for valid pixels
    ndwi[valid_mask] = (nir[valid_mask] - swir[valid_mask]) / denominator[valid_mask]
    
    # Clip values to valid range
    ndwi = np.clip(ndwi, -1.0, 1.0)
    
    return ndwi

def calculate_ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    Calculate the Normalized Difference Moisture Index.
    
    NDMI = (NIR - SWIR1) / (NIR + SWIR1)
    
    Args:
        nir: Near-infrared band (B8 for Sentinel-2)
        swir1: Short-wave infrared band (B11 for Sentinel-2)
        
    Returns:
        NDMI array with values between -1.0 and 1.0
    """
    # Avoid division by zero
    denominator = nir + swir1
    valid_mask = denominator > 0
    
    # Initialize output array with NaN values
    ndmi = np.full_like(nir, np.nan, dtype=np.float32)
    
    # Calculate NDMI only for valid pixels
    ndmi[valid_mask] = (nir[valid_mask] - swir1[valid_mask]) / denominator[valid_mask]
    
    # Clip values to valid range
    ndmi = np.clip(ndmi, -1.0, 1.0)
    
    return ndmi

def calculate_nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """
    Calculate the Normalized Burn Ratio.
    
    NBR = (NIR - SWIR2) / (NIR + SWIR2)
    
    Args:
        nir: Near-infrared band (B8 for Sentinel-2)
        swir2: Short-wave infrared band (B12 for Sentinel-2)
        
    Returns:
        NBR array with values between -1.0 and 1.0
    """
    # Avoid division by zero
    denominator = nir + swir2
    valid_mask = denominator > 0
    
    # Initialize output array with NaN values
    nbr = np.full_like(nir, np.nan, dtype=np.float32)
    
    # Calculate NBR only for valid pixels
    nbr[valid_mask] = (nir[valid_mask] - swir2[valid_mask]) / denominator[valid_mask]
    
    # Clip values to valid range
    nbr = np.clip(nbr, -1.0, 1.0)
    
    return nbr

def apply_cloud_mask(
    image: np.ndarray, 
    scl: np.ndarray,
    valid_classes: List[int] = [4, 5, 6, 11]  # Vegetation, Bare Soil, Water, Snow/Ice
) -> np.ndarray:
    """
    Apply cloud mask to an image using the Scene Classification Layer (SCL).
    
    Args:
        image: Image array to mask
        scl: Scene Classification Layer array
        valid_classes: List of valid SCL classes to keep
        
    Returns:
        Masked array with invalid pixels set to NaN
    """
    # Create a mask where True means the pixel is valid
    valid_mask = np.isin(scl, valid_classes)
    
    # Make a copy of the input image
    masked_image = image.copy()
    
    # Set invalid pixels to NaN
    masked_image[~valid_mask] = np.nan
    
    logger.info(f"Applied cloud mask: {np.sum(~valid_mask) / valid_mask.size:.2%} of pixels masked")
    return masked_image

def calculate_zonal_statistics(
    image: np.ndarray, 
    mask: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate zonal statistics for an image, optionally within a mask.
    
    Args:
        image: Image array
        mask: Optional binary mask array (1 for valid pixels, 0 for invalid)
        
    Returns:
        Dictionary with statistics (min, max, mean, median, std)
    """
    # Apply mask if provided
    if mask is not None:
        valid_pixels = image[mask == 1]
    else:
        valid_pixels = image[~np.isnan(image)]
    
    # Calculate statistics only if there are valid pixels
    if len(valid_pixels) > 0:
        stats = {
            "min": float(np.nanmin(valid_pixels)),
            "max": float(np.nanmax(valid_pixels)),
            "mean": float(np.nanmean(valid_pixels)),
            "median": float(np.nanmedian(valid_pixels)),
            "std": float(np.nanstd(valid_pixels)),
            "count": int(len(valid_pixels)),
            "percent_valid": float(len(valid_pixels) / image.size)
        }
    else:
        stats = {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "count": 0,
            "percent_valid": 0.0
        }
    
    return stats

def extract_time_series(
    images: List[np.ndarray],
    dates: List[datetime.datetime],
    mask: np.ndarray = None,
    stat: str = "mean"
) -> Dict[str, float]:
    """
    Extract time series from a list of images.
    
    Args:
        images: List of image arrays
        dates: List of dates corresponding to the images
        mask: Optional binary mask array (1 for valid pixels, 0 for invalid)
        stat: Statistic to extract ('mean', 'median', 'min', 'max')
        
    Returns:
        Dictionary mapping date strings to values
    """
    stat_functions = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "min": np.nanmin,
        "max": np.nanmax
    }
    
    if stat not in stat_functions:
        raise ValueError(f"Invalid statistic: {stat}. Must be one of {list(stat_functions.keys())}")
    
    # Calculate the requested statistic for each image
    time_series = {}
    for i, (image, date) in enumerate(zip(images, dates)):
        if mask is not None:
            valid_pixels = image[mask == 1]
        else:
            valid_pixels = image[~np.isnan(image)]
        
        if len(valid_pixels) > 0:
            value = float(stat_functions[stat](valid_pixels))
        else:
            value = None
        
        # Use ISO format date as key
        time_series[date.isoformat()] = value
    
    return time_series

def detect_anomalies(
    time_series: Dict[str, float],
    method: str = "zscore",
    threshold: float = 2.0
) -> Dict[str, Dict[str, Any]]:
    """
    Detect anomalies in a time series.
    
    Args:
        time_series: Dictionary mapping date strings to values
        method: Method for anomaly detection ('zscore', 'iqr', 'rolling')
        threshold: Threshold for detecting anomalies
        
    Returns:
        Dictionary with detected anomalies and their scores
    """
    # Convert time series to pandas Series
    dates = list(time_series.keys())
    values = list(time_series.values())
    
    # Remove NaN values
    valid_indices = [i for i, v in enumerate(values) if v is not None]
    valid_dates = [dates[i] for i in valid_indices]
    valid_values = [values[i] for i in valid_indices]
    
    if len(valid_values) < 3:
        logger.warning("Not enough valid data points for anomaly detection")
        return {}
    
    series = pd.Series(valid_values, index=pd.to_datetime(valid_dates))
    
    # Detect anomalies using the specified method
    anomalies = {}
    
    if method == "zscore":
        # Z-score method
        mean = np.mean(valid_values)
        std = np.std(valid_values)
        
        if std == 0:
            logger.warning("Standard deviation is zero, cannot calculate Z-scores")
            return {}
        
        z_scores = [(v - mean) / std for v in valid_values]
        
        for i, (date, value, z) in enumerate(zip(valid_dates, valid_values, z_scores)):
            if abs(z) > threshold:
                anomalies[date] = {
                    "value": value,
                    "score": z,
                    "type": "high" if z > 0 else "low"
                }
    
    elif method == "iqr":
        # Interquartile Range method
        q1 = np.percentile(valid_values, 25)
        q3 = np.percentile(valid_values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        for i, (date, value) in enumerate(zip(valid_dates, valid_values)):
            if value < lower_bound:
                anomalies[date] = {
                    "value": value,
                    "score": (lower_bound - value) / iqr,
                    "type": "low"
                }
            elif value > upper_bound:
                anomalies[date] = {
                    "value": value,
                    "score": (value - upper_bound) / iqr,
                    "type": "high"
                }
    
    elif method == "rolling":
        # Rolling window method
        window_size = min(5, len(valid_values) // 2)
        if window_size < 2:
            window_size = 2
        
        rolling_mean = series.rolling(window=window_size, center=True).mean()
        rolling_std = series.rolling(window=window_size, center=True).std()
        
        for date, value in zip(valid_dates, valid_values):
            idx = pd.to_datetime(date)
            if idx in rolling_mean.index and not np.isnan(rolling_mean[idx]) and not np.isnan(rolling_std[idx]):
                z = (value - rolling_mean[idx]) / rolling_std[idx]
                if abs(z) > threshold:
                    anomalies[date] = {
                        "value": value,
                        "score": z,
                        "type": "high" if z > 0 else "low"
                    }
    
    else:
        raise ValueError(f"Invalid anomaly detection method: {method}")
    
    return anomalies

def save_processed_data(
    field_name: str,
    data_type: str,
    data: Any,
    metadata: Dict = None
) -> str:
    """
    Save processed data to file.
    
    Args:
        field_name: Name of the field
        data_type: Type of data ('ndvi', 'evi', 'time_series', etc.)
        data: Data to save
        metadata: Optional metadata dictionary
        
    Returns:
        Path to the saved file
    """
    # Create field directory if it doesn't exist
    field_dir = os.path.join(PROCESSED_DATA_DIR, field_name)
    os.makedirs(field_dir, exist_ok=True)
    
    # Determine file format based on data type
    if data_type.lower() in ['ndvi', 'evi', 'ndwi', 'ndmi', 'nbr']:
        # For raster data, save as NumPy array
        filename = os.path.join(field_dir, f"{data_type.lower()}.npy")
        np.save(filename, data)
    elif data_type.lower() in ['time_series', 'anomalies', 'statistics', 'metadata']:
        # For time series and metadata, save as JSON
        filename = os.path.join(field_dir, f"{data_type.lower()}.json")
        
        # Handle NumPy types in the data
        def convert_np(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                                np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
            return obj
        
        with open(filename, 'w') as f:
            json.dump(data, f, default=convert_np, indent=2)
    else:
        # For unknown data types, save as pickle
        filename = os.path.join(field_dir, f"{data_type.lower()}.pkl")
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    logger.info(f"Saved {data_type} data for field {field_name} to {filename}")
    
    # Save metadata if provided
    if metadata:
        metadata_filename = os.path.join(field_dir, f"{data_type.lower()}_metadata.json")
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, default=convert_np, indent=2)
        logger.info(f"Saved {data_type} metadata for field {field_name} to {metadata_filename}")
    
    return filename

def load_processed_data(field_name: str, data_type: str) -> Tuple[Any, Optional[Dict]]:
    """
    Load processed data from file.
    
    Args:
        field_name: Name of the field
        data_type: Type of data ('ndvi', 'evi', 'time_series', etc.)
        
    Returns:
        Tuple of (data, metadata)
    """
    # Create paths for data and metadata files
    field_dir = os.path.join(PROCESSED_DATA_DIR, field_name)
    
    # Check if field directory exists
    if not os.path.exists(field_dir):
        logger.warning(f"Field directory does not exist: {field_dir}")
        return None, None
    
    # Determine file format based on data type
    data = None
    metadata = None
    
    if data_type.lower() in ['ndvi', 'evi', 'ndwi', 'ndmi', 'nbr']:
        # For raster data, load from NumPy array
        filename = os.path.join(field_dir, f"{data_type.lower()}.npy")
        if os.path.exists(filename):
            data = np.load(filename)
            logger.info(f"Loaded {data_type} data for field {field_name} from {filename}")
        else:
            logger.warning(f"File does not exist: {filename}")
    
    elif data_type.lower() in ['time_series', 'anomalies', 'statistics', 'metadata']:
        # For time series and metadata, load from JSON
        filename = os.path.join(field_dir, f"{data_type.lower()}.json")
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded {data_type} data for field {field_name} from {filename}")
        else:
            logger.warning(f"File does not exist: {filename}")
    
    else:
        # For unknown data types, load from pickle
        filename = os.path.join(field_dir, f"{data_type.lower()}.pkl")
        if os.path.exists(filename):
            import pickle
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded {data_type} data for field {field_name} from {filename}")
        else:
            logger.warning(f"File does not exist: {filename}")
    
    # Try to load metadata if it exists
    metadata_filename = os.path.join(field_dir, f"{data_type.lower()}_metadata.json")
    if os.path.exists(metadata_filename):
        with open(metadata_filename, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded {data_type} metadata for field {field_name} from {metadata_filename}")
    
    return data, metadata