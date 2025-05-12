import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import datetime
from pathlib import Path
import json
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, box, Polygon
import geopandas as gpd
from diskcache import Cache

# Configure logger
logger = logging.getLogger(__name__)

# Configure cache
cache_dir = Path("./cache")
cache_dir.mkdir(exist_ok=True)
cache = Cache(str(cache_dir))

def calculate_ndvi(red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index (NDVI).
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        red_band: Red band (B04) from Sentinel-2
        nir_band: Near Infrared band (B08) from Sentinel-2
        
    Returns:
        NDVI array with values between -1 and 1
    """
    # Avoid division by zero
    denominator = nir_band + red_band
    ndvi = np.where(
        denominator > 0,
        (nir_band - red_band) / denominator,
        0
    )
    
    return ndvi

def calculate_evi(red_band: np.ndarray, nir_band: np.ndarray, blue_band: np.ndarray) -> np.ndarray:
    """
    Calculate Enhanced Vegetation Index (EVI).
    
    EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
    
    Args:
        red_band: Red band (B04) from Sentinel-2
        nir_band: Near Infrared band (B08) from Sentinel-2
        blue_band: Blue band (B02) from Sentinel-2
        
    Returns:
        EVI array
    """
    denominator = nir_band + (6 * red_band) - (7.5 * blue_band) + 1
    
    # Avoid division by zero
    evi = np.where(
        denominator > 0,
        2.5 * ((nir_band - red_band) / denominator),
        0
    )
    
    return evi

def calculate_ndwi(green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Water Index (NDWI).
    
    NDWI = (Green - NIR) / (Green + NIR)
    
    Args:
        green_band: Green band (B03) from Sentinel-2
        nir_band: Near Infrared band (B08) from Sentinel-2
        
    Returns:
        NDWI array with values between -1 and 1
    """
    # Avoid division by zero
    denominator = green_band + nir_band
    ndwi = np.where(
        denominator > 0,
        (green_band - nir_band) / denominator,
        0
    )
    
    return ndwi

def extract_bands_from_sentinel(sentinel_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract individual bands from Sentinel-2 data.
    
    Args:
        sentinel_data: Multi-band Sentinel-2 image array
        
    Returns:
        Dictionary with individual bands
    """
    # Sentinel-2 bands in the order from fetch_sentinel_data:
    # B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
    
    bands = {
        "blue": sentinel_data[:, :, 0],      # B02
        "green": sentinel_data[:, :, 1],     # B03
        "red": sentinel_data[:, :, 2],       # B04
        "red_edge_1": sentinel_data[:, :, 3], # B05
        "red_edge_2": sentinel_data[:, :, 4], # B06
        "red_edge_3": sentinel_data[:, :, 5], # B07
        "nir": sentinel_data[:, :, 6],       # B08
        "nir_narrow": sentinel_data[:, :, 7], # B8A
        "swir_1": sentinel_data[:, :, 8],    # B11
        "swir_2": sentinel_data[:, :, 9],    # B12
    }
    
    return bands

def calculate_vegetation_indices(sentinel_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate vegetation indices from Sentinel-2 data.
    
    Args:
        sentinel_data: Multi-band Sentinel-2 image array
        
    Returns:
        Dictionary with vegetation indices
    """
    bands = extract_bands_from_sentinel(sentinel_data)
    
    # Calculate indices
    indices = {
        "ndvi": calculate_ndvi(bands["red"], bands["nir"]),
        "evi": calculate_evi(bands["red"], bands["nir"], bands["blue"]),
        "ndwi": calculate_ndwi(bands["green"], bands["nir"]),
    }
    
    return indices

def calculate_zonal_statistics(
    index_array: np.ndarray, 
    geometry: Polygon,
    transform=None,
    nodata=None
) -> Dict[str, float]:
    """
    Calculate zonal statistics for a given index within a polygon.
    
    Args:
        index_array: Array containing index values
        geometry: Shapely polygon defining the zone
        transform: Affine transform for the array
        nodata: Value to treat as nodata
        
    Returns:
        Dictionary with statistics
    """
    # Convert polygon to GeoJSON format
    geoms = [geometry.__geo_interface__]
    
    # Create a mask for the polygon
    if transform is not None:
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=index_array.shape[0],
                width=index_array.shape[1],
                count=1,
                dtype=index_array.dtype,
                transform=transform,
                nodata=nodata
            ) as dataset:
                dataset.write(index_array, 1)
                masked_array, mask_transform = mask(dataset, geoms, crop=True, nodata=nodata)
    else:
        # For testing/simple cases without transform
        masked_array = index_array
    
    # Flatten and remove nodata values
    valid_data = masked_array.flatten()
    if nodata is not None:
        valid_data = valid_data[valid_data != nodata]
    
    # Calculate statistics
    stats = {
        "min": float(np.nanmin(valid_data)) if len(valid_data) > 0 else None,
        "max": float(np.nanmax(valid_data)) if len(valid_data) > 0 else None,
        "mean": float(np.nanmean(valid_data)) if len(valid_data) > 0 else None,
        "median": float(np.nanmedian(valid_data)) if len(valid_data) > 0 else None,
        "std": float(np.nanstd(valid_data)) if len(valid_data) > 0 else None,
        "count": int(np.count_nonzero(~np.isnan(valid_data)))
    }
    
    return stats

def detect_anomalies(
    time_series: List[float], 
    threshold: float = 2.0
) -> List[int]:
    """
    Detect anomalies in a time series using z-score.
    
    Args:
        time_series: List of values
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        List of indices where anomalies were detected
    """
    # Convert to numpy array
    array = np.array(time_series)
    
    # Calculate z-scores
    mean = np.mean(array)
    std = np.std(array)
    
    if std == 0:
        return []
    
    z_scores = (array - mean) / std
    
    # Find anomalies
    anomalies = np.where(np.abs(z_scores) > threshold)[0].tolist()
    
    return anomalies

def detect_drought(ndvi_time_series: List[float], dates: List[str]) -> List[Dict[str, Any]]:
    """
    Detect drought conditions based on NDVI time series.
    
    Drought condition: NDVI decrease >= 0.15 for two or more consecutive periods.
    
    Args:
        ndvi_time_series: List of NDVI values
        dates: List of dates corresponding to NDVI values
        
    Returns:
        List of drought events with start date, end date, and severity
    """
    drought_events = []
    
    # Calculate NDVI differences
    ndvi_diff = np.diff(ndvi_time_series)
    
    # Find potential drought starts (NDVI decrease >= 0.15)
    drought_starts = np.where(ndvi_diff <= -0.15)[0]
    
    if len(drought_starts) == 0:
        return []
    
    # Check for consecutive periods
    current_drought = None
    
    for i in range(len(drought_starts)):
        idx = drought_starts[i]
        
        # Check if this is a continuation of the current drought
        if current_drought and idx == current_drought["end_idx"] + 1:
            # Extend the current drought
            current_drought["end_idx"] = idx + 1
            current_drought["end_date"] = dates[idx + 1]
            current_drought["duration"] += 1
            current_drought["severity"] = max(
                current_drought["severity"],
                abs(ndvi_time_series[idx + 1] - ndvi_time_series[idx])
            )
        else:
            # If we have a previous drought event with duration >= 2, save it
            if current_drought and current_drought["duration"] >= 2:
                drought_events.append(current_drought)
            
            # Start a new drought event
            current_drought = {
                "start_idx": idx,
                "start_date": dates[idx],
                "end_idx": idx + 1,
                "end_date": dates[idx + 1],
                "duration": 1,
                "severity": abs(ndvi_time_series[idx + 1] - ndvi_time_series[idx])
            }
    
    # Don't forget to add the last drought if it meets the criteria
    if current_drought and current_drought["duration"] >= 2:
        drought_events.append(current_drought)
    
    return drought_events

def calculate_rolling_variance(
    time_series: List[float], 
    window_size: int = 3
) -> List[float]:
    """
    Calculate rolling variance for a time series.
    
    Args:
        time_series: List of values
        window_size: Size of the rolling window
        
    Returns:
        List of rolling variance values
    """
    # Convert to pandas Series for easy rolling calculations
    series = pd.Series(time_series)
    
    # Calculate rolling variance
    rolling_var = series.rolling(window=window_size).var().tolist()
    
    # Fill NaN values at the beginning with the first valid variance
    first_valid = next((x for x in rolling_var if not pd.isna(x)), 0)
    rolling_var = [first_valid if pd.isna(x) else x for x in rolling_var]
    
    return rolling_var

def perform_pca_clustering(
    indices_matrix: np.ndarray,
    n_clusters: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform PCA and K-means clustering on multi-season indices.
    
    Args:
        indices_matrix: Matrix with dimensions (pixels, features)
        n_clusters: Number of clusters for K-means
        
    Returns:
        Tuple of (pca_components, explained_variance, cluster_labels)
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    
    # Remove any rows with NaN values
    valid_indices = ~np.isnan(indices_matrix).any(axis=1)
    valid_data = indices_matrix[valid_indices]
    
    if len(valid_data) == 0:
        return None, None, None
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(valid_data)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_result[:, :2])  # Use first two components
    
    # Create full cluster map (including NaN pixels)
    full_cluster_map = np.full(len(indices_matrix), -1)  # -1 for invalid pixels
    full_cluster_map[valid_indices] = cluster_labels
    
    return pca_result, pca.explained_variance_ratio_, full_cluster_map
