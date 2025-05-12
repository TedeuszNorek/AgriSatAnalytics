"""
Module for visualizing satellite data and agricultural metrics.
"""
import os
import json
import logging
import datetime
import base64
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import folium
from folium import plugins
from shapely.geometry import Polygon, shape

from config import (
    REPORTS_DIR,
    SATELLITE_INDICES
)

# Set up logging
logger = logging.getLogger(__name__)

# Colormap definitions for different indices
COLORMAPS = {
    'NDVI': plt.cm.RdYlGn,    # Red-Yellow-Green
    'EVI': plt.cm.RdYlGn,     # Red-Yellow-Green
    'NDWI': plt.cm.Blues,     # Blue
    'NDMI': plt.cm.Blues,     # Blue
    'NBR': plt.cm.RdYlBu_r,   # Reversed Red-Yellow-Blue
    'RGB': None,              # True color
    'SCL': plt.cm.tab20       # Categorical
}

# Color ranges for different indices
COLOR_RANGES = {
    'NDVI': (-0.1, 1.0),
    'EVI': (-0.1, 1.0),
    'NDWI': (-0.3, 1.0),
    'NDMI': (-0.3, 1.0),
    'NBR': (-0.5, 1.0)
}

def create_colorbar(index_type: str) -> Tuple[plt.cm.ScalarMappable, Tuple[float, float]]:
    """
    Create a colorbar for a given index type.
    
    Args:
        index_type: Type of index ('NDVI', 'EVI', etc.)
        
    Returns:
        Tuple of (colormap, value range)
    """
    cmap = COLORMAPS.get(index_type, plt.cm.viridis)
    value_range = COLOR_RANGES.get(index_type, (-1.0, 1.0))
    
    norm = mcolors.Normalize(vmin=value_range[0], vmax=value_range[1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    return sm, value_range

def create_index_map(
    index_array: np.ndarray,
    index_type: str = 'NDVI',
    title: str = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Figure:
    """
    Create a map visualization for a satellite index.
    
    Args:
        index_array: Array with index values
        index_type: Type of index ('NDVI', 'EVI', etc.)
        title: Title for the map
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colormap and value range for the index
    cmap = COLORMAPS.get(index_type, plt.cm.viridis)
    value_range = COLOR_RANGES.get(index_type, (-1.0, 1.0))
    
    # Create image
    im = ax.imshow(index_array, cmap=cmap, vmin=value_range[0], vmax=value_range[1])
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label(SATELLITE_INDICES.get(index_type, index_type))
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{SATELLITE_INDICES.get(index_type, index_type)}")
    
    # Remove axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add scale bar (placeholder)
    ax.text(0.02, 0.02, "Scale: 1 pixel = 10m", transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.7))
    
    # Add north arrow (placeholder)
    ax.text(0.98, 0.02, "↑N", transform=ax.transAxes, ha='right',
            bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_rgb_image(
    rgb_array: np.ndarray,
    title: str = "RGB Composite",
    figsize: Tuple[int, int] = (10, 8)
) -> Figure:
    """
    Create an RGB image visualization.
    
    Args:
        rgb_array: Array with RGB values (shape: height, width, 3)
        title: Title for the image
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display RGB image
    ax.imshow(rgb_array)
    
    # Set title
    ax.set_title(title)
    
    # Remove axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add scale bar (placeholder)
    ax.text(0.02, 0.02, "Scale: 1 pixel = 10m", transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.7))
    
    # Add north arrow (placeholder)
    ax.text(0.98, 0.02, "↑N", transform=ax.transAxes, ha='right',
            bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_multi_temporal_figure(
    time_series: Dict[str, float],
    title: str = "Temporal Changes",
    y_label: str = "Value",
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Create a multi-temporal visualization from time series data.
    
    Args:
        time_series: Dictionary mapping date strings to values
        title: Title for the figure
        y_label: Label for y-axis
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert dates to datetime objects and handle missing values
    dates = []
    values = []
    
    for date_str, value in time_series.items():
        if value is not None:
            try:
                date = datetime.datetime.fromisoformat(date_str)
                dates.append(date)
                values.append(value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid date format: {date_str}")
    
    if not dates:
        ax.text(0.5, 0.5, "No valid data points", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    # Plot data
    ax.plot(dates, values, 'o-', linewidth=2, markersize=6)
    
    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight min and max values
    if values:
        min_value = min(values)
        max_value = max(values)
        
        min_index = values.index(min_value)
        max_index = values.index(max_value)
        
        ax.plot(dates[min_index], min_value, 'v', color='blue', 
                markersize=10, label=f'Min: {min_value:.3f}')
        ax.plot(dates[max_index], max_value, '^', color='red', 
                markersize=10, label=f'Max: {max_value:.3f}')
        
        ax.legend()
    
    plt.tight_layout()
    return fig

def create_anomaly_figure(
    time_series: Dict[str, float],
    anomalies: Dict[str, Dict[str, Any]],
    title: str = "Anomaly Detection",
    y_label: str = "Value",
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Create an anomaly detection visualization.
    
    Args:
        time_series: Dictionary mapping date strings to values
        anomalies: Dictionary mapping date strings to anomaly information
        title: Title for the figure
        y_label: Label for y-axis
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert dates to datetime objects and handle missing values
    dates = []
    values = []
    
    for date_str, value in time_series.items():
        if value is not None:
            try:
                date = datetime.datetime.fromisoformat(date_str)
                dates.append(date)
                values.append(value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid date format: {date_str}")
    
    if not dates:
        ax.text(0.5, 0.5, "No valid data points", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    # Plot regular data points
    ax.plot(dates, values, 'o-', color='black', linewidth=1, 
            markersize=5, label='Normal')
    
    # Plot anomalies
    anomaly_dates = []
    anomaly_values = []
    anomaly_types = []
    
    for date_str, anomaly_info in anomalies.items():
        try:
            date = datetime.datetime.fromisoformat(date_str)
            value = anomaly_info["value"]
            anomaly_type = anomaly_info.get("type", "unknown")
            
            anomaly_dates.append(date)
            anomaly_values.append(value)
            anomaly_types.append(anomaly_type)
        except (ValueError, TypeError, KeyError):
            logger.warning(f"Invalid anomaly data for date: {date_str}")
    
    # Plot high and low anomalies with different colors
    high_dates = [date for date, atype in zip(anomaly_dates, anomaly_types) if atype == "high"]
    high_values = [value for value, atype in zip(anomaly_values, anomaly_types) if atype == "high"]
    
    low_dates = [date for date, atype in zip(anomaly_dates, anomaly_types) if atype == "low"]
    low_values = [value for value, atype in zip(anomaly_values, anomaly_types) if atype == "low"]
    
    if high_dates:
        ax.scatter(high_dates, high_values, color='red', s=100, 
                   marker='^', label='High Anomaly', zorder=10)
    
    if low_dates:
        ax.scatter(low_dates, low_values, color='blue', s=100, 
                   marker='v', label='Low Anomaly', zorder=10)
    
    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_folium_map(
    lat: float,
    lon: float,
    zoom: int = 13,
    tiles: str = 'OpenStreetMap'
) -> folium.Map:
    """
    Create a Folium map centered at a specified location.
    
    Args:
        lat: Latitude for map center
        lon: Longitude for map center
        zoom: Initial zoom level
        tiles: Tile provider for the map base layer
        
    Returns:
        Folium map object
    """
    # Create map
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=tiles)
    
    # Add scale
    folium.plugins.MeasureControl(position='bottomleft').add_to(m)
    
    # Add fullscreen option
    folium.plugins.Fullscreen(position='topleft').add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright').add_to(m)
    
    return m

def add_geojson_to_map(
    m: folium.Map,
    geojson_data: Dict,
    name: str = "Field Boundary",
    style: Dict = None
) -> folium.Map:
    """
    Add a GeoJSON layer to a Folium map.
    
    Args:
        m: Folium map object
        geojson_data: GeoJSON data as a dictionary
        name: Name for the layer
        style: Style for the GeoJSON layer
        
    Returns:
        Updated Folium map
    """
    # Default style if none provided
    if style is None:
        style = {
            'fillColor': '#28a745',
            'color': '#28a745',
            'weight': 3,
            'fillOpacity': 0.3
        }
    
    # Add GeoJSON layer
    folium.GeoJson(
        geojson_data,
        name=name,
        style_function=lambda x: style
    ).add_to(m)
    
    return m

def create_comparison_figure(
    before_image: np.ndarray,
    after_image: np.ndarray,
    before_date: str,
    after_date: str,
    index_type: str = 'NDVI',
    figsize: Tuple[int, int] = (16, 8)
) -> Figure:
    """
    Create a side-by-side comparison of images from two dates.
    
    Args:
        before_image: Array for the earlier date
        after_image: Array for the later date
        before_date: Date string for the earlier image
        after_date: Date string for the later image
        index_type: Type of index ('NDVI', 'EVI', etc.)
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Get colormap and value range for the index
    cmap = COLORMAPS.get(index_type, plt.cm.viridis)
    value_range = COLOR_RANGES.get(index_type, (-1.0, 1.0))
    
    # Display before image
    im1 = ax1.imshow(before_image, cmap=cmap, vmin=value_range[0], vmax=value_range[1])
    ax1.set_title(f"{before_date}")
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Display after image
    im2 = ax2.imshow(after_image, cmap=cmap, vmin=value_range[0], vmax=value_range[1])
    ax2.set_title(f"{after_date}")
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Add common colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label(SATELLITE_INDICES.get(index_type, index_type))
    
    # Add super title
    fig.suptitle(f"{SATELLITE_INDICES.get(index_type, index_type)} Comparison", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    return fig

def create_difference_figure(
    before_image: np.ndarray,
    after_image: np.ndarray,
    before_date: str,
    after_date: str,
    index_type: str = 'NDVI',
    figsize: Tuple[int, int] = (12, 8)
) -> Figure:
    """
    Create a difference map between two dates.
    
    Args:
        before_image: Array for the earlier date
        after_image: Array for the later date
        before_date: Date string for the earlier image
        after_date: Date string for the later image
        index_type: Type of index ('NDVI', 'EVI', etc.)
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate difference
    diff = after_image - before_image
    
    # Create color map for difference
    diff_cmap = plt.cm.RdBu_r
    max_diff = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
    diff_range = (-max_diff, max_diff)
    
    # Display difference
    im = ax.imshow(diff, cmap=diff_cmap, vmin=diff_range[0], vmax=diff_range[1])
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    cbar.set_label(f"{index_type} Difference")
    
    # Set title
    ax.set_title(f"{index_type} Change: {before_date} to {after_date}")
    
    # Remove axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def create_histogram_figure(
    index_array: np.ndarray,
    index_type: str = 'NDVI',
    title: str = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Create a histogram visualization for index values.
    
    Args:
        index_array: Array with index values
        index_type: Type of index ('NDVI', 'EVI', etc.)
        title: Title for the histogram
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get value range for the index
    value_range = COLOR_RANGES.get(index_type, (-1.0, 1.0))
    
    # Flatten the array and remove NaN values
    valid_values = index_array[~np.isnan(index_array)].flatten()
    
    if len(valid_values) > 0:
        # Create histogram
        n, bins, patches = ax.hist(
            valid_values, 
            bins=50, 
            range=value_range, 
            alpha=0.7, 
            color='steelblue',
            density=True
        )
        
        # Add a kernel density estimate
        from scipy import stats
        kde = stats.gaussian_kde(valid_values)
        x = np.linspace(value_range[0], value_range[1], 100)
        ax.plot(x, kde(x), 'r-', linewidth=2)
        
        # Add vertical lines for statistics
        mean_val = np.mean(valid_values)
        median_val = np.median(valid_values)
        
        ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='orange', linestyle='-.', linewidth=2, 
                   label=f'Median: {median_val:.3f}')
        
        # Add legend
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid data points", 
                ha='center', va='center', transform=ax.transAxes)
    
    # Set labels and title
    ax.set_xlabel(SATELLITE_INDICES.get(index_type, index_type))
    ax.set_ylabel('Density')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{SATELLITE_INDICES.get(index_type, index_type)} Distribution")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def create_correlation_figure(
    x_data: Dict[str, float],
    y_data: Dict[str, float],
    x_label: str = "Variable X",
    y_label: str = "Variable Y",
    title: str = "Correlation Analysis",
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Create a correlation scatter plot between two variables.
    
    Args:
        x_data: Dictionary mapping dates to x values
        y_data: Dictionary mapping dates to y values
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Title for the figure
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract common dates and corresponding values
    common_dates = set(x_data.keys()) & set(y_data.keys())
    
    x_values = []
    y_values = []
    dates = []
    
    for date in common_dates:
        x_val = x_data.get(date)
        y_val = y_data.get(date)
        
        if x_val is not None and y_val is not None:
            x_values.append(x_val)
            y_values.append(y_val)
            dates.append(date)
    
    if not x_values or not y_values:
        ax.text(0.5, 0.5, "No matching data points", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    # Create scatter plot
    scatter = ax.scatter(x_values, y_values, alpha=0.7, s=50)
    
    # Calculate and plot linear regression
    if len(x_values) > 1:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
        
        x_line = np.array([min(x_values), max(x_values)])
        y_line = slope * x_line + intercept
        
        ax.plot(x_line, y_line, 'r-', linewidth=2, 
                label=f'y = {slope:.3f}x + {intercept:.3f}')
        
        # Add correlation coefficient
        ax.text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.3g}', 
                transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend if regression line exists
    if len(x_values) > 1:
        ax.legend()
    
    plt.tight_layout()
    return fig

def fig_to_base64(fig: Figure) -> str:
    """
    Convert a matplotlib figure to a base64 encoded string.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64 encoded string
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def save_figure(
    fig: Figure,
    filename: str,
    dpi: int = 150,
    bbox_inches: str = 'tight'
) -> str:
    """
    Save a figure to file.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box in inches
        
    Returns:
        Path to the saved file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save figure
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    logger.info(f"Figure saved to {filename}")
    
    return filename