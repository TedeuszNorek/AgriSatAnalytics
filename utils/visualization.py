"""
Funkcje wizualizacji dla aplikacji Agro Insight.
"""
import os
import logging
import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import folium
from folium.plugins import HeatMap, MarkerCluster

logger = logging.getLogger(__name__)

# Palety kolorów dla różnych typów wizualizacji
COLOR_PALETTES = {
    "ndvi": {
        "cmap": "RdYlGn",  # Red-Yellow-Green
        "vmin": -0.2,
        "vmax": 1.0,
        "bad_color": "grey"
    },
    "evi": {
        "cmap": "RdYlGn",  # Red-Yellow-Green
        "vmin": -0.2,
        "vmax": 1.0,
        "bad_color": "grey"
    },
    "soil_moisture": {
        "cmap": "Blues",  # Blues
        "vmin": 0.0,
        "vmax": 1.0,
        "bad_color": "grey"
    },
    "temperature": {
        "cmap": "Blues",  # Blues
        "vmin": 0.0,
        "vmax": 40.0,
        "bad_color": "grey"
    },
    "anomaly": {
        "cmap": "RdYlBu_r",  # Red-Yellow-Blue reversed
        "vmin": -3.0,
        "vmax": 3.0,
        "bad_color": "grey"
    },
    "crops": {
        "cmap": "tab20",  # Discrete colors for crop classification
        "vmin": 0,
        "vmax": 20,
        "bad_color": "grey"
    }
}

def plot_satellite_image(
    image: np.ndarray,
    image_type: str = "ndvi",
    title: str = "Satellite Image",
    colorbar_label: str = "Value",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot a satellite image with the appropriate colormap.
    
    Args:
        image: The image array to plot
        image_type: Type of image (ndvi, evi, soil_moisture, etc.)
        title: Plot title
        colorbar_label: Label for the colorbar
        figsize: Figure size in inches
        
    Returns:
        Matplotlib figure
    """
    # Get color palette for the image type
    palette = COLOR_PALETTES.get(image_type.lower(), {
        "cmap": "viridis",
        "vmin": None,
        "vmax": None,
        "bad_color": "grey"
    })
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create masked array for NaN values
    masked_image = np.ma.masked_invalid(image)
    
    # Plot the image
    im = ax.imshow(
        masked_image,
        cmap=palette["cmap"],
        vmin=palette["vmin"],
        vmax=palette["vmax"]
    )
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label=colorbar_label or image_type.upper())
    
    # Set title and remove axes ticks
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_time_series(
    dates: List[Union[str, datetime.datetime]],
    values: List[float],
    series_type: str = "ndvi",
    title: str = None,
    xlabel: str = "Date",
    ylabel: str = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot a time series.
    
    Args:
        dates: List of dates
        values: List of values
        series_type: Type of series (ndvi, evi, soil_moisture, etc.)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size in inches
        
    Returns:
        Matplotlib figure
    """
    # Convert string dates to datetime objects if needed
    if dates and isinstance(dates[0], str):
        dates = [datetime.datetime.fromisoformat(d) if "T" in d else datetime.datetime.strptime(d, "%Y-%m-%d") for d in dates]
    
    # Get color for the series type
    palette = COLOR_PALETTES.get(series_type.lower(), {
        "cmap": "viridis",
        "vmin": None,
        "vmax": None,
        "bad_color": "grey"
    })
    
    # Create colormap and get a color for the line
    if isinstance(palette["cmap"], str):
        cmap = plt.get_cmap(palette["cmap"])
        color = cmap(0.7)  # Get a color from the colormap
    else:
        color = "steelblue"
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the time series
    ax.plot(dates, values, marker='o', linestyle='-', color=color, markersize=4)
    
    # Format the x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Get min and max values for y-axis limits with some margin
    if values:
        ymin, ymax = min(values), max(values)
        y_margin = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
        ax.set_ylim(ymin - y_margin, ymax + y_margin)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or series_type.upper())
    ax.set_title(title or f"{series_type.upper()} Time Series")
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_field_map(
    lat: float,
    lon: float,
    geojson: Dict = None,
    zoom_start: int = 13,
    width: int = 800,
    height: int = 500,
    tooltip: str = None
) -> folium.Map:
    """
    Create a Folium map centered on a field location.
    
    Args:
        lat: Field center latitude
        lon: Field center longitude
        geojson: GeoJSON data for the field boundary
        zoom_start: Initial zoom level
        width: Map width in pixels
        height: Map height in pixels
        tooltip: Tooltip text to display on hover
        
    Returns:
        Folium map object
    """
    # Create map centered at the field location
    m = folium.Map(
        location=[lat, lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap"
    )
    
    # Add marker for the field center
    folium.Marker(
        location=[lat, lon],
        popup=f"Center: {lat:.6f}, {lon:.6f}",
        tooltip=tooltip,
        icon=folium.Icon(icon="leaf", prefix="fa", color="green")
    ).add_to(m)
    
    # Add field boundary if available
    if geojson:
        folium.GeoJson(
            geojson,
            name="Field Boundary",
            style_function=lambda x: {
                "fillColor": "#00ff00",
                "color": "#00aa00",
                "weight": 2,
                "fillOpacity": 0.2
            },
            tooltip=tooltip
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def plot_field_statistics(
    field_data: Dict[str, Any],
    title: str = "Field Statistics",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a dashboard of field statistics.
    
    Args:
        field_data: Dictionary containing field statistics
        title: Dashboard title
        figsize: Figure size in inches
        
    Returns:
        Matplotlib figure
    """
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Define grid layout
    gs = fig.add_gridspec(3, 4)
    
    # NDVI time series (larger plot)
    if "ndvi_time_series" in field_data and field_data["ndvi_time_series"]:
        ts_data = field_data["ndvi_time_series"]
        dates = list(ts_data.keys())
        values = list(ts_data.values())
        
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(dates, values, marker='o', linestyle='-', color='green', markersize=4)
        ax1.set_title("NDVI Time Series")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("NDVI")
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    
    # Weather data (if available)
    if "weather_data" in field_data and field_data["weather_data"]:
        weather = field_data["weather_data"]
        w_dates = list(weather.keys())
        
        # Temperature
        if "temperature" in weather[w_dates[0]]:
            temp_values = [weather[d]["temperature"] for d in w_dates]
            ax2 = fig.add_subplot(gs[1, :2])
            ax2.plot(w_dates, temp_values, marker='o', linestyle='-', color='red', markersize=3)
            ax2.set_title("Temperature")
            ax2.set_ylabel("Temperature (°C)")
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        
        # Precipitation
        if "precipitation" in weather[w_dates[0]]:
            precip_values = [weather[d]["precipitation"] for d in w_dates]
            ax3 = fig.add_subplot(gs[1, 2:])
            ax3.bar(w_dates, precip_values, color='blue', alpha=0.7)
            ax3.set_title("Precipitation")
            ax3.set_ylabel("Precipitation (mm)")
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
    
    # Yield forecast (if available)
    if "yield_forecast" in field_data and field_data["yield_forecast"]:
        forecast = field_data["yield_forecast"]
        f_dates = list(forecast.keys())
        f_values = list(forecast.values())
        
        ax4 = fig.add_subplot(gs[2, :2])
        ax4.plot(f_dates, f_values, marker='s', linestyle='-', color='purple', markersize=5)
        ax4.set_title("Yield Forecast")
        ax4.set_ylabel("Yield (t/ha)")
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")
    
    # Market signals (if available)
    if "market_signals" in field_data and field_data["market_signals"]:
        signals = field_data["market_signals"]
        
        # Extract dates and signal types
        s_dates = [s["date"] for s in signals]
        s_actions = [s["action"] for s in signals]
        s_confidences = [s["confidence"] for s in signals]
        
        ax5 = fig.add_subplot(gs[2, 2:])
        colors = ['green' if a == "LONG" else 'red' for a in s_actions]
        
        ax5.scatter(s_dates, s_confidences, c=colors, s=50, alpha=0.7)
        ax5.set_title("Market Signals")
        ax5.set_ylabel("Confidence")
        ax5.set_ylim(0, 1.1)
        ax5.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='LONG'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='SHORT')
        ]
        ax5.legend(handles=legend_elements, loc='upper right')
        
        plt.setp(ax5.get_xticklabels(), rotation=45, ha="right")
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

def create_heatmap_layer(
    data: List[Tuple[float, float, float]],
    name: str = "Heatmap",
    radius: int = 15
) -> HeatMap:
    """
    Create a heatmap layer for a Folium map.
    
    Args:
        data: List of (lat, lon, value) tuples
        name: Layer name
        radius: Heatmap radius
        
    Returns:
        HeatMap layer
    """
    return HeatMap(
        data,
        name=name,
        radius=radius,
        min_opacity=0.2,
        max_opacity=0.8,
        blur=10
    )

def plot_anomaly_detection(
    dates: List[Union[str, datetime.datetime]],
    values: List[float],
    anomalies: Dict[str, float] = None,
    title: str = "Anomaly Detection",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot a time series with detected anomalies.
    
    Args:
        dates: List of dates
        values: List of values
        anomalies: Dictionary mapping dates to anomaly scores
        title: Plot title
        figsize: Figure size in inches
        
    Returns:
        Matplotlib figure
    """
    # Convert string dates to datetime objects if needed
    if dates and isinstance(dates[0], str):
        dates = [datetime.datetime.fromisoformat(d) if "T" in d else datetime.datetime.strptime(d, "%Y-%m-%d") for d in dates]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the time series
    ax.plot(dates, values, marker='o', linestyle='-', color='steelblue', markersize=4, alpha=0.7, label="Values")
    
    # Plot anomalies if provided
    if anomalies:
        # Get anomaly dates and scores
        anomaly_dates = []
        anomaly_values = []
        anomaly_scores = []
        
        for i, d in enumerate(dates):
            # Convert to string format matching the anomalies dict
            date_str = d.strftime("%Y-%m-%d") if isinstance(d, datetime.datetime) else d
            
            if date_str in anomalies:
                anomaly_dates.append(d)
                anomaly_values.append(values[i])
                anomaly_scores.append(anomalies[date_str])
        
        # Plot the anomalies with size proportional to anomaly score
        sizes = [max(20, s * 100) for s in anomaly_scores]
        ax.scatter(anomaly_dates, anomaly_values, s=sizes, color='red', alpha=0.7, label="Anomalies")
    
    # Format the x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Set labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend()
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def create_choropleth_map(
    geojson_data: Dict,
    values: Dict[str, float],
    id_property: str,
    value_name: str = "Value",
    title: str = "Choropleth Map",
    colormap: str = "YlOrRd"
) -> folium.Map:
    """
    Create a choropleth map for regions with values.
    
    Args:
        geojson_data: GeoJSON data with region polygons
        values: Dictionary mapping region IDs to values
        id_property: Property in the GeoJSON that corresponds to region IDs
        value_name: Name of the value being displayed
        title: Map title
        colormap: Colormap to use
        
    Returns:
        Folium map with choropleth layer
    """
    # Find the center of the map
    import numpy as np
    lats = []
    lons = []
    
    for feature in geojson_data["features"]:
        if feature["geometry"]["type"] == "Polygon":
            for coord in feature["geometry"]["coordinates"][0]:
                lons.append(coord[0])
                lats.append(coord[1])
        elif feature["geometry"]["type"] == "MultiPolygon":
            for polygon in feature["geometry"]["coordinates"]:
                for coord in polygon[0]:
                    lons.append(coord[0])
                    lats.append(coord[1])
    
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    # Create the map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="OpenStreetMap"
    )
    
    # Add the choropleth layer
    folium.Choropleth(
        geo_data=geojson_data,
        name="Choropleth",
        data=values,
        columns=["Region", value_name],
        key_on=f"feature.properties.{id_property}",
        fill_color=colormap,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=value_name
    ).add_to(m)
    
    # Add tooltips
    folium.GeoJson(
        geojson_data,
        name="Labels",
        tooltip=folium.GeoJsonTooltip(
            fields=["name", value_name],
            aliases=["Region", value_name],
            localize=True
        )
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def create_index_map(
    image_data: np.ndarray,
    index_type: str = "ndvi",
    title: str = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = None,
    vmin: float = None,
    vmax: float = None
) -> plt.Figure:
    """
    Create a map visualization of a satellite index.
    
    Args:
        image_data: Numpy array containing the index values
        index_type: Type of index (ndvi, evi, etc.)
        title: Map title
        figsize: Figure size in inches
        cmap: Optional custom colormap
        vmin: Optional minimum value for color scaling
        vmax: Optional maximum value for color scaling
        
    Returns:
        Matplotlib figure
    """
    # Get color palette for the index type
    palette = COLOR_PALETTES.get(index_type.lower(), {
        "cmap": "viridis",
        "vmin": None,
        "vmax": None,
        "bad_color": "grey"
    })
    
    # Override with custom values if provided
    if cmap:
        palette["cmap"] = cmap
    if vmin is not None:
        palette["vmin"] = vmin
    if vmax is not None:
        palette["vmax"] = vmax
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create masked array for NaN values
    masked_data = np.ma.masked_invalid(image_data)
    
    # Plot the image
    im = ax.imshow(
        masked_data,
        cmap=palette["cmap"],
        vmin=palette["vmin"],
        vmax=palette["vmax"]
    )
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label=index_type.upper())
    
    # Set title and remove axes ticks
    ax.set_title(title or f"{index_type.upper()} Map")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def create_multi_temporal_figure(
    images: List[np.ndarray],
    dates: List[Union[str, datetime.datetime]],
    index_type: str = "ndvi",
    title: str = "Multi-temporal Analysis",
    figsize: Tuple[int, int] = (15, 10),
    max_cols: int = 3
) -> plt.Figure:
    """
    Create a multi-temporal visualization of satellite indices.
    
    Args:
        images: List of numpy arrays containing index values
        dates: List of dates corresponding to the images
        index_type: Type of index (ndvi, evi, etc.)
        title: Figure title
        figsize: Figure size in inches
        max_cols: Maximum number of columns in the grid
        
    Returns:
        Matplotlib figure
    """
    # Get color palette for the index type
    palette = COLOR_PALETTES.get(index_type.lower(), {
        "cmap": "viridis",
        "vmin": None,
        "vmax": None,
        "bad_color": "grey"
    })
    
    # Calculate grid dimensions
    n_images = len(images)
    n_cols = min(max_cols, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure and axes grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Ensure axes is always a 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each image
    for i, (img, date) in enumerate(zip(images, dates)):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Format the date
        if isinstance(date, datetime.datetime):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = date
        
        # Create masked array for NaN values
        masked_img = np.ma.masked_invalid(img)
        
        # Plot the image
        im = ax.imshow(
            masked_img,
            cmap=palette["cmap"],
            vmin=palette["vmin"],
            vmax=palette["vmax"]
        )
        
        # Set title and remove axes ticks
        ax.set_title(date_str)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    # Add a single colorbar for the whole figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label=index_type.upper())
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.9, top=0.9)
    
    return fig

def create_anomaly_figure(
    base_image: np.ndarray,
    anomaly_image: np.ndarray,
    title: str = "Anomaly Detection",
    figsize: Tuple[int, int] = (15, 6)
) -> plt.Figure:
    """
    Create a visualization comparing base image with anomaly detection.
    
    Args:
        base_image: Numpy array containing the original index values
        anomaly_image: Numpy array containing the anomaly scores
        title: Figure title
        figsize: Figure size in inches
        
    Returns:
        Matplotlib figure
    """
    # Get color palettes
    base_palette = COLOR_PALETTES.get("ndvi", {
        "cmap": "RdYlGn",
        "vmin": -0.2,
        "vmax": 1.0,
        "bad_color": "grey"
    })
    
    anomaly_palette = COLOR_PALETTES.get("anomaly", {
        "cmap": "RdYlBu_r",
        "vmin": -3.0,
        "vmax": 3.0,
        "bad_color": "grey"
    })
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Plot the base image
    masked_base = np.ma.masked_invalid(base_image)
    im1 = ax1.imshow(
        masked_base,
        cmap=base_palette["cmap"],
        vmin=base_palette["vmin"],
        vmax=base_palette["vmax"]
    )
    ax1.set_title("Original Image")
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Add colorbar
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("NDVI")
    
    # Plot the anomaly image
    masked_anomaly = np.ma.masked_invalid(anomaly_image)
    im2 = ax2.imshow(
        masked_anomaly,
        cmap=anomaly_palette["cmap"],
        vmin=anomaly_palette["vmin"],
        vmax=anomaly_palette["vmax"]
    )
    ax2.set_title("Anomaly Detection")
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Add colorbar
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Z-Score")
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

def create_histogram_figure(
    image_data: np.ndarray,
    title: str = "Value Distribution",
    figsize: Tuple[int, int] = (8, 6),
    bins: int = 50,
    color: str = "steelblue"
) -> plt.Figure:
    """
    Create a histogram of pixel values.
    
    Args:
        image_data: Numpy array containing the values
        title: Figure title
        figsize: Figure size in inches
        bins: Number of histogram bins
        color: Bar color
        
    Returns:
        Matplotlib figure
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Flatten the image data and remove NaNs
    valid_data = image_data.flatten()
    valid_data = valid_data[~np.isnan(valid_data)]
    
    # Plot histogram
    ax.hist(valid_data, bins=bins, color=color, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def fig_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to base64 string.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64 encoded string
    """
    import io
    import base64
    
    # Save figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    
    # Encode as base64
    img_str = base64.b64encode(buf.read()).decode("ascii")
    
    return img_str

def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Correlation Heatmap",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create a correlation heatmap.
    
    Args:
        correlation_matrix: Pandas DataFrame with the correlation matrix
        title: Plot title
        figsize: Figure size in inches
        
    Returns:
        Matplotlib figure
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label="Correlation Coefficient")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(correlation_matrix.columns)))
    ax.set_yticks(np.arange(len(correlation_matrix.index)))
    ax.set_xticklabels(correlation_matrix.columns)
    ax.set_yticklabels(correlation_matrix.index)
    
    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            value = correlation_matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", 
                   color="white" if abs(value) > 0.5 else "black",
                   fontsize=8)
    
    # Set title
    ax.set_title(title)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_market_signals(
    prices: pd.DataFrame,
    signals: List[Dict],
    title: str = "Market Signals and Prices",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot commodity prices and market signals.
    
    Args:
        prices: DataFrame with price data (Date as index, commodities as columns)
        signals: List of signal dictionaries
        title: Plot title
        figsize: Figure size in inches
        
    Returns:
        Matplotlib figure
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot prices for each commodity
    for column in prices.columns:
        ax.plot(prices.index, prices[column], label=column, linewidth=1.5, alpha=0.8)
    
    # Plot signals
    for signal in signals:
        signal_date = signal["signal_date"]
        signal_price = prices.loc[signal_date, signal["commodity"]] if signal_date in prices.index and signal["commodity"] in prices.columns else None
        
        if signal_price is not None:
            if signal["action"] == "LONG":
                ax.scatter(signal_date, signal_price, color="green", s=100, marker="^", zorder=5,
                         label="LONG Signal" if "LONG Signal" not in [l.get_label() for l in ax.get_lines()] else "")
            else:  # "SHORT"
                ax.scatter(signal_date, signal_price, color="red", s=100, marker="v", zorder=5,
                         label="SHORT Signal" if "SHORT Signal" not in [l.get_label() for l in ax.get_lines()] else "")
    
    # Format the x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Add vertical line for current date
    current_date = datetime.datetime.now().date()
    if min(prices.index) <= current_date <= max(prices.index):
        ax.axvline(x=current_date, color='black', linestyle='--', alpha=0.5, label="Current Date")
    
    # Add key price levels if available in signals
    for signal in signals:
        if "key_levels" in signal and signal["key_levels"]:
            for level_name, level_value in signal["key_levels"].items():
                ax.axhline(y=level_value, color='gray', linestyle='-.', alpha=0.5,
                         label=f"{level_name} ({level_value})" if f"{level_name}" not in [l.get_label() for l in ax.get_lines()] else "")
    
    # Set labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    ax.legend(unique_handles, unique_labels, loc="best")
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def create_insight_badge(
    title: str,
    icon: str,
    level: int = 0,
    progress: int = 0,
    max_progress: int = 100,
    description: str = ""
) -> str:
    """
    Create an HTML badge for insights and achievements.
    
    Args:
        title: Badge title
        icon: Badge icon (emoji or font awesome code)
        level: Badge level (0-3)
        progress: Current progress
        max_progress: Maximum progress
        description: Badge description
        
    Returns:
        HTML string for the badge
    """
    # Calculate progress percentage
    progress_pct = (progress / max_progress) * 100 if max_progress > 0 else 0
    
    # Generate stars based on level
    stars = "★" * level + "☆" * (3 - level)
    
    # Determine badge color based on level
    colors = {
        0: "#6c757d",  # Gray
        1: "#28a745",  # Green
        2: "#17a2b8",  # Teal
        3: "#ffc107"   # Gold
    }
    
    badge_color = colors.get(level, "#6c757d")
    
    # Create HTML for the badge
    badge_html = f"""
    <div style="border: 2px solid {badge_color}; border-radius: 10px; padding: 10px; margin: 10px 0; background-color: white;">
        <div style="display: flex; align-items: center;">
            <div style="font-size: 2.5em; margin-right: 10px;">{icon}</div>
            <div style="flex-grow: 1;">
                <div style="font-weight: bold; font-size: 1.2em;">{title}</div>
                <div style="color: {badge_color}; font-weight: bold;">{stars} Level {level}</div>
                <div style="margin-top: 5px; font-size: 0.9em;">{description}</div>
                <div style="margin-top: 5px; width: 100%; background-color: #e9ecef; border-radius: 5px; height: 10px;">
                    <div style="width: {progress_pct}%; height: 100%; background-color: {badge_color}; border-radius: 5px;"></div>
                </div>
                <div style="font-size: 0.8em; text-align: right; margin-top: 2px;">{progress}/{max_progress}</div>
            </div>
        </div>
    </div>
    """
    
    return badge_html