import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster, HeatMap
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from shapely.geometry import Polygon, Point, shape

def plot_ndvi_map(ndvi_array: np.ndarray, title: str = "NDVI Map") -> plt.Figure:
    """
    Create a matplotlib figure with NDVI visualization.
    
    Args:
        ndvi_array: Array containing NDVI values (-1 to 1)
        title: Title for the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # NDVI colormap
    cmap = plt.cm.RdYlGn
    
    # Plot NDVI with colormap
    im = ax.imshow(ndvi_array, cmap=cmap, vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('NDVI Value')
    
    # Add title
    ax.set_title(title)
    
    # Remove axis labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig

def plot_interactive_ndvi_map(
    ndvi_array: np.ndarray, 
    lat_min: float, 
    lon_min: float, 
    lat_max: float, 
    lon_max: float,
    title: str = "NDVI Map"
) -> go.Figure:
    """
    Create an interactive Plotly figure with NDVI visualization.
    
    Args:
        ndvi_array: Array containing NDVI values (-1 to 1)
        lat_min, lon_min, lat_max, lon_max: Boundaries of the map
        title: Title for the plot
        
    Returns:
        Plotly figure
    """
    # Create a heatmap using Plotly
    fig = px.imshow(
        ndvi_array,
        zmin=-1, 
        zmax=1,
        color_continuous_scale='RdYlGn',
        title=title,
        labels={"color": "NDVI Value"}
    )
    
    # Add layout settings
    fig.update_layout(
        height=600,
        width=800,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # Update axes to show lat/lon instead of pixel coordinates
    fig.update_xaxes(
        title_text="Longitude",
        showgrid=True,
        range=[lon_min, lon_max]
    )
    fig.update_yaxes(
        title_text="Latitude",
        showgrid=True,
        range=[lat_max, lat_min],  # Reversed to match geographic coordinates
        scaleanchor="x",
        scaleratio=1
    )
    
    return fig

def plot_time_series(
    dates: List[str], 
    values: List[float], 
    title: str = "Time Series", 
    y_label: str = "Value",
    anomalies: List[int] = None
) -> go.Figure:
    """
    Create an interactive time series plot with optional anomaly highlighting.
    
    Args:
        dates: List of date strings
        values: List of values corresponding to dates
        title: Title for the plot
        y_label: Label for y-axis
        anomalies: List of indices where anomalies occur
        
    Returns:
        Plotly figure
    """
    # Create main time series plot
    fig = go.Figure()
    
    # Add line plot
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        name=y_label,
        line=dict(color='royalblue', width=2)
    ))
    
    # Add anomalies if provided
    if anomalies:
        anomaly_dates = [dates[i] for i in anomalies]
        anomaly_values = [values[i] for i in anomalies]
        
        fig.add_trace(go.Scatter(
            x=anomaly_dates,
            y=anomaly_values,
            mode='markers',
            name='Anomalies',
            marker=dict(
                color='red',
                size=10,
                symbol='circle',
                line=dict(
                    color='black',
                    width=2
                )
            )
        ))
    
    # Set layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=y_label,
        height=500,
        width=800,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def plot_drought_detection(
    dates: List[str],
    ndvi_values: List[float],
    drought_events: List[Dict[str, Any]]
) -> go.Figure:
    """
    Create a plot highlighting drought events in NDVI time series.
    
    Args:
        dates: List of date strings
        ndvi_values: List of NDVI values
        drought_events: List of drought events detected
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add NDVI time series
    fig.add_trace(go.Scatter(
        x=dates,
        y=ndvi_values,
        mode='lines+markers',
        name='NDVI',
        line=dict(color='green', width=2)
    ))
    
    # Add drought periods as colored rectangles
    for event in drought_events:
        start_date = event["start_date"]
        end_date = event["end_date"]
        severity = event["severity"]
        
        # Calculate color based on severity (darker red for more severe)
        color_intensity = min(1.0, severity / 0.3)  # Normalize to 0-1
        color = f'rgba(255, 0, 0, {color_intensity})'
        
        fig.add_shape(
            type="rect",
            x0=start_date,
            y0=min(ndvi_values) - 0.05,
            x1=end_date,
            y1=max(ndvi_values) + 0.05,
            fillcolor=color,
            opacity=0.3,
            layer="below",
            line=dict(width=0)
        )
    
    # Set layout
    fig.update_layout(
        title='NDVI Time Series with Drought Detection',
        xaxis_title='Date',
        yaxis_title='NDVI Value',
        height=500,
        width=800,
        template='plotly_white',
        hovermode='x',
        shapes=[
            # Add horizontal line at NDVI = 0.3 (typical drought threshold)
            dict(
                type='line',
                x0=dates[0],
                y0=0.3,
                x1=dates[-1],
                y1=0.3,
                line=dict(
                    color='rgba(255, 0, 0, 0.5)',
                    width=2,
                    dash='dash'
                )
            )
        ],
        annotations=[
            dict(
                x=dates[0],
                y=0.3,
                xref='x',
                yref='y',
                text='Drought Threshold',
                showarrow=False,
                font=dict(
                    color='red',
                    size=12
                ),
                bgcolor='white',
                opacity=0.7
            )
        ]
    )
    
    return fig

def create_folium_map(
    center_lat: float, 
    center_lon: float, 
    zoom: int = 10
) -> folium.Map:
    """
    Create a Folium map centered at the given coordinates.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        zoom: Initial zoom level
        
    Returns:
        Folium map object
    """
    # Create the map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles='CartoDB positron',
        control_scale=True
    )
    
    # Add tile options
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB positron').add_to(m)
    folium.TileLayer('CartoDB dark_matter').add_to(m)
    folium.TileLayer('Stamen Terrain').add_to(m)
    folium.TileLayer('Stamen Toner').add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def add_geojson_to_map(
    m: folium.Map, 
    geojson_data: Dict, 
    style_function: Optional[callable] = None, 
    highlight_function: Optional[callable] = None,
    popup_function: Optional[callable] = None,
    name: str = "GeoJSON"
) -> folium.Map:
    """
    Add GeoJSON data to a Folium map.
    
    Args:
        m: Folium map object
        geojson_data: GeoJSON data as dictionary
        style_function: Function to style GeoJSON features
        highlight_function: Function to style GeoJSON features on hover
        popup_function: Function to create popup content
        name: Name for the GeoJSON layer
        
    Returns:
        Updated Folium map
    """
    # Default style function
    if style_function is None:
        style_function = lambda x: {
            'fillColor': '#28a745',
            'color': '#28a745',
            'fillOpacity': 0.3,
            'weight': 2
        }
    
    # Default highlight function
    if highlight_function is None:
        highlight_function = lambda x: {
            'fillColor': '#28a745',
            'color': '#000000',
            'fillOpacity': 0.5,
            'weight': 3
        }
    
    # Add GeoJSON to map
    if popup_function:
        gjson = folium.GeoJson(
            geojson_data,
            style_function=style_function,
            highlight_function=highlight_function,
            name=name,
            tooltip=folium.GeoJsonTooltip(fields=list(geojson_data['features'][0]['properties'].keys())),
            popup=folium.GeoJsonPopup(fields=list(geojson_data['features'][0]['properties'].keys()))
        )
    else:
        gjson = folium.GeoJson(
            geojson_data,
            style_function=style_function,
            highlight_function=highlight_function,
            name=name,
            tooltip=folium.GeoJsonTooltip(fields=list(geojson_data['features'][0]['properties'].keys()))
        )
    
    gjson.add_to(m)
    
    return m

def overlay_raster_on_map(
    m: folium.Map,
    raster_data: np.ndarray,
    bounds: List[float],
    colormap: str = 'RdYlGn',
    opacity: float = 0.7,
    name: str = "Raster Layer"
) -> folium.Map:
    """
    Overlay a raster image on a Folium map.
    
    Args:
        m: Folium map object
        raster_data: 2D array with raster values
        bounds: [min_lon, min_lat, max_lon, max_lat]
        colormap: Matplotlib colormap name
        opacity: Opacity of the overlay
        name: Name for the raster layer
        
    Returns:
        Updated Folium map
    """
    import matplotlib.cm as cm
    
    # Normalize data to 0-1 range for colormap
    if raster_data.min() != raster_data.max():
        norm_data = (raster_data - raster_data.min()) / (raster_data.max() - raster_data.min())
    else:
        norm_data = raster_data / raster_data.max() if raster_data.max() > 0 else raster_data
    
    # Get colormap
    cmap = cm.get_cmap(colormap)
    
    # Convert to RGBA
    rgba_img = cmap(norm_data)
    
    # Convert to PIL Image
    from PIL import Image
    img = Image.fromarray((rgba_img * 255).astype(np.uint8))
    
    # Create image overlay
    image_overlay = folium.raster_layers.ImageOverlay(
        image=img,
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        opacity=opacity,
        name=name,
        zindex=1
    )
    
    # Add to map
    image_overlay.add_to(m)
    
    return m

def plot_yield_forecast(
    dates: List[str],
    historical_yield: List[float],
    forecasted_yield: Dict[str, float],
    crop_type: str = "Crop"
) -> go.Figure:
    """
    Create a plot showing historical yield data and forecasts.
    
    Args:
        dates: List of historical dates
        historical_yield: List of historical yield values
        forecasted_yield: Dictionary of forecasted dates and yields
        crop_type: Type of crop
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add historical yield
    fig.add_trace(go.Scatter(
        x=dates,
        y=historical_yield,
        mode='lines+markers',
        name='Historical Yield',
        line=dict(color='blue', width=2)
    ))
    
    # Add forecasted yield
    forecast_dates = list(forecasted_yield.keys())
    forecast_values = list(forecasted_yield.values())
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='markers',
        name='Forecasted Yield',
        marker=dict(
            color='red',
            size=10,
            symbol='star',
            line=dict(
                color='black',
                width=1
            )
        )
    ))
    
    # Add a line connecting the last historical point to first forecast point
    if dates and forecast_dates:
        fig.add_trace(go.Scatter(
            x=[dates[-1], forecast_dates[0]],
            y=[historical_yield[-1], forecast_values[0]],
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False
        ))
    
    # Set layout
    fig.update_layout(
        title=f'{crop_type} Yield Forecast',
        xaxis_title='Date',
        yaxis_title='Yield (t/ha)',
        height=500,
        width=800,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def plot_market_signals(
    dates: List[str],
    prices: List[float],
    signals: List[Dict[str, Any]]
) -> go.Figure:
    """
    Create a plot showing market prices and trading signals.
    
    Args:
        dates: List of dates
        prices: List of market prices
        signals: List of signal events with date, action, confidence
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add price chart
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='black', width=2)
    ))
    
    # Add buy signals
    buy_dates = [s['date'] for s in signals if s['action'] == 'LONG']
    buy_prices = [prices[dates.index(d)] if d in dates else None for d in buy_dates]
    buy_confidences = [s['confidence'] for s in signals if s['action'] == 'LONG']
    
    if buy_dates:
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=buy_prices,
            mode='markers',
            name='BUY Signal',
            marker=dict(
                color='green',
                size=[c * 20 for c in buy_confidences],  # Size based on confidence
                symbol='triangle-up',
                line=dict(
                    color='darkgreen',
                    width=1
                )
            ),
            text=[f"Confidence: {c:.2f}" for c in buy_confidences],
            hoverinfo='text+x+y'
        ))
    
    # Add sell signals
    sell_dates = [s['date'] for s in signals if s['action'] == 'SHORT']
    sell_prices = [prices[dates.index(d)] if d in dates else None for d in sell_dates]
    sell_confidences = [s['confidence'] for s in signals if s['action'] == 'SHORT']
    
    if sell_dates:
        fig.add_trace(go.Scatter(
            x=sell_dates,
            y=sell_prices,
            mode='markers',
            name='SELL Signal',
            marker=dict(
                color='red',
                size=[c * 20 for c in sell_confidences],  # Size based on confidence
                symbol='triangle-down',
                line=dict(
                    color='darkred',
                    width=1
                )
            ),
            text=[f"Confidence: {c:.2f}" for c in sell_confidences],
            hoverinfo='text+x+y'
        ))
    
    # Set layout
    fig.update_layout(
        title='Market Prices and Trading Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        width=900,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def plot_feature_importance(
    features: List[str],
    importance_values: List[float],
    title: str = "Feature Importance"
) -> go.Figure:
    """
    Create a horizontal bar chart of feature importance.
    
    Args:
        features: List of feature names
        importance_values: List of importance values
        title: Title for the plot
        
    Returns:
        Plotly figure
    """
    # Sort features by importance
    sorted_idx = np.argsort(importance_values)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = [importance_values[i] for i in sorted_idx]
    
    # Create figure
    fig = go.Figure(go.Bar(
        x=sorted_importance,
        y=sorted_features,
        orientation='h',
        marker=dict(
            color=sorted_importance,
            colorscale='YlGnBu',
            colorbar=dict(title='Importance')
        )
    ))
    
    # Set layout
    fig.update_layout(
        title=title,
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500,
        width=800,
        template='plotly_white'
    )
    
    return fig

def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Correlation Matrix"
) -> go.Figure:
    """
    Create a heatmap of correlation values.
    
    Args:
        correlation_matrix: Pandas DataFrame with correlation values
        title: Title for the plot
        
    Returns:
        Plotly figure
    """
    fig = px.imshow(
        correlation_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title=title
    )
    
    # Set layout
    fig.update_layout(
        height=700,
        width=700,
        template='plotly_white'
    )
    
    return fig

def plot_pca_clusters(
    pca_results: np.ndarray,
    cluster_labels: np.ndarray,
    explained_variance: List[float],
    title: str = "PCA and Clustering Results"
) -> go.Figure:
    """
    Create a scatter plot of PCA results with cluster labels.
    
    Args:
        pca_results: PCA transformed data (n_samples, n_components)
        cluster_labels: Cluster labels for each sample
        explained_variance: Explained variance ratio for each component
        title: Title for the plot
        
    Returns:
        Plotly figure
    """
    # Create a dataframe for easy plotting
    df = pd.DataFrame({
        'PC1': pca_results[:, 0],
        'PC2': pca_results[:, 1],
        'Cluster': cluster_labels
    })
    
    # Create figure
    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color='Cluster',
        title=title,
        color_discrete_sequence=px.colors.qualitative.G10,
        labels={
            'PC1': f'PC1 ({explained_variance[0]:.1%} variance)',
            'PC2': f'PC2 ({explained_variance[1]:.1%} variance)'
        }
    )
    
    # Set layout
    fig.update_layout(
        height=600,
        width=800,
        template='plotly_white',
        legend_title='Cluster'
    )
    
    return fig
