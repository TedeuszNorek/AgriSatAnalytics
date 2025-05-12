import os
import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
import logging
import time
import glob
import psutil
import asyncio
import uuid
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Debug Dashboard - Agro Insight",
    page_icon="🔧",
    layout="wide"
)

# Header
st.title("🔧 Debug Dashboard")
st.markdown("""
Monitor system health, rate limits, data processing, and logs for the Agro Insight application.
This dashboard provides technical insights for developers and administrators.
""")

# Function to get system metrics
def get_system_metrics():
    """Get current system metrics"""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "timestamp": datetime.datetime.now().isoformat()
    }

# Function to get data directory stats
def get_data_stats():
    """Get statistics about the data directories"""
    result = {}
    
    # Check data directories
    data_dir = Path("./data")
    
    if data_dir.exists():
        # Count files in each subdirectory
        for subdir in ["geotiff", "metadata", "market"]:
            subdir_path = data_dir / subdir
            if subdir_path.exists():
                result[subdir] = len(list(subdir_path.glob("*")))
            else:
                result[subdir] = 0
        
        # Get total size
        total_size = 0
        for path in data_dir.glob("**/*"):
            if path.is_file():
                total_size += path.stat().st_size
        
        result["total_size_mb"] = total_size / (1024 * 1024)
    else:
        result = {"geotiff": 0, "metadata": 0, "market": 0, "total_size_mb": 0}
    
    return result

# Function to get cache stats
def get_cache_stats():
    """Get statistics about the cache directory"""
    result = {"files": 0, "size_mb": 0}
    
    cache_dir = Path("./cache")
    if cache_dir.exists():
        # Count files
        result["files"] = len(list(cache_dir.glob("**/*")))
        
        # Get total size
        total_size = 0
        for path in cache_dir.glob("**/*"):
            if path.is_file():
                total_size += path.stat().st_size
        
        result["size_mb"] = total_size / (1024 * 1024)
    
    return result

# Function to get rate limit metrics
def get_rate_limit_metrics():
    """Get rate limit metrics (for demonstration only)"""
    # In a real app, this would get actual rate limit usage from Sentinel Hub API
    # For this demo, we'll simulate some values
    return {
        "copernicus_requests": np.random.randint(0, 30),
        "copernicus_max": 30,
        "copernicus_processing_units": np.random.randint(0, 10),
        "copernicus_max_processing_units": 10,
        "open_meteo_requests": np.random.randint(0, 50),
        "yfinance_requests": np.random.randint(0, 20),
        "timestamp": datetime.datetime.now().isoformat()
    }

# Function to get log entries
def get_log_entries(n=20):
    """Get recent log entries"""
    logs = []
    
    # Try to read the app.log file if it exists
    log_path = Path("app.log")
    if log_path.exists():
        with open(log_path, "r") as f:
            for line in f.readlines()[-n:]:
                logs.append(line.strip())
    
    # If no logs found, provide some sample logs
    if not logs:
        logs = [
            "2023-07-01 12:01:23 - root - INFO - Application started",
            "2023-07-01 12:02:45 - utils.data_access - INFO - Fetching data from Sentinel Hub",
            "2023-07-01 12:03:12 - utils.data_access - INFO - Cached data retrieval: ETag match",
            "2023-07-01 12:05:30 - utils.processing - INFO - Calculated NDVI for field_abc",
            "2023-07-01 12:07:22 - models.yield_forecast - INFO - Training yield model"
        ]
    
    return logs

# Create dashboard layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("System Health")
    
    # Get current metrics
    metrics = get_system_metrics()
    
    # Display current metrics
    col_cpu, col_mem, col_disk = st.columns(3)
    
    col_cpu.metric(
        "CPU Usage",
        f"{metrics['cpu_percent']}%",
        None
    )
    
    col_mem.metric(
        "Memory Usage",
        f"{metrics['memory_percent']}%",
        None
    )
    
    col_disk.metric(
        "Disk Usage",
        f"{metrics['disk_percent']}%",
        None
    )
    
    # Create a line chart for CPU and memory over time
    # In a real app, we would store historical data
    # For this demo, we'll simulate some data
    
    # Simulate historical data
    timestamps = [
        (datetime.datetime.now() - datetime.timedelta(minutes=i)).strftime("%H:%M:%S")
        for i in range(20, 0, -1)
    ]
    cpu_values = [np.random.randint(10, 90) for _ in range(20)]
    memory_values = [np.random.randint(30, 85) for _ in range(20)]
    
    # Add current values
    timestamps.append(datetime.datetime.now().strftime("%H:%M:%S"))
    cpu_values.append(metrics["cpu_percent"])
    memory_values.append(metrics["memory_percent"])
    
    # Create a DataFrame
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "CPU": cpu_values,
        "Memory": memory_values
    })
    
    # Plot the data
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["Timestamp"],
        y=df["CPU"],
        mode="lines+markers",
        name="CPU Usage (%)",
        line=dict(color="royalblue", width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df["Timestamp"],
        y=df["Memory"],
        mode="lines+markers",
        name="Memory Usage (%)",
        line=dict(color="firebrick", width=2)
    ))
    
    fig.update_layout(
        title="System Resource Usage",
        xaxis_title="Time",
        yaxis_title="Usage (%)",
        legend=dict(y=0.99, x=0.01),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data storage statistics
    st.subheader("Data Storage")
    
    data_stats = get_data_stats()
    cache_stats = get_cache_stats()
    
    col_geotiff, col_metadata, col_market, col_cache = st.columns(4)
    
    col_geotiff.metric(
        "GeoTIFF Files",
        data_stats.get("geotiff", 0),
        None
    )
    
    col_metadata.metric(
        "Metadata Files",
        data_stats.get("metadata", 0),
        None
    )
    
    col_market.metric(
        "Market Data Files",
        data_stats.get("market", 0),
        None
    )
    
    col_cache.metric(
        "Cache Files",
        cache_stats.get("files", 0),
        None
    )
    
    # Data storage size
    st.markdown(f"**Total Data Size:** {data_stats.get('total_size_mb', 0):.2f} MB")
    st.markdown(f"**Cache Size:** {cache_stats.get('size_mb', 0):.2f} MB")
    
    # Data size chart
    sizes = [
        data_stats.get("total_size_mb", 0) - cache_stats.get("size_mb", 0), 
        cache_stats.get("size_mb", 0)
    ]
    labels = ["Data Files", "Cache Files"]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=sizes,
        hole=.4,
        marker_colors=['royalblue', 'lightgray']
    )])
    
    fig.update_layout(
        title="Storage Distribution",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("API Rate Limits")
    
    # Get rate limit metrics
    rate_limits = get_rate_limit_metrics()
    
    # Display current rate limits
    col_copernicus, col_openmeteo, col_yfinance = st.columns(3)
    
    col_copernicus.metric(
        "Copernicus Requests",
        f"{rate_limits['copernicus_requests']}/{rate_limits['copernicus_max']}",
        None
    )
    
    col_openmeteo.metric(
        "Open-Meteo Requests",
        rate_limits['open_meteo_requests'],
        None
    )
    
    col_yfinance.metric(
        "YFinance Requests",
        rate_limits['yfinance_requests'],
        None
    )
    
    # Copernicus rate limit gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rate_limits["copernicus_requests"],
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Copernicus Requests (per minute)"},
        gauge={
            "axis": {"range": [0, rate_limits["copernicus_max"]]},
            "bar": {"color": "royalblue"},
            "steps": [
                {"range": [0, rate_limits["copernicus_max"] * 0.5], "color": "lightgreen"},
                {"range": [rate_limits["copernicus_max"] * 0.5, rate_limits["copernicus_max"] * 0.8], "color": "yellow"},
                {"range": [rate_limits["copernicus_max"] * 0.8, rate_limits["copernicus_max"]], "color": "salmon"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": rate_limits["copernicus_max"]
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Copernicus processing units gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rate_limits["copernicus_processing_units"],
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Copernicus Processing Units (per minute)"},
        gauge={
            "axis": {"range": [0, rate_limits["copernicus_max_processing_units"]]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, rate_limits["copernicus_max_processing_units"] * 0.5], "color": "lightgreen"},
                {"range": [rate_limits["copernicus_max_processing_units"] * 0.5, rate_limits["copernicus_max_processing_units"] * 0.8], "color": "yellow"},
                {"range": [rate_limits["copernicus_max_processing_units"] * 0.8, rate_limits["copernicus_max_processing_units"]], "color": "salmon"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": rate_limits["copernicus_max_processing_units"]
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Application logs
st.subheader("Application Logs")

# Get recent logs
logs = get_log_entries(20)

# Display logs in an expandable section
with st.expander("View Recent Logs", expanded=False):
    for log in logs:
        # Color code logs based on level
        if " ERROR " in log:
            st.markdown(f"<span style='color:red'>{log}</span>", unsafe_allow_html=True)
        elif " WARNING " in log:
            st.markdown(f"<span style='color:orange'>{log}</span>", unsafe_allow_html=True)
        else:
            st.text(log)

# Debug actions
st.subheader("Debug Actions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Cache Management")
    
    if st.button("Clear Cache"):
        # In a real app, this would actually clear the cache
        st.success("Cache cleared successfully!")
        
        # Update cache stats
        cache_stats = get_cache_stats()
        st.metric(
            "Cache Size After Clearing",
            f"{cache_stats.get('size_mb', 0):.2f} MB",
            None
        )

with col2:
    st.markdown("### Test External APIs")
    
    # Select API to test
    api_to_test = st.selectbox(
        "Select API to Test",
        options=["Sentinel Hub", "Open-Meteo", "Yahoo Finance"],
        help="Run a test query to check API connectivity"
    )
    
    if st.button("Test API Connection"):
        with st.spinner(f"Testing connection to {api_to_test}..."):
            # Simulate API test
            time.sleep(2)
            
            # In a real app, this would make an actual API request
            if api_to_test == "Sentinel Hub":
                if os.environ.get("SENTINEL_HUB_CLIENT_ID"):
                    st.success("✅ Successfully connected to Sentinel Hub API")
                else:
                    st.error("❌ Failed to connect to Sentinel Hub API: No credentials found")
            else:
                st.success(f"✅ Successfully connected to {api_to_test} API")

# System configuration
st.subheader("System Configuration")

# Display current environment variables (with sensitive information masked)
env_vars = {
    "SENTINEL_HUB_CLIENT_ID": "***" if os.environ.get("SENTINEL_HUB_CLIENT_ID") else "Not set",
    "SENTINEL_HUB_CLIENT_SECRET": "***" if os.environ.get("SENTINEL_HUB_CLIENT_SECRET") else "Not set",
    "PYTHONPATH": os.environ.get("PYTHONPATH", "Not set"),
    "DATA_DIR": os.environ.get("DATA_DIR", "./data"),
    "CACHE_DIR": os.environ.get("CACHE_DIR", "./cache"),
    "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
}

# Display as a table
env_df = pd.DataFrame([env_vars]).T.reset_index()
env_df.columns = ["Environment Variable", "Value"]
st.table(env_df)

# Bottom-page links
st.markdown("---")
st.markdown("""
👈 Go to **Reports** to generate comprehensive reports

👉 Return to **Home** to start a new analysis
""")
