# Agro Insight 🛰️

A satellite data analytics module for agricultural insights and market predictions using free Copernicus/Sentinel data.

## Overview

Agro Insight is a powerful tool that leverages free, open-source Sentinel-2 satellite data to provide agricultural field monitoring, crop health assessment, yield forecasting, and market signal detection. The application is designed to help farmers, agronomists, agricultural traders, and investors make data-driven decisions based on real satellite observations.

![Satellite imagery](https://pixabay.com/get/g4efcf32e7e1316c10c5547bc49ca7c6fe56ddd2b0787c5f35e822bf6392641e24ddb0209cb35a405581d490e27e587e7_1280.jpg)

## Features

- **Data Ingest:** Import field boundaries and fetch Sentinel-2 L2A satellite data
- **Field Analysis:** Monitor vegetation indices (NDVI, EVI, NDWI) and detect anomalies
- **Yield Forecast:** Predict crop yields using machine learning (LightGBM + Prophet)
- **Market Signals:** Correlate satellite data with commodity prices to generate trading signals
- **Reports:** Generate comprehensive PDF/Markdown reports with insights and recommendations
- **Debug Dashboard:** Monitor system health, rate limits, and application logs

## Data Sources

- **Satellite Data:** Copernicus Sentinel-2 (10-20m/pixel resolution, 5-day revisit)
- **Weather Data:** Open-Meteo API (free tier)
- **Market Data:** Yahoo Finance API (futures prices)

## Technical Stack

- **Backend:** Python 3.10+
- **Web Framework:** Streamlit
- **Data Science:** NumPy, Pandas, GeoPandas, Rasterio, SciPy
- **Machine Learning:** scikit-learn, LightGBM, Prophet
- **Visualization:** Plotly, Folium, Matplotlib
- **Satellite API:** sentinelhub-py (free Basic plan)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/agro-insight.git
   cd agro-insight
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file with the following variables:
   ```
   SENTINEL_HUB_CLIENT_ID=your_client_id
   SENTINEL_HUB_CLIENT_SECRET=your_client_secret
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

## Getting Started

1. **Obtain Sentinel Hub Credentials:**
   - Create a free account at [Sentinel Hub](https://www.sentinel-hub.com/)
   - Generate OAuth client credentials (Client ID and Client Secret)
   - Set these as environment variables

2. **Data Ingest:**
   - Upload a GeoJSON file with field boundaries or use a country code
   - Select the date range and maximum cloud coverage
   - Choose vegetation indices to calculate (NDVI, EVI, NDWI)

3. **Analyze and Forecast:**
   - Explore vegetation indices and time series
   - Detect anomalies and potential drought conditions
   - Generate yield forecasts and market signals

## Data Workflow

