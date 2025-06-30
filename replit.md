# Agro Insight - Satellite Data Analytics for Agriculture

## Overview

Agro Insight is a comprehensive satellite data analytics platform designed for agricultural field monitoring, crop health assessment, yield forecasting, and market signal detection. The system leverages free, open-source Copernicus Sentinel-2 satellite data combined with machine learning models to provide actionable insights for agricultural operations and market trading decisions.

## System Architecture

### Frontend Architecture
- **Web Framework**: Streamlit-based multi-page application
- **Interface**: Wide layout with sidebar navigation and responsive design
- **Visualization**: Interactive maps using Folium, charts with Plotly and Matplotlib
- **User Experience**: Real-time dashboards with customizable analysis parameters

### Backend Architecture
- **Core Language**: Python 3.10+
- **Database Layer**: SQLAlchemy ORM with dual database support (PostgreSQL/SQLite fallback)
- **API Integration**: Asynchronous data fetching with rate limiting and retry mechanisms
- **Processing Pipeline**: Modular architecture with separate utilities for data access, processing, and visualization

### Data Storage Solutions
- **Primary Database**: PostgreSQL with SQLite fallback for development
- **File Storage**: Hierarchical directory structure for satellite data, processed results, and reports
- **Cache Management**: Local disk cache with `diskcache` for reducing API quota usage
- **Data Organization**: Separate directories for satellite data, processed data, models, and reports

## Key Components

### Data Access Layer (`utils/data_access.py`)
- Sentinel Hub API integration with OAuth2 authentication
- Rate limiting and retry mechanisms using `tenacity`
- Global semaphore to respect 30 req/min free tier limits
- ETag checking to minimize unnecessary downloads

### Processing Engine (`utils/processing.py`)
- Satellite image processing for vegetation indices (NDVI, EVI, NDWI)
- Cloud masking and anomaly detection
- Zonal statistics calculation for field boundaries
- Time series extraction and analysis

### Machine Learning Models
- **Yield Forecasting**: LightGBM + Prophet models for crop yield prediction
- **Market Signals**: Correlation analysis between satellite data and commodity futures
- **AI Intelligence**: Google Gemini integration for advanced analysis

### Visualization Framework (`utils/visualization.py`)
- Interactive maps with field boundaries and satellite overlays
- Time series charts with anomaly highlighting
- Market correlation heatmaps and trading signals
- Multi-temporal analysis dashboards

### Database Models (`database.py`)
- Field management with geometry and metadata storage
- Satellite image tracking with STAC-compliant metadata
- Time series data for vegetation indices
- Yield forecasts and market signals storage

## Data Flow

1. **Field Definition**: Users define agricultural fields through interactive maps or file upload
2. **Data Ingestion**: System fetches Sentinel-2 L2A data for specified time periods and regions
3. **Processing Pipeline**: Calculates vegetation indices, applies cloud masks, and extracts statistics
4. **Analysis Engine**: Runs ML models for yield prediction and market signal generation
5. **Visualization**: Renders interactive dashboards with insights and recommendations
6. **Reporting**: Generates comprehensive PDF/Markdown reports with actionable intelligence

## External Dependencies

### Satellite Data Sources
- **Primary**: Copernicus Sentinel-2 via Sentinel Hub API (Basic plan, free tier)
- **Alternative**: AWS S3 public Sentinel-2 bucket for backup access
- **Resolution**: 10-20m/pixel with 5-day revisit frequency

### Market Data
- **Futures Prices**: Yahoo Finance API for commodity market data
- **Weather Data**: Open-Meteo API for meteorological information
- **Energy Markets**: Multiple sources for energy price correlations

### AI/ML Services
- **Google Gemini**: Advanced AI analysis and market intelligence
- **Local Models**: scikit-learn, LightGBM, and Prophet for predictions

## Deployment Strategy

### Development Environment
- Streamlit development server with auto-reload functionality
- SQLite database for rapid prototyping and testing
- Environment variable configuration for API credentials

### Production Considerations
- PostgreSQL database with connection pooling
- SSL-enabled connections with proper credential management
- Monitoring and logging infrastructure for system health
- Automated backup and recovery procedures

## Changelog

- June 30, 2025. Fixed market data integrity - removed all fake/mock data and implemented real Yahoo Finance data fetching
- June 29, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.