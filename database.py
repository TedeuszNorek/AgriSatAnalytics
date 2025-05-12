import os
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

# Create SQLAlchemy engine
if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    logger.info("Database connection established")
else:
    logger.warning("No DATABASE_URL environment variable found. Using SQLite database")
    engine = create_engine("sqlite:///agro_insight.db")

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Define models
class Field(Base):
    """Field model for storing agricultural field information"""
    __tablename__ = "fields"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    geojson = Column(JSON)  # Store GeoJSON representation of field boundary
    center_lat = Column(Float)
    center_lon = Column(Float)
    area_hectares = Column(Float)
    crop_type = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    satellite_images = relationship("SatelliteImage", back_populates="field", cascade="all, delete-orphan")
    time_series = relationship("TimeSeries", back_populates="field", cascade="all, delete-orphan")
    yield_forecasts = relationship("YieldForecast", back_populates="field", cascade="all, delete-orphan")
    market_signals = relationship("MarketSignal", back_populates="field", cascade="all, delete-orphan")

class SatelliteImage(Base):
    """Model for storing processed satellite images"""
    __tablename__ = "satellite_images"
    
    id = Column(Integer, primary_key=True, index=True)
    field_id = Column(Integer, ForeignKey("fields.id"))
    image_type = Column(String)  # e.g., "NDVI", "EVI", "NDWI"
    image_path = Column(String)  # Path to stored GeoTIFF
    metadata_path = Column(String)  # Path to STAC metadata
    acquisition_date = Column(DateTime)
    cloud_cover_percentage = Column(Float)
    scene_id = Column(String)
    stats = Column(JSON)  # Statistics for the image (min, max, mean, etc.)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    field = relationship("Field", back_populates="satellite_images")

class TimeSeries(Base):
    """Model for storing time series data for fields"""
    __tablename__ = "time_series"
    
    id = Column(Integer, primary_key=True, index=True)
    field_id = Column(Integer, ForeignKey("fields.id"))
    series_type = Column(String)  # e.g., "NDVI", "EVI", "NDWI", "temperature"
    data = Column(JSON)  # JSON object with dates and values
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    stats = Column(JSON)  # Statistics for the time series
    anomalies = Column(JSON, nullable=True)  # Detected anomalies
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    field = relationship("Field", back_populates="time_series")

class YieldForecast(Base):
    """Model for storing yield forecasts"""
    __tablename__ = "yield_forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    field_id = Column(Integer, ForeignKey("fields.id"))
    crop_type = Column(String)
    forecast_date = Column(DateTime)
    forecast_data = Column(JSON)  # JSON object with dates and predicted yields
    model_metrics = Column(JSON, nullable=True)  # Model evaluation metrics
    feature_importance = Column(JSON, nullable=True)  # Feature importance if available
    weather_data = Column(JSON, nullable=True)  # Weather data used for forecast
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    field = relationship("Field", back_populates="yield_forecasts")

class MarketSignal(Base):
    """Model for storing market signals"""
    __tablename__ = "market_signals"
    
    id = Column(Integer, primary_key=True, index=True)
    field_id = Column(Integer, ForeignKey("fields.id"))
    commodity = Column(String)  # e.g., "ZW=F" (Wheat), "ZC=F" (Corn)
    signal_date = Column(DateTime)
    action = Column(String)  # "LONG" or "SHORT"
    confidence = Column(Float)
    reason = Column(Text)
    correlation_data = Column(JSON, nullable=True)  # Correlation analysis data
    price_data = Column(JSON, nullable=True)  # Price data used for analysis
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    field = relationship("Field", back_populates="market_signals")

class Report(Base):
    """Model for storing generated reports"""
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    field_id = Column(Integer, ForeignKey("fields.id", ondelete="CASCADE"))
    report_type = Column(String)  # e.g., "Executive Summary", "Comprehensive Analysis"
    report_format = Column(String)  # e.g., "Markdown", "HTML", "PDF"
    report_path = Column(String)  # Path to stored report file
    report_data = Column(JSON, nullable=True)  # Data used to generate the report
    created_at = Column(DateTime, default=datetime.utcnow)
    
    field = relationship("Field")

# Create functions for database operations
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")

# Initialize database if run directly
if __name__ == "__main__":
    init_db()