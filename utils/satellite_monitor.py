"""
Module for automatic satellite data monitoring, prediction updates, and correlation analysis.
"""
import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import time
import threading
import schedule
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib

from utils.data_access import validate_credentials
from config import (
    SENTINEL_HUB_CLIENT_ID,
    SENTINEL_HUB_CLIENT_SECRET,
    PLANET_API_KEY
)
from models.yield_forecast import YieldForecastModel
from models.market_signals import MarketSignalModel

# Initialize logging
logger = logging.getLogger(__name__)

class SatelliteMonitor:
    """
    Class for monitoring satellite data updates, recording predictions,
    and analyzing correlations from historical data.
    """
    
    def __init__(self):
        """Initialize the satellite monitor."""
        # Sprawdzanie poświadczeń zamiast inicjalizacji klas
        self.sentinel_credentials_valid = validate_credentials("sentinel_hub")
        self.planet_credentials_valid = validate_credentials("planet")
        self.yield_model = YieldForecastModel()
        self.market_model = MarketSignalModel()
        
        # Create prediction storage directories if they don't exist
        self.predictions_dir = Path("data/predictions")
        self.predictions_dir.mkdir(exist_ok=True, parents=True)
        
        self.correlation_dir = Path("data/correlations")
        self.correlation_dir.mkdir(exist_ok=True, parents=True)
        
        self.charts_dir = Path("data/charts")
        self.charts_dir.mkdir(exist_ok=True, parents=True)
        
        # Track the last update time for each field
        self.last_update_times = {}
        self.load_last_update_times()
        
        # Set up a scheduler to check for updates
        self.scheduler_thread = None
        self.is_running = False
    
    def load_last_update_times(self):
        """Load the last update times from file."""
        last_update_file = self.predictions_dir / "last_updates.json"
        if last_update_file.exists():
            try:
                with open(last_update_file, 'r') as f:
                    self.last_update_times = json.load(f)
                logger.info(f"Loaded last update times for {len(self.last_update_times)} fields")
            except Exception as e:
                logger.error(f"Error loading last update times: {str(e)}")
    
    def save_last_update_times(self):
        """Save the last update times to file."""
        last_update_file = self.predictions_dir / "last_updates.json"
        try:
            with open(last_update_file, 'w') as f:
                json.dump(self.last_update_times, f)
            logger.info(f"Saved last update times for {len(self.last_update_times)} fields")
        except Exception as e:
            logger.error(f"Error saving last update times: {str(e)}")
    
    def start_monitoring(self, interval_hours=24):
        """
        Start monitoring for satellite data updates.
        
        Args:
            interval_hours: Number of hours between checks
        """
        if self.is_running:
            logger.warning("Monitoring is already running")
            return
        
        def run_scheduler():
            schedule.every(interval_hours).hours.do(self.check_for_updates)
            self.is_running = True
            
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute if there are pending scheduled tasks
        
        self.scheduler_thread = threading.Thread(target=run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info(f"Started satellite monitoring with {interval_hours} hour interval")
        
        # Run an initial check
        self.check_for_updates()
    
    def stop_monitoring(self):
        """Stop monitoring for satellite data updates."""
        if not self.is_running:
            logger.warning("Monitoring is not running")
            return
        
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=1)
        
        logger.info("Stopped satellite monitoring")
    
    def check_for_updates(self):
        """Check for updates to satellite data for all fields."""
        logger.info("Checking for satellite data updates")
        
        # Get all fields
        fields = self.get_available_fields()
        if not fields:
            logger.warning("No fields found for monitoring")
            return
        
        for field_name in fields:
            self.check_field_update(field_name)
    
    def check_field_update(self, field_name: str):
        """
        Check if there are new satellite data for a specific field.
        
        Args:
            field_name: Name of the field to check
        """
        logger.info(f"Checking for updates for field: {field_name}")
        
        # Get field geojson from file if available
        field_info = self.get_field_info(field_name)
        if not field_info or "geojson" not in field_info:
            logger.warning(f"Could not find geojson for field: {field_name}")
            return
        
        # Get the last update time for this field
        last_update = self.last_update_times.get(field_name, "1970-01-01T00:00:00Z")
        last_update_dt = datetime.fromisoformat(last_update.replace('Z', '+00:00')) if 'Z' in last_update else datetime.fromisoformat(last_update)
        
        # Check for new Sentinel-2 data since last update
        has_new_data = False
        try:
            # Query latest Sentinel-2 image for the field
            latest_image_info = self.sentinel_access.get_latest_image_info(
                field_info["geojson"],
                start_date=(last_update_dt - timedelta(days=1)).strftime("%Y-%m-%d"),
                end_date=datetime.now().strftime("%Y-%m-%d")
            )
            
            if latest_image_info and 'properties' in latest_image_info and 'datetime' in latest_image_info['properties']:
                img_date = latest_image_info['properties']['datetime']
                img_datetime = datetime.fromisoformat(img_date.replace('Z', '+00:00')) if 'Z' in img_date else datetime.fromisoformat(img_date)
                
                if img_datetime > last_update_dt:
                    logger.info(f"Found new satellite data for {field_name} from {img_date}")
                    
                    # Update the last update time
                    self.last_update_times[field_name] = img_date
                    self.save_last_update_times()
                    
                    # Process new data and update predictions
                    has_new_data = True
                    self.update_field_predictions(field_name, latest_image_info)
                else:
                    logger.info(f"No new data for field {field_name} since {last_update}")
            else:
                logger.warning(f"No image information found for field {field_name}")
        
        except Exception as e:
            logger.error(f"Error checking for updates for field {field_name}: {str(e)}")
        
        if has_new_data:
            # After updating predictions, analyze correlations
            self.analyze_correlations(field_name)
            
            # Generate updated charts
            self.generate_charts(field_name)
    
    def update_field_predictions(self, field_name: str, image_info: Dict[str, Any]):
        """
        Update predictions for a field based on new satellite data.
        
        Args:
            field_name: Name of the field to update
            image_info: Information about the new satellite image
        """
        logger.info(f"Updating predictions for field: {field_name}")
        
        try:
            # Extract date from image info
            img_date = image_info['properties']['datetime']
            img_date_str = img_date.split('T')[0]  # Just the date part
            
            # Get NDVI for the new image
            ndvi_data = self.get_or_calculate_ndvi(field_name, image_info)
            if not ndvi_data:
                logger.warning(f"Could not get NDVI data for field {field_name}")
                return
            
            # Update NDVI time series
            self.update_ndvi_time_series(field_name, img_date_str, ndvi_data)
            
            # Calculate yield forecast
            ndvi_time_series = self.get_ndvi_time_series(field_name)
            yield_forecast = self.predict_yield(field_name, ndvi_time_series)
            
            # Update yield forecast time series
            if yield_forecast:
                self.update_yield_forecast(field_name, img_date_str, yield_forecast)
            
            # Generate market signals
            market_signals = self.generate_market_signals(field_name, ndvi_time_series)
            
            # Update market signals
            if market_signals:
                self.update_market_signals(field_name, img_date_str, market_signals)
            
            logger.info(f"Successfully updated predictions for field {field_name}")
        
        except Exception as e:
            logger.error(f"Error updating predictions for field {field_name}: {str(e)}")
    
    def get_or_calculate_ndvi(self, field_name: str, image_info: Dict[str, Any]) -> Optional[float]:
        """
        Get NDVI from the image or calculate it if it's not available.
        
        Args:
            field_name: Name of the field
            image_info: Information about the satellite image
            
        Returns:
            NDVI value or None if it couldn't be calculated
        """
        # Try to get precomputed NDVI from image info
        if 'properties' in image_info and 'ndvi' in image_info['properties']:
            return image_info['properties']['ndvi']
        
        # If not available, check if we have the NDVI file
        field_info = self.get_field_info(field_name)
        if not field_info:
            return None
        
        # Try to calculate NDVI from the image
        try:
            # This would require implementing the actual NDVI calculation
            # based on the specifics of your satellite data processing
            logger.warning("NDVI calculation from raw imagery not implemented")
            # Placeholder for demonstration
            return 0.75  # This would be replaced with actual calculation
        except Exception as e:
            logger.error(f"Error calculating NDVI: {str(e)}")
            return None
    
    def update_ndvi_time_series(self, field_name: str, date_str: str, ndvi_value: float):
        """
        Update the NDVI time series for a field.
        
        Args:
            field_name: Name of the field
            date_str: Date string (YYYY-MM-DD)
            ndvi_value: NDVI value
        """
        # Load existing NDVI time series if available
        ndvi_file = Path(f"data/{field_name}_ndvi.json")
        ndvi_data = {}
        
        if ndvi_file.exists():
            try:
                with open(ndvi_file, 'r') as f:
                    ndvi_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading NDVI data for {field_name}: {str(e)}")
        
        # Add the new NDVI value
        ndvi_data[date_str] = ndvi_value
        
        # Save the updated time series
        try:
            with open(ndvi_file, 'w') as f:
                json.dump(ndvi_data, f)
            logger.info(f"Updated NDVI time series for field {field_name}")
        except Exception as e:
            logger.error(f"Error saving NDVI data for {field_name}: {str(e)}")
    
    def get_ndvi_time_series(self, field_name: str) -> Dict[str, float]:
        """
        Get the NDVI time series for a field.
        
        Args:
            field_name: Name of the field
            
        Returns:
            Dictionary mapping dates to NDVI values
        """
        ndvi_file = Path(f"data/{field_name}_ndvi.json")
        if ndvi_file.exists():
            try:
                with open(ndvi_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading NDVI data for {field_name}: {str(e)}")
        
        return {}
    
    def predict_yield(self, field_name: str, ndvi_time_series: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        Predict yield based on NDVI time series.
        
        Args:
            field_name: Name of the field
            ndvi_time_series: Dictionary mapping dates to NDVI values
            
        Returns:
            Dictionary with yield predictions for different crops
        """
        if not ndvi_time_series:
            logger.warning(f"No NDVI time series available for field {field_name}")
            return None
        
        try:
            # Convert NDVI time series to dataframe
            ndvi_df = pd.DataFrame(list(ndvi_time_series.items()), columns=['date', 'ndvi'])
            ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])
            
            # Get crop type from field info
            field_info = self.get_field_info(field_name)
            crop_type = field_info.get('crop_type', 'unknown')
            
            # Prepare forecast dates (next 30, 60, 90 days)
            last_date = ndvi_df['date'].max()
            forecast_dates = [
                (last_date + timedelta(days=30)).strftime('%Y-%m-%d'),
                (last_date + timedelta(days=60)).strftime('%Y-%m-%d'),
                (last_date + timedelta(days=90)).strftime('%Y-%m-%d')
            ]
            
            # Simplified yield prediction based on NDVI trend
            ndvi_values = ndvi_df['ndvi'].values
            if len(ndvi_values) < 2:
                return None
            
            # Calculate basic statistics
            ndvi_mean = np.mean(ndvi_values)
            ndvi_trend = (ndvi_values[-1] - ndvi_values[0]) / len(ndvi_values)
            
            # Base yields for different crops (tons per hectare)
            base_yields = {
                'Wheat': 3.5,
                'Corn': 9.0,
                'Soybean': 3.0,
                'Barley': 3.2,
                'Oats': 2.5,
                'Rice': 4.5
            }
            
            # Default to wheat if crop type is unknown
            if crop_type.lower() not in [k.lower() for k in base_yields.keys()]:
                crop_type = 'Wheat'
            
            # Find the matching crop type case-insensitively
            crop_key = next((k for k in base_yields.keys() if k.lower() == crop_type.lower()), 'Wheat')
            base_yield = base_yields[crop_key]
            
            # Adjust yield based on NDVI
            ndvi_factor = 1.0 + ((ndvi_mean - 0.5) * 0.5)  # NDVI effect on yield
            trend_factor = 1.0 + (ndvi_trend * 20)  # Trend effect on yield
            
            # Calculate yield predictions for each forecast date
            predictions = {}
            for crop, base in base_yields.items():
                crop_predictions = {}
                for i, date in enumerate(forecast_dates):
                    # Different prediction horizons have different uncertainties
                    horizon_factor = 1.0 - (i * 0.05)  # Reduce confidence for farther predictions
                    yield_prediction = base * ndvi_factor * trend_factor * horizon_factor
                    crop_predictions[date] = round(yield_prediction, 2)
                
                predictions[crop] = crop_predictions
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error predicting yield for field {field_name}: {str(e)}")
            return None
    
    def update_yield_forecast(self, field_name: str, date_str: str, yield_forecast: Dict[str, Any]):
        """
        Update the yield forecast time series for a field.
        
        Args:
            field_name: Name of the field
            date_str: Date string (YYYY-MM-DD)
            yield_forecast: Dictionary with yield predictions for different crops
        """
        # Load existing yield forecast if available
        forecast_file = Path(f"data/{field_name}_yield_forecast.json")
        forecast_data = {
            "date_updated": date_str,
            "forecasts": {}
        }
        
        if forecast_file.exists():
            try:
                with open(forecast_file, 'r') as f:
                    existing_data = json.load(f)
                    # Keep existing structure but update forecasts
                    if "forecasts" in existing_data:
                        forecast_data["forecasts"] = existing_data["forecasts"]
            except Exception as e:
                logger.error(f"Error loading yield forecast for {field_name}: {str(e)}")
        
        # Update forecasts with new data
        for crop, predictions in yield_forecast.items():
            if crop not in forecast_data["forecasts"]:
                forecast_data["forecasts"][crop] = {}
            
            # Add the new predictions
            for pred_date, pred_value in predictions.items():
                forecast_data["forecasts"][crop][pred_date] = pred_value
        
        # Save the updated forecast
        try:
            with open(forecast_file, 'w') as f:
                json.dump(forecast_data, f)
            logger.info(f"Updated yield forecast for field {field_name}")
        except Exception as e:
            logger.error(f"Error saving yield forecast for {field_name}: {str(e)}")
    
    def generate_market_signals(self, field_name: str, ndvi_time_series: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Generate market signals based on NDVI time series and commodity prices.
        
        Args:
            field_name: Name of the field
            ndvi_time_series: Dictionary mapping dates to NDVI values
            
        Returns:
            Dictionary with market signals for different commodities
        """
        if not ndvi_time_series:
            logger.warning(f"No NDVI time series available for field {field_name}")
            return None
        
        try:
            # Convert NDVI time series to dataframe
            ndvi_df = pd.DataFrame(list(ndvi_time_series.items()), columns=['date', 'ndvi'])
            ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])
            
            # Get field info to determine crop type
            field_info = self.get_field_info(field_name)
            crop_type = field_info.get('crop_type', 'unknown')
            
            # Map crop type to commodity symbols
            crop_to_symbol = {
                'Wheat': 'ZW=F',
                'Corn': 'ZC=F',
                'Soybean': 'ZS=F',
                'Oats': 'ZO=F',
                'Rice': 'ZR=F'
            }
            
            # Default to wheat if crop type is unknown
            if crop_type.lower() not in [k.lower() for k in crop_to_symbol.keys()]:
                crop_type = 'Wheat'
            
            # Find the matching crop type case-insensitively
            crop_key = next((k for k in crop_to_symbol.keys() if k.lower() == crop_type.lower()), 'Wheat')
            primary_symbol = crop_to_symbol[crop_key]
            
            # For the primary crop and a few other main commodities
            commodity_symbols = [primary_symbol, 'ZW=F', 'ZC=F', 'ZS=F']
            
            # Calculate NDVI trend
            ndvi_df = ndvi_df.sort_values('date')
            if len(ndvi_df) < 5:
                return None
            
            # Calculate short, medium, and long term NDVI trends
            short_window = min(5, len(ndvi_df))
            medium_window = min(10, len(ndvi_df))
            long_window = len(ndvi_df)
            
            short_trend = (ndvi_df['ndvi'].iloc[-1] - ndvi_df['ndvi'].iloc[-short_window]) / short_window
            medium_trend = (ndvi_df['ndvi'].iloc[-1] - ndvi_df['ndvi'].iloc[-medium_window]) / medium_window
            long_trend = (ndvi_df['ndvi'].iloc[-1] - ndvi_df['ndvi'].iloc[0]) / long_window
            
            # Use the market model to generate signals
            market_signals = {}
            
            # Simple rule-based signals
            for symbol in commodity_symbols:
                # Determine the signal action based on NDVI trends
                if short_trend > 0.01:  # Strong positive NDVI trend
                    action = "SHORT"  # Expect price decrease due to good crop conditions
                    confidence = min(0.7, 0.5 + (short_trend * 10))
                    reason = f"Strong positive NDVI trend (+{short_trend:.4f}) suggests good crop conditions, potentially leading to increased supply and lower prices."
                elif short_trend < -0.01:  # Strong negative NDVI trend
                    action = "LONG"  # Expect price increase due to poor crop conditions
                    confidence = min(0.7, 0.5 + (abs(short_trend) * 10))
                    reason = f"Strong negative NDVI trend ({short_trend:.4f}) suggests poor crop conditions, potentially leading to reduced supply and higher prices."
                else:  # Neutral NDVI trend
                    action = "NEUTRAL"
                    confidence = 0.5
                    reason = f"Neutral NDVI trend ({short_trend:.4f}) suggests stable crop conditions with no strong market signal."
                
                # Adjust confidence based on alignment of different timeframes
                if (short_trend > 0 and medium_trend > 0 and long_trend > 0) or \
                   (short_trend < 0 and medium_trend < 0 and long_trend < 0):
                    confidence += 0.1  # Increase confidence if all trends align
                
                # Create signal
                today = datetime.now().strftime('%Y-%m-%d')
                signal = {
                    "date": today,
                    "action": action,
                    "confidence": min(0.9, confidence),  # Cap at 0.9
                    "reason": reason,
                    "ndvi_short_trend": float(short_trend),
                    "ndvi_medium_trend": float(medium_trend),
                    "ndvi_long_trend": float(long_trend)
                }
                
                # Add to signals
                if symbol not in market_signals:
                    market_signals[symbol] = []
                market_signals[symbol].append(signal)
            
            return {"signals": market_signals}
        
        except Exception as e:
            logger.error(f"Error generating market signals for field {field_name}: {str(e)}")
            return None
    
    def update_market_signals(self, field_name: str, date_str: str, market_signals: Dict[str, Any]):
        """
        Update the market signals for a field.
        
        Args:
            field_name: Name of the field
            date_str: Date string (YYYY-MM-DD)
            market_signals: Dictionary with market signals for different commodities
        """
        # Load existing market signals if available
        signals_file = Path(f"data/{field_name}_market_signals.json")
        signals_data = {
            "date_updated": date_str,
            "signals": {}
        }
        
        if signals_file.exists():
            try:
                with open(signals_file, 'r') as f:
                    existing_data = json.load(f)
                    # Keep existing structure but update signals
                    if "signals" in existing_data:
                        signals_data["signals"] = existing_data["signals"]
            except Exception as e:
                logger.error(f"Error loading market signals for {field_name}: {str(e)}")
        
        # Update signals with new data
        if "signals" in market_signals:
            for commodity, signals in market_signals["signals"].items():
                if commodity not in signals_data["signals"]:
                    signals_data["signals"][commodity] = []
                
                # Add the new signals
                for signal in signals:
                    signals_data["signals"][commodity].append(signal)
                
                # Sort signals by date
                signals_data["signals"][commodity] = sorted(
                    signals_data["signals"][commodity], 
                    key=lambda x: x.get("date", "")
                )
                
                # Keep only the last 30 signals
                signals_data["signals"][commodity] = signals_data["signals"][commodity][-30:]
        
        # Save the updated signals
        try:
            with open(signals_file, 'w') as f:
                json.dump(signals_data, f)
            logger.info(f"Updated market signals for field {field_name}")
        except Exception as e:
            logger.error(f"Error saving market signals for {field_name}: {str(e)}")
    
    def analyze_correlations(self, field_name: str):
        """
        Analyze correlations between NDVI, yields, and market prices for a field.
        
        Args:
            field_name: Name of the field
        """
        logger.info(f"Analyzing correlations for field: {field_name}")
        
        try:
            # Get NDVI time series
            ndvi_time_series = self.get_ndvi_time_series(field_name)
            if not ndvi_time_series:
                logger.warning(f"No NDVI time series available for field {field_name}")
                return
            
            # Convert to DataFrame
            ndvi_df = pd.DataFrame(list(ndvi_time_series.items()), columns=['date', 'ndvi'])
            ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])
            ndvi_df = ndvi_df.sort_values('date')
            
            # Get field info to determine crop type
            field_info = self.get_field_info(field_name)
            crop_type = field_info.get('crop_type', 'unknown')
            
            # Map crop type to commodity symbols
            crop_to_symbol = {
                'Wheat': 'ZW=F',
                'Corn': 'ZC=F',
                'Soybean': 'ZS=F',
                'Oats': 'ZO=F',
                'Rice': 'ZR=F'
            }
            
            # Get primary commodity symbol
            if crop_type.lower() not in [k.lower() for k in crop_to_symbol.keys()]:
                crop_type = 'Wheat'
            crop_key = next((k for k in crop_to_symbol.keys() if k.lower() == crop_type.lower()), 'Wheat')
            primary_symbol = crop_to_symbol[crop_key]
            
            # Try to get market data
            from models.market_signals import MarketSignalModel
            market_model = MarketSignalModel()
            
            try:
                # Get market data for the past year
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                end_date = datetime.now().strftime('%Y-%m-%d')
                
                # Run in a separate thread or event loop since it's an async function
                import asyncio
                price_data = asyncio.run(market_model.fetch_futures_prices([primary_symbol], '1y'))
                
                if price_data is not None and not price_data.empty:
                    # Calculate rolling correlations between NDVI and prices
                    # First, resample NDVI data to daily frequency and forward fill
                    ndvi_df = ndvi_df.set_index('date')
                    ndvi_daily = ndvi_df.resample('D').ffill()
                    
                    # Convert price_data index to datetime if it's not already
                    if not isinstance(price_data.index, pd.DatetimeIndex):
                        price_data.index = pd.to_datetime(price_data.index)
                    
                    # Align dates and merge
                    start_date = max(ndvi_daily.index.min(), price_data.index.min())
                    end_date = min(ndvi_daily.index.max(), price_data.index.max())
                    
                    # Filter data to common date range
                    ndvi_filtered = ndvi_daily[(ndvi_daily.index >= start_date) & (ndvi_daily.index <= end_date)]
                    price_filtered = price_data[(price_data.index >= start_date) & (price_data.index <= end_date)]
                    
                    # Merge data
                    merged_data = pd.DataFrame(index=ndvi_filtered.index)
                    merged_data['ndvi'] = ndvi_filtered['ndvi']
                    
                    # Add price data
                    for column in price_filtered.columns:
                        merged_data[column] = price_filtered[column]
                    
                    # Calculate correlations
                    correlations = {}
                    
                    # Calculate overall correlation
                    corr_matrix = merged_data.corr()
                    overall_corr = corr_matrix.loc['ndvi', primary_symbol] if primary_symbol in corr_matrix.columns else 0
                    correlations['overall'] = overall_corr
                    
                    # Calculate lagged correlations
                    max_lag = 30  # Check correlations with up to 30 days lag
                    lag_correlations = []
                    
                    for lag in range(1, max_lag + 1):
                        lagged_ndvi = merged_data['ndvi'].shift(lag)
                        valid_data = merged_data.dropna()
                        
                        if len(valid_data) > 10:  # Need at least 10 data points
                            if primary_symbol in valid_data.columns:
                                lag_corr = valid_data['ndvi'].corr(valid_data[primary_symbol])
                                lag_correlations.append((lag, lag_corr))
                    
                    correlations['lagged'] = lag_correlations
                    
                    # Save correlation results
                    correlation_file = self.correlation_dir / f"{field_name}_{primary_symbol}_correlation.json"
                    
                    correlation_data = {
                        "field_name": field_name,
                        "commodity": primary_symbol,
                        "date_analyzed": datetime.now().strftime('%Y-%m-%d'),
                        "overall_correlation": float(overall_corr),
                        "lag_correlations": [{"lag": lag, "correlation": float(corr)} for lag, corr in lag_correlations]
                    }
                    
                    with open(correlation_file, 'w') as f:
                        json.dump(correlation_data, f)
                    
                    logger.info(f"Saved correlation analysis for {field_name} with {primary_symbol}")
                    
                    # Find the strongest correlation
                    strongest_lag = max(lag_correlations, key=lambda x: abs(x[1]), default=(0, 0))
                    logger.info(f"Strongest correlation for {field_name} with {primary_symbol}: lag={strongest_lag[0]} days, corr={strongest_lag[1]:.4f}")
                
                else:
                    logger.warning(f"No price data available for {primary_symbol}")
            
            except Exception as e:
                logger.error(f"Error fetching price data: {str(e)}")
            
            # Also analyze correlation between NDVI and yield
            # This would require historical yield data which might not be available
            # For now, we'll use model predictions as proxy
            
            logger.info(f"Completed correlation analysis for field {field_name}")
        
        except Exception as e:
            logger.error(f"Error analyzing correlations for field {field_name}: {str(e)}")
    
    def generate_charts(self, field_name: str):
        """
        Generate charts for a field based on satellite data and predictions.
        
        Args:
            field_name: Name of the field
        """
        logger.info(f"Generating charts for field: {field_name}")
        
        try:
            # Create a directory for field charts
            field_charts_dir = self.charts_dir / field_name
            field_charts_dir.mkdir(exist_ok=True, parents=True)
            
            # Get NDVI time series
            ndvi_time_series = self.get_ndvi_time_series(field_name)
            if not ndvi_time_series:
                logger.warning(f"No NDVI time series available for field {field_name}")
                return
            
            # Convert to DataFrame
            ndvi_df = pd.DataFrame(list(ndvi_time_series.items()), columns=['date', 'ndvi'])
            ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])
            ndvi_df = ndvi_df.sort_values('date')
            
            # 1. NDVI time series chart
            plt.figure(figsize=(12, 6))
            plt.plot(ndvi_df['date'], ndvi_df['ndvi'], marker='o', linestyle='-', linewidth=2)
            plt.title(f'NDVI Time Series - {field_name}')
            plt.xlabel('Date')
            plt.ylabel('NDVI')
            plt.grid(True)
            plt.tight_layout()
            
            # Save chart
            ndvi_chart_path = field_charts_dir / 'ndvi_time_series.png'
            plt.savefig(ndvi_chart_path)
            plt.close()
            
            # 2. Load yield forecasts if available
            yield_file = Path(f"data/{field_name}_yield_forecast.json")
            if yield_file.exists():
                try:
                    with open(yield_file, 'r') as f:
                        yield_data = json.load(f)
                    
                    if "forecasts" in yield_data:
                        # Get field info to determine crop type
                        field_info = self.get_field_info(field_name)
                        crop_type = field_info.get('crop_type', 'Wheat')
                        
                        # Find the matching crop type case-insensitively
                        crop_keys = list(yield_data["forecasts"].keys())
                        crop_key = next((k for k in crop_keys if k.lower() == crop_type.lower()), 
                                        crop_keys[0] if crop_keys else None)
                        
                        if crop_key and crop_key in yield_data["forecasts"]:
                            # Convert yield forecasts to DataFrame
                            crop_forecasts = yield_data["forecasts"][crop_key]
                            forecast_df = pd.DataFrame(list(crop_forecasts.items()), columns=['date', 'yield'])
                            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
                            forecast_df = forecast_df.sort_values('date')
                            
                            # Create yield forecast chart
                            plt.figure(figsize=(12, 6))
                            plt.plot(forecast_df['date'], forecast_df['yield'], marker='s', linestyle='-', linewidth=2)
                            plt.title(f'Yield Forecast - {crop_key} - {field_name}')
                            plt.xlabel('Forecast Date')
                            plt.ylabel('Predicted Yield (t/ha)')
                            plt.grid(True)
                            plt.tight_layout()
                            
                            # Save chart
                            yield_chart_path = field_charts_dir / f'yield_forecast_{crop_key.lower()}.png'
                            plt.savefig(yield_chart_path)
                            plt.close()
                            
                            logger.info(f"Generated yield forecast chart for {field_name} - {crop_key}")
                
                except Exception as e:
                    logger.error(f"Error generating yield forecast chart for {field_name}: {str(e)}")
            
            # 3. Load market signals if available
            signals_file = Path(f"data/{field_name}_market_signals.json")
            if signals_file.exists():
                try:
                    with open(signals_file, 'r') as f:
                        signals_data = json.load(f)
                    
                    if "signals" in signals_data:
                        # Get field info to determine crop type
                        field_info = self.get_field_info(field_name)
                        crop_type = field_info.get('crop_type', 'Wheat')
                        
                        # Map crop type to commodity symbols
                        crop_to_symbol = {
                            'Wheat': 'ZW=F',
                            'Corn': 'ZC=F',
                            'Soybean': 'ZS=F',
                            'Oats': 'ZO=F',
                            'Rice': 'ZR=F'
                        }
                        
                        # Get primary commodity symbol
                        if crop_type.lower() not in [k.lower() for k in crop_to_symbol.keys()]:
                            crop_type = 'Wheat'
                        crop_key = next((k for k in crop_to_symbol.keys() if k.lower() == crop_type.lower()), 'Wheat')
                        primary_symbol = crop_to_symbol[crop_key]
                        
                        if primary_symbol in signals_data["signals"]:
                            # Convert signals to DataFrame
                            signals = signals_data["signals"][primary_symbol]
                            signal_rows = []
                            
                            for signal in signals:
                                if "date" in signal and "action" in signal and "confidence" in signal:
                                    signal_rows.append({
                                        "date": signal["date"],
                                        "action": signal["action"],
                                        "confidence": signal["confidence"],
                                        "value": 1 if signal["action"] == "LONG" else 
                                                 (-1 if signal["action"] == "SHORT" else 0)
                                    })
                            
                            if signal_rows:
                                signal_df = pd.DataFrame(signal_rows)
                                signal_df['date'] = pd.to_datetime(signal_df['date'])
                                signal_df = signal_df.sort_values('date')
                                
                                # Create market signals chart
                                plt.figure(figsize=(12, 6))
                                
                                # Scatter plot for signals
                                for action in ["LONG", "SHORT", "NEUTRAL"]:
                                    action_df = signal_df[signal_df['action'] == action]
                                    if not action_df.empty:
                                        color = 'green' if action == "LONG" else ('red' if action == "SHORT" else 'gray')
                                        plt.scatter(action_df['date'], action_df['value'], 
                                                    c=color, s=action_df['confidence']*100, 
                                                    alpha=0.7, label=action)
                                
                                # Add a line connecting the values
                                plt.plot(signal_df['date'], signal_df['value'], 'k-', alpha=0.3)
                                
                                plt.title(f'Market Signals - {primary_symbol} - {field_name}')
                                plt.xlabel('Date')
                                plt.ylabel('Signal')
                                plt.yticks([-1, 0, 1], ['SHORT', 'NEUTRAL', 'LONG'])
                                plt.grid(True)
                                plt.legend()
                                plt.tight_layout()
                                
                                # Save chart
                                signals_chart_path = field_charts_dir / f'market_signals_{primary_symbol}.png'
                                plt.savefig(signals_chart_path)
                                plt.close()
                                
                                logger.info(f"Generated market signals chart for {field_name} - {primary_symbol}")
                
                except Exception as e:
                    logger.error(f"Error generating market signals chart for {field_name}: {str(e)}")
            
            # 4. Load correlation data if available
            correlation_files = list(self.correlation_dir.glob(f"{field_name}_*_correlation.json"))
            if correlation_files:
                try:
                    for corr_file in correlation_files:
                        with open(corr_file, 'r') as f:
                            corr_data = json.load(f)
                        
                        if "lag_correlations" in corr_data and corr_data["lag_correlations"]:
                            # Extract commodity from filename
                            commodity = corr_data.get("commodity", corr_file.stem.split('_')[1])
                            
                            # Convert correlations to DataFrame
                            lag_corrs = corr_data["lag_correlations"]
                            lag_df = pd.DataFrame(lag_corrs)
                            
                            # Create correlation chart
                            plt.figure(figsize=(12, 6))
                            plt.bar(lag_df['lag'], lag_df['correlation'], color='skyblue')
                            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                            plt.title(f'NDVI-Price Correlation by Lag Days - {commodity} - {field_name}')
                            plt.xlabel('Lag (days)')
                            plt.ylabel('Correlation Coefficient')
                            plt.grid(True, axis='y')
                            plt.tight_layout()
                            
                            # Save chart
                            corr_chart_path = field_charts_dir / f'ndvi_price_correlation_{commodity}.png'
                            plt.savefig(corr_chart_path)
                            plt.close()
                            
                            logger.info(f"Generated correlation chart for {field_name} - {commodity}")
                
                except Exception as e:
                    logger.error(f"Error generating correlation chart for {field_name}: {str(e)}")
            
            logger.info(f"Completed generating charts for field {field_name}")
        
        except Exception as e:
            logger.error(f"Error generating charts for field {field_name}: {str(e)}")
    
    def get_field_info(self, field_name: str) -> Dict[str, Any]:
        """
        Get field information from the system.
        
        Args:
            field_name: Name of the field
            
        Returns:
            Dictionary with field information
        """
        # First check if we have a JSON file with field info
        field_info_file = Path(f"data/{field_name}_info.json")
        if field_info_file.exists():
            try:
                with open(field_info_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading field info for {field_name}: {str(e)}")
        
        # If not, check the database
        from database import Field, get_db
        
        try:
            db = get_db()
            field = db.query(Field).filter(Field.name == field_name).first()
            
            if field:
                field_info = {
                    "name": field.name,
                    "geojson": field.geojson,
                    "center_lat": field.center_lat,
                    "center_lon": field.center_lon,
                    "area_hectares": field.area_hectares,
                    "crop_type": field.crop_type
                }
                return field_info
        except Exception as e:
            logger.error(f"Error getting field info from database for {field_name}: {str(e)}")
        
        # If all else fails, return an empty dict
        return {}
    
    def get_available_fields(self) -> List[str]:
        """
        Get a list of all available fields in the system.
        
        Returns:
            List of field names
        """
        # Get fields from database
        from database import Field, get_db
        
        field_names = []
        
        try:
            db = get_db()
            fields = db.query(Field).all()
            field_names = [field.name for field in fields]
        except Exception as e:
            logger.error(f"Error getting fields from database: {str(e)}")
        
        # Also check the data directory for field data files
        data_dir = Path("data")
        if data_dir.exists():
            for file_path in data_dir.glob("*_ndvi.json"):
                field_name = file_path.stem.split('_')[0]
                if field_name not in field_names:
                    field_names.append(field_name)
        
        return field_names

# Singleton instance
satellite_monitor = SatelliteMonitor()

def get_satellite_monitor() -> SatelliteMonitor:
    """Get the satellite monitor singleton instance."""
    return satellite_monitor

def start_monitoring(interval_hours=24):
    """Start the satellite monitoring process."""
    satellite_monitor.start_monitoring(interval_hours)

def stop_monitoring():
    """Stop the satellite monitoring process."""
    satellite_monitor.stop_monitoring()

def check_for_updates_now():
    """Manually trigger checking for updates."""
    satellite_monitor.check_for_updates()

def generate_charts_for_all_fields():
    """Generate charts for all available fields."""
    fields = satellite_monitor.get_available_fields()
    for field_name in fields:
        satellite_monitor.generate_charts(field_name)