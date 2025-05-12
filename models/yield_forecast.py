import pandas as pd
import numpy as np
import lightgbm as lgb
from prophet import Prophet
from typing import Dict, List, Tuple, Any, Optional
import datetime
import logging
import asyncio
import aiohttp
import json
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logger
logger = logging.getLogger(__name__)

class YieldForecastModel:
    """Model for forecasting crop yields based on satellite data."""
    
    def __init__(self):
        self.prophet_model = None
        self.lgbm_model = None
        self.feature_importance = None
        self.metrics = None
        self.best_hyperparams = None
        
        # Create models directory if it doesn't exist
        self.models_dir = Path("./models/saved")
        self.models_dir.mkdir(exist_ok=True, parents=True)
    
    def prepare_training_data(
        self, 
        ndvi_time_series: Dict[str, List[float]], 
        weather_data: Dict[str, Dict[str, float]],
        historical_yields: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        Prepare training data by combining NDVI time series and weather data.
        
        Args:
            ndvi_time_series: Dictionary mapping dates to NDVI values
            weather_data: Dictionary mapping dates to weather variables
            historical_yields: Dictionary mapping years to yield values
            
        Returns:
            Pandas DataFrame ready for model training
        """
        # Create a DataFrame from NDVI time series
        ndvi_df = pd.DataFrame(
            [(date, value) for date, value in ndvi_time_series.items()],
            columns=['date', 'ndvi']
        )
        
        # Convert date strings to datetime
        ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])
        
        # Sort by date
        ndvi_df = ndvi_df.sort_values('date')
        
        # Create a DataFrame from weather data
        weather_rows = []
        for date, variables in weather_data.items():
            row = {'date': date}
            row.update(variables)
            weather_rows.append(row)
            
        weather_df = pd.DataFrame(weather_rows)
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        # Merge NDVI and weather data
        merged_df = pd.merge(ndvi_df, weather_df, on='date', how='outer')
        
        # Fill missing values using forward/backward fill
        merged_df = merged_df.sort_values('date').ffill().bfill()
        
        # Add time-based features
        merged_df['year'] = merged_df['date'].dt.year
        merged_df['month'] = merged_df['date'].dt.month
        merged_df['day'] = merged_df['date'].dt.day
        merged_df['day_of_year'] = merged_df['date'].dt.dayofyear
        
        # Calculate rolling statistics for NDVI
        merged_df['ndvi_rolling_mean_30d'] = merged_df['ndvi'].rolling(window=6).mean()  # Assuming ~5 days between NDVI measurements
        merged_df['ndvi_rolling_std_30d'] = merged_df['ndvi'].rolling(window=6).std()
        
        # Handle missing values from rolling calculations
        merged_df = merged_df.fillna(method='bfill').fillna(method='ffill')
        
        # Add historical yields if available
        if historical_yields:
            yield_df = pd.DataFrame(
                [(int(year), value) for year, value in historical_yields.items()],
                columns=['year', 'yield']
            )
            
            merged_df = pd.merge(merged_df, yield_df, on='year', how='left')
            
            # Forward fill yields within the same year
            merged_df['yield'] = merged_df.groupby('year')['yield'].transform(lambda x: x.ffill().bfill())
        
        return merged_df
    
    async def fetch_weather_data(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Fetch historical weather data from Open-Meteo API.
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping dates to weather variables
        """
        try:
            # Format dates
            start_date_obj = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            end_date_obj = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            
            # Build URL
            url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={lat}&longitude={lon}&"
                f"start_date={start_date}&end_date={end_date}&"
                f"daily=temperature_2m_max,temperature_2m_min,precipitation_sum,et0_fao_evapotranspiration"
            )
            
            # Make async request
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    data = await response.json()
            
            # Process the response
            if 'daily' not in data:
                logger.error(f"Invalid response from Open-Meteo API: {data}")
                return {}
            
            daily_data = data['daily']
            dates = daily_data.get('time', [])
            
            # Create the result dictionary
            result = {}
            for i, date in enumerate(dates):
                result[date] = {
                    'temp_max': daily_data.get('temperature_2m_max', [])[i],
                    'temp_min': daily_data.get('temperature_2m_min', [])[i],
                    'precip': daily_data.get('precipitation_sum', [])[i],
                    'evapotrans': daily_data.get('et0_fao_evapotranspiration', [])[i]
                }
            
            return result
                
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return {}
    
    def train_prophet_model(
        self,
        time_series_df: pd.DataFrame,
        target_column: str = 'ndvi',
        date_column: str = 'date',
        forecast_periods: int = 90  # 3 months
    ) -> Dict[str, Any]:
        """
        Train a Prophet model for time series forecasting.
        
        Args:
            time_series_df: DataFrame with time series data
            target_column: Name of the target column
            date_column: Name of the date column
            forecast_periods: Number of days to forecast
            
        Returns:
            Dictionary with model, forecast, and metrics
        """
        # Prepare data for Prophet
        prophet_df = time_series_df[[date_column, target_column]].copy()
        prophet_df.columns = ['ds', 'y']  # Prophet requires these column names
        
        # Train the model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        model.fit(prophet_df)
        
        # Make forecast
        future = model.make_future_dataframe(periods=forecast_periods)
        forecast = model.predict(future)
        
        # Calculate metrics on the historical data
        historical_forecast = forecast[forecast['ds'].isin(prophet_df['ds'])]
        metrics = {
            'mae': mean_absolute_error(prophet_df['y'], historical_forecast['yhat']),
            'rmse': np.sqrt(mean_squared_error(prophet_df['y'], historical_forecast['yhat'])),
            'r2': r2_score(prophet_df['y'], historical_forecast['yhat'])
        }
        
        # Save the model
        self.prophet_model = model
        
        # Return results
        return {
            'model': model,
            'forecast': forecast,
            'metrics': metrics
        }
    
    def train_lightgbm_model(
        self,
        features_df: pd.DataFrame,
        target_column: str = 'yield',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train a LightGBM model for yield prediction.
        
        Args:
            features_df: DataFrame with features
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed
            
        Returns:
            Dictionary with model, predictions, and metrics
        """
        # Drop date column if present
        if 'date' in features_df.columns:
            X = features_df.drop(['date', target_column], axis=1)
        else:
            X = features_df.drop([target_column], axis=1)
        
        y = features_df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train the model
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Make predictions
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Calculate feature importance
        feature_importance = model.feature_importance(importance_type='gain')
        feature_names = X.columns.tolist()
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Save the model
        self.lgbm_model = model
        self.feature_importance = importance_df
        self.metrics = metrics
        
        # Return results
        return {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'feature_importance': importance_df
        }
    
    def forecast_yield(
        self,
        current_data: pd.DataFrame,
        forecast_dates: List[str]
    ) -> Dict[str, float]:
        """
        Forecast yields for future dates.
        
        Args:
            current_data: DataFrame with current data
            forecast_dates: List of dates to forecast for (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping dates to forecasted yields
        """
        if self.lgbm_model is None:
            logger.error("LightGBM model not trained yet")
            return {}
        
        # Create future data based on current data
        forecast_rows = []
        
        for date_str in forecast_dates:
            # Parse date
            date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            
            # Create a copy of the latest data row
            latest_row = current_data.iloc[-1].copy()
            
            # Update date-related features
            latest_row['year'] = date.year
            latest_row['month'] = date.month
            latest_row['day'] = date.day
            latest_row['day_of_year'] = date.timetuple().tm_yday
            
            # Use the Prophet model to forecast NDVI if available
            if self.prophet_model is not None:
                future_date = pd.DataFrame({'ds': [date]})
                ndvi_forecast = self.prophet_model.predict(future_date)
                latest_row['ndvi'] = ndvi_forecast['yhat'].values[0]
            
            forecast_rows.append(latest_row)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame(forecast_rows)
        
        # Drop target column if present
        if 'yield' in forecast_df.columns:
            forecast_df = forecast_df.drop('yield', axis=1)
        
        # Drop date column if present
        if 'date' in forecast_df.columns:
            forecast_df = forecast_df.drop('date', axis=1)
        
        # Make predictions using LightGBM model
        predictions = self.lgbm_model.predict(forecast_df[self.lgbm_model.feature_name()])
        
        # Create result dictionary
        result = {date: pred for date, pred in zip(forecast_dates, predictions)}
        
        return result
    
    def save_model(self, filename: str = "yield_forecast_model") -> str:
        """
        Save the trained models to files.
        
        Args:
            filename: Base filename for the models
            
        Returns:
            Path to the saved model directory
        """
        if self.lgbm_model is None and self.prophet_model is None:
            logger.error("No trained models to save")
            return ""
        
        # Create directory if it doesn't exist
        model_dir = self.models_dir / filename
        model_dir.mkdir(exist_ok=True, parents=True)
        
        # Save LightGBM model if available
        if self.lgbm_model is not None:
            lgbm_path = model_dir / "lightgbm_model.txt"
            self.lgbm_model.save_model(str(lgbm_path))
            
            # Save feature importance
            if self.feature_importance is not None:
                importance_path = model_dir / "feature_importance.csv"
                self.feature_importance.to_csv(str(importance_path), index=False)
            
            # Save metrics
            if self.metrics is not None:
                metrics_path = model_dir / "metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(self.metrics, f, indent=2)
        
        # Save Prophet model if available
        if self.prophet_model is not None:
            prophet_path = model_dir / "prophet_model.json"
            with open(prophet_path, 'w') as f:
                json.dump(self.prophet_model.to_json(), f, indent=2)
        
        return str(model_dir)
    
    def load_model(self, model_dir: str) -> bool:
        """
        Load trained models from files.
        
        Args:
            model_dir: Directory containing the saved models
            
        Returns:
            True if loading was successful, False otherwise
        """
        model_path = Path(model_dir)
        
        try:
            # Load LightGBM model if available
            lgbm_path = model_path / "lightgbm_model.txt"
            if lgbm_path.exists():
                self.lgbm_model = lgb.Booster(model_file=str(lgbm_path))
                
                # Load feature importance
                importance_path = model_path / "feature_importance.csv"
                if importance_path.exists():
                    self.feature_importance = pd.read_csv(importance_path)
                
                # Load metrics
                metrics_path = model_path / "metrics.json"
                if metrics_path.exists():
                    with open(metrics_path, 'r') as f:
                        self.metrics = json.load(f)
            
            # Load Prophet model if available
            prophet_path = model_path / "prophet_model.json"
            if prophet_path.exists():
                with open(prophet_path, 'r') as f:
                    prophet_json = json.load(f)
                
                self.prophet_model = Prophet.from_json(prophet_json)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
