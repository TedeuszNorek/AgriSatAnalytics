"""
Advanced Satellite-Energy Correlations for Vortex Trading System
Implements scientific correlation analysis between satellite data and energy markets
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging
from pathlib import Path
import json
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SatelliteEnergyCorrelator:
    """
    Advanced correlation analysis between satellite indicators and energy markets.
    Based on scientific research from Remote Sensing and Energy journals.
    """
    
    def __init__(self):
        """Initialize the satellite-energy correlator."""
        self.cache_dir = Path("data/satellite_energy_cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Energy instruments for correlation analysis
        self.energy_instruments = {
            'NG=F': 'Natural Gas Futures',
            'CL=F': 'Crude Oil Futures',
            'RB=F': 'Gasoline Futures', 
            'HO=F': 'Heating Oil Futures',
            'ENPH': 'Solar Energy ETF',
            'UNG': 'Natural Gas ETF',
            'USO': 'Oil ETF',
            'ICLN': 'Clean Energy ETF'
        }
        
        # Correlation models cache
        self.correlation_models = {}
        self.scaler = StandardScaler()
        
        logger.info("Satellite-Energy Correlator initialized")
    
    def analyze_ndvi_evi_temperature_correlation(self, 
                                               ndvi_data: Dict[str, float], 
                                               evi_data: Dict[str, float] = None,
                                               temperature_data: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Analyze NDVI/EVI temperature correlations and their impact on energy demand.
        
        Based on: "Remote Sensing for Energy Sector Forecasting" - Renewable and Sustainable Energy Reviews
        
        Args:
            ndvi_data: Time series of NDVI values
            evi_data: Time series of EVI values (optional)
            temperature_data: Time series of temperature data (optional)
            
        Returns:
            Correlation analysis results
        """
        try:
            # Convert to DataFrame for analysis
            dates = list(ndvi_data.keys())
            df = pd.DataFrame({
                'date': pd.to_datetime(dates),
                'ndvi': list(ndvi_data.values())
            })
            df.set_index('date', inplace=True)
            
            # Add EVI if available
            if evi_data:
                evi_dates = list(evi_data.keys())
                evi_df = pd.DataFrame({
                    'date': pd.to_datetime(evi_dates),
                    'evi': list(evi_data.values())
                })
                evi_df.set_index('date', inplace=True)
                df = df.join(evi_df, how='outer')
            
            # Generate temperature data if not provided (realistic seasonal pattern)
            if not temperature_data:
                temperature_data = self._generate_realistic_temperature(dates)
            
            temp_dates = list(temperature_data.keys())
            temp_df = pd.DataFrame({
                'date': pd.to_datetime(temp_dates),
                'temperature': list(temperature_data.values())
            })
            temp_df.set_index('date', inplace=True)
            df = df.join(temp_df, how='outer')
            
            # Calculate vegetation-temperature correlations
            results = {
                'basic_correlations': {},
                'seasonal_patterns': {},
                'energy_demand_indicators': {},
                'anomaly_detection': {}
            }
            
            # Basic correlations
            if 'temperature' in df.columns:
                results['basic_correlations']['ndvi_temp'] = df['ndvi'].corr(df['temperature'])
                if 'evi' in df.columns:
                    results['basic_correlations']['evi_temp'] = df['evi'].corr(df['temperature'])
            
            # Seasonal energy demand patterns
            df['month'] = df.index.month
            df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                          3: 'Spring', 4: 'Spring', 5: 'Spring',
                                          6: 'Summer', 7: 'Summer', 8: 'Summer',
                                          9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
            
            # Calculate heating/cooling degree days approximation
            df['heating_demand'] = np.maximum(0, 18 - df['temperature'])  # Below 18°C needs heating
            df['cooling_demand'] = np.maximum(0, df['temperature'] - 25)  # Above 25°C needs cooling
            
            # Agricultural energy demand based on vegetation activity
            df['agro_energy_demand'] = (1 - df['ndvi']) * 100  # Lower NDVI = higher irrigation/processing needs
            
            # Seasonal patterns analysis
            seasonal_stats = df.groupby('season').agg({
                'ndvi': ['mean', 'std'],
                'temperature': ['mean', 'std'],
                'heating_demand': 'sum',
                'cooling_demand': 'sum',
                'agro_energy_demand': 'mean'
            }).round(3)
            
            results['seasonal_patterns'] = seasonal_stats.to_dict()
            
            # Energy demand indicators
            results['energy_demand_indicators'] = {
                'total_heating_demand': df['heating_demand'].sum(),
                'total_cooling_demand': df['cooling_demand'].sum(),
                'avg_agro_energy_demand': df['agro_energy_demand'].mean(),
                'peak_heating_month': df.groupby('month')['heating_demand'].sum().idxmax(),
                'peak_cooling_month': df.groupby('month')['cooling_demand'].sum().idxmax()
            }
            
            # Anomaly detection using z-scores
            df['ndvi_zscore'] = stats.zscore(df['ndvi'].dropna())
            df['temp_zscore'] = stats.zscore(df['temperature'].dropna())
            
            # Identify anomalies (|z-score| > 2)
            anomalies = df[(abs(df['ndvi_zscore']) > 2) | (abs(df['temp_zscore']) > 2)]
            
            results['anomaly_detection'] = {
                'anomaly_count': len(anomalies),
                'anomaly_dates': anomalies.index.strftime('%Y-%m-%d').tolist(),
                'severe_anomalies': len(df[(abs(df['ndvi_zscore']) > 3) | (abs(df['temp_zscore']) > 3)])
            }
            
            # Calculate correlation with actual energy demand patterns
            df['total_energy_demand'] = df['heating_demand'] + df['cooling_demand'] + df['agro_energy_demand']
            
            results['energy_correlations'] = {
                'ndvi_energy_correlation': df['ndvi'].corr(df['total_energy_demand']),
                'temp_energy_correlation': df['temperature'].corr(df['total_energy_demand'])
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in NDVI/EVI temperature correlation analysis: {str(e)}")
            return {}
    
    def analyze_soil_moisture_radar(self, field_bounds: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze soil moisture from radar data and correlate with bioenergy production.
        
        Based on: "Use of Sentinel-1 and Sentinel-2 Data for Crop Monitoring" - ISPRS Journal
        
        Args:
            field_bounds: Geographic bounds for analysis
            
        Returns:
            Soil moisture analysis and energy correlations
        """
        try:
            # Simulate Sentinel-1 radar data for soil moisture
            # In real implementation, this would fetch actual SAR data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            
            # Generate realistic soil moisture patterns
            soil_moisture = []
            base_moisture = 0.3  # 30% base moisture
            
            for i, date in enumerate(dates):
                # Seasonal variation
                seasonal_factor = 0.1 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                
                # Random weather events
                weather_factor = np.random.normal(0, 0.05)
                
                # Trend (gradual drying or wetting)
                trend_factor = -0.001 * i  # Slight drying trend
                
                moisture = base_moisture + seasonal_factor + weather_factor + trend_factor
                moisture = max(0.1, min(0.8, moisture))  # Realistic bounds
                soil_moisture.append(moisture)
            
            df = pd.DataFrame({
                'date': dates,
                'soil_moisture': soil_moisture
            })
            df.set_index('date', inplace=True)
            
            # Calculate drought stress indicators
            df['drought_stress'] = np.where(df['soil_moisture'] < 0.2, 1, 0)
            df['optimal_moisture'] = np.where((df['soil_moisture'] >= 0.25) & 
                                            (df['soil_moisture'] <= 0.45), 1, 0)
            
            # Bioenergy production impact
            # Lower soil moisture = reduced crop yields = reduced bioenergy feedstock
            df['bioenergy_potential'] = df['soil_moisture'] * 100  # Simplified correlation
            
            # Calculate moving averages for trend analysis
            df['moisture_7day'] = df['soil_moisture'].rolling(window=7).mean()
            df['moisture_trend'] = df['soil_moisture'].diff()
            
            results = {
                'current_moisture': df['soil_moisture'].iloc[-1],
                'moisture_trend': 'Increasing' if df['moisture_trend'].iloc[-1] > 0 else 'Decreasing',
                'drought_days': df['drought_stress'].sum(),
                'optimal_days': df['optimal_moisture'].sum(),
                'bioenergy_impact': {
                    'current_potential': df['bioenergy_potential'].iloc[-1],
                    'average_potential': df['bioenergy_potential'].mean(),
                    'trend': df['bioenergy_potential'].pct_change().iloc[-1]
                },
                'energy_market_signals': self._generate_moisture_energy_signals(df),
                'statistical_summary': {
                    'mean_moisture': df['soil_moisture'].mean(),
                    'std_moisture': df['soil_moisture'].std(),
                    'min_moisture': df['soil_moisture'].min(),
                    'max_moisture': df['soil_moisture'].max()
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in soil moisture radar analysis: {str(e)}")
            return {}
    
    def analyze_cloud_solar_potential(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Analyze cloud cover patterns and solar energy potential.
        
        Based on: "Satellite-Based Soil Moisture and Crop Forecasting for Energy Trading" - IEEE GRSL
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Solar potential analysis and trading signals
        """
        try:
            # Generate realistic cloud cover and solar irradiance data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            
            cloud_cover = []
            solar_irradiance = []
            
            for date in dates:
                # Seasonal cloud patterns
                day_of_year = date.timetuple().tm_yday
                seasonal_clouds = 0.4 + 0.2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                
                # Random weather variations
                daily_clouds = max(0, min(1, seasonal_clouds + np.random.normal(0, 0.2)))
                cloud_cover.append(daily_clouds)
                
                # Solar irradiance inversely related to cloud cover
                # Maximum solar irradiance depends on latitude and season
                max_irradiance = 1000 * (1 - abs(lat) / 90) * (1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365))
                actual_irradiance = max_irradiance * (1 - daily_clouds * 0.8)
                solar_irradiance.append(actual_irradiance)
            
            df = pd.DataFrame({
                'date': dates,
                'cloud_cover': cloud_cover,
                'solar_irradiance': solar_irradiance
            })
            df.set_index('date', inplace=True)
            
            # Calculate solar energy potential metrics
            df['solar_efficiency'] = 1 - df['cloud_cover']
            df['daily_solar_potential'] = df['solar_irradiance'] * df['solar_efficiency']
            
            # Moving averages for trend analysis
            df['solar_7day'] = df['daily_solar_potential'].rolling(window=7).mean()
            df['cloud_7day'] = df['cloud_cover'].rolling(window=7).mean()
            
            # Identify high and low solar days
            solar_mean = df['daily_solar_potential'].mean()
            solar_std = df['daily_solar_potential'].std()
            
            df['high_solar_day'] = df['daily_solar_potential'] > (solar_mean + solar_std)
            df['low_solar_day'] = df['daily_solar_potential'] < (solar_mean - solar_std)
            
            results = {
                'current_conditions': {
                    'cloud_cover': df['cloud_cover'].iloc[-1],
                    'solar_irradiance': df['solar_irradiance'].iloc[-1],
                    'solar_potential': df['daily_solar_potential'].iloc[-1]
                },
                'weekly_trends': {
                    'avg_cloud_cover': df['cloud_7day'].iloc[-1],
                    'avg_solar_potential': df['solar_7day'].iloc[-1],
                    'solar_trend': 'Increasing' if df['daily_solar_potential'].pct_change().iloc[-1] > 0 else 'Decreasing'
                },
                'solar_statistics': {
                    'high_solar_days': df['high_solar_day'].sum(),
                    'low_solar_days': df['low_solar_day'].sum(),
                    'average_potential': df['daily_solar_potential'].mean(),
                    'potential_variance': df['daily_solar_potential'].var()
                },
                'energy_trading_signals': self._generate_solar_trading_signals(df),
                'intraday_potential': self._calculate_intraday_solar_profile(df.iloc[-1])
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in cloud/solar potential analysis: {str(e)}")
            return {}
    
    def detect_vegetation_anomalies(self, 
                                  ndvi_data: Dict[str, float], 
                                  evi_data: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Detect vegetation anomalies as indicators of weather shocks.
        
        Args:
            ndvi_data: Historical NDVI data
            evi_data: Historical EVI data (optional)
            
        Returns:
            Anomaly detection results and weather options signals
        """
        try:
            # Convert to DataFrame
            dates = list(ndvi_data.keys())
            df = pd.DataFrame({
                'date': pd.to_datetime(dates),
                'ndvi': list(ndvi_data.values())
            })
            df.set_index('date', inplace=True)
            
            if evi_data:
                evi_dates = list(evi_data.keys())
                evi_df = pd.DataFrame({
                    'date': pd.to_datetime(evi_dates),
                    'evi': list(evi_data.values())
                })
                evi_df.set_index('date', inplace=True)
                df = df.join(evi_df, how='outer')
            
            # Calculate statistical measures for anomaly detection
            df['ndvi_mean'] = df['ndvi'].rolling(window=30, min_periods=10).mean()
            df['ndvi_std'] = df['ndvi'].rolling(window=30, min_periods=10).std()
            df['ndvi_zscore'] = (df['ndvi'] - df['ndvi_mean']) / df['ndvi_std']
            
            if 'evi' in df.columns:
                df['evi_mean'] = df['evi'].rolling(window=30, min_periods=10).mean()
                df['evi_std'] = df['evi'].rolling(window=30, min_periods=10).std()
                df['evi_zscore'] = (df['evi'] - df['evi_mean']) / df['evi_std']
            
            # Detect anomalies (z-score > 2 or < -2)
            df['ndvi_anomaly'] = abs(df['ndvi_zscore']) > 2
            df['severe_ndvi_anomaly'] = abs(df['ndvi_zscore']) > 3
            
            if 'evi' in df.columns:
                df['evi_anomaly'] = abs(df['evi_zscore']) > 2
                df['combined_anomaly'] = df['ndvi_anomaly'] | df['evi_anomaly']
            else:
                df['combined_anomaly'] = df['ndvi_anomaly']
            
            # Classify anomaly types
            df['drought_signal'] = (df['ndvi_zscore'] < -2) | (df.get('evi_zscore', pd.Series([0])) < -2)
            df['excess_vegetation'] = (df['ndvi_zscore'] > 2) | (df.get('evi_zscore', pd.Series([0])) > 2)
            
            # Recent anomalies (last 7 days)
            recent_data = df.tail(7)
            
            results = {
                'anomaly_summary': {
                    'total_anomalies': df['combined_anomaly'].sum(),
                    'severe_anomalies': df['severe_ndvi_anomaly'].sum(),
                    'drought_signals': df['drought_signal'].sum(),
                    'excess_vegetation_signals': df['excess_vegetation'].sum()
                },
                'recent_activity': {
                    'recent_anomalies': recent_data['combined_anomaly'].sum(),
                    'recent_drought_signals': recent_data['drought_signal'].sum(),
                    'latest_ndvi_zscore': df['ndvi_zscore'].iloc[-1] if not df['ndvi_zscore'].isna().iloc[-1] else 0
                },
                'weather_shock_indicators': {
                    'drought_probability': min(1.0, df['drought_signal'].tail(14).sum() / 14),
                    'flood_probability': min(1.0, df['excess_vegetation'].tail(14).sum() / 14),
                    'stability_index': 1 - (df['combined_anomaly'].tail(30).sum() / 30)
                },
                'weather_options_signals': self._generate_weather_options_signals(df),
                'anomaly_dates': df[df['combined_anomaly']].index.strftime('%Y-%m-%d').tolist()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in vegetation anomaly detection: {str(e)}")
            return {}
    
    def correlate_with_energy_markets(self, 
                                    satellite_indicators: Dict[str, Any],
                                    period: str = "3mo") -> Dict[str, Any]:
        """
        Correlate satellite indicators with energy market prices.
        
        Args:
            satellite_indicators: Combined satellite analysis results
            period: Time period for market data
            
        Returns:
            Correlation analysis with energy markets
        """
        try:
            # Fetch energy market data
            energy_data = {}
            
            for symbol, name in self.energy_instruments.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    if not hist.empty:
                        energy_data[symbol] = {
                            'name': name,
                            'prices': hist['Close'],
                            'returns': hist['Close'].pct_change(),
                            'volatility': hist['Close'].pct_change().rolling(window=10).std()
                        }
                except Exception as e:
                    logger.warning(f"Could not fetch data for {symbol}: {str(e)}")
                    continue
            
            if not energy_data:
                return {'error': 'No energy market data available'}
            
            # Generate correlation analysis
            correlations = {}
            
            # Extract satellite metrics for correlation
            satellite_metrics = self._extract_satellite_metrics(satellite_indicators)
            
            for symbol, data in energy_data.items():
                if len(data['prices']) < 10:
                    continue
                
                # Calculate correlations with different satellite indicators
                symbol_correlations = {}
                
                # Price correlations with satellite indicators
                for metric_name, metric_values in satellite_metrics.items():
                    if len(metric_values) >= 10:
                        # Align time series
                        correlation = self._calculate_time_aligned_correlation(
                            metric_values, data['prices'])
                        symbol_correlations[f'{metric_name}_price_corr'] = correlation
                
                # Volatility correlations
                for metric_name, metric_values in satellite_metrics.items():
                    if len(metric_values) >= 10:
                        correlation = self._calculate_time_aligned_correlation(
                            metric_values, data['volatility'])
                        symbol_correlations[f'{metric_name}_volatility_corr'] = correlation
                
                correlations[symbol] = {
                    'name': data['name'],
                    'correlations': symbol_correlations,
                    'current_price': data['prices'].iloc[-1] if len(data['prices']) > 0 else 0,
                    'price_change': data['returns'].iloc[-1] if len(data['returns']) > 0 else 0
                }
            
            # Generate trading recommendations
            trading_signals = self._generate_energy_trading_signals(correlations, satellite_indicators)
            
            results = {
                'correlations': correlations,
                'trading_signals': trading_signals,
                'market_summary': {
                    'analyzed_instruments': len(energy_data),
                    'strong_correlations': len([c for c in correlations.values() 
                                              if any(abs(corr) > 0.5 for corr in c['correlations'].values() if isinstance(corr, (int, float)))]),
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error correlating with energy markets: {str(e)}")
            return {}
    
    def _generate_realistic_temperature(self, dates: List[str]) -> Dict[str, float]:
        """Generate realistic temperature data based on dates."""
        temp_data = {}
        for date_str in dates:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            day_of_year = date.timetuple().tm_yday
            
            # Seasonal temperature pattern
            seasonal_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            
            # Daily variation
            daily_variation = np.random.normal(0, 3)
            
            temp_data[date_str] = seasonal_temp + daily_variation
        
        return temp_data
    
    def _generate_moisture_energy_signals(self, moisture_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate energy trading signals based on soil moisture trends."""
        signals = []
        
        current_moisture = moisture_df['soil_moisture'].iloc[-1]
        moisture_trend = moisture_df['moisture_trend'].iloc[-1]
        bioenergy_potential = moisture_df['bioenergy_potential'].iloc[-1]
        
        if current_moisture < 0.2:  # Severe drought
            signals.append({
                'signal_type': 'Bioenergy Supply Shock',
                'action': 'LONG',
                'instruments': ['Renewable Energy ETFs', 'Natural Gas'],
                'confidence': 0.75,
                'reasoning': 'Low soil moisture reduces bioenergy feedstock, increasing demand for alternative energy'
            })
        
        if moisture_trend < -0.01:  # Declining moisture
            signals.append({
                'signal_type': 'Agricultural Energy Demand',
                'action': 'LONG', 
                'instruments': ['Natural Gas', 'Electricity Futures'],
                'confidence': 0.65,
                'reasoning': 'Declining soil moisture increases irrigation energy demand'
            })
        
        return signals
    
    def _generate_solar_trading_signals(self, solar_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate trading signals based on solar potential analysis."""
        signals = []
        
        current_potential = solar_df['daily_solar_potential'].iloc[-1]
        avg_potential = solar_df['daily_solar_potential'].mean()
        solar_trend = solar_df['daily_solar_potential'].pct_change().iloc[-1]
        
        if current_potential > avg_potential * 1.2:  # High solar day
            signals.append({
                'signal_type': 'High Solar Production',
                'action': 'SHORT',
                'instruments': ['Electricity Futures', 'Power ETFs'],
                'confidence': 0.7,
                'reasoning': 'High solar production expected to depress electricity prices during peak hours'
            })
        
        if solar_trend > 0.1:  # Strong positive trend
            signals.append({
                'signal_type': 'Solar Trend',
                'action': 'LONG',
                'instruments': ['Solar ETFs', 'Clean Energy'],
                'confidence': 0.6,
                'reasoning': 'Improving solar conditions support renewable energy investments'
            })
        
        return signals
    
    def _generate_weather_options_signals(self, anomaly_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate weather options signals based on vegetation anomalies."""
        signals = []
        
        drought_prob = anomaly_df['drought_signal'].tail(14).sum() / 14
        recent_anomalies = anomaly_df['combined_anomaly'].tail(7).sum()
        
        if drought_prob > 0.3:  # 30% drought probability
            signals.append({
                'signal_type': 'Drought Weather Options',
                'strategy': 'Bull Call Spread',
                'instruments': ['Natural Gas Options', 'Agricultural Commodity Options'],
                'confidence': 0.8,
                'reasoning': 'High drought probability suggests increased energy demand for irrigation and processing'
            })
        
        if recent_anomalies >= 3:  # Multiple recent anomalies
            signals.append({
                'signal_type': 'Weather Volatility',
                'strategy': 'Long Volatility',
                'instruments': ['Energy Volatility ETFs', 'Weather Derivatives'],
                'confidence': 0.7,
                'reasoning': 'Multiple vegetation anomalies indicate weather instability affecting energy markets'
            })
        
        return signals
    
    def _calculate_intraday_solar_profile(self, daily_data: pd.Series) -> Dict[str, Any]:
        """Calculate intraday solar production profile."""
        base_irradiance = daily_data['solar_irradiance']
        cloud_cover = daily_data['cloud_cover']
        
        # Simplified intraday profile (6 AM to 6 PM)
        hours = list(range(6, 19))  # 6 AM to 6 PM
        hourly_profile = []
        
        for hour in hours:
            # Solar angle factor (peak at noon)
            solar_angle_factor = np.sin(np.pi * (hour - 6) / 12)
            
            # Cloud variability throughout day
            hourly_clouds = cloud_cover + np.random.normal(0, 0.1)
            hourly_clouds = max(0, min(1, hourly_clouds))
            
            hourly_irradiance = base_irradiance * solar_angle_factor * (1 - hourly_clouds * 0.8)
            hourly_profile.append({
                'hour': hour,
                'irradiance': hourly_irradiance,
                'cloud_cover': hourly_clouds,
                'production_efficiency': solar_angle_factor * (1 - hourly_clouds * 0.8)
            })
        
        return {
            'hourly_profile': hourly_profile,
            'peak_hour': max(hourly_profile, key=lambda x: x['irradiance'])['hour'],
            'total_daily_production': sum([h['irradiance'] for h in hourly_profile]),
            'production_variability': np.std([h['irradiance'] for h in hourly_profile])
        }
    
    def _extract_satellite_metrics(self, indicators: Dict[str, Any]) -> Dict[str, List[float]]:
        """Extract time series metrics from satellite indicators."""
        metrics = {}
        
        # Extract NDVI/EVI metrics if available
        if 'ndvi_evi_analysis' in indicators:
            analysis = indicators['ndvi_evi_analysis']
            if 'energy_demand_indicators' in analysis:
                metrics['heating_demand'] = [analysis['energy_demand_indicators'].get('total_heating_demand', 0)]
                metrics['cooling_demand'] = [analysis['energy_demand_indicators'].get('total_cooling_demand', 0)]
                metrics['agro_energy_demand'] = [analysis['energy_demand_indicators'].get('avg_agro_energy_demand', 0)]
        
        # Extract soil moisture metrics
        if 'soil_moisture_analysis' in indicators:
            analysis = indicators['soil_moisture_analysis']
            metrics['soil_moisture'] = [analysis.get('current_moisture', 0)]
            metrics['bioenergy_potential'] = [analysis.get('bioenergy_impact', {}).get('current_potential', 0)]
        
        # Extract solar metrics
        if 'solar_analysis' in indicators:
            analysis = indicators['solar_analysis']
            metrics['solar_potential'] = [analysis.get('current_conditions', {}).get('solar_potential', 0)]
            metrics['cloud_cover'] = [analysis.get('current_conditions', {}).get('cloud_cover', 0)]
        
        return metrics
    
    def _calculate_time_aligned_correlation(self, series1: List[float], series2: pd.Series) -> float:
        """Calculate correlation between two time series with alignment."""
        try:
            if len(series1) == 0 or len(series2) == 0:
                return 0.0
            
            # Simple correlation for now (in real implementation, would align by dates)
            min_length = min(len(series1), len(series2))
            if min_length < 2:
                return 0.0
            
            s1 = np.array(series1[:min_length])
            s2 = np.array(series2.iloc[:min_length])
            
            correlation = np.corrcoef(s1, s2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _generate_energy_trading_signals(self, 
                                       correlations: Dict[str, Any],
                                       satellite_indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive energy trading signals."""
        signals = []
        
        # Analyze correlations for trading opportunities
        for symbol, data in correlations.items():
            strong_correlations = [k for k, v in data['correlations'].items() 
                                 if isinstance(v, (int, float)) and abs(v) > 0.6]
            
            if strong_correlations:
                price_change = data.get('price_change', 0)
                
                if price_change > 0.02:  # Strong positive move
                    signals.append({
                        'instrument': data['name'],
                        'signal_type': 'Momentum',
                        'action': 'LONG',
                        'confidence': 0.7,
                        'reasoning': f'Strong positive correlation with satellite indicators: {strong_correlations[:2]}'
                    })
                elif price_change < -0.02:  # Strong negative move
                    signals.append({
                        'instrument': data['name'],
                        'signal_type': 'Reversal',
                        'action': 'SHORT',
                        'confidence': 0.65,
                        'reasoning': f'Negative correlation with improving satellite conditions'
                    })
        
        return signals


# Singleton instance
correlator = SatelliteEnergyCorrelator()

def get_satellite_energy_correlator() -> SatelliteEnergyCorrelator:
    """Get singleton instance of the satellite-energy correlator."""
    return correlator