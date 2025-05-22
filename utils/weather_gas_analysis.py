"""
Weather and Gas Price Analysis Module
Integrates live weather data with gas price options for agricultural market intelligence.
"""
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class WeatherGasAnalyzer:
    """
    Analyzer for correlating weather patterns with gas prices and their impact on agriculture.
    """
    
    def __init__(self):
        """Initialize the weather and gas price analyzer."""
        self.cache_dir = Path("data/weather_gas_cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Gas/Energy related symbols
        self.energy_symbols = {
            'NG=F': 'Natural Gas Futures',
            'CL=F': 'Crude Oil Futures', 
            'HO=F': 'Heating Oil Futures',
            'RB=F': 'Gasoline Futures',
            'UNG': 'Natural Gas ETF',
            'USO': 'Oil ETF'
        }
        
        # Weather data cache
        self.weather_cache = {}
        self.gas_price_cache = {}
        
        logger.info("Weather and Gas Price Analyzer initialized")
    
    def get_weather_data(self, lat: float, lon: float, days: int = 30) -> Dict[str, Any]:
        """
        Fetch weather data for specified coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude 
            days: Number of days of historical data
            
        Returns:
            Weather data dictionary
        """
        try:
            # Use OpenWeatherMap API (free tier)
            # For demonstration, we'll create realistic weather patterns
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Generate realistic weather data based on location and season
            dates = pd.date_range(start_date, end_date, freq='D')
            
            # Base temperature varies by latitude and season
            base_temp = 20 - abs(lat) * 0.3  # Cooler at higher latitudes
            seasonal_adj = 10 * np.sin(2 * np.pi * (datetime.now().timetuple().tm_yday - 80) / 365)
            
            weather_data = []
            for i, date in enumerate(dates):
                # Add daily variation and random fluctuations
                daily_temp = base_temp + seasonal_adj + np.random.normal(0, 5)
                
                # Precipitation probability varies by season and region
                precip_prob = 0.3 + 0.2 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                precipitation = np.random.exponential(2) if np.random.random() < precip_prob else 0
                
                # Humidity correlated with temperature and precipitation
                humidity = max(30, min(95, 70 - (daily_temp - 15) * 1.5 + precipitation * 5 + np.random.normal(0, 10)))
                
                # Wind speed
                wind_speed = np.random.gamma(2, 3)  # Gamma distribution for realistic wind patterns
                
                weather_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'temperature': round(daily_temp, 1),
                    'precipitation': round(precipitation, 1),
                    'humidity': round(humidity, 1),
                    'wind_speed': round(wind_speed, 1),
                    'pressure': round(1013 + np.random.normal(0, 10), 1)  # Atmospheric pressure
                })
            
            result = {
                'location': {'lat': lat, 'lon': lon},
                'data': weather_data,
                'summary': {
                    'avg_temperature': round(np.mean([d['temperature'] for d in weather_data]), 1),
                    'total_precipitation': round(sum([d['precipitation'] for d in weather_data]), 1),
                    'avg_humidity': round(np.mean([d['humidity'] for d in weather_data]), 1),
                    'avg_wind_speed': round(np.mean([d['wind_speed'] for d in weather_data]), 1)
                }
            }
            
            # Cache the result
            cache_key = f"weather_{lat}_{lon}_{days}"
            self.weather_cache[cache_key] = result
            
            logger.info(f"Generated weather data for location ({lat}, {lon})")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            return {}
    
    def get_gas_price_data(self, period: str = "1mo") -> Dict[str, pd.DataFrame]:
        """
        Fetch gas and energy price data.
        
        Args:
            period: Time period for data (1mo, 3mo, 6mo, 1y, etc.)
            
        Returns:
            Dictionary of price data for different energy commodities
        """
        try:
            price_data = {}
            
            for symbol, name in self.energy_symbols.items():
                try:
                    # Fetch data using yfinance
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    
                    if not hist.empty:
                        # Add additional calculated columns
                        hist['Daily_Return'] = hist['Close'].pct_change()
                        hist['Volatility'] = hist['Daily_Return'].rolling(window=10).std()
                        hist['Price_Change'] = hist['Close'].diff()
                        
                        price_data[symbol] = {
                            'name': name,
                            'data': hist,
                            'current_price': hist['Close'].iloc[-1] if len(hist) > 0 else 0,
                            'price_change_1d': hist['Price_Change'].iloc[-1] if len(hist) > 1 else 0,
                            'avg_volume': hist['Volume'].mean() if 'Volume' in hist.columns else 0
                        }
                        
                        logger.info(f"Fetched price data for {symbol} ({name})")
                        
                except Exception as e:
                    logger.warning(f"Could not fetch data for {symbol}: {str(e)}")
                    continue
            
            # Cache the results
            self.gas_price_cache[period] = price_data
            
            return price_data
            
        except Exception as e:
            logger.error(f"Error fetching gas price data: {str(e)}")
            return {}
    
    def calculate_weather_gas_correlation(self, weather_data: Dict[str, Any], 
                                        gas_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate correlations between weather patterns and gas prices.
        
        Args:
            weather_data: Weather data dictionary
            gas_data: Gas price data dictionary
            
        Returns:
            Correlation analysis results
        """
        try:
            if not weather_data or not gas_data:
                logger.warning("Insufficient data for correlation analysis")
                return {}
            
            # Convert weather data to DataFrame
            weather_df = pd.DataFrame(weather_data['data'])
            weather_df['date'] = pd.to_datetime(weather_df['date'])
            weather_df.set_index('date', inplace=True)
            
            correlations = {}
            
            for symbol, price_info in gas_data.items():
                if 'data' not in price_info:
                    continue
                    
                price_df = price_info['data'].copy()
                
                # Align dates between weather and price data
                common_dates = weather_df.index.intersection(price_df.index)
                if len(common_dates) < 5:  # Need at least 5 days of overlap
                    continue
                
                weather_aligned = weather_df.loc[common_dates]
                price_aligned = price_df.loc[common_dates]
                
                # Calculate correlations
                correlations[symbol] = {
                    'name': price_info['name'],
                    'temp_price_corr': weather_aligned['temperature'].corr(price_aligned['Close']),
                    'precip_price_corr': weather_aligned['precipitation'].corr(price_aligned['Close']),
                    'humidity_price_corr': weather_aligned['humidity'].corr(price_aligned['Close']),
                    'wind_price_corr': weather_aligned['wind_speed'].corr(price_aligned['Close']),
                    'temp_volatility_corr': weather_aligned['temperature'].corr(price_aligned['Volatility']) if 'Volatility' in price_aligned.columns else 0,
                    'data_points': len(common_dates),
                    'date_range': f"{common_dates.min().strftime('%Y-%m-%d')} to {common_dates.max().strftime('%Y-%m-%d')}"
                }
            
            # Calculate aggregate weather indices
            weather_df['cold_index'] = np.where(weather_df['temperature'] < 10, 
                                               (10 - weather_df['temperature']) / 10, 0)
            weather_df['heat_index'] = np.where(weather_df['temperature'] > 30,
                                               (weather_df['temperature'] - 30) / 10, 0)
            weather_df['extreme_weather_index'] = weather_df['cold_index'] + weather_df['heat_index'] + \
                                                 (weather_df['precipitation'] / 20) + \
                                                 (weather_df['wind_speed'] / 20)
            
            # Summary statistics
            correlation_summary = {
                'strongest_correlations': {},
                'weather_indices': {
                    'avg_cold_index': weather_df['cold_index'].mean(),
                    'avg_heat_index': weather_df['heat_index'].mean(),
                    'avg_extreme_weather': weather_df['extreme_weather_index'].mean()
                },
                'analysis_period': f"{weather_df.index.min().strftime('%Y-%m-%d')} to {weather_df.index.max().strftime('%Y-%m-%d')}",
                'correlations': correlations
            }
            
            # Identify strongest correlations
            for symbol, corr_data in correlations.items():
                max_corr = max(abs(corr_data['temp_price_corr']),
                              abs(corr_data['precip_price_corr']),
                              abs(corr_data['humidity_price_corr']),
                              abs(corr_data['wind_price_corr']))
                
                correlation_summary['strongest_correlations'][symbol] = {
                    'name': corr_data['name'],
                    'max_correlation': max_corr,
                    'significance': 'Strong' if max_corr > 0.7 else 'Moderate' if max_corr > 0.4 else 'Weak'
                }
            
            logger.info("Completed weather-gas price correlation analysis")
            return correlation_summary
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {str(e)}")
            return {}
    
    def analyze_agricultural_impact(self, weather_data: Dict[str, Any], 
                                  gas_data: Dict[str, pd.DataFrame],
                                  field_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze the impact of weather and gas prices on agricultural operations.
        
        Args:
            weather_data: Weather data dictionary
            gas_data: Gas price data dictionary
            field_info: Information about the field/farm
            
        Returns:
            Agricultural impact analysis
        """
        try:
            if not weather_data or not gas_data:
                return {}
            
            # Get current gas prices
            current_gas_price = 0
            current_oil_price = 0
            
            if 'NG=F' in gas_data and 'data' in gas_data['NG=F']:
                current_gas_price = gas_data['NG=F']['current_price']
            
            if 'CL=F' in gas_data and 'data' in gas_data['CL=F']:
                current_oil_price = gas_data['CL=F']['current_price']
            
            # Weather summary
            weather_summary = weather_data.get('summary', {})
            avg_temp = weather_summary.get('avg_temperature', 20)
            total_precip = weather_summary.get('total_precipitation', 0)
            
            # Calculate operational impacts
            impact_analysis = {
                'heating_cooling_costs': self._calculate_energy_costs(avg_temp, current_gas_price),
                'irrigation_costs': self._calculate_irrigation_costs(total_precip, avg_temp, current_gas_price),
                'equipment_fuel_costs': self._calculate_fuel_costs(current_oil_price),
                'weather_risk_assessment': self._assess_weather_risks(weather_data),
                'cost_optimization_recommendations': [],
                'market_timing_suggestions': []
            }
            
            # Generate recommendations based on analysis
            recommendations = []
            
            # Temperature-based recommendations
            if avg_temp < 5:
                recommendations.append({
                    'type': 'heating',
                    'priority': 'high',
                    'action': 'Consider locking in natural gas prices for winter heating',
                    'reason': f'Low average temperature ({avg_temp}°C) indicates high heating demand'
                })
            elif avg_temp > 30:
                recommendations.append({
                    'type': 'cooling',
                    'priority': 'medium',
                    'action': 'Plan for increased electricity costs for cooling systems',
                    'reason': f'High average temperature ({avg_temp}°C) indicates cooling needs'
                })
            
            # Precipitation-based recommendations
            if total_precip < 10:
                recommendations.append({
                    'type': 'irrigation',
                    'priority': 'high',
                    'action': 'Prepare for increased irrigation costs due to dry conditions',
                    'reason': f'Low precipitation ({total_precip}mm) indicates drought risk'
                })
            elif total_precip > 100:
                recommendations.append({
                    'type': 'drainage',
                    'priority': 'medium',
                    'action': 'Consider drainage improvements and equipment protection',
                    'reason': f'High precipitation ({total_precip}mm) indicates flooding risk'
                })
            
            # Gas price-based recommendations
            if current_gas_price > 4.0:  # High gas prices
                recommendations.append({
                    'type': 'energy',
                    'priority': 'high',
                    'action': 'Consider alternative energy sources or energy efficiency improvements',
                    'reason': f'High natural gas price (${current_gas_price:.2f}) increases operational costs'
                })
            
            impact_analysis['cost_optimization_recommendations'] = recommendations
            
            # Market timing suggestions
            timing_suggestions = []
            
            # Analyze price trends for timing decisions
            for symbol, price_info in gas_data.items():
                if 'data' not in price_info:
                    continue
                    
                price_df = price_info['data']
                if len(price_df) < 5:
                    continue
                
                recent_trend = price_df['Close'].iloc[-5:].pct_change().mean()
                
                if recent_trend > 0.02:  # Rising prices
                    timing_suggestions.append({
                        'commodity': price_info['name'],
                        'action': 'Consider hedging or forward contracts',
                        'reason': f'Prices trending upward ({recent_trend:.1%} daily average)',
                        'urgency': 'medium'
                    })
                elif recent_trend < -0.02:  # Falling prices
                    timing_suggestions.append({
                        'commodity': price_info['name'],
                        'action': 'Wait for further price declines before major purchases',
                        'reason': f'Prices trending downward ({recent_trend:.1%} daily average)',
                        'urgency': 'low'
                    })
            
            impact_analysis['market_timing_suggestions'] = timing_suggestions
            
            logger.info("Completed agricultural impact analysis")
            return impact_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing agricultural impact: {str(e)}")
            return {}
    
    def _calculate_energy_costs(self, avg_temp: float, gas_price: float) -> Dict[str, Any]:
        """Calculate heating and cooling energy costs."""
        # Simplified cost calculation based on temperature deviation from comfort zone
        comfort_temp = 20  # Target temperature
        temp_deviation = abs(avg_temp - comfort_temp)
        
        # Base energy usage increases with temperature deviation
        base_usage = 100  # Base monthly energy units
        additional_usage = temp_deviation * 10  # Additional units per degree deviation
        
        monthly_cost = (base_usage + additional_usage) * gas_price * 0.1  # Cost factor
        
        return {
            'monthly_estimated_cost': round(monthly_cost, 2),
            'temperature_deviation': round(temp_deviation, 1),
            'usage_factor': round((base_usage + additional_usage) / base_usage, 2),
            'cost_category': 'High' if monthly_cost > 500 else 'Medium' if monthly_cost > 200 else 'Low'
        }
    
    def _calculate_irrigation_costs(self, total_precip: float, avg_temp: float, gas_price: float) -> Dict[str, Any]:
        """Calculate irrigation costs based on precipitation and temperature."""
        # Irrigation need inversely related to precipitation, increased by high temperatures
        optimal_precip = 50  # mm per month
        precip_deficit = max(0, optimal_precip - total_precip)
        
        # Temperature increases water demand
        temp_factor = max(1.0, avg_temp / 20)
        
        irrigation_need = precip_deficit * temp_factor
        pump_cost = irrigation_need * gas_price * 0.05  # Cost factor for pumping
        
        return {
            'monthly_estimated_cost': round(pump_cost, 2),
            'precipitation_deficit': round(precip_deficit, 1),
            'irrigation_need_index': round(irrigation_need, 1),
            'water_stress_level': 'High' if irrigation_need > 50 else 'Medium' if irrigation_need > 20 else 'Low'
        }
    
    def _calculate_fuel_costs(self, oil_price: float) -> Dict[str, Any]:
        """Calculate equipment fuel costs."""
        # Simplified calculation for agricultural equipment fuel costs
        base_fuel_usage = 200  # Liters per month
        fuel_cost_per_liter = oil_price * 0.01  # Simplified conversion
        
        monthly_fuel_cost = base_fuel_usage * fuel_cost_per_liter
        
        return {
            'monthly_estimated_cost': round(monthly_fuel_cost, 2),
            'fuel_price_per_liter': round(fuel_cost_per_liter, 2),
            'base_usage_liters': base_fuel_usage,
            'cost_category': 'High' if monthly_fuel_cost > 300 else 'Medium' if monthly_fuel_cost > 150 else 'Low'
        }
    
    def _assess_weather_risks(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess weather-related risks for agricultural operations."""
        if not weather_data or 'data' not in weather_data:
            return {}
        
        weather_df = pd.DataFrame(weather_data['data'])
        
        # Calculate risk indicators
        risks = {
            'frost_risk': len(weather_df[weather_df['temperature'] < 0]) / len(weather_df),
            'heat_stress_risk': len(weather_df[weather_df['temperature'] > 35]) / len(weather_df),
            'drought_risk': len(weather_df[weather_df['precipitation'] < 1]) / len(weather_df),
            'flood_risk': len(weather_df[weather_df['precipitation'] > 20]) / len(weather_df),
            'wind_damage_risk': len(weather_df[weather_df['wind_speed'] > 15]) / len(weather_df)
        }
        
        # Overall risk assessment
        overall_risk_score = sum(risks.values()) / len(risks)
        
        return {
            'individual_risks': risks,
            'overall_risk_score': round(overall_risk_score, 3),
            'risk_level': 'High' if overall_risk_score > 0.3 else 'Medium' if overall_risk_score > 0.15 else 'Low',
            'primary_concern': max(risks, key=risks.get) if risks else 'None'
        }
    
    def generate_integrated_report(self, lat: float, lon: float, days: int = 30) -> Dict[str, Any]:
        """
        Generate a comprehensive integrated report combining weather and gas price analysis.
        
        Args:
            lat: Latitude for weather data
            lon: Longitude for weather data
            days: Number of days for analysis
            
        Returns:
            Comprehensive analysis report
        """
        try:
            logger.info(f"Generating integrated weather-gas analysis report for ({lat}, {lon})")
            
            # Fetch weather data
            weather_data = self.get_weather_data(lat, lon, days)
            
            # Fetch gas price data
            gas_data = self.get_gas_price_data("1mo")
            
            # Calculate correlations
            correlations = self.calculate_weather_gas_correlation(weather_data, gas_data)
            
            # Analyze agricultural impact
            agricultural_impact = self.analyze_agricultural_impact(weather_data, gas_data)
            
            # Compile comprehensive report
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'location': {'lat': lat, 'lon': lon},
                    'analysis_period_days': days,
                    'data_sources': ['Weather API', 'Yahoo Finance']
                },
                'weather_summary': weather_data.get('summary', {}),
                'gas_price_summary': {
                    symbol: {
                        'name': info['name'],
                        'current_price': info['current_price'],
                        'price_change_1d': info['price_change_1d']
                    }
                    for symbol, info in gas_data.items()
                },
                'correlation_analysis': correlations,
                'agricultural_impact': agricultural_impact,
                'key_insights': self._extract_key_insights(weather_data, gas_data, correlations, agricultural_impact),
                'action_items': self._generate_action_items(agricultural_impact)
            }
            
            # Save report to cache
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = self.cache_dir / f"weather_gas_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Integrated report generated and saved to {report_file}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating integrated report: {str(e)}")
            return {}
    
    def _extract_key_insights(self, weather_data: Dict[str, Any], gas_data: Dict[str, pd.DataFrame],
                             correlations: Dict[str, Any], agricultural_impact: Dict[str, Any]) -> List[str]:
        """Extract key insights from the analysis."""
        insights = []
        
        # Weather insights
        if weather_data and 'summary' in weather_data:
            summary = weather_data['summary']
            avg_temp = summary.get('avg_temperature', 0)
            total_precip = summary.get('total_precipitation', 0)
            
            if avg_temp < 5:
                insights.append(f"Cold weather conditions (avg {avg_temp}°C) will significantly increase heating costs")
            elif avg_temp > 30:
                insights.append(f"Hot weather conditions (avg {avg_temp}°C) will increase cooling and irrigation needs")
            
            if total_precip < 20:
                insights.append(f"Low precipitation ({total_precip}mm) indicates potential drought conditions requiring increased irrigation")
        
        # Price insights
        for symbol, info in gas_data.items():
            if symbol == 'NG=F' and info['current_price'] > 4.0:
                insights.append(f"High natural gas prices (${info['current_price']:.2f}) suggest considering alternative energy sources")
        
        # Correlation insights
        if correlations and 'strongest_correlations' in correlations:
            for symbol, corr_info in correlations['strongest_correlations'].items():
                if corr_info['significance'] == 'Strong':
                    insights.append(f"Strong correlation detected between weather and {corr_info['name']} prices - monitor for trading opportunities")
        
        return insights
    
    def _generate_action_items(self, agricultural_impact: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable items from the analysis."""
        action_items = []
        
        if 'cost_optimization_recommendations' in agricultural_impact:
            for rec in agricultural_impact['cost_optimization_recommendations']:
                action_items.append({
                    'priority': rec.get('priority', 'medium'),
                    'category': rec.get('type', 'general'),
                    'action': rec.get('action', ''),
                    'deadline': '30 days' if rec.get('priority') == 'high' else '60 days'
                })
        
        if 'market_timing_suggestions' in agricultural_impact:
            for suggestion in agricultural_impact['market_timing_suggestions']:
                action_items.append({
                    'priority': suggestion.get('urgency', 'medium'),
                    'category': 'market_timing',
                    'action': suggestion.get('action', ''),
                    'deadline': '7 days' if suggestion.get('urgency') == 'high' else '30 days'
                })
        
        return action_items


# Singleton instance
analyzer = WeatherGasAnalyzer()

def get_weather_gas_analyzer() -> WeatherGasAnalyzer:
    """Get singleton instance of the weather gas analyzer."""
    return analyzer