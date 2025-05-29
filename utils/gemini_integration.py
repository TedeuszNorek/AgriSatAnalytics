"""
Google Gemini AI Integration for Agricultural Analytics
Enhanced satellite data analysis and market intelligence using Gemini
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import google.generativeai as genai

logger = logging.getLogger(__name__)

class GeminiAnalyzer:
    """
    Google Gemini AI integration for advanced agricultural analysis.
    """
    
    def __init__(self):
        """Initialize Gemini AI with API key."""
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel('gemini-pro')
        
        logger.info("Gemini AI analyzer initialized successfully")
    
    def analyze_satellite_data(self, 
                             field_data: Dict[str, Any],
                             ndvi_data: Dict[str, float],
                             weather_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze satellite data using Gemini AI for advanced insights.
        
        Args:
            field_data: Field information including location and crop type
            ndvi_data: NDVI time series data
            weather_data: Weather information (optional)
            
        Returns:
            AI-generated analysis and recommendations
        """
        try:
            # Prepare context for Gemini
            context = self._prepare_satellite_context(field_data, ndvi_data, weather_data)
            
            prompt = f"""
            As an expert agricultural analyst with deep knowledge of satellite remote sensing and precision agriculture, 
            analyze the following satellite data and provide comprehensive insights:

            {context}

            Please provide:
            1. Detailed analysis of the NDVI trends and what they indicate about crop health
            2. Identification of any concerning patterns or anomalies
            3. Specific agricultural recommendations for the farmer
            4. Risk assessment for potential yield impacts
            5. Optimal timing recommendations for agricultural operations
            
            Focus on actionable insights that can improve farm productivity and profitability.
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                'analysis_type': 'satellite_data_analysis',
                'ai_insights': response.text,
                'field_name': field_data.get('name', 'Unknown'),
                'analysis_timestamp': datetime.now().isoformat(),
                'data_quality': self._assess_data_quality(ndvi_data),
                'confidence_level': 'high' if len(ndvi_data) > 10 else 'medium'
            }
            
        except Exception as e:
            logger.error(f"Error in Gemini satellite analysis: {str(e)}")
            return {'error': str(e)}
    
    def generate_market_intelligence(self,
                                   crop_type: str,
                                   satellite_indicators: Dict[str, Any],
                                   market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate market intelligence using Gemini AI.
        
        Args:
            crop_type: Type of crop being analyzed
            satellite_indicators: Satellite-derived indicators
            market_data: Current market conditions (optional)
            
        Returns:
            AI-generated market intelligence and trading insights
        """
        try:
            context = self._prepare_market_context(crop_type, satellite_indicators, market_data)
            
            prompt = f"""
            As an expert agricultural commodity analyst with expertise in satellite data and market dynamics,
            provide comprehensive market intelligence based on the following information:

            {context}

            Please provide:
            1. Market impact analysis of the satellite indicators
            2. Price trend predictions for {crop_type} 
            3. Supply and demand implications
            4. Trading recommendations and optimal timing
            5. Risk factors and hedging strategies
            6. Correlation analysis with related commodities
            
            Focus on actionable trading insights and market positioning strategies.
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                'analysis_type': 'market_intelligence',
                'ai_insights': response.text,
                'crop_type': crop_type,
                'analysis_timestamp': datetime.now().isoformat(),
                'market_conditions': market_data.get('current_conditions', 'unknown') if market_data else 'unknown',
                'confidence_level': 'high'
            }
            
        except Exception as e:
            logger.error(f"Error in Gemini market analysis: {str(e)}")
            return {'error': str(e)}
    
    def optimize_strategy_parameters(self,
                                   strategy_results: Dict[str, Any],
                                   performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use Gemini AI to optimize trading strategy parameters.
        
        Args:
            strategy_results: Results from strategy backtesting
            performance_metrics: Performance metrics and risk analysis
            
        Returns:
            AI-generated optimization recommendations
        """
        try:
            context = self._prepare_strategy_context(strategy_results, performance_metrics)
            
            prompt = f"""
            As an expert quantitative analyst specializing in agricultural commodity trading strategies,
            analyze the following strategy performance and provide optimization recommendations:

            {context}

            Please provide:
            1. Analysis of current strategy performance strengths and weaknesses
            2. Specific parameter optimization suggestions
            3. Risk management improvements
            4. Portfolio allocation recommendations
            5. Alternative strategy variations to consider
            6. Market regime considerations and adaptations
            
            Focus on practical improvements that can enhance risk-adjusted returns.
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                'analysis_type': 'strategy_optimization',
                'ai_recommendations': response.text,
                'strategy_name': strategy_results.get('strategy_name', 'Unknown'),
                'analysis_timestamp': datetime.now().isoformat(),
                'performance_summary': {
                    'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                    'max_drawdown': performance_metrics.get('max_drawdown', 0),
                    'total_return': performance_metrics.get('total_return', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Gemini strategy optimization: {str(e)}")
            return {'error': str(e)}
    
    def generate_comprehensive_report(self,
                                    field_data: Dict[str, Any],
                                    satellite_analysis: Dict[str, Any],
                                    market_analysis: Dict[str, Any],
                                    strategy_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive AI-powered report combining all analyses.
        
        Args:
            field_data: Field information
            satellite_analysis: Satellite data analysis results
            market_analysis: Market intelligence results
            strategy_results: Trading strategy results (optional)
            
        Returns:
            Comprehensive AI-generated report
        """
        try:
            context = self._prepare_comprehensive_context(
                field_data, satellite_analysis, market_analysis, strategy_results
            )
            
            prompt = f"""
            As a senior agricultural advisor and investment strategist, create a comprehensive executive report
            that synthesizes satellite data, market intelligence, and trading strategy analysis:

            {context}

            Please create a structured report with:
            1. Executive Summary (key findings and recommendations)
            2. Agricultural Analysis (crop health, yield prospects, operational recommendations)
            3. Market Intelligence (price outlook, supply/demand dynamics, trading opportunities)
            4. Risk Assessment (production risks, market risks, mitigation strategies)
            5. Strategic Recommendations (immediate actions, medium-term planning, long-term strategy)
            6. Performance Metrics (if strategy data available)
            
            Write in a professional tone suitable for farm management and investment decision-making.
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                'report_type': 'comprehensive_analysis',
                'ai_report': response.text,
                'field_name': field_data.get('name', 'Unknown'),
                'report_timestamp': datetime.now().isoformat(),
                'data_sources': ['satellite_data', 'market_data', 'ai_analysis'],
                'confidence_level': 'high'
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            return {'error': str(e)}
    
    def _prepare_satellite_context(self, 
                                 field_data: Dict[str, Any],
                                 ndvi_data: Dict[str, float],
                                 weather_data: Dict[str, Any] = None) -> str:
        """Prepare context for satellite data analysis."""
        context = f"""
        FIELD INFORMATION:
        - Field Name: {field_data.get('name', 'Unknown')}
        - Crop Type: {field_data.get('crop_type', 'Unknown')}
        - Location: {field_data.get('center_lat', 'Unknown')}, {field_data.get('center_lon', 'Unknown')}
        - Area: {field_data.get('area_hectares', 'Unknown')} hectares
        
        NDVI DATA (Recent 30 days):
        """
        
        # Add NDVI data points
        if ndvi_data:
            sorted_dates = sorted(ndvi_data.keys())[-30:]  # Last 30 data points
            for date in sorted_dates:
                context += f"- {date}: {ndvi_data[date]:.3f}\n"
        
        # Add weather context if available
        if weather_data:
            context += f"""
        WEATHER CONDITIONS:
        - Average Temperature: {weather_data.get('avg_temperature', 'Unknown')}°C
        - Total Precipitation: {weather_data.get('total_precipitation', 'Unknown')}mm
        - Average Humidity: {weather_data.get('avg_humidity', 'Unknown')}%
        """
        
        return context
    
    def _prepare_market_context(self,
                              crop_type: str,
                              satellite_indicators: Dict[str, Any],
                              market_data: Dict[str, Any] = None) -> str:
        """Prepare context for market intelligence."""
        context = f"""
        CROP INFORMATION:
        - Crop Type: {crop_type}
        
        SATELLITE INDICATORS:
        """
        
        # Add satellite indicators
        for key, value in satellite_indicators.items():
            if isinstance(value, dict):
                context += f"- {key.replace('_', ' ').title()}:\n"
                for subkey, subvalue in value.items():
                    context += f"  - {subkey.replace('_', ' ').title()}: {subvalue}\n"
            else:
                context += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        # Add market data if available
        if market_data:
            context += f"""
        CURRENT MARKET CONDITIONS:
        - Current Price: {market_data.get('current_price', 'Unknown')}
        - Price Change: {market_data.get('price_change', 'Unknown')}
        - Market Trend: {market_data.get('trend', 'Unknown')}
        """
        
        return context
    
    def _prepare_strategy_context(self,
                                strategy_results: Dict[str, Any],
                                performance_metrics: Dict[str, Any]) -> str:
        """Prepare context for strategy optimization."""
        context = f"""
        STRATEGY INFORMATION:
        - Strategy Name: {strategy_results.get('strategy_name', 'Unknown')}
        - Strategy Type: {strategy_results.get('strategy_type', 'Unknown')}
        - Simulation Period: {strategy_results.get('simulation_period', {}).get('duration_days', 'Unknown')} days
        
        PERFORMANCE METRICS:
        - Total Return: {performance_metrics.get('total_return', 0):.2%}
        - Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.3f}
        - Maximum Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}
        - Win Rate: {performance_metrics.get('win_rate', 0):.1%}
        - Volatility: {performance_metrics.get('volatility', 0):.2%}
        - Number of Trades: {performance_metrics.get('num_trades', 0)}
        
        RISK METRICS:
        - Value at Risk (95%): {performance_metrics.get('value_at_risk_95', 0):.2%}
        - Value at Risk (99%): {performance_metrics.get('value_at_risk_99', 0):.2%}
        - Expected Shortfall: {performance_metrics.get('expected_shortfall_95', 0):.2%}
        """
        
        return context
    
    def _prepare_comprehensive_context(self,
                                     field_data: Dict[str, Any],
                                     satellite_analysis: Dict[str, Any],
                                     market_analysis: Dict[str, Any],
                                     strategy_results: Dict[str, Any] = None) -> str:
        """Prepare context for comprehensive report."""
        context = f"""
        FIELD OVERVIEW:
        - Field: {field_data.get('name', 'Unknown')}
        - Crop: {field_data.get('crop_type', 'Unknown')}
        - Size: {field_data.get('area_hectares', 'Unknown')} hectares
        
        SATELLITE ANALYSIS SUMMARY:
        """
        
        # Add satellite analysis insights
        if satellite_analysis.get('ai_insights'):
            context += f"- AI Analysis: {satellite_analysis['ai_insights'][:500]}...\n"
        
        # Add market analysis insights
        context += "\nMARKET INTELLIGENCE SUMMARY:\n"
        if market_analysis.get('ai_insights'):
            context += f"- Market Analysis: {market_analysis['ai_insights'][:500]}...\n"
        
        # Add strategy results if available
        if strategy_results:
            context += f"""
        STRATEGY PERFORMANCE:
        - Strategy: {strategy_results.get('strategy_name', 'Unknown')}
        - Performance: {strategy_results.get('performance_metrics', {}).get('total_return', 0):.2%} return
        - Risk Level: {strategy_results.get('risk_analysis', {}).get('maximum_drawdown', 0):.2%} max drawdown
        """
        
        return context
    
    def _assess_data_quality(self, ndvi_data: Dict[str, float]) -> str:
        """Assess the quality of NDVI data."""
        if not ndvi_data:
            return 'poor'
        elif len(ndvi_data) < 5:
            return 'limited'
        elif len(ndvi_data) < 15:
            return 'good'
        else:
            return 'excellent'


# Singleton instance
gemini_analyzer = GeminiAnalyzer()

def get_gemini_analyzer() -> GeminiAnalyzer:
    """Get singleton instance of Gemini analyzer."""
    return gemini_analyzer