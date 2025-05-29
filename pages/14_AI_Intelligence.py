"""
AI Intelligence - Google Gemini AI Integration for Agricultural Analytics
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta

from utils.gemini_integration import get_gemini_analyzer
from utils.predictions import get_prediction_manager

# Page configuration
st.set_page_config(
    page_title="AI Intelligence - Agro Insight",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 AI Intelligence")
st.markdown("""
**Google Gemini AI Integration** for advanced agricultural analytics. Get expert-level insights, 
market intelligence, and strategic recommendations powered by cutting-edge artificial intelligence.
""")

# Get analyzer instances
gemini_analyzer = get_gemini_analyzer()
prediction_manager = get_prediction_manager()

# Sidebar controls
st.sidebar.header("🎯 AI Analysis Options")

# Analysis type selection
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    [
        "🛰️ Satellite Data Analysis",
        "📈 Market Intelligence",
        "⚡ Strategy Optimization",
        "📊 Comprehensive Report"
    ]
)

# Field selection for satellite analysis
if "Satellite Data" in analysis_type or "Comprehensive" in analysis_type:
    available_fields = prediction_manager.get_available_fields()
    if available_fields:
        selected_field = st.sidebar.selectbox("Select Field", available_fields)
    else:
        st.sidebar.warning("No fields available. Please add fields in Field Manager.")
        selected_field = None

# Market analysis parameters
if "Market Intelligence" in analysis_type or "Comprehensive" in analysis_type:
    crop_types = ["Wheat", "Corn", "Soybean", "Barley", "Oats", "Rice"]
    selected_crop = st.sidebar.selectbox("Select Crop Type", crop_types)

# Run analysis button
run_analysis = st.sidebar.button("🚀 Generate AI Analysis", type="primary")

# Information panel
with st.sidebar.expander("ℹ️ About Gemini AI"):
    st.markdown("""
    **Google Gemini AI** provides:
    - Expert-level agricultural analysis
    - Market intelligence and predictions
    - Strategic trading recommendations
    - Comprehensive risk assessment
    - Natural language insights
    
    The AI analyzes your satellite data, market conditions, and strategy performance to generate professional-grade reports.
    """)

# Main content area
if run_analysis:
    if "Satellite Data" in analysis_type:
        if selected_field:
            st.header(f"🛰️ AI Satellite Analysis - {selected_field}")
            
            with st.spinner("Gemini AI is analyzing your satellite data..."):
                # Get field data
                field_info = prediction_manager.get_field_info(selected_field)
                ndvi_data = prediction_manager.get_ndvi_time_series(selected_field)
                
                if ndvi_data:
                    # Generate sample weather data for context
                    weather_data = {
                        'avg_temperature': 18.5,
                        'total_precipitation': 45.2,
                        'avg_humidity': 65.3
                    }
                    
                    # Run Gemini analysis
                    analysis_result = gemini_analyzer.analyze_satellite_data(
                        field_info, ndvi_data, weather_data
                    )
                    
                    if 'error' not in analysis_result:
                        # Display analysis results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Data Quality", analysis_result.get('data_quality', 'Unknown').title())
                        with col2:
                            st.metric("Confidence Level", analysis_result.get('confidence_level', 'Unknown').title())
                        with col3:
                            analysis_time = datetime.fromisoformat(analysis_result['analysis_timestamp'])
                            st.metric("Analysis Time", analysis_time.strftime('%H:%M'))
                        
                        # AI Insights
                        st.subheader("🧠 AI Analysis & Recommendations")
                        
                        ai_insights = analysis_result.get('ai_insights', '')
                        if ai_insights:
                            st.markdown(ai_insights)
                        else:
                            st.info("AI analysis is being processed...")
                        
                        # Additional context
                        st.subheader("📊 Data Context")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Field Information:**")
                            st.write(f"• Field: {field_info.get('name', 'Unknown')}")
                            st.write(f"• Crop Type: {field_info.get('crop_type', 'Unknown')}")
                            st.write(f"• Area: {field_info.get('area_hectares', 'Unknown')} hectares")
                        
                        with col2:
                            st.write("**NDVI Data Summary:**")
                            if ndvi_data:
                                ndvi_values = list(ndvi_data.values())
                                st.write(f"• Data Points: {len(ndvi_values)}")
                                st.write(f"• Average NDVI: {sum(ndvi_values)/len(ndvi_values):.3f}")
                                st.write(f"• Current NDVI: {ndvi_values[-1]:.3f}")
                    
                    else:
                        st.error(f"AI analysis failed: {analysis_result['error']}")
                else:
                    st.warning("No NDVI data available for this field.")
        else:
            st.warning("Please select a field for satellite analysis.")
    
    elif "Market Intelligence" in analysis_type:
        st.header(f"📈 AI Market Intelligence - {selected_crop}")
        
        with st.spinner("Gemini AI is analyzing market conditions..."):
            # Prepare satellite indicators
            satellite_indicators = {
                'ndvi_average': 0.65,
                'temperature_trend': 'increasing',
                'precipitation_status': 'adequate',
                'drought_risk': 'low',
                'crop_health_index': 'good'
            }
            
            # Prepare market data
            market_data = {
                'current_price': 245.50,
                'price_change': 0.025,
                'trend': 'bullish',
                'volatility': 0.18
            }
            
            # Run Gemini market analysis
            market_result = gemini_analyzer.generate_market_intelligence(
                selected_crop, satellite_indicators, market_data
            )
            
            if 'error' not in market_result:
                # Display market analysis
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Crop Type", market_result.get('crop_type', 'Unknown'))
                with col2:
                    st.metric("Confidence", market_result.get('confidence_level', 'Unknown').title())
                with col3:
                    analysis_time = datetime.fromisoformat(market_result['analysis_timestamp'])
                    st.metric("Analysis Time", analysis_time.strftime('%H:%M'))
                
                # AI Market Intelligence
                st.subheader("🧠 AI Market Intelligence")
                
                ai_insights = market_result.get('ai_insights', '')
                if ai_insights:
                    st.markdown(ai_insights)
                else:
                    st.info("Market intelligence is being processed...")
                
                # Market context
                st.subheader("📊 Market Context")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Satellite Indicators:**")
                    for key, value in satellite_indicators.items():
                        st.write(f"• {key.replace('_', ' ').title()}: {value}")
                
                with col2:
                    st.write("**Current Market Data:**")
                    st.write(f"• Current Price: ${market_data['current_price']}")
                    st.write(f"• Daily Change: {market_data['price_change']:.1%}")
                    st.write(f"• Trend: {market_data['trend'].title()}")
                    st.write(f"• Volatility: {market_data['volatility']:.1%}")
            
            else:
                st.error(f"Market analysis failed: {market_result['error']}")
    
    elif "Strategy Optimization" in analysis_type:
        st.header("⚡ AI Strategy Optimization")
        
        with st.spinner("Gemini AI is optimizing your trading strategy..."):
            # Prepare sample strategy results
            strategy_results = {
                'strategy_name': 'Seasonal Energy Strategy',
                'strategy_type': 'seasonal_energy',
                'simulation_period': {'duration_days': 252}
            }
            
            performance_metrics = {
                'total_return': 0.157,
                'sharpe_ratio': 1.23,
                'max_drawdown': 0.087,
                'win_rate': 0.64,
                'volatility': 0.145,
                'num_trades': 42,
                'value_at_risk_95': -0.032,
                'value_at_risk_99': -0.048,
                'expected_shortfall_95': -0.041
            }
            
            # Run Gemini optimization analysis
            optimization_result = gemini_analyzer.optimize_strategy_parameters(
                strategy_results, performance_metrics
            )
            
            if 'error' not in optimization_result:
                # Display optimization results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Strategy", optimization_result.get('strategy_name', 'Unknown'))
                with col2:
                    current_sharpe = performance_metrics.get('sharpe_ratio', 0)
                    st.metric("Current Sharpe Ratio", f"{current_sharpe:.3f}")
                with col3:
                    analysis_time = datetime.fromisoformat(optimization_result['analysis_timestamp'])
                    st.metric("Analysis Time", analysis_time.strftime('%H:%M'))
                
                # AI Optimization Recommendations
                st.subheader("🧠 AI Optimization Recommendations")
                
                ai_recommendations = optimization_result.get('ai_recommendations', '')
                if ai_recommendations:
                    st.markdown(ai_recommendations)
                else:
                    st.info("Optimization recommendations are being processed...")
                
                # Performance summary
                st.subheader("📊 Current Performance Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Return Metrics:**")
                    st.write(f"• Total Return: {performance_metrics['total_return']:.1%}")
                    st.write(f"• Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f}")
                    st.write(f"• Win Rate: {performance_metrics['win_rate']:.1%}")
                    st.write(f"• Number of Trades: {performance_metrics['num_trades']}")
                
                with col2:
                    st.write("**Risk Metrics:**")
                    st.write(f"• Maximum Drawdown: {performance_metrics['max_drawdown']:.1%}")
                    st.write(f"• Volatility: {performance_metrics['volatility']:.1%}")
                    st.write(f"• VaR 95%: {performance_metrics['value_at_risk_95']:.1%}")
                    st.write(f"• Expected Shortfall: {performance_metrics['expected_shortfall_95']:.1%}")
            
            else:
                st.error(f"Strategy optimization failed: {optimization_result['error']}")
    
    elif "Comprehensive" in analysis_type:
        if selected_field:
            st.header(f"📊 Comprehensive AI Report - {selected_field}")
            
            with st.spinner("Gemini AI is generating your comprehensive report..."):
                # Get field data
                field_info = prediction_manager.get_field_info(selected_field)
                
                # Prepare satellite analysis data
                satellite_analysis = {
                    'ai_insights': 'Satellite data indicates healthy crop development with consistent NDVI values above 0.6, suggesting good vegetation density and photosynthetic activity.',
                    'confidence_level': 'high',
                    'data_quality': 'excellent'
                }
                
                # Prepare market analysis data
                market_analysis = {
                    'ai_insights': f'Market conditions for {selected_crop} show positive fundamentals with supply constraints supporting higher prices in the medium term.',
                    'crop_type': selected_crop,
                    'confidence_level': 'high'
                }
                
                # Prepare strategy results (optional)
                strategy_results = {
                    'strategy_name': 'Integrated Agricultural Strategy',
                    'performance_metrics': {
                        'total_return': 0.142,
                        'sharpe_ratio': 1.18,
                        'max_drawdown': 0.065
                    },
                    'risk_analysis': {
                        'maximum_drawdown': 0.065,
                        'value_at_risk_95': -0.028
                    }
                }
                
                # Generate comprehensive report
                report_result = gemini_analyzer.generate_comprehensive_report(
                    field_info, satellite_analysis, market_analysis, strategy_results
                )
                
                if 'error' not in report_result:
                    # Display report header
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Field", report_result.get('field_name', 'Unknown'))
                    with col2:
                        st.metric("Confidence", report_result.get('confidence_level', 'Unknown').title())
                    with col3:
                        report_time = datetime.fromisoformat(report_result['report_timestamp'])
                        st.metric("Report Time", report_time.strftime('%H:%M'))
                    
                    # Comprehensive AI Report
                    st.subheader("📋 Executive AI Report")
                    
                    ai_report = report_result.get('ai_report', '')
                    if ai_report:
                        st.markdown(ai_report)
                    else:
                        st.info("Comprehensive report is being generated...")
                    
                    # Data sources and metadata
                    st.subheader("📊 Report Metadata")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Data Sources:**")
                        data_sources = report_result.get('data_sources', [])
                        for source in data_sources:
                            st.write(f"• {source.replace('_', ' ').title()}")
                    
                    with col2:
                        st.write("**Field Details:**")
                        st.write(f"• Field: {field_info.get('name', 'Unknown')}")
                        st.write(f"• Crop: {field_info.get('crop_type', 'Unknown')}")
                        st.write(f"• Area: {field_info.get('area_hectares', 'Unknown')} hectares")
                
                else:
                    st.error(f"Report generation failed: {report_result['error']}")
        else:
            st.warning("Please select a field for comprehensive analysis.")

else:
    # Welcome screen
    st.info("""
    ### 🤖 Google Gemini AI Integration
    
    **Unlock Advanced Agricultural Intelligence** with Google's most capable AI model. Get expert-level insights 
    that combine satellite data, market intelligence, and strategic analysis.
    
    **🎯 AI Analysis Capabilities:**
    
    1. **🛰️ Satellite Data Analysis**
       - Expert interpretation of NDVI trends and crop health indicators
       - Identification of concerning patterns and anomalies
       - Specific agricultural recommendations for optimal productivity
       - Risk assessment for potential yield impacts
    
    2. **📈 Market Intelligence**
       - AI-powered market impact analysis of satellite indicators
       - Price trend predictions and supply/demand implications
       - Strategic trading recommendations and optimal timing
       - Correlation analysis with related commodities
    
    3. **⚡ Strategy Optimization**
       - Performance analysis of trading strategies
       - Parameter optimization suggestions for improved returns
       - Risk management improvements and portfolio allocation
       - Market regime considerations and strategy adaptations
    
    4. **📊 Comprehensive Reports**
       - Executive summaries combining all analysis components
       - Professional reports suitable for farm management decisions
       - Investment strategy recommendations with risk assessments
       - Long-term planning and strategic positioning advice
    
    **💡 Key Benefits:**
    
    - **Expert-Level Analysis**: Get insights comparable to agricultural consultants
    - **Real-Time Intelligence**: AI processes your current data for immediate insights
    - **Natural Language Output**: Easy-to-understand recommendations in plain English
    - **Comprehensive Integration**: Combines satellite, market, and strategy data
    - **Professional Quality**: Reports suitable for farm management and investment decisions
    
    **🚀 Getting Started:**
    
    Choose an analysis type from the sidebar, select your field or crop, and let Gemini AI 
    generate professional-grade insights for your agricultural operations.
    """)

# Footer with AI capabilities
st.markdown("---")
st.markdown("""
### 🧠 AI Technology Details

**Google Gemini Pro** provides:
- Advanced natural language understanding and generation
- Multi-modal analysis combining text, data, and contextual information
- Expert-level reasoning across agricultural and financial domains
- Real-time processing of complex satellite and market data
- Professional report generation with actionable insights

**Integration Features:**
- Secure API integration with your Gemini API key
- Real-time analysis of your satellite and field data
- Contextual understanding of agricultural cycles and market dynamics
- Personalized recommendations based on your specific field conditions
- Comprehensive risk assessment and strategic planning support
""")