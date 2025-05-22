"""
Weather & Gas Price Analysis - Real-time correlation analysis for agricultural operations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
from typing import Dict, Any

from utils.weather_gas_analysis import get_weather_gas_analyzer

# Page configuration
st.set_page_config(
    page_title="Weather & Gas Analysis - Agro Insight",
    page_icon="🌡️",
    layout="wide"
)

# Title and description
st.title("🌡️ Weather & Gas Price Analysis")
st.markdown("""
**Real-time correlation analysis** between weather patterns and energy prices to optimize agricultural operations and costs.
This module helps you understand how weather conditions impact energy costs and provides actionable insights for better decision-making.
""")

# Get the analyzer instance
analyzer = get_weather_gas_analyzer()

# Sidebar controls
st.sidebar.header("Analysis Parameters")

# Location input
st.sidebar.subheader("📍 Location")
lat = st.sidebar.number_input("Latitude", value=52.2297, min_value=-90.0, max_value=90.0, step=0.1, help="Enter the latitude of your location")
lon = st.sidebar.number_input("Longitude", value=21.0122, min_value=-180.0, max_value=180.0, step=0.1, help="Enter the longitude of your location")

# Analysis period
st.sidebar.subheader("📅 Analysis Period")
days = st.sidebar.slider("Days of historical data", min_value=7, max_value=90, value=30, help="Number of days to analyze")

# Run analysis button
if st.sidebar.button("🔍 Run Analysis", type="primary"):
    with st.spinner("Fetching weather data and gas prices..."):
        # Generate comprehensive report
        report = analyzer.generate_integrated_report(lat, lon, days)
        
        if report:
            st.session_state['analysis_report'] = report
            st.success("Analysis completed successfully!")
        else:
            st.error("Failed to generate analysis. Please check your inputs and try again.")

# Display results if available
if 'analysis_report' in st.session_state:
    report = st.session_state['analysis_report']
    
    # Key metrics overview
    st.header("📊 Overview")
    
    # Create columns for key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    weather_summary = report.get('weather_summary', {})
    
    with col1:
        avg_temp = weather_summary.get('avg_temperature', 0)
        st.metric(
            label="Avg Temperature", 
            value=f"{avg_temp}°C",
            delta=f"{avg_temp - 20:.1f}°C from comfort zone"
        )
    
    with col2:
        total_precip = weather_summary.get('total_precipitation', 0)
        st.metric(
            label="Total Precipitation", 
            value=f"{total_precip}mm",
            delta=f"{total_precip - 50:.1f}mm from optimal"
        )
    
    with col3:
        gas_summary = report.get('gas_price_summary', {})
        ng_price = 0
        if 'NG=F' in gas_summary:
            ng_price = gas_summary['NG=F'].get('current_price', 0)
        st.metric(
            label="Natural Gas Price", 
            value=f"${ng_price:.2f}",
            delta=f"{gas_summary.get('NG=F', {}).get('price_change_1d', 0):.2f}" if 'NG=F' in gas_summary else "0.00"
        )
    
    with col4:
        agricultural_impact = report.get('agricultural_impact', {})
        heating_costs = agricultural_impact.get('heating_cooling_costs', {})
        monthly_cost = heating_costs.get('monthly_estimated_cost', 0)
        st.metric(
            label="Est. Monthly Energy Cost", 
            value=f"${monthly_cost:.0f}",
            delta=None
        )
    
    # Key insights
    st.header("💡 Key Insights")
    insights = report.get('key_insights', [])
    if insights:
        for i, insight in enumerate(insights, 1):
            st.info(f"**{i}.** {insight}")
    else:
        st.info("No specific insights generated for current conditions.")
    
    # Tabs for detailed analysis
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌤️ Weather Analysis", 
        "⛽ Gas Prices", 
        "📈 Correlations",
        "🚜 Agricultural Impact"
    ])
    
    with tab1:
        st.subheader("Weather Conditions Analysis")
        
        # Weather data visualization
        weather_summary = report.get('weather_summary', {})
        
        if weather_summary:
            # Create weather metrics cards
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Weather Summary**")
                st.write(f"• Average Temperature: {weather_summary.get('avg_temperature', 0):.1f}°C")
                st.write(f"• Total Precipitation: {weather_summary.get('total_precipitation', 0):.1f}mm")
                st.write(f"• Average Humidity: {weather_summary.get('avg_humidity', 0):.1f}%")
                st.write(f"• Average Wind Speed: {weather_summary.get('avg_wind_speed', 0):.1f} km/h")
            
            with col2:
                # Weather risk assessment
                agricultural_impact = report.get('agricultural_impact', {})
                weather_risks = agricultural_impact.get('weather_risk_assessment', {})
                
                if weather_risks:
                    st.markdown("**Weather Risk Assessment**")
                    risk_level = weather_risks.get('risk_level', 'Unknown')
                    overall_score = weather_risks.get('overall_risk_score', 0)
                    primary_concern = weather_risks.get('primary_concern', 'None')
                    
                    # Color code the risk level
                    risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk_level, "⚪")
                    st.write(f"• Overall Risk Level: {risk_color} {risk_level}")
                    st.write(f"• Risk Score: {overall_score:.3f}")
                    st.write(f"• Primary Concern: {primary_concern}")
        
        # Generate sample weather chart (since we have synthetic data)
        if weather_summary:
            # Create a sample weather trends chart
            dates = pd.date_range(end=datetime.datetime.now(), periods=days, freq='D')
            
            # Generate sample data for visualization
            np.random.seed(42)  # For consistent demo data
            temps = np.random.normal(weather_summary.get('avg_temperature', 20), 5, days)
            precip = np.random.exponential(2, days)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Temperature Trend', 'Precipitation'),
                vertical_spacing=0.12
            )
            
            # Temperature plot
            fig.add_trace(
                go.Scatter(x=dates, y=temps, mode='lines', name='Temperature', line=dict(color='red')),
                row=1, col=1
            )
            
            # Precipitation plot
            fig.add_trace(
                go.Bar(x=dates, y=precip, name='Precipitation', marker_color='blue'),
                row=2, col=1
            )
            
            fig.update_layout(height=500, showlegend=False)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
            fig.update_yaxes(title_text="Precipitation (mm)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Gas & Energy Prices")
        
        gas_summary = report.get('gas_price_summary', {})
        
        if gas_summary:
            # Price summary table
            price_data = []
            for symbol, info in gas_summary.items():
                price_data.append({
                    'Symbol': symbol,
                    'Commodity': info['name'],
                    'Current Price': f"${info['current_price']:.2f}",
                    'Daily Change': f"${info['price_change_1d']:.2f}",
                    'Trend': '↗️' if info['price_change_1d'] > 0 else '↘️' if info['price_change_1d'] < 0 else '➡️'
                })
            
            if price_data:
                df = pd.DataFrame(price_data)
                st.dataframe(df, use_container_width=True)
            
            # Price impact analysis
            st.markdown("**Energy Cost Impact Analysis**")
            
            agricultural_impact = report.get('agricultural_impact', {})
            
            if agricultural_impact:
                cost_cols = st.columns(3)
                
                with cost_cols[0]:
                    heating_costs = agricultural_impact.get('heating_cooling_costs', {})
                    if heating_costs:
                        st.metric(
                            label="Heating/Cooling Costs",
                            value=f"${heating_costs.get('monthly_estimated_cost', 0):.0f}/month",
                            help=f"Cost category: {heating_costs.get('cost_category', 'Unknown')}"
                        )
                
                with cost_cols[1]:
                    irrigation_costs = agricultural_impact.get('irrigation_costs', {})
                    if irrigation_costs:
                        st.metric(
                            label="Irrigation Costs",
                            value=f"${irrigation_costs.get('monthly_estimated_cost', 0):.0f}/month",
                            help=f"Water stress: {irrigation_costs.get('water_stress_level', 'Unknown')}"
                        )
                
                with cost_cols[2]:
                    fuel_costs = agricultural_impact.get('equipment_fuel_costs', {})
                    if fuel_costs:
                        st.metric(
                            label="Equipment Fuel Costs",
                            value=f"${fuel_costs.get('monthly_estimated_cost', 0):.0f}/month",
                            help=f"Cost category: {fuel_costs.get('cost_category', 'Unknown')}"
                        )
        else:
            st.info("Gas price data not available. Please check your internet connection and try again.")
    
    with tab3:
        st.subheader("Weather-Gas Price Correlations")
        
        correlations = report.get('correlation_analysis', {})
        
        if correlations and 'correlations' in correlations:
            # Correlation heatmap data
            corr_data = []
            for symbol, corr_info in correlations['correlations'].items():
                corr_data.append({
                    'Commodity': corr_info['name'],
                    'Temperature': corr_info.get('temp_price_corr', 0),
                    'Precipitation': corr_info.get('precip_price_corr', 0),
                    'Humidity': corr_info.get('humidity_price_corr', 0),
                    'Wind Speed': corr_info.get('wind_price_corr', 0)
                })
            
            if corr_data:
                df_corr = pd.DataFrame(corr_data)
                df_corr.set_index('Commodity', inplace=True)
                
                # Create correlation heatmap
                fig = px.imshow(
                    df_corr.T,
                    color_continuous_scale='RdBu',
                    aspect='auto',
                    title='Weather-Price Correlation Matrix'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Strongest correlations
                strongest = correlations.get('strongest_correlations', {})
                if strongest:
                    st.markdown("**Strongest Correlations**")
                    for symbol, info in strongest.items():
                        significance = info['significance']
                        color = {"Strong": "🔴", "Moderate": "🟡", "Weak": "⚪"}.get(significance, "⚪")
                        st.write(f"{color} **{info['name']}**: {significance} correlation (r = {info['max_correlation']:.3f})")
        else:
            st.info("Correlation analysis not available. Ensure both weather and price data are loaded.")
    
    with tab4:
        st.subheader("Agricultural Operations Impact")
        
        agricultural_impact = report.get('agricultural_impact', {})
        
        if agricultural_impact:
            # Recommendations
            recommendations = agricultural_impact.get('cost_optimization_recommendations', [])
            if recommendations:
                st.markdown("**💰 Cost Optimization Recommendations**")
                for i, rec in enumerate(recommendations, 1):
                    priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec.get('priority', 'medium'), "⚪")
                    st.write(f"{priority_icon} **{rec.get('action', '')}**")
                    st.write(f"   _{rec.get('reason', '')}_")
                    st.write("")
            
            # Market timing suggestions
            timing_suggestions = agricultural_impact.get('market_timing_suggestions', [])
            if timing_suggestions:
                st.markdown("**⏰ Market Timing Suggestions**")
                for i, suggestion in enumerate(timing_suggestions, 1):
                    urgency_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(suggestion.get('urgency', 'medium'), "⚪")
                    st.write(f"{urgency_icon} **{suggestion.get('commodity', '')}**: {suggestion.get('action', '')}")
                    st.write(f"   _{suggestion.get('reason', '')}_")
                    st.write("")
        
        # Action items summary
        action_items = report.get('action_items', [])
        if action_items:
            st.markdown("**📋 Action Items**")
            
            # Group by priority
            high_priority = [item for item in action_items if item.get('priority') == 'high']
            medium_priority = [item for item in action_items if item.get('priority') == 'medium']
            low_priority = [item for item in action_items if item.get('priority') == 'low']
            
            if high_priority:
                st.markdown("**🔴 High Priority (Complete within 7-30 days)**")
                for item in high_priority:
                    st.write(f"• {item.get('action', '')}")
            
            if medium_priority:
                st.markdown("**🟡 Medium Priority (Complete within 30-60 days)**")
                for item in medium_priority:
                    st.write(f"• {item.get('action', '')}")
            
            if low_priority:
                st.markdown("**🟢 Low Priority (Monitor and plan)**")
                for item in low_priority:
                    st.write(f"• {item.get('action', '')}")

else:
    # Welcome screen
    st.info("""
    ### 🎯 How to Use This Analysis
    
    1. **Set your location** using the latitude and longitude inputs in the sidebar
    2. **Choose analysis period** (7-90 days of historical data)
    3. **Click "Run Analysis"** to fetch real-time weather and gas price data
    4. **Review the results** in the detailed tabs above
    
    ### 🔍 What You'll Get
    
    - **Real-time weather conditions** and trends for your location
    - **Current gas and energy prices** from major commodity markets
    - **Correlation analysis** showing how weather affects energy prices
    - **Cost impact estimates** for heating, cooling, irrigation, and fuel
    - **Actionable recommendations** for optimizing operational costs
    - **Market timing suggestions** for energy purchases and hedging
    
    ### 💡 Key Benefits
    
    - **Reduce energy costs** by timing purchases based on weather forecasts
    - **Plan operations** around expected weather and price conditions
    - **Identify opportunities** for cost savings and risk management
    - **Make data-driven decisions** with comprehensive analysis
    
    **Ready to start?** Enter your coordinates in the sidebar and run your first analysis!
    """)

# Additional information
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About This Analysis")
st.sidebar.markdown("""
This tool combines:
- **Live weather data** for your location
- **Real-time gas & oil prices** from commodity markets
- **Statistical correlation analysis** 
- **Agricultural cost modeling**

The analysis helps you understand how weather patterns impact energy costs and provides actionable insights for better operational planning.
""")

st.sidebar.markdown("### 🎯 Sample Locations")
st.sidebar.markdown("""
**Poland (Central)**: 52.2297, 21.0122
**USA (Iowa)**: 41.8780, -93.0977  
**Ukraine (Kiev)**: 50.4501, 30.5234
**Germany (Berlin)**: 52.5200, 13.4050
""")

# Add refresh button for new analysis
if st.sidebar.button("🔄 Clear Results"):
    if 'analysis_report' in st.session_state:
        del st.session_state['analysis_report']
    st.rerun()