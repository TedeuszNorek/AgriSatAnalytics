"""
Investment Strategy Simulator - Real-time backtesting and optimization
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from utils.strategy_optimizer import get_strategy_optimizer

# Page configuration
st.set_page_config(
    page_title="Strategy Simulator - Agro Insight",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("🎯 Investment Strategy Simulator")
st.markdown("""
**Real-time strategy backtesting and optimization** for satellite-energy correlations. Run simulations, optimize parameters, 
and conduct Monte Carlo stress testing to maximize your trading performance.
""")

# Get optimizer instance
optimizer = get_strategy_optimizer()

# Sidebar controls
st.sidebar.header("🎮 Simulation Controls")

# Simulation type selection
simulation_type = st.sidebar.selectbox(
    "Simulation Type",
    [
        "🎯 Single Strategy Backtest",
        "⚡ Strategy Optimization", 
        "🎲 Monte Carlo Simulation",
        "🏆 Strategy Comparison"
    ]
)

# Strategy selection
available_strategies = list(optimizer.strategy_templates.keys())
strategy_names = [optimizer.strategy_templates[s]['name'] for s in available_strategies]

if "Single Strategy" in simulation_type or "Strategy Optimization" in simulation_type:
    selected_strategy_name = st.sidebar.selectbox("Select Strategy", strategy_names)
    selected_strategy = available_strategies[strategy_names.index(selected_strategy_name)]
elif "Strategy Comparison" in simulation_type:
    selected_strategies_names = st.sidebar.multiselect("Select Strategies to Compare", strategy_names, default=strategy_names[:2])
    selected_strategies = [available_strategies[strategy_names.index(name)] for name in selected_strategies_names]

# Time parameters
st.sidebar.subheader("⏰ Time Parameters")
if "Monte Carlo" in simulation_type:
    simulation_days = st.sidebar.slider("Simulation Days", min_value=30, max_value=365, value=252)
    num_simulations = st.sidebar.slider("Number of Simulations", min_value=100, max_value=2000, value=1000, step=100)
else:
    period_days = st.sidebar.slider("Backtest Period (Days)", min_value=30, max_value=730, value=252)

# Capital parameters
initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=10000, max_value=1000000, value=100000, step=10000)

# Run simulation button
run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary")

# Strategy information
with st.sidebar.expander("ℹ️ Strategy Information"):
    if "Single Strategy" in simulation_type or "Strategy Optimization" in simulation_type:
        if 'selected_strategy' in locals():
            strategy_info = optimizer.strategy_templates[selected_strategy]
            st.write(f"**{strategy_info['name']}**")
            st.write(strategy_info['description'])
            st.write(f"**Instruments:** {', '.join(strategy_info['instruments'])}")
            st.write(f"**Timeframe:** {strategy_info['timeframe']}")

# Main content area
if run_simulation:
    if "Single Strategy" in simulation_type:
        st.header(f"📊 {selected_strategy_name} Backtest Results")
        
        with st.spinner("Running strategy backtest..."):
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # Generate sample satellite data for simulation
            sample_satellite_data = {
                'ndvi_average': 0.65,
                'temperature_average': 18.5,
                'soil_moisture': 0.35,
                'cloud_cover': 0.45
            }
            
            # Run simulation
            results = optimizer.simulate_strategy(
                selected_strategy,
                start_date,
                end_date,
                initial_capital,
                sample_satellite_data
            )
            
            if 'error' not in results:
                # Display key performance metrics
                performance = results['performance_metrics']
                risk_analysis = results['risk_analysis']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_return = performance.get('total_return', 0)
                    st.metric(
                        "Total Return", 
                        f"{total_return:.2%}",
                        delta=f"vs benchmark" if total_return > 0.1 else None
                    )
                
                with col2:
                    sharpe_ratio = performance.get('sharpe_ratio', 0)
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
                
                with col3:
                    max_drawdown = performance.get('max_drawdown', 0)
                    st.metric("Max Drawdown", f"{max_drawdown:.2%}")
                
                with col4:
                    win_rate = performance.get('win_rate', 0)
                    st.metric("Win Rate", f"{win_rate:.1%}")
                
                # Portfolio value chart
                st.subheader("📈 Portfolio Performance")
                
                portfolio_values = results['portfolio_results'].get('daily_values', [])
                if portfolio_values:
                    # Create date range for chart
                    chart_dates = pd.date_range(start=start_date, periods=len(portfolio_values), freq='D')
                    
                    fig = go.Figure()
                    
                    # Portfolio value line
                    fig.add_trace(go.Scatter(
                        x=chart_dates,
                        y=portfolio_values,
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Benchmark line (buy and hold)
                    benchmark_values = [initial_capital * (1 + 0.07 * i / 252) for i in range(len(portfolio_values))]
                    fig.add_trace(go.Scatter(
                        x=chart_dates,
                        y=benchmark_values,
                        mode='lines',
                        name='Benchmark (7% annual)',
                        line=dict(color='gray', width=1, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='Portfolio Value Over Time',
                        xaxis_title='Date',
                        yaxis_title='Portfolio Value ($)',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Additional metrics and analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Performance Breakdown")
                    
                    metrics_data = [
                        ["Annualized Return", f"{performance.get('annualized_return', 0):.2%}"],
                        ["Volatility", f"{performance.get('volatility', 0):.2%}"],
                        ["Sortino Ratio", f"{performance.get('sortino_ratio', 0):.3f}"],
                        ["Number of Trades", f"{performance.get('num_trades', 0)}"],
                        ["Best Day", f"{performance.get('best_day', 0):.2%}"],
                        ["Worst Day", f"{performance.get('worst_day', 0):.2%}"]
                    ]
                    
                    df_metrics = pd.DataFrame(metrics_data, columns=['Metric', 'Value'])
                    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
                
                with col2:
                    st.subheader("⚠️ Risk Analysis")
                    
                    risk_data = [
                        ["Value at Risk (95%)", f"{risk_analysis.get('value_at_risk_95', 0):.2%}"],
                        ["Value at Risk (99%)", f"{risk_analysis.get('value_at_risk_99', 0):.2%}"],
                        ["Expected Shortfall", f"{risk_analysis.get('expected_shortfall_95', 0):.2%}"],
                        ["Skewness", f"{risk_analysis.get('skewness', 0):.3f}"],
                        ["Kurtosis", f"{risk_analysis.get('kurtosis', 0):.3f}"]
                    ]
                    
                    df_risk = pd.DataFrame(risk_data, columns=['Risk Metric', 'Value'])
                    st.dataframe(df_risk, use_container_width=True, hide_index=True)
                
                # Trading activity
                trades = results['portfolio_results'].get('trades', [])
                if trades:
                    st.subheader("💼 Trading Activity")
                    
                    # Convert trades to DataFrame
                    trades_df = pd.DataFrame(trades)
                    
                    # Show recent trades
                    st.write(f"**Total Trades:** {len(trades)}")
                    
                    if len(trades_df) > 0:
                        # Add color coding for buy/sell
                        trades_display = trades_df.copy()
                        trades_display['Date'] = pd.to_datetime(trades_display['date']).dt.strftime('%Y-%m-%d')
                        trades_display = trades_display[['Date', 'instrument', 'action', 'shares', 'price', 'value', 'reason']]
                        trades_display.columns = ['Date', 'Instrument', 'Action', 'Shares', 'Price', 'Value', 'Reason']
                        
                        st.dataframe(trades_display.tail(10), use_container_width=True, hide_index=True)
            else:
                st.error(f"Simulation failed: {results['error']}")
    
    elif "Strategy Optimization" in simulation_type:
        st.header(f"⚡ {selected_strategy_name} Optimization")
        
        with st.spinner("Optimizing strategy parameters..."):
            # Generate sample satellite data
            sample_satellite_data = {
                'ndvi_average': 0.65,
                'temperature_average': 18.5,
                'soil_moisture': 0.35,
                'cloud_cover': 0.45
            }
            
            # Run optimization
            optimization_results = optimizer.optimize_strategy(
                selected_strategy,
                period_days,
                sample_satellite_data
            )
            
            if 'error' not in optimization_results:
                # Display optimization results
                improvement = optimization_results['performance_improvement']
                optimal_params = optimization_results['optimal_parameters']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    baseline_sharpe = improvement.get('baseline_sharpe', 0)
                    st.metric("Baseline Sharpe", f"{baseline_sharpe:.3f}")
                
                with col2:
                    optimized_sharpe = improvement.get('optimized_sharpe', 0)
                    st.metric("Optimized Sharpe", f"{optimized_sharpe:.3f}")
                
                with col3:
                    improvement_value = improvement.get('improvement', 0)
                    st.metric("Improvement", f"{improvement_value:.3f}", delta=f"+{improvement_value:.3f}")
                
                # Optimal parameters
                st.subheader("🎯 Optimal Parameters")
                
                if optimal_params:
                    params_data = [[param, f"{value:.4f}"] for param, value in optimal_params.items()]
                    df_params = pd.DataFrame(params_data, columns=['Parameter', 'Optimal Value'])
                    st.dataframe(df_params, use_container_width=True, hide_index=True)
                
                # Optimization details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Optimization Details:**")
                    st.write(f"• Success: {'✅' if optimization_results.get('optimization_success', False) else '❌'}")
                    st.write(f"• Iterations: {optimization_results.get('iterations', 0)}")
                    st.write(f"• Optimization Period: {optimization_results['optimization_period']['duration_days']} days")
                
                with col2:
                    st.write("**Performance Comparison:**")
                    baseline_return = (baseline_sharpe * 0.15) / np.sqrt(252)  # Approximate annual return
                    optimized_return = (optimized_sharpe * 0.15) / np.sqrt(252)
                    
                    st.write(f"• Baseline Annual Return: {baseline_return:.2%}")
                    st.write(f"• Optimized Annual Return: {optimized_return:.2%}")
                    st.write(f"• Performance Gain: {(optimized_return - baseline_return):.2%}")
            else:
                st.error(f"Optimization failed: {optimization_results['error']}")
    
    elif "Monte Carlo" in simulation_type:
        st.header("🎲 Monte Carlo Simulation Results")
        
        if 'selected_strategy' in locals():
            with st.spinner(f"Running {num_simulations} Monte Carlo simulations..."):
                
                # Run Monte Carlo simulation
                mc_results = optimizer.run_monte_carlo_simulation(
                    selected_strategy,
                    num_simulations,
                    simulation_days
                )
                
                if 'error' not in mc_results:
                    # Display key statistics
                    return_stats = mc_results['return_statistics']
                    risk_stats = mc_results['risk_statistics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        mean_return = return_stats.get('mean_return', 0)
                        st.metric("Mean Return", f"{mean_return:.2%}")
                    
                    with col2:
                        prob_positive = risk_stats.get('probability_positive', 0)
                        st.metric("Probability of Profit", f"{prob_positive:.1%}")
                    
                    with col3:
                        var_95 = mc_results['value_at_risk'].get('var_95', 0)
                        st.metric("VaR 95%", f"{var_95:.2%}")
                    
                    with col4:
                        worst_drawdown = risk_stats.get('worst_case_drawdown', 0)
                        st.metric("Worst Case Drawdown", f"{worst_drawdown:.2%}")
                    
                    # Return distribution histogram
                    st.subheader("📊 Return Distribution")
                    
                    # Generate sample distribution for visualization
                    np.random.seed(42)
                    sample_returns = np.random.normal(
                        return_stats.get('mean_return', 0),
                        return_stats.get('std_return', 0.1),
                        num_simulations
                    )
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=sample_returns,
                        nbinsx=50,
                        name='Return Distribution',
                        marker_color='lightblue',
                        opacity=0.7
                    ))
                    
                    # Add VaR lines
                    fig.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text="VaR 95%")
                    
                    fig.update_layout(
                        title='Distribution of Strategy Returns',
                        xaxis_title='Total Return',
                        yaxis_title='Frequency',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk analysis table
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📈 Return Statistics")
                        
                        return_data = [
                            ["Mean Return", f"{return_stats.get('mean_return', 0):.2%}"],
                            ["Median Return", f"{return_stats.get('median_return', 0):.2%}"],
                            ["Standard Deviation", f"{return_stats.get('std_return', 0):.2%}"],
                            ["Minimum Return", f"{return_stats.get('min_return', 0):.2%}"],
                            ["Maximum Return", f"{return_stats.get('max_return', 0):.2%}"],
                            ["5th Percentile", f"{return_stats.get('percentile_5', 0):.2%}"],
                            ["95th Percentile", f"{return_stats.get('percentile_95', 0):.2%}"]
                        ]
                        
                        df_returns = pd.DataFrame(return_data, columns=['Statistic', 'Value'])
                        st.dataframe(df_returns, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.subheader("⚠️ Risk Analysis")
                        
                        risk_data = [
                            ["Probability of Loss > 5%", f"{risk_stats.get('probability_loss_5pct', 0):.1%}"],
                            ["Probability of Loss > 10%", f"{risk_stats.get('probability_loss_10pct', 0):.1%}"],
                            ["Expected Max Drawdown", f"{risk_stats.get('expected_max_drawdown', 0):.2%}"],
                            ["Worst Case Drawdown", f"{risk_stats.get('worst_case_drawdown', 0):.2%}"],
                            ["Value at Risk 99%", f"{mc_results['value_at_risk'].get('var_99', 0):.2%}"],
                            ["Expected Shortfall", f"{mc_results['value_at_risk'].get('expected_shortfall_95', 0):.2%}"]
                        ]
                        
                        df_risk_mc = pd.DataFrame(risk_data, columns=['Risk Metric', 'Value'])
                        st.dataframe(df_risk_mc, use_container_width=True, hide_index=True)
                else:
                    st.error(f"Monte Carlo simulation failed: {mc_results['error']}")
    
    elif "Strategy Comparison" in simulation_type:
        st.header("🏆 Strategy Comparison")
        
        if 'selected_strategies' in locals() and selected_strategies:
            with st.spinner("Comparing strategies..."):
                
                # Run strategy comparison
                comparison_results = optimizer.compare_strategies(
                    selected_strategies,
                    period_days
                )
                
                if 'error' not in comparison_results:
                    strategies_data = comparison_results['strategies']
                    rankings = comparison_results['rankings']
                    
                    # Comparison table
                    st.subheader("📊 Performance Comparison")
                    
                    comparison_data = []
                    for strategy_key, strategy_data in strategies_data.items():
                        name = strategy_data['name']
                        performance = strategy_data['performance']
                        risk = strategy_data['risk']
                        
                        comparison_data.append({
                            'Strategy': name,
                            'Total Return': f"{performance.get('total_return', 0):.2%}",
                            'Sharpe Ratio': f"{performance.get('sharpe_ratio', 0):.3f}",
                            'Max Drawdown': f"{performance.get('max_drawdown', 0):.2%}",
                            'Volatility': f"{performance.get('volatility', 0):.2%}",
                            'Win Rate': f"{performance.get('win_rate', 0):.1%}",
                            'Final Value': f"${strategy_data.get('final_value', 100000):,.0f}"
                        })
                    
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                    
                    # Rankings
                    st.subheader("🥇 Strategy Rankings")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write("**By Total Return:**")
                        for i, strategy in enumerate(rankings.get('by_return', []), 1):
                            strategy_name = strategies_data[strategy]['name']
                            emoji = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
                            st.write(f"{emoji} {strategy_name}")
                    
                    with col2:
                        st.write("**By Sharpe Ratio:**")
                        for i, strategy in enumerate(rankings.get('by_sharpe', []), 1):
                            strategy_name = strategies_data[strategy]['name']
                            emoji = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
                            st.write(f"{emoji} {strategy_name}")
                    
                    with col3:
                        st.write("**By Drawdown (Lower is Better):**")
                        for i, strategy in enumerate(rankings.get('by_drawdown', []), 1):
                            strategy_name = strategies_data[strategy]['name']
                            emoji = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
                            st.write(f"{emoji} {strategy_name}")
                    
                    with col4:
                        st.write("**Overall Ranking:**")
                        for i, strategy in enumerate(rankings.get('overall', []), 1):
                            strategy_name = strategies_data[strategy]['name']
                            emoji = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
                            st.write(f"{emoji} {strategy_name}")
                    
                    # Performance comparison chart
                    st.subheader("📈 Risk-Return Comparison")
                    
                    fig = go.Figure()
                    
                    for strategy_key, strategy_data in strategies_data.items():
                        name = strategy_data['name']
                        performance = strategy_data['performance']
                        risk = strategy_data['risk']
                        
                        fig.add_trace(go.Scatter(
                            x=[performance.get('volatility', 0)],
                            y=[performance.get('total_return', 0)],
                            mode='markers+text',
                            name=name,
                            text=[name],
                            textposition="top center",
                            marker=dict(size=15)
                        ))
                    
                    fig.update_layout(
                        title='Risk vs Return Profile',
                        xaxis_title='Volatility (Risk)',
                        yaxis_title='Total Return',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(f"Strategy comparison failed: {comparison_results['error']}")
        else:
            st.warning("Please select at least one strategy to compare.")

else:
    # Welcome screen
    st.info("""
    ### 🎯 Advanced Strategy Simulation & Optimization
    
    This powerful simulation engine helps you optimize your satellite-energy trading strategies through:
    
    **🎮 Available Simulations:**
    
    1. **📊 Single Strategy Backtest**
       - Complete performance analysis with real market data
       - Risk metrics, drawdown analysis, and trade details
       - Portfolio value tracking over time
    
    2. **⚡ Strategy Optimization**
       - Automatically optimize strategy parameters
       - Maximize Sharpe ratio using scientific optimization
       - Compare baseline vs optimized performance
    
    3. **🎲 Monte Carlo Simulation**
       - Stress test strategies under 1000+ random scenarios
       - Calculate probability of profit and risk of loss
       - Value at Risk (VaR) and Expected Shortfall analysis
    
    4. **🏆 Strategy Comparison**
       - Side-by-side comparison of multiple strategies
       - Rankings by return, Sharpe ratio, and risk metrics
       - Risk-return visualization
    
    **🚀 Available Strategies:**
    
    - **Seasonal Energy Strategy**: Long/short energy based on NDVI-temperature correlations
    - **Solar Intraday Strategy**: Electricity trading based on cloud cover forecasts  
    - **Weather Options Strategy**: Options strategies based on vegetation anomalies
    - **Bioenergy Momentum Strategy**: Long/short bioenergy stocks based on soil moisture
    
    **📈 Key Features:**
    
    - Real market data integration with Yahoo Finance
    - Advanced risk metrics (VaR, Expected Shortfall, Sharpe ratio)
    - Parameter optimization using scientific methods
    - Monte Carlo stress testing with customizable scenarios
    - Comprehensive trade analysis and portfolio tracking
    
    **💡 Getting Started:**
    
    Choose a simulation type from the sidebar, select your strategy, set your parameters, and click "Run Simulation" to begin!
    """)

# Footer information
st.markdown("---")
st.markdown("""
### 📚 Technical Details

**Optimization Engine:**
- Uses L-BFGS-B algorithm for parameter optimization
- Maximizes risk-adjusted returns (Sharpe ratio)
- Handles bounded parameter constraints

**Risk Analysis:**
- Value at Risk (VaR) at 95% and 99% confidence levels
- Expected Shortfall (Conditional VaR) calculations
- Maximum drawdown and volatility analysis
- Skewness and kurtosis for distribution analysis

**Monte Carlo Framework:**
- Generates realistic market scenarios from historical data
- Incorporates satellite data variability
- Provides comprehensive statistical analysis of outcomes
- Calculates probability distributions for all metrics

**Data Sources:**
- Real-time market data from Yahoo Finance
- Satellite indicators from your field analysis
- Weather and energy correlation data
- Historical price and volatility patterns
""")