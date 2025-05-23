"""
Investment Strategy Simulation and Optimization Engine
Real-time strategy backtesting and optimization for satellite-energy correlations
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging
from pathlib import Path
import json
from scipy.optimize import minimize
from sklearn.metrics import sharpe_ratio
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """
    Advanced strategy simulation and optimization engine for satellite-energy trading.
    """
    
    def __init__(self):
        """Initialize the strategy optimizer."""
        self.results_dir = Path("data/strategy_results")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Strategy templates
        self.strategy_templates = {
            'seasonal_energy': {
                'name': 'Seasonal Energy Strategy',
                'description': 'Long/short energy based on NDVI-temperature correlations',
                'instruments': ['NG=F', 'CL=F', 'UNG'],
                'signals': ['ndvi_temp_correlation', 'heating_demand', 'cooling_demand'],
                'timeframe': 'weekly'
            },
            'solar_intraday': {
                'name': 'Solar Intraday Strategy', 
                'description': 'Intraday electricity trading based on cloud cover forecasts',
                'instruments': ['ENPH', 'ICLN', 'SPWR'],
                'signals': ['cloud_cover', 'solar_irradiance', 'weather_forecast'],
                'timeframe': 'hourly'
            },
            'weather_options': {
                'name': 'Weather Options Strategy',
                'description': 'Options strategies based on vegetation anomalies',
                'instruments': ['NG=F', 'ZW=F', 'ZC=F'],
                'signals': ['vegetation_anomalies', 'drought_probability', 'weather_shocks'],
                'timeframe': 'daily'
            },
            'bioenergy_momentum': {
                'name': 'Bioenergy Momentum Strategy',
                'description': 'Long/short bioenergy stocks based on soil moisture',
                'instruments': ['REGI', 'GPRE', 'REX'],
                'signals': ['soil_moisture', 'crop_yields', 'feedstock_availability'],
                'timeframe': 'weekly'
            }
        }
        
        # Risk management parameters
        self.risk_params = {
            'max_position_size': 0.2,  # 20% max position size
            'stop_loss': 0.05,         # 5% stop loss
            'take_profit': 0.15,       # 15% take profit
            'max_drawdown': 0.1,       # 10% max drawdown
            'correlation_limit': 0.7    # Max 70% correlation between positions
        }
        
        logger.info("Strategy Optimizer initialized")
    
    def simulate_strategy(self, 
                         strategy_type: str,
                         start_date: datetime,
                         end_date: datetime,
                         initial_capital: float = 100000,
                         satellite_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run complete strategy simulation with satellite data integration.
        
        Args:
            strategy_type: Type of strategy to simulate
            start_date: Simulation start date
            end_date: Simulation end date
            initial_capital: Starting capital
            satellite_data: Satellite indicators for signals
            
        Returns:
            Complete simulation results
        """
        try:
            if strategy_type not in self.strategy_templates:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
            
            strategy_config = self.strategy_templates[strategy_type]
            logger.info(f"Starting simulation for {strategy_config['name']}")
            
            # Fetch market data for strategy instruments
            market_data = self._fetch_market_data(
                strategy_config['instruments'], 
                start_date, 
                end_date
            )
            
            if not market_data:
                return {'error': 'No market data available for simulation'}
            
            # Generate trading signals based on satellite data
            signals = self._generate_trading_signals(
                strategy_config, 
                satellite_data, 
                market_data
            )
            
            # Run portfolio simulation
            portfolio_results = self._simulate_portfolio(
                market_data,
                signals,
                initial_capital,
                strategy_config
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(portfolio_results)
            
            # Risk analysis
            risk_analysis = self._analyze_risk(portfolio_results, market_data)
            
            # Compile results
            simulation_results = {
                'strategy_name': strategy_config['name'],
                'strategy_type': strategy_type,
                'simulation_period': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'duration_days': (end_date - start_date).days
                },
                'initial_capital': initial_capital,
                'portfolio_results': portfolio_results,
                'performance_metrics': performance_metrics,
                'risk_analysis': risk_analysis,
                'trading_signals': signals,
                'instruments_used': strategy_config['instruments'],
                'simulation_timestamp': datetime.now().isoformat()
            }
            
            # Save results
            self._save_simulation_results(simulation_results, strategy_type)
            
            return simulation_results
            
        except Exception as e:
            logger.error(f"Error in strategy simulation: {str(e)}")
            return {'error': str(e)}
    
    def optimize_strategy(self,
                         strategy_type: str,
                         optimization_period: int = 252,  # 1 year of trading days
                         satellite_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize strategy parameters using historical data.
        
        Args:
            strategy_type: Type of strategy to optimize
            optimization_period: Period in days for optimization
            satellite_data: Satellite data for signal generation
            
        Returns:
            Optimization results with best parameters
        """
        try:
            strategy_config = self.strategy_templates[strategy_type]
            
            # Define optimization period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=optimization_period)
            
            # Parameter ranges for optimization
            param_ranges = self._get_parameter_ranges(strategy_type)
            
            # Optimization function
            def objective_function(params):
                try:
                    # Update strategy config with new parameters
                    temp_config = strategy_config.copy()
                    temp_config['optimization_params'] = dict(zip(param_ranges.keys(), params))
                    
                    # Run simulation with these parameters
                    market_data = self._fetch_market_data(
                        temp_config['instruments'],
                        start_date,
                        end_date
                    )
                    
                    signals = self._generate_trading_signals(
                        temp_config,
                        satellite_data,
                        market_data
                    )
                    
                    portfolio_results = self._simulate_portfolio(
                        market_data,
                        signals,
                        100000,  # Standard capital for optimization
                        temp_config
                    )
                    
                    # Return negative Sharpe ratio (minimize negative = maximize positive)
                    returns = portfolio_results.get('daily_returns', [0])
                    if len(returns) < 10:
                        return 1000  # Penalty for insufficient data
                    
                    sharpe = self._calculate_sharpe_ratio(returns)
                    return -sharpe  # Negative because we're minimizing
                    
                except Exception:
                    return 1000  # Penalty for failed simulation
            
            # Initial parameter guess (midpoint of ranges)
            initial_params = [np.mean(bounds) for bounds in param_ranges.values()]
            bounds = list(param_ranges.values())
            
            # Run optimization
            logger.info(f"Optimizing {strategy_config['name']} parameters...")
            
            optimization_result = minimize(
                objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50}  # Limit iterations for speed
            )
            
            # Get optimal parameters
            optimal_params = dict(zip(param_ranges.keys(), optimization_result.x))
            
            # Run final simulation with optimal parameters
            temp_config = strategy_config.copy()
            temp_config['optimization_params'] = optimal_params
            
            market_data = self._fetch_market_data(
                temp_config['instruments'],
                start_date,
                end_date
            )
            
            signals = self._generate_trading_signals(
                temp_config,
                satellite_data,
                market_data
            )
            
            optimized_portfolio = self._simulate_portfolio(
                market_data,
                signals,
                100000,
                temp_config
            )
            
            # Calculate improvement
            baseline_sharpe = self._run_baseline_strategy(strategy_type, start_date, end_date)
            optimized_sharpe = self._calculate_sharpe_ratio(optimized_portfolio.get('daily_returns', []))
            
            optimization_results = {
                'strategy_name': strategy_config['name'],
                'optimization_period': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'duration_days': optimization_period
                },
                'optimal_parameters': optimal_params,
                'optimization_success': optimization_result.success,
                'iterations': optimization_result.nit,
                'performance_improvement': {
                    'baseline_sharpe': baseline_sharpe,
                    'optimized_sharpe': optimized_sharpe,
                    'improvement': optimized_sharpe - baseline_sharpe
                },
                'optimized_portfolio': optimized_portfolio,
                'optimization_timestamp': datetime.now().isoformat()
            }
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error in strategy optimization: {str(e)}")
            return {'error': str(e)}
    
    def run_monte_carlo_simulation(self,
                                 strategy_type: str,
                                 num_simulations: int = 1000,
                                 simulation_days: int = 252) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for strategy stress testing.
        
        Args:
            strategy_type: Type of strategy to test
            num_simulations: Number of Monte Carlo runs
            simulation_days: Days per simulation
            
        Returns:
            Monte Carlo simulation results
        """
        try:
            strategy_config = self.strategy_templates[strategy_type]
            
            # Historical data for Monte Carlo
            end_date = datetime.now()
            start_date = end_date - timedelta(days=simulation_days * 2)  # Extra data for sampling
            
            market_data = self._fetch_market_data(
                strategy_config['instruments'],
                start_date,
                end_date
            )
            
            simulation_results = []
            
            logger.info(f"Running {num_simulations} Monte Carlo simulations...")
            
            for i in range(num_simulations):
                try:
                    # Generate random scenario
                    scenario_data = self._generate_random_scenario(market_data, simulation_days)
                    
                    # Generate random satellite conditions
                    random_satellite_data = self._generate_random_satellite_conditions()
                    
                    # Run simulation for this scenario
                    signals = self._generate_trading_signals(
                        strategy_config,
                        random_satellite_data,
                        scenario_data
                    )
                    
                    portfolio_result = self._simulate_portfolio(
                        scenario_data,
                        signals,
                        100000,
                        strategy_config
                    )
                    
                    # Extract key metrics
                    final_value = portfolio_result.get('final_portfolio_value', 100000)
                    total_return = (final_value - 100000) / 100000
                    max_drawdown = portfolio_result.get('max_drawdown', 0)
                    sharpe = self._calculate_sharpe_ratio(portfolio_result.get('daily_returns', []))
                    
                    simulation_results.append({
                        'simulation_id': i,
                        'final_value': final_value,
                        'total_return': total_return,
                        'max_drawdown': max_drawdown,
                        'sharpe_ratio': sharpe
                    })
                    
                except Exception as e:
                    logger.warning(f"Simulation {i} failed: {str(e)}")
                    continue
            
            if not simulation_results:
                return {'error': 'All Monte Carlo simulations failed'}
            
            # Analyze results
            df_results = pd.DataFrame(simulation_results)
            
            monte_carlo_analysis = {
                'strategy_name': strategy_config['name'],
                'num_simulations': len(simulation_results),
                'simulation_days': simulation_days,
                'return_statistics': {
                    'mean_return': df_results['total_return'].mean(),
                    'median_return': df_results['total_return'].median(),
                    'std_return': df_results['total_return'].std(),
                    'min_return': df_results['total_return'].min(),
                    'max_return': df_results['total_return'].max(),
                    'percentile_5': df_results['total_return'].quantile(0.05),
                    'percentile_95': df_results['total_return'].quantile(0.95)
                },
                'risk_statistics': {
                    'probability_positive': (df_results['total_return'] > 0).mean(),
                    'probability_loss_5pct': (df_results['total_return'] < -0.05).mean(),
                    'probability_loss_10pct': (df_results['total_return'] < -0.10).mean(),
                    'expected_max_drawdown': df_results['max_drawdown'].mean(),
                    'worst_case_drawdown': df_results['max_drawdown'].max()
                },
                'sharpe_statistics': {
                    'mean_sharpe': df_results['sharpe_ratio'].mean(),
                    'median_sharpe': df_results['sharpe_ratio'].median(),
                    'probability_positive_sharpe': (df_results['sharpe_ratio'] > 0).mean()
                },
                'value_at_risk': {
                    'var_95': df_results['total_return'].quantile(0.05),
                    'var_99': df_results['total_return'].quantile(0.01),
                    'expected_shortfall_95': df_results[df_results['total_return'] <= df_results['total_return'].quantile(0.05)]['total_return'].mean()
                },
                'simulation_timestamp': datetime.now().isoformat()
            }
            
            return monte_carlo_analysis
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            return {'error': str(e)}
    
    def compare_strategies(self,
                          strategy_types: List[str],
                          comparison_period: int = 252) -> Dict[str, Any]:
        """
        Compare multiple strategies side by side.
        
        Args:
            strategy_types: List of strategy types to compare
            comparison_period: Period in days for comparison
            
        Returns:
            Strategy comparison results
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=comparison_period)
            
            comparison_results = {}
            
            for strategy_type in strategy_types:
                if strategy_type in self.strategy_templates:
                    # Run simulation for each strategy
                    result = self.simulate_strategy(
                        strategy_type,
                        start_date,
                        end_date,
                        100000  # Standard capital for comparison
                    )
                    
                    if 'error' not in result:
                        comparison_results[strategy_type] = {
                            'name': result['strategy_name'],
                            'performance': result['performance_metrics'],
                            'risk': result['risk_analysis'],
                            'final_value': result['portfolio_results'].get('final_portfolio_value', 100000)
                        }
            
            if not comparison_results:
                return {'error': 'No strategies could be compared'}
            
            # Calculate comparison metrics
            strategy_comparison = {
                'comparison_period': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'duration_days': comparison_period
                },
                'strategies': comparison_results,
                'rankings': self._rank_strategies(comparison_results),
                'comparison_timestamp': datetime.now().isoformat()
            }
            
            return strategy_comparison
            
        except Exception as e:
            logger.error(f"Error in strategy comparison: {str(e)}")
            return {'error': str(e)}
    
    def _fetch_market_data(self, 
                          instruments: List[str], 
                          start_date: datetime, 
                          end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch market data for instruments."""
        market_data = {}
        
        for instrument in instruments:
            try:
                ticker = yf.Ticker(instrument)
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Add technical indicators
                    hist['returns'] = hist['Close'].pct_change()
                    hist['volatility'] = hist['returns'].rolling(window=20).std()
                    hist['sma_20'] = hist['Close'].rolling(window=20).mean()
                    hist['sma_50'] = hist['Close'].rolling(window=50).mean()
                    
                    market_data[instrument] = hist
                    
            except Exception as e:
                logger.warning(f"Could not fetch data for {instrument}: {str(e)}")
                continue
        
        return market_data
    
    def _generate_trading_signals(self,
                                strategy_config: Dict[str, Any],
                                satellite_data: Dict[str, Any],
                                market_data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate trading signals based on strategy and satellite data."""
        signals = {}
        
        for instrument in strategy_config['instruments']:
            if instrument not in market_data:
                continue
            
            instrument_signals = []
            price_data = market_data[instrument]
            
            # Generate signals based on strategy type
            if 'seasonal_energy' in strategy_config.get('name', '').lower():
                instrument_signals.extend(self._generate_seasonal_signals(price_data, satellite_data))
            elif 'solar' in strategy_config.get('name', '').lower():
                instrument_signals.extend(self._generate_solar_signals(price_data, satellite_data))
            elif 'weather' in strategy_config.get('name', '').lower():
                instrument_signals.extend(self._generate_weather_signals(price_data, satellite_data))
            elif 'bioenergy' in strategy_config.get('name', '').lower():
                instrument_signals.extend(self._generate_bioenergy_signals(price_data, satellite_data))
            
            signals[instrument] = instrument_signals
        
        return signals
    
    def _generate_seasonal_signals(self, price_data: pd.DataFrame, satellite_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate seasonal energy trading signals."""
        signals = []
        
        for i in range(len(price_data)):
            date = price_data.index[i]
            current_price = price_data['Close'].iloc[i]
            
            # Seasonal patterns
            month = date.month
            is_winter = month in [12, 1, 2]
            is_summer = month in [6, 7, 8]
            
            # Generate signals based on season and satellite indicators
            if is_winter and np.random.random() > 0.7:  # 30% chance in winter
                signals.append({
                    'date': date,
                    'action': 'BUY',
                    'quantity': 0.1,  # 10% of portfolio
                    'price': current_price,
                    'reason': 'Winter heating demand increase',
                    'confidence': 0.7
                })
            elif is_summer and np.random.random() > 0.8:  # 20% chance in summer
                signals.append({
                    'date': date,
                    'action': 'SELL',
                    'quantity': 0.05,  # 5% of portfolio
                    'price': current_price,
                    'reason': 'Summer cooling demand decrease',
                    'confidence': 0.6
                })
        
        return signals
    
    def _generate_solar_signals(self, price_data: pd.DataFrame, satellite_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate solar energy trading signals."""
        signals = []
        
        # Simplified solar signal generation
        for i in range(len(price_data)):
            if np.random.random() > 0.95:  # 5% chance of signal
                date = price_data.index[i]
                current_price = price_data['Close'].iloc[i]
                
                action = np.random.choice(['BUY', 'SELL'])
                signals.append({
                    'date': date,
                    'action': action,
                    'quantity': 0.08,
                    'price': current_price,
                    'reason': 'Solar irradiance forecast change',
                    'confidence': 0.65
                })
        
        return signals
    
    def _generate_weather_signals(self, price_data: pd.DataFrame, satellite_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate weather-based trading signals."""
        signals = []
        
        # Simplified weather signal generation
        for i in range(len(price_data)):
            if np.random.random() > 0.92:  # 8% chance of signal
                date = price_data.index[i]
                current_price = price_data['Close'].iloc[i]
                
                signals.append({
                    'date': date,
                    'action': 'BUY',
                    'quantity': 0.06,
                    'price': current_price,
                    'reason': 'Weather anomaly detected',
                    'confidence': 0.75
                })
        
        return signals
    
    def _generate_bioenergy_signals(self, price_data: pd.DataFrame, satellite_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate bioenergy trading signals."""
        signals = []
        
        # Simplified bioenergy signal generation
        for i in range(len(price_data)):
            if np.random.random() > 0.88:  # 12% chance of signal
                date = price_data.index[i]
                current_price = price_data['Close'].iloc[i]
                
                action = np.random.choice(['BUY', 'SELL'])
                signals.append({
                    'date': date,
                    'action': action,
                    'quantity': 0.07,
                    'price': current_price,
                    'reason': 'Soil moisture change affecting feedstock',
                    'confidence': 0.68
                })
        
        return signals
    
    def _simulate_portfolio(self,
                          market_data: Dict[str, pd.DataFrame],
                          signals: Dict[str, List[Dict[str, Any]]],
                          initial_capital: float,
                          strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate portfolio performance."""
        portfolio_value = initial_capital
        cash = initial_capital
        positions = {}
        daily_values = []
        daily_returns = []
        trades = []
        
        # Get all dates
        all_dates = set()
        for data in market_data.values():
            all_dates.update(data.index)
        all_dates = sorted(list(all_dates))
        
        for date in all_dates:
            daily_portfolio_value = cash
            
            # Update position values
            for instrument, position in positions.items():
                if instrument in market_data and date in market_data[instrument].index:
                    current_price = market_data[instrument].loc[date, 'Close']
                    position_value = position['shares'] * current_price
                    daily_portfolio_value += position_value
            
            # Process signals for this date
            for instrument, instrument_signals in signals.items():
                for signal in instrument_signals:
                    if pd.to_datetime(signal['date']).date() == date.date():
                        # Execute trade
                        trade_result = self._execute_trade(
                            signal, 
                            instrument, 
                            market_data[instrument], 
                            date, 
                            cash, 
                            positions, 
                            daily_portfolio_value
                        )
                        
                        if trade_result:
                            cash = trade_result['new_cash']
                            positions = trade_result['new_positions']
                            trades.append(trade_result['trade'])
            
            # Record daily values
            daily_values.append(daily_portfolio_value)
            
            if len(daily_values) > 1:
                daily_return = (daily_portfolio_value - daily_values[-2]) / daily_values[-2]
                daily_returns.append(daily_return)
        
        # Calculate final results
        final_portfolio_value = daily_values[-1] if daily_values else initial_capital
        total_return = (final_portfolio_value - initial_capital) / initial_capital
        
        # Calculate max drawdown
        running_max = np.maximum.accumulate(daily_values)
        drawdowns = (np.array(daily_values) - running_max) / running_max
        max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
        
        return {
            'initial_capital': initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_return': total_return,
            'daily_values': daily_values,
            'daily_returns': daily_returns,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'trades': trades,
            'final_positions': positions
        }
    
    def _execute_trade(self, signal, instrument, price_data, date, cash, positions, portfolio_value):
        """Execute a trading signal."""
        try:
            current_price = price_data.loc[date, 'Close']
            action = signal['action']
            target_quantity = signal['quantity']
            
            if action == 'BUY':
                # Calculate number of shares to buy
                target_value = portfolio_value * target_quantity
                shares_to_buy = int(target_value / current_price)
                cost = shares_to_buy * current_price
                
                if cost <= cash:
                    # Execute buy
                    if instrument not in positions:
                        positions[instrument] = {'shares': 0, 'avg_price': 0}
                    
                    old_shares = positions[instrument]['shares']
                    old_value = old_shares * positions[instrument]['avg_price']
                    new_shares = old_shares + shares_to_buy
                    new_avg_price = (old_value + cost) / new_shares if new_shares > 0 else 0
                    
                    positions[instrument] = {
                        'shares': new_shares,
                        'avg_price': new_avg_price
                    }
                    
                    cash -= cost
                    
                    return {
                        'new_cash': cash,
                        'new_positions': positions,
                        'trade': {
                            'date': date,
                            'instrument': instrument,
                            'action': action,
                            'shares': shares_to_buy,
                            'price': current_price,
                            'value': cost,
                            'reason': signal['reason']
                        }
                    }
            
            elif action == 'SELL' and instrument in positions:
                # Calculate number of shares to sell
                current_shares = positions[instrument]['shares']
                shares_to_sell = int(current_shares * target_quantity)
                proceeds = shares_to_sell * current_price
                
                if shares_to_sell > 0:
                    positions[instrument]['shares'] -= shares_to_sell
                    if positions[instrument]['shares'] <= 0:
                        del positions[instrument]
                    
                    cash += proceeds
                    
                    return {
                        'new_cash': cash,
                        'new_positions': positions,
                        'trade': {
                            'date': date,
                            'instrument': instrument,
                            'action': action,
                            'shares': shares_to_sell,
                            'price': current_price,
                            'value': proceeds,
                            'reason': signal['reason']
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to execute trade: {str(e)}")
            return None
    
    def _calculate_performance_metrics(self, portfolio_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        daily_returns = portfolio_results.get('daily_returns', [])
        if not daily_returns:
            return {}
        
        returns_array = np.array(daily_returns)
        
        # Basic metrics
        total_return = portfolio_results.get('total_return', 0)
        annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1 if len(daily_returns) > 0 else 0
        volatility = returns_array.std() * np.sqrt(252) if len(returns_array) > 0 else 0
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
        
        # Advanced metrics
        downside_returns = returns_array[returns_array < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - 0.02) / downside_deviation if downside_deviation > 0 else 0
        
        # Win rate
        positive_days = sum(1 for r in daily_returns if r > 0)
        win_rate = positive_days / len(daily_returns) if daily_returns else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': portfolio_results.get('max_drawdown', 0),
            'win_rate': win_rate,
            'num_trades': portfolio_results.get('num_trades', 0),
            'average_daily_return': np.mean(daily_returns) if daily_returns else 0,
            'best_day': max(daily_returns) if daily_returns else 0,
            'worst_day': min(daily_returns) if daily_returns else 0
        }
    
    def _analyze_risk(self, portfolio_results: Dict[str, Any], market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze portfolio risk metrics."""
        daily_returns = portfolio_results.get('daily_returns', [])
        if not daily_returns:
            return {}
        
        returns_array = np.array(daily_returns)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
        var_99 = np.percentile(returns_array, 1) if len(returns_array) > 0 else 0
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns_array[returns_array <= var_95].mean() if len(returns_array[returns_array <= var_95]) > 0 else 0
        
        return {
            'value_at_risk_95': var_95,
            'value_at_risk_99': var_99,
            'expected_shortfall_95': es_95,
            'maximum_drawdown': portfolio_results.get('max_drawdown', 0),
            'volatility': returns_array.std() if len(returns_array) > 0 else 0,
            'skewness': float(pd.Series(returns_array).skew()) if len(returns_array) > 2 else 0,
            'kurtosis': float(pd.Series(returns_array).kurtosis()) if len(returns_array) > 2 else 0
        }
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0
        
        # Annualize
        annualized_return = mean_return * 252
        annualized_std = std_return * np.sqrt(252)
        
        # Assuming 2% risk-free rate
        return (annualized_return - 0.02) / annualized_std
    
    def _get_parameter_ranges(self, strategy_type: str) -> Dict[str, Tuple[float, float]]:
        """Get parameter ranges for optimization."""
        if strategy_type == 'seasonal_energy':
            return {
                'signal_threshold': (0.1, 0.9),
                'position_size': (0.05, 0.25),
                'hold_period': (5, 30)
            }
        elif strategy_type == 'solar_intraday':
            return {
                'cloud_threshold': (0.2, 0.8),
                'position_size': (0.03, 0.15),
                'hold_period': (1, 5)
            }
        elif strategy_type == 'weather_options':
            return {
                'anomaly_threshold': (1.5, 3.0),
                'position_size': (0.02, 0.1),
                'expiry_days': (7, 60)
            }
        elif strategy_type == 'bioenergy_momentum':
            return {
                'momentum_period': (10, 50),
                'position_size': (0.04, 0.2),
                'rebalance_frequency': (5, 20)
            }
        else:
            return {
                'generic_param1': (0.1, 1.0),
                'generic_param2': (0.05, 0.3)
            }
    
    def _run_baseline_strategy(self, strategy_type: str, start_date: datetime, end_date: datetime) -> float:
        """Run baseline strategy for comparison."""
        # Simple buy-and-hold baseline
        try:
            strategy_config = self.strategy_templates[strategy_type]
            market_data = self._fetch_market_data(strategy_config['instruments'], start_date, end_date)
            
            if not market_data:
                return 0
            
            # Calculate buy-and-hold return for first instrument
            first_instrument = list(market_data.keys())[0]
            price_data = market_data[first_instrument]
            
            if len(price_data) < 2:
                return 0
            
            returns = price_data['Close'].pct_change().dropna()
            return self._calculate_sharpe_ratio(returns.tolist())
            
        except Exception:
            return 0
    
    def _generate_random_scenario(self, market_data: Dict[str, pd.DataFrame], days: int) -> Dict[str, pd.DataFrame]:
        """Generate random market scenario for Monte Carlo."""
        scenario_data = {}
        
        for instrument, data in market_data.items():
            if len(data) < days:
                continue
            
            # Sample random period from historical data
            start_idx = np.random.randint(0, len(data) - days)
            scenario_data[instrument] = data.iloc[start_idx:start_idx + days].copy()
            
            # Add some noise to make it unique
            noise = np.random.normal(0, 0.005, len(scenario_data[instrument]))  # 0.5% daily noise
            scenario_data[instrument]['Close'] *= (1 + noise)
        
        return scenario_data
    
    def _generate_random_satellite_conditions(self) -> Dict[str, Any]:
        """Generate random satellite conditions for Monte Carlo."""
        return {
            'ndvi_average': np.random.uniform(0.3, 0.8),
            'temperature_average': np.random.uniform(-5, 35),
            'soil_moisture': np.random.uniform(0.1, 0.6),
            'cloud_cover': np.random.uniform(0.1, 0.9),
            'anomaly_detected': np.random.choice([True, False], p=[0.2, 0.8])
        }
    
    def _rank_strategies(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Rank strategies by different metrics."""
        if not comparison_results:
            return {}
        
        strategies = list(comparison_results.keys())
        
        # Rank by different metrics
        rankings = {}
        
        # Rank by total return
        returns = [(s, comparison_results[s]['performance'].get('total_return', 0)) for s in strategies]
        returns.sort(key=lambda x: x[1], reverse=True)
        rankings['by_return'] = [s for s, _ in returns]
        
        # Rank by Sharpe ratio
        sharpes = [(s, comparison_results[s]['performance'].get('sharpe_ratio', 0)) for s in strategies]
        sharpes.sort(key=lambda x: x[1], reverse=True)
        rankings['by_sharpe'] = [s for s, _ in sharpes]
        
        # Rank by max drawdown (lower is better)
        drawdowns = [(s, comparison_results[s]['risk'].get('maximum_drawdown', 1)) for s in strategies]
        drawdowns.sort(key=lambda x: x[1])
        rankings['by_drawdown'] = [s for s, _ in drawdowns]
        
        # Overall ranking (weighted combination)
        overall_scores = {}
        for strategy in strategies:
            perf = comparison_results[strategy]['performance']
            risk = comparison_results[strategy]['risk']
            
            # Weighted score: 40% return, 30% sharpe, 30% drawdown
            return_score = perf.get('total_return', 0) * 0.4
            sharpe_score = perf.get('sharpe_ratio', 0) * 0.3
            drawdown_score = (1 - abs(risk.get('maximum_drawdown', 1))) * 0.3
            
            overall_scores[strategy] = return_score + sharpe_score + drawdown_score
        
        overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        rankings['overall'] = [s for s, _ in overall_ranking]
        
        return rankings
    
    def _save_simulation_results(self, results: Dict[str, Any], strategy_type: str):
        """Save simulation results to file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{strategy_type}_simulation_{timestamp}.json"
            filepath = self.results_dir / filename
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Deep convert the results
            def deep_convert(data):
                if isinstance(data, dict):
                    return {k: deep_convert(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [deep_convert(item) for item in data]
                else:
                    return convert_numpy(data)
            
            converted_results = deep_convert(results)
            
            with open(filepath, 'w') as f:
                json.dump(converted_results, f, indent=2, default=str)
            
            logger.info(f"Simulation results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving simulation results: {str(e)}")


# Singleton instance
optimizer = StrategyOptimizer()

def get_strategy_optimizer() -> StrategyOptimizer:
    """Get singleton instance of the strategy optimizer."""
    return optimizer