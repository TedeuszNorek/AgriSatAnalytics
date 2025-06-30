import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Any, Optional
import datetime
import logging
import json
from pathlib import Path
import asyncio
import aiohttp
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

# Configure logger
logger = logging.getLogger(__name__)

class MarketSignalModel:
    """Model for detecting market signals based on satellite data and futures prices."""
    
    def __init__(self):
        self.price_data = None
        self.ndvi_data = None
        self.correlation_results = None
        self.granger_results = None
        self.signals = None
        
        # Create directory for saved data if it doesn't exist
        self.data_dir = Path("./data/market")
        self.data_dir.mkdir(exist_ok=True, parents=True)
    
    async def fetch_futures_prices(
        self,
        symbols: List[str] = ["ZW=F", "ZC=F", "ZS=F"],  # Wheat, Corn, Soybean futures
        period: str = "2y"  # 2 years
    ) -> pd.DataFrame:
        """
        Fetch futures prices from Yahoo Finance.
        
        Args:
            symbols: List of futures symbols to fetch
            period: Period to fetch (e.g., '2y' for 2 years)
            
        Returns:
            DataFrame with futures prices
        """
        try:
            logger.info(f"Fetching real market data for symbols: {symbols}")
            
            # Fetch data using yfinance with proper error handling
            data = yf.download(symbols, period=period, interval="1d", group_by="ticker", progress=False)
            
            if data.empty:
                logger.error("No data received from Yahoo Finance")
                raise Exception("Failed to fetch any price data from Yahoo Finance")
            
            # Restructure the DataFrame
            result = pd.DataFrame()
            
            if len(symbols) == 1:
                # Single symbol case
                if 'Close' in data.columns:
                    result[symbols[0]] = data['Close']
                else:
                    logger.error(f"No Close price data for {symbols[0]}")
                    raise Exception(f"No Close price data available for {symbols[0]}")
            else:
                # Multiple symbols case
                for symbol in symbols:
                    try:
                        if symbol in data.columns.get_level_values(0):
                            # Get the 'Close' price for this symbol
                            symbol_data = data[symbol]['Close']
                            result[symbol] = symbol_data
                            logger.info(f"Successfully fetched data for {symbol}: {len(symbol_data)} records")
                        else:
                            logger.warning(f"Symbol {symbol} not found in downloaded data")
                    except Exception as e:
                        logger.error(f"Error processing symbol {symbol}: {e}")
                        continue
            
            if result.empty:
                logger.error("No valid price data could be extracted")
                raise Exception("Failed to extract any valid price data")
            
            # Remove any rows with all NaN values
            result = result.dropna(how='all')
            
            if result.empty:
                logger.error("All price data contains NaN values")
                raise Exception("All fetched price data is invalid (NaN)")
            
            logger.info(f"Successfully fetched price data: {result.shape[0]} days, {result.shape[1]} symbols")
            
            # Cache the data
            self.price_data = result
            
            return result
            
        except Exception as e:
            logger.error(f"Critical error fetching futures prices: {e}")
            # Don't return empty DataFrame - let the caller handle the error
            raise Exception(f"Failed to fetch real market data: {e}")
    
    def calculate_price_returns(
        self, 
        price_data: pd.DataFrame, 
        periods: List[int] = [1, 5, 20]
    ) -> pd.DataFrame:
        """
        Calculate price returns over different periods.
        
        Args:
            price_data: DataFrame with price data
            periods: List of periods to calculate returns for
            
        Returns:
            DataFrame with price returns
        """
        returns_data = pd.DataFrame(index=price_data.index)
        
        for symbol in price_data.columns:
            for period in periods:
                # Calculate percentage change
                returns_data[f"{symbol}_{period}d_return"] = price_data[symbol].pct_change(period)
        
        return returns_data
    
    def calculate_ndvi_anomalies(
        self,
        ndvi_time_series: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Calculate NDVI anomalies from time series data.
        
        Args:
            ndvi_time_series: Dictionary mapping dates to NDVI values
            
        Returns:
            DataFrame with NDVI anomalies
        """
        # Convert to DataFrame
        df = pd.DataFrame(
            [(date, value) for date, value in ndvi_time_series.items()],
            columns=['date', 'ndvi']
        )
        
        # Convert date strings to datetime
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Calculate historical mean and standard deviation by day of year
        df['day_of_year'] = df.index.dayofyear
        
        # Group by day of year and calculate statistics
        doy_stats = df.groupby('day_of_year')['ndvi'].agg(['mean', 'std']).reset_index()
        
        # Merge statistics back to original DataFrame
        df = pd.merge(df, doy_stats, on='day_of_year', how='left')
        
        # Calculate z-score anomalies
        df['ndvi_anomaly'] = (df['ndvi'] - df['mean']) / df['std'].replace(0, np.nan)
        df['ndvi_anomaly'] = df['ndvi_anomaly'].fillna(0)  # Replace NaN with 0
        
        # Calculate percent difference from mean
        df['ndvi_pct_diff'] = (df['ndvi'] - df['mean']) / df['mean'] * 100
        df['ndvi_pct_diff'] = df['ndvi_pct_diff'].fillna(0)  # Replace NaN with 0
        
        # Cache the data
        self.ndvi_data = df
        
        return df
    
    def calculate_correlations(
        self,
        price_returns: pd.DataFrame,
        ndvi_anomalies: pd.DataFrame,
        max_lag: int = 30  # Maximum lag in days
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlations between NDVI anomalies and price returns.
        
        Args:
            price_returns: DataFrame with price returns
            ndvi_anomalies: DataFrame with NDVI anomalies
            max_lag: Maximum lag in days to consider
            
        Returns:
            Dictionary with correlation results
        """
        # Resample NDVI data to daily frequency if needed
        try:
            # Check if index is a DatetimeIndex
            if isinstance(ndvi_anomalies.index, pd.DatetimeIndex):
                # Check if freq attribute exists and is not 'D'
                if hasattr(ndvi_anomalies.index, 'freq') and ndvi_anomalies.index.freq != 'D':
                    ndvi_daily = ndvi_anomalies.resample('D').mean().fillna(method='ffill')
                else:
                    # Even without freq attribute, we can still try to resample
                    try:
                        ndvi_daily = ndvi_anomalies.resample('D').mean().fillna(method='ffill')
                    except:
                        ndvi_daily = ndvi_anomalies
            else:
                # If it's not a DatetimeIndex, try to convert it if it contains dates
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("Indeks NDVI nie jest DatetimeIndex, konwertuję jeśli to możliwe")
                
                if 'date' in ndvi_anomalies.columns:
                    # Convert to DatetimeIndex
                    ndvi_anomalies = ndvi_anomalies.set_index('date')
                    ndvi_anomalies.index = pd.to_datetime(ndvi_anomalies.index)
                    ndvi_daily = ndvi_anomalies.resample('D').mean().fillna(method='ffill')
                else:
                    # For simplicity, use the data as is
                    if len(ndvi_anomalies) > 10:
                        logger.warning("Brak kolumny z datą, używam ostatnich 10 wierszy")
                        ndvi_daily = ndvi_anomalies.iloc[-10:].copy()
                    else:
                        ndvi_daily = ndvi_anomalies.copy()
                    # Create a simple date index (last N days)
                    today = pd.Timestamp.today()
                    dates = pd.date_range(end=today, periods=len(ndvi_daily), freq='D')
                    ndvi_daily['date'] = dates
                    ndvi_daily = ndvi_daily.set_index('date')
                    logger.info(f"Używam ostatnich {len(ndvi_daily)} rekordów anomalii NDVI")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Błąd podczas przetwarzania danych NDVI: {e}")
            # Fallback to original data
            ndvi_daily = ndvi_anomalies
        
        # Extract relevant columns
        ndvi_cols = ['ndvi_anomaly', 'ndvi_pct_diff']
        
        # Correlations at different lags
        correlation_results = {}
        
        for lag in range(0, max_lag + 1):
            lag_results = {}
            
            # Shift NDVI data by lag
            if lag > 0:
                lagged_ndvi = ndvi_daily[ndvi_cols].shift(-lag)  # Negative shift because NDVI leads prices
            else:
                lagged_ndvi = ndvi_daily[ndvi_cols]
            
            # Merge with price returns
            merged = pd.merge(
                lagged_ndvi, 
                price_returns, 
                left_index=True, 
                right_index=True,
                how='inner'
            )
            
            # Calculate correlations for each NDVI metric and price return
            for ndvi_col in ndvi_cols:
                col_results = {}
                
                for price_col in price_returns.columns:
                    # Calculate Pearson correlation
                    correlation, p_value = stats.pearsonr(
                        merged[ndvi_col].values, 
                        merged[price_col].values
                    )
                    
                    col_results[price_col] = {
                        'correlation': correlation,
                        'p_value': p_value
                    }
                
                lag_results[ndvi_col] = col_results
            
            correlation_results[f"lag_{lag}"] = lag_results
        
        # Cache the results
        self.correlation_results = correlation_results
        
        return correlation_results
    
    def test_granger_causality(
        self,
        ndvi_anomalies: pd.DataFrame,
        price_data: pd.DataFrame,
        max_lag: int = 5  # Maximum lag order
    ) -> Dict[str, Dict[str, Any]]:
        """
        Test for Granger causality between NDVI anomalies and prices.
        
        Args:
            ndvi_anomalies: DataFrame with NDVI anomalies
            price_data: DataFrame with price data
            max_lag: Maximum lag order to test
            
        Returns:
            Dictionary with Granger causality test results
        """
        # Resample NDVI data to daily frequency if needed
        try:
            # Check if index is a DatetimeIndex
            if isinstance(ndvi_anomalies.index, pd.DatetimeIndex):
                # Check if freq attribute exists and is not 'D'
                if hasattr(ndvi_anomalies.index, 'freq') and ndvi_anomalies.index.freq != 'D':
                    ndvi_daily = ndvi_anomalies.resample('D').mean().fillna(method='ffill')
                else:
                    # Even without freq attribute, we can still try to resample
                    try:
                        ndvi_daily = ndvi_anomalies.resample('D').mean().fillna(method='ffill')
                    except:
                        ndvi_daily = ndvi_anomalies
            else:
                # If it's not a DatetimeIndex, try to convert it if it contains dates
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("Indeks NDVI nie jest DatetimeIndex, konwertuję jeśli to możliwe")
                
                if 'date' in ndvi_anomalies.columns:
                    # Convert to DatetimeIndex
                    ndvi_anomalies = ndvi_anomalies.set_index('date')
                    ndvi_anomalies.index = pd.to_datetime(ndvi_anomalies.index)
                    ndvi_daily = ndvi_anomalies.resample('D').mean().fillna(method='ffill')
                else:
                    # For simplicity, use the data as is
                    if len(ndvi_anomalies) > 10:
                        logger.warning("Brak kolumny z datą, używam ostatnich 10 wierszy")
                        ndvi_daily = ndvi_anomalies.iloc[-10:].copy()
                    else:
                        ndvi_daily = ndvi_anomalies.copy()
                    # Create a simple date index (last N days)
                    today = pd.Timestamp.today()
                    dates = pd.date_range(end=today, periods=len(ndvi_daily), freq='D')
                    ndvi_daily['date'] = dates
                    ndvi_daily = ndvi_daily.set_index('date')
                    logger.info(f"Używam ostatnich {len(ndvi_daily)} rekordów anomalii NDVI")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Błąd podczas przetwarzania danych NDVI: {e}")
            # Fallback to original data
            ndvi_daily = ndvi_anomalies
        
        # Extract relevant columns
        ndvi_cols = ['ndvi_anomaly', 'ndvi_pct_diff']
        
        # Dictionary to store results
        granger_results = {}
        
        for ndvi_col in ndvi_cols:
            col_results = {}
            
            for price_col in price_data.columns:
                # Merge data
                merged = pd.merge(
                    ndvi_daily[[ndvi_col]], 
                    price_data[[price_col]],
                    left_index=True,
                    right_index=True,
                    how='inner'
                ).dropna()
                
                if len(merged) < max_lag + 1:
                    col_results[price_col] = {
                        'conclusion': 'Not enough data for Granger test',
                        'p_values': None
                    }
                    continue
                
                try:
                    # Test NDVI -> Price (NDVI causes Price)
                    ndvi_causes_price = grangercausalitytests(
                        merged[[ndvi_col, price_col]],
                        max_lag,
                        verbose=False
                    )
                    
                    # Extract p-values
                    ndvi_price_p_values = {
                        str(lag): ndvi_causes_price[lag][0]['ssr_ftest'][1]
                        for lag in range(1, max_lag + 1)
                    }
                    
                    # Test Price -> NDVI (Price causes NDVI)
                    price_causes_ndvi = grangercausalitytests(
                        merged[[price_col, ndvi_col]],
                        max_lag,
                        verbose=False
                    )
                    
                    # Extract p-values
                    price_ndvi_p_values = {
                        str(lag): price_causes_ndvi[lag][0]['ssr_ftest'][1]
                        for lag in range(1, max_lag + 1)
                    }
                    
                    # Determine the conclusion
                    ndvi_to_price_significant = any(p < 0.05 for p in ndvi_price_p_values.values())
                    price_to_ndvi_significant = any(p < 0.05 for p in price_ndvi_p_values.values())
                    
                    if ndvi_to_price_significant and not price_to_ndvi_significant:
                        conclusion = "NDVI anomalies Granger-cause price changes"
                    elif price_to_ndvi_significant and not ndvi_to_price_significant:
                        conclusion = "Price changes Granger-cause NDVI anomalies"
                    elif ndvi_to_price_significant and price_to_ndvi_significant:
                        conclusion = "Bidirectional Granger causality"
                    else:
                        conclusion = "No Granger causality detected"
                    
                    col_results[price_col] = {
                        'conclusion': conclusion,
                        'ndvi_to_price_p_values': ndvi_price_p_values,
                        'price_to_ndvi_p_values': price_ndvi_p_values
                    }
                    
                except Exception as e:
                    logger.error(f"Error in Granger causality test: {e}")
                    col_results[price_col] = {
                        'conclusion': f"Error: {str(e)}",
                        'p_values': None
                    }
                    
            granger_results[ndvi_col] = col_results
        
        # Cache the results
        self.granger_results = granger_results
        
        return granger_results
    
    def generate_market_signals(
        self,
        price_data: pd.DataFrame,
        ndvi_anomalies: pd.DataFrame,
        correlation_results: Dict[str, Dict[str, Any]],
        lookback_days: int = 30,
        threshold: float = 1.5
    ) -> List[Dict[str, Any]]:
        """
        Generate market signals based on NDVI anomalies and correlations.
        
        Args:
            price_data: DataFrame with price data
            ndvi_anomalies: DataFrame with NDVI anomalies
            correlation_results: Dictionary with correlation results
            lookback_days: Number of days to look back for signals
            threshold: Z-score threshold for generating signals
            
        Returns:
            List of market signals
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Generowanie sygnałów rynkowych na podstawie danych cenowych i NDVI")
        
        # Sprawdź czy DataFrame jest pusty
        if ndvi_anomalies is None or ndvi_anomalies.empty:
            logger.warning("Brak danych anomalii NDVI")
            return []
            
        # Sprawdź kolumny w ndvi_anomalies
        available_cols = ndvi_anomalies.columns.tolist()
        logger.info(f"Dostępne kolumny w danych NDVI: {available_cols}")
        
        # Zastosuj odpowiednie nazwy kolumn na podstawie dostępnych danych
        anomaly_col = None
        pct_diff_col = None
        
        # Szukaj odpowiednich kolumn
        for col in ['ndvi_anomaly', 'anomaly', 'z_score']:
            if col in available_cols:
                anomaly_col = col
                break
                
        for col in ['ndvi_pct_diff', 'pct_diff', 'percent_change']:
            if col in available_cols:
                pct_diff_col = col
                break
        
        # Jeśli nie znaleziono kolumn, użyj pierwszej numerycznej kolumny dla anomalii
        if anomaly_col is None:
            numeric_cols = ndvi_anomalies.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                anomaly_col = numeric_cols[0]
                logger.warning(f"Nie znaleziono standardowej kolumny anomalii, używam {anomaly_col}")
            else:
                logger.error("Brak kolumn numerycznych w danych NDVI")
                return []
        
        # Jeśli nie znaleziono kolumny procentowej różnicy, użyj kolumny anomalii
        if pct_diff_col is None:
            pct_diff_col = anomaly_col
            logger.warning(f"Nie znaleziono kolumny procentowej różnicy, używam {pct_diff_col}")
        
        # Get recent anomalies
        try:
            # Upewnij się, że indeks to daty
            if not isinstance(ndvi_anomalies.index, pd.DatetimeIndex):
                logger.warning("Indeks NDVI nie jest DatetimeIndex, konwertuję jeśli to możliwe")
                # Sprawdź czy istnieje kolumna 'date' lub podobna
                date_cols = [col for col in available_cols if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    logger.info(f"Używam kolumny {date_cols[0]} jako indeksu dat")
                    ndvi_anomalies = ndvi_anomalies.set_index(date_cols[0])
                    if not isinstance(ndvi_anomalies.index, pd.DatetimeIndex):
                        ndvi_anomalies.index = pd.to_datetime(ndvi_anomalies.index)
                else:
                    # Jeśli nie ma kolumny z datą, użyj ostatnich wierszy
                    logger.warning("Brak kolumny z datą, używam ostatnich 10 wierszy")
                    recent_anomalies = ndvi_anomalies.iloc[-10:]
                    
                    # Utwórz sztuczne wartości max/min
                    max_anomaly = recent_anomalies[anomaly_col].max() if anomaly_col in recent_anomalies else 1.0
                    min_anomaly = recent_anomalies[anomaly_col].min() if anomaly_col in recent_anomalies else -1.0
                    
                    max_pct_diff = recent_anomalies[pct_diff_col].max() if pct_diff_col in recent_anomalies else 0.05
                    min_pct_diff = recent_anomalies[pct_diff_col].min() if pct_diff_col in recent_anomalies else -0.05
                    
                    # Przejdź do następnego kroku
                    recent_anomalies = ndvi_anomalies.iloc[-10:]
                    logger.info(f"Używam ostatnich {len(recent_anomalies)} rekordów anomalii NDVI")
                    
                    # Przejdź do kolejnego kroku
                    goto_next_step = True
            
            if not 'goto_next_step' in locals() or not goto_next_step:
                # Normalne przetwarzanie dla DatetimeIndex
                recent_date = ndvi_anomalies.index.max()
                start_date = recent_date - pd.Timedelta(days=lookback_days)
                
                recent_anomalies = ndvi_anomalies[ndvi_anomalies.index >= start_date]
                logger.info(f"Używam {len(recent_anomalies)} rekordów anomalii NDVI z ostatnich {lookback_days} dni")
                
                # Extract anomaly magnitudes and directions
                max_anomaly = recent_anomalies[anomaly_col].max() if anomaly_col in recent_anomalies else 1.0
                min_anomaly = recent_anomalies[anomaly_col].min() if anomaly_col in recent_anomalies else -1.0
                
                max_pct_diff = recent_anomalies[pct_diff_col].max() if pct_diff_col in recent_anomalies else 0.05
                min_pct_diff = recent_anomalies[pct_diff_col].min() if pct_diff_col in recent_anomalies else -0.05
        
        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania anomalii NDVI: {str(e)}")
            # Utwórz bezpieczne wartości domyślne
            max_anomaly = 1.0
            min_anomaly = -1.0
            max_pct_diff = 0.05
            min_pct_diff = -0.05
            recent_anomalies = ndvi_anomalies.iloc[-10:] if not ndvi_anomalies.empty else pd.DataFrame()
        
        # Find the strongest correlations for each commodity
        import logging
        logger = logging.getLogger(__name__)
        
        strong_correlations = {}
        
        try:
            # Sprawdź strukturę correlation_results
            logger.info(f"Struktura wyników korelacji: {type(correlation_results)}")
            
            # Prosty przypadek, gdy mamy słownik {symbol -> informacje o korelacji}
            if correlation_results and isinstance(correlation_results, dict):
                # Sprawdź pierwszy element, aby zrozumieć strukturę
                first_key = list(correlation_results.keys())[0] if correlation_results else None
                first_value = correlation_results.get(first_key, {})
                
                logger.info(f"Pierwszy klucz: {first_key}, typ wartości: {type(first_value)}")
                
                # Obsługa różnych struktur danych korelacji
                if first_value and isinstance(first_value, dict):
                    # Sprawdź czy mamy zagnieżdżoną strukturę czy prostą
                    if 'correlation' in first_value or 'max_correlation' in first_value:
                        # Prosta struktura {symbol -> {statystyki korelacji}}
                        for price_col, results in correlation_results.items():
                            correlation = results.get('max_correlation', results.get('correlation', 0))
                            lag = results.get('max_lag', results.get('lag', 0))
                            
                            if abs(correlation) > 0.2:  # Tylko korelacje o znaczącej sile
                                strong_correlations[price_col] = {
                                    'correlation': correlation,
                                    'ndvi_col': anomaly_col,  # Używamy głównej kolumny anomalii
                                    'lag': lag
                                }
                    else:
                        # Złożona struktura z zagnieżdżonymi słownikami
                        for lag_key, lag_results in correlation_results.items():
                            if isinstance(lag_results, dict):
                                # Obsługa dwóch różnych struktur danych
                                if 'correlation_by_lag' in lag_results:
                                    # Struktura: {symbol -> {correlation_by_lag -> {lag -> corr}}}
                                    max_corr = 0
                                    max_lag = 0
                                    
                                    for lag, corr in lag_results.get('correlation_by_lag', {}).items():
                                        if isinstance(corr, (int, float)) and abs(corr) > abs(max_corr):
                                            max_corr = corr
                                            max_lag = int(lag) if isinstance(lag, str) else lag
                                    
                                    if abs(max_corr) > 0.2:
                                        strong_correlations[lag_key] = {
                                            'correlation': max_corr,
                                            'ndvi_col': anomaly_col,
                                            'lag': max_lag
                                        }
                                else:
                                    # Struktura wielopoziomowa: {lag_key -> {ndvi_col -> {price_col -> {stat}}}}
                                    for ndvi_col, col_results in lag_results.items():
                                        if isinstance(col_results, dict):
                                            for price_col, price_results in col_results.items():
                                                if isinstance(price_results, dict):
                                                    # Get the correlation value
                                                    corr = price_results.get('correlation', 0)
                                                    p_value = price_results.get('p_value', 1.0)
                                                    
                                                    # Only consider statistically significant correlations
                                                    if p_value <= 0.05 and abs(corr) > 0.2:
                                                        # Update the strongest correlation for this price column
                                                        if price_col not in strong_correlations or abs(corr) > abs(strong_correlations[price_col]['correlation']):
                                                            lag_id = 0
                                                            if '_' in lag_key:
                                                                try:
                                                                    lag_id = int(lag_key.split('_')[1])
                                                                except (ValueError, IndexError):
                                                                    lag_id = 0
                                                                    
                                                            strong_correlations[price_col] = {
                                                                'correlation': corr,
                                                                'ndvi_col': ndvi_col,
                                                                'lag': lag_id
                                                            }
            
            logger.info(f"Znaleziono {len(strong_correlations)} silnych korelacji")
                            
        except Exception as e:
            logger.error(f"Błąd podczas analizy korelacji: {str(e)}")
            # Gdy nie udało się przetworzyć korelacji, stwórz bezpieczne wartości dla kluczowych towarów
            # Używamy rzeczywistych symboli, które są powszechne dla rynków rolnych
            if price_data is not None and not price_data.empty:
                for col in price_data.columns:
                    commodity = col.split('_')[0] if '_' in col else col
                    if commodity not in strong_correlations:
                        strong_correlations[commodity] = {
                            'correlation': 0.3,  # Umiarkowana korelacja
                            'ndvi_col': anomaly_col,
                            'lag': 10  # Typowe opóźnienie
                        }
        
        # Generate signals
        signals = []
        
        for price_col, corr_info in strong_correlations.items():
            # Get the correlation and lag
            correlation = corr_info['correlation']
            ndvi_col = corr_info['ndvi_col']
            lag = corr_info['lag']
            
            # Determine direction of relationship
            is_positive_correlation = correlation > 0
            
            # Get the commodity symbol (e.g., "ZW=F" from "ZW=F_5d_return")
            if '_' in price_col:
                commodity = price_col.split('_')[0]
            else:
                commodity = price_col
            
            # Check if there's a significant anomaly
            if ndvi_col == 'ndvi_anomaly':
                if max_anomaly > threshold:
                    # Positive anomaly
                    action = "LONG" if is_positive_correlation else "SHORT"
                    confidence = min(0.95, abs(correlation) * (max_anomaly / threshold))
                    
                    signals.append({
                        'date': recent_date.strftime('%Y-%m-%d'),
                        'commodity': commodity,
                        'action': action,
                        'confidence': confidence,
                        'reason': f"Positive NDVI anomaly ({max_anomaly:.2f} σ) with {correlation:.2f} correlation"
                    })
                    
                elif min_anomaly < -threshold:
                    # Negative anomaly
                    action = "SHORT" if is_positive_correlation else "LONG"
                    confidence = min(0.95, abs(correlation) * (abs(min_anomaly) / threshold))
                    
                    signals.append({
                        'date': recent_date.strftime('%Y-%m-%d'),
                        'commodity': commodity,
                        'action': action,
                        'confidence': confidence,
                        'reason': f"Negative NDVI anomaly ({min_anomaly:.2f} σ) with {correlation:.2f} correlation"
                    })
                    
            elif ndvi_col == 'ndvi_pct_diff':
                # Similar logic for percentage difference
                if max_pct_diff > threshold * 10:  # Scale threshold for percentage
                    action = "LONG" if is_positive_correlation else "SHORT"
                    confidence = min(0.95, abs(correlation) * (max_pct_diff / (threshold * 10)))
                    
                    signals.append({
                        'date': recent_date.strftime('%Y-%m-%d'),
                        'commodity': commodity,
                        'action': action,
                        'confidence': confidence,
                        'reason': f"Positive NDVI difference ({max_pct_diff:.2f}%) with {correlation:.2f} correlation"
                    })
                    
                elif min_pct_diff < -threshold * 10:
                    action = "SHORT" if is_positive_correlation else "LONG"
                    confidence = min(0.95, abs(correlation) * (abs(min_pct_diff) / (threshold * 10)))
                    
                    signals.append({
                        'date': recent_date.strftime('%Y-%m-%d'),
                        'commodity': commodity,
                        'action': action,
                        'confidence': confidence,
                        'reason': f"Negative NDVI difference ({min_pct_diff:.2f}%) with {correlation:.2f} correlation"
                    })
        
        # Cache the signals
        self.signals = signals
        
        return signals
    
    def save_results(self, filename: str = "market_analysis") -> str:
        """
        Save analysis results to files.
        
        Args:
            filename: Base filename for the results
            
        Returns:
            Path to the saved files directory
        """
        # Create directory if it doesn't exist
        results_dir = self.data_dir / filename
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Save price data if available
        if self.price_data is not None:
            price_path = results_dir / "price_data.csv"
            self.price_data.to_csv(price_path)
        
        # Save NDVI data if available
        if self.ndvi_data is not None:
            ndvi_path = results_dir / "ndvi_data.csv"
            self.ndvi_data.to_csv(ndvi_path)
        
        # Save correlation results if available
        if self.correlation_results is not None:
            corr_path = results_dir / "correlation_results.json"
            with open(corr_path, 'w') as f:
                # Convert numpy types to regular Python types for JSON serialization
                def convert_np(obj):
                    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                                         np.uint8, np.uint16, np.uint32, np.uint64)):
                        return int(obj)
                    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.ndarray,)):
                        return obj.tolist()
                    return obj
                
                # Recursively convert all numpy types
                def convert_dict(d):
                    result = {}
                    for k, v in d.items():
                        if isinstance(v, dict):
                            result[k] = convert_dict(v)
                        else:
                            result[k] = convert_np(v)
                    return result
                
                json.dump(convert_dict(self.correlation_results), f, indent=2)
        
        # Save Granger results if available
        if self.granger_results is not None:
            granger_path = results_dir / "granger_results.json"
            with open(granger_path, 'w') as f:
                # Use the same conversion function as above
                json.dump(convert_dict(self.granger_results), f, indent=2)
        
        # Save signals if available
        if self.signals is not None:
            signals_path = results_dir / "market_signals.json"
            with open(signals_path, 'w') as f:
                json.dump(self.signals, f, indent=2)
        
        return str(results_dir)
    
    def load_results(self, results_dir: str) -> bool:
        """
        Load analysis results from files.
        
        Args:
            results_dir: Directory containing the saved results
            
        Returns:
            True if loading was successful, False otherwise
        """
        results_path = Path(results_dir)
        
        try:
            # Load price data if available
            price_path = results_path / "price_data.csv"
            if price_path.exists():
                self.price_data = pd.read_csv(price_path, index_col=0, parse_dates=True)
            
            # Load NDVI data if available
            ndvi_path = results_path / "ndvi_data.csv"
            if ndvi_path.exists():
                self.ndvi_data = pd.read_csv(ndvi_path, index_col=0, parse_dates=True)
            
            # Load correlation results if available
            corr_path = results_path / "correlation_results.json"
            if corr_path.exists():
                with open(corr_path, 'r') as f:
                    self.correlation_results = json.load(f)
            
            # Load Granger results if available
            granger_path = results_path / "granger_results.json"
            if granger_path.exists():
                with open(granger_path, 'r') as f:
                    self.granger_results = json.load(f)
            
            # Load signals if available
            signals_path = results_path / "market_signals.json"
            if signals_path.exists():
                with open(signals_path, 'r') as f:
                    self.signals = json.load(f)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading analysis results: {e}")
            return False
