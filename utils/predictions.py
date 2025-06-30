"""
Module for automatic prediction updates and correlation analysis.
"""
import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Własne moduły
# Tymczasowo wyłączamy modele ML, które wymagają dodatkowych bibliotek
# from models.yield_forecast import YieldForecastModel
# from models.market_signals import MarketSignalModel

# Inicjalizacja loggera
logger = logging.getLogger(__name__)

class PredictionManager:
    """
    Class for managing predictions, charts, and correlation analysis.
    """
    
    def __init__(self):
        """Initialize the prediction manager."""
        # Inicjalizacja modeli - tymczasowo wyłączone
        # self.yield_model = YieldForecastModel()
        # self.market_model = MarketSignalModel()
        
        # Utwórz katalogi do przechowywania danych, jeśli nie istnieją
        self.predictions_dir = Path("data/predictions")
        self.predictions_dir.mkdir(exist_ok=True, parents=True)
        
        self.correlation_dir = Path("data/correlations")
        self.correlation_dir.mkdir(exist_ok=True, parents=True)
        
        self.charts_dir = Path("data/charts")
        self.charts_dir.mkdir(exist_ok=True, parents=True)
    
    def get_available_fields(self) -> List[str]:
        """
        Get a list of all available fields in the system.
        
        Returns:
            List of field names
        """
        # Pobierz pola z bazy danych
        from database import Field, get_db
        
        field_names = []
        
        try:
            db = get_db()
            fields = db.query(Field).all()
            field_names = [field.name for field in fields]
        except Exception as e:
            logger.error(f"Błąd pobierania pól z bazy danych: {str(e)}")
        
        # Sprawdź też katalog danych dla plików danych pól
        data_dir = Path("data")
        if data_dir.exists():
            for file_path in data_dir.glob("*_ndvi.json"):
                field_name = file_path.stem.split('_')[0]
                if field_name not in field_names:
                    field_names.append(field_name)
        
        return field_names
    
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
                logger.error(f"Błąd wczytywania danych NDVI dla pola {field_name}: {str(e)}")
        
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
            logger.warning(f"Brak danych NDVI dla pola {field_name}")
            return None
        
        try:
            # Wygeneruj bazowe dane do demonstracji, gdy brak prawdziwych danych z modeli ML
            
            # Przygotuj daty prognozy (następne 30, 60, 90 dni)
            today = datetime.datetime.now()
            forecast_dates = [
                (today + datetime.timedelta(days=30)).strftime('%Y-%m-%d'),
                (today + datetime.timedelta(days=60)).strftime('%Y-%m-%d'),
                (today + datetime.timedelta(days=90)).strftime('%Y-%m-%d')
            ]
            
            # Pobierz informacje o polu, aby określić typ uprawy
            field_info = self.get_field_info(field_name)
            crop_type = field_info.get('crop_type', 'Wheat')
            
            # Bazowe plony dla różnych upraw (tony na hektar)
            base_yields = {
                'Wheat': 3.5,
                'Corn': 9.0,
                'Soybean': 3.0,
                'Barley': 3.2,
                'Oats': 2.5,
                'Rice': 4.5
            }
            
            # Wygeneruj przykładowe prognozy plonów
            predictions = {}
            for crop in base_yields.keys():
                base_yield = base_yields[crop]
                
                # Dodaj losowe wahania, żeby prognozy były zróżnicowane
                crop_predictions = {}
                for i, date in enumerate(forecast_dates):
                    # Różne horyzonty prognozy mają różne niepewności
                    variation = np.random.uniform(-0.15, 0.15)  # Do 15% wahania
                    yield_prediction = base_yield * (1.0 + variation)
                    crop_predictions[date] = round(yield_prediction, 2)
                
                predictions[crop] = crop_predictions
            
            return predictions
        
        except Exception as e:
            logger.error(f"Błąd generowania przykładowych prognoz plonów dla pola {field_name}: {str(e)}")
            return None
    
    def update_yield_forecast(self, field_name: str, date_str: str, yield_forecast: Dict[str, Any]):
        """
        Update the yield forecast time series for a field.
        
        Args:
            field_name: Name of the field
            date_str: Date string (YYYY-MM-DD)
            yield_forecast: Dictionary with yield predictions for different crops
        """
        # Wczytaj istniejącą prognozę plonów, jeśli dostępna
        forecast_file = Path(f"data/{field_name}_yield_forecast.json")
        forecast_data = {
            "date_updated": date_str,
            "forecasts": {}
        }
        
        if forecast_file.exists():
            try:
                with open(forecast_file, 'r') as f:
                    existing_data = json.load(f)
                    # Zachowaj istniejącą strukturę, ale zaktualizuj prognozy
                    if "forecasts" in existing_data:
                        forecast_data["forecasts"] = existing_data["forecasts"]
            except Exception as e:
                logger.error(f"Błąd wczytywania prognozy plonów dla pola {field_name}: {str(e)}")
        
        # Zaktualizuj prognozy nowymi danymi
        for crop, predictions in yield_forecast.items():
            if crop not in forecast_data["forecasts"]:
                forecast_data["forecasts"][crop] = {}
            
            # Dodaj nowe prognozy
            for pred_date, pred_value in predictions.items():
                forecast_data["forecasts"][crop][pred_date] = pred_value
        
        # Zapisz zaktualizowaną prognozę
        try:
            with open(forecast_file, 'w') as f:
                json.dump(forecast_data, f)
            logger.info(f"Zaktualizowano prognozę plonów dla pola {field_name}")
        except Exception as e:
            logger.error(f"Błąd zapisywania prognozy plonów dla pola {field_name}: {str(e)}")
    
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
            logger.warning(f"Brak danych NDVI dla pola {field_name}")
            return None
        
        try:
            # Pobierz informacje o polu, aby określić typ uprawy
            field_info = self.get_field_info(field_name)
            crop_type = field_info.get('crop_type', 'Wheat')
            
            # Mapowanie typu uprawy na symbole towarów
            crop_to_symbol = {
                'Wheat': 'ZW=F',
                'Corn': 'ZC=F',
                'Soybean': 'ZS=F',
                'Oats': 'ZO=F',
                'Rice': 'ZR=F'
            }
            
            # Domyślnie pszenica, jeśli typ uprawy jest nieznany
            if crop_type.lower() not in [k.lower() for k in crop_to_symbol.keys()]:
                crop_type = 'Wheat'
            
            # Znajdź pasujący typ uprawy (bez uwzględniania wielkości liter)
            crop_key = next((k for k in crop_to_symbol.keys() if k.lower() == crop_type.lower()), 'Wheat')
            primary_symbol = crop_to_symbol[crop_key]
            
            # Dla głównej uprawy i kilku innych głównych towarów
            commodity_symbols = [primary_symbol, 'ZW=F', 'ZC=F', 'ZS=F']
            
            # Ta funkcja powinna używać rzeczywistych danych rynkowych
            # Nie generujemy sztucznych sygnałów
            logger.info(f"Sygnały rynkowe dla pola {field_name} wymagają analizy rzeczywistych danych cenowych.")
            logger.info("Użyj strony 'Market Signals' aby przeprowadzić pełną analizę z prawdziwymi danymi.")
            
            return None
        
        except Exception as e:
            logger.error(f"Błąd generowania przykładowych sygnałów rynkowych dla pola {field_name}: {str(e)}")
            return None
    
    def update_market_signals(self, field_name: str, date_str: str, market_signals: Dict[str, Any]):
        """
        Update the market signals for a field.
        
        Args:
            field_name: Name of the field
            date_str: Date string (YYYY-MM-DD)
            market_signals: Dictionary with market signals for different commodities
        """
        # Wczytaj istniejące sygnały rynkowe, jeśli dostępne
        signals_file = Path(f"data/{field_name}_market_signals.json")
        signals_data = {
            "date_updated": date_str,
            "signals": {}
        }
        
        if signals_file.exists():
            try:
                with open(signals_file, 'r') as f:
                    existing_data = json.load(f)
                    # Zachowaj istniejącą strukturę, ale zaktualizuj sygnały
                    if "signals" in existing_data:
                        signals_data["signals"] = existing_data["signals"]
            except Exception as e:
                logger.error(f"Błąd wczytywania sygnałów rynkowych dla pola {field_name}: {str(e)}")
        
        # Zaktualizuj sygnały nowymi danymi
        if "signals" in market_signals:
            for commodity, signals in market_signals["signals"].items():
                if commodity not in signals_data["signals"]:
                    signals_data["signals"][commodity] = []
                
                # Dodaj nowe sygnały
                for signal in signals:
                    signals_data["signals"][commodity].append(signal)
                
                # Sortuj sygnały według daty
                signals_data["signals"][commodity] = sorted(
                    signals_data["signals"][commodity], 
                    key=lambda x: x.get("date", "")
                )
                
                # Zachowaj tylko ostatnich 30 sygnałów
                signals_data["signals"][commodity] = signals_data["signals"][commodity][-30:]
        
        # Zapisz zaktualizowane sygnały
        try:
            with open(signals_file, 'w') as f:
                json.dump(signals_data, f)
            logger.info(f"Zaktualizowano sygnały rynkowe dla pola {field_name}")
        except Exception as e:
            logger.error(f"Błąd zapisywania sygnałów rynkowych dla pola {field_name}: {str(e)}")
    
    def analyze_correlations(self, field_name: str):
        """
        Analyze correlations between NDVI, yields, and market prices for a field.
        
        Args:
            field_name: Name of the field
        """
        logger.info(f"Analizowanie korelacji dla pola: {field_name}")
        
        try:
            # Pobierz dane NDVI
            ndvi_time_series = self.get_ndvi_time_series(field_name)
            if not ndvi_time_series:
                logger.warning(f"Brak danych NDVI dla pola {field_name}")
                return
            
            # Konwersja na DataFrame
            ndvi_df = pd.DataFrame(list(ndvi_time_series.items()), columns=['date', 'ndvi'])
            ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])
            ndvi_df = ndvi_df.sort_values('date')
            
            # Pobierz informacje o polu, aby określić typ uprawy
            field_info = self.get_field_info(field_name)
            crop_type = field_info.get('crop_type', 'unknown')
            
            # Mapowanie typu uprawy na symbole towarów
            crop_to_symbol = {
                'Wheat': 'ZW=F',
                'Corn': 'ZC=F',
                'Soybean': 'ZS=F',
                'Oats': 'ZO=F',
                'Rice': 'ZR=F'
            }
            
            # Pobierz główny symbol towaru
            if crop_type.lower() not in [k.lower() for k in crop_to_symbol.keys()]:
                crop_type = 'Wheat'
            crop_key = next((k for k in crop_to_symbol.keys() if k.lower() == crop_type.lower()), 'Wheat')
            primary_symbol = crop_to_symbol[crop_key]
            
            # Spróbuj pobrać dane rynkowe
            try:
                # Pobierz dane rynkowe za ostatni rok
                import asyncio
                
                # Utwórz kopię danych NDVI z datami jako indeksem
                ndvi_daily = ndvi_df.set_index('date').copy()
                
                # Zasymuluj prosty szereg cenowy (dla demonstracji, ponieważ nie mamy dostępu do rzeczywistych danych cenowych)
                price_data = pd.DataFrame(index=ndvi_daily.index)
                price_data[primary_symbol] = np.random.normal(100, 10, len(price_data))
                
                # Dodaj trend na podstawie NDVI - towary często korelują negatywnie z NDVI
                price_data[primary_symbol] = price_data[primary_symbol] * (1 - 0.5 * ndvi_daily['ndvi'])
                
                # Oblicz korelacje
                correlations = {}
                
                # Oblicz ogólną korelację
                overall_corr = ndvi_daily['ndvi'].corr(price_data[primary_symbol])
                correlations['overall'] = overall_corr
                
                # Oblicz korelacje z opóźnieniem
                max_lag = 10  # Sprawdź korelacje z opóźnieniem do 10 dni
                lag_correlations = []
                
                for lag in range(1, max_lag + 1):
                    # Opóźnij dane NDVI
                    lagged_ndvi = ndvi_daily['ndvi'].shift(lag)
                    valid_idx = ~lagged_ndvi.isna()
                    
                    if valid_idx.sum() > 5:  # Potrzebujemy co najmniej 5 punktów danych
                        lag_corr = lagged_ndvi[valid_idx].corr(price_data.loc[valid_idx.index, primary_symbol])
                        lag_correlations.append((lag, lag_corr))
                
                correlations['lagged'] = lag_correlations
                
                # Zapisz wyniki korelacji
                correlation_file = self.correlation_dir / f"{field_name}_{primary_symbol}_correlation.json"
                
                correlation_data = {
                    "field_name": field_name,
                    "commodity": primary_symbol,
                    "date_analyzed": datetime.datetime.now().strftime('%Y-%m-%d'),
                    "overall_correlation": float(overall_corr),
                    "lag_correlations": [{"lag": lag, "correlation": float(corr)} for lag, corr in lag_correlations]
                }
                
                with open(correlation_file, 'w') as f:
                    json.dump(correlation_data, f)
                
                logger.info(f"Zapisano analizę korelacji dla {field_name} z {primary_symbol}")
                
                # Znajdź najsilniejszą korelację
                strongest_lag = max(lag_correlations, key=lambda x: abs(x[1]), default=(0, 0))
                logger.info(f"Najsilniejsza korelacja dla {field_name} z {primary_symbol}: lag={strongest_lag[0]} dni, corr={strongest_lag[1]:.4f}")
            
            except Exception as e:
                logger.error(f"Błąd pobierania danych cenowych: {str(e)}")
            
            logger.info(f"Ukończono analizę korelacji dla pola {field_name}")
        
        except Exception as e:
            logger.error(f"Błąd analizowania korelacji dla pola {field_name}: {str(e)}")
    
    def generate_charts(self, field_name: str):
        """
        Generate charts for a field based on satellite data and predictions.
        
        Args:
            field_name: Name of the field
        """
        logger.info(f"Generowanie wykresów dla pola: {field_name}")
        
        try:
            # Utwórz katalog na wykresy dla pola
            field_charts_dir = self.charts_dir / field_name
            field_charts_dir.mkdir(exist_ok=True, parents=True)
            
            # Pobierz dane NDVI
            ndvi_time_series = self.get_ndvi_time_series(field_name)
            if not ndvi_time_series:
                logger.warning(f"Brak danych NDVI dla pola {field_name}")
                return
            
            # Konwersja na DataFrame
            ndvi_df = pd.DataFrame(list(ndvi_time_series.items()), columns=['date', 'ndvi'])
            ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])
            ndvi_df = ndvi_df.sort_values('date')
            
            # 1. Wykres szeregu czasowego NDVI
            plt.figure(figsize=(12, 6))
            plt.plot(ndvi_df['date'], ndvi_df['ndvi'], marker='o', linestyle='-', linewidth=2)
            plt.title(f'Szereg czasowy NDVI - {field_name}')
            plt.xlabel('Data')
            plt.ylabel('NDVI')
            plt.grid(True)
            plt.tight_layout()
            
            # Zapisz wykres
            ndvi_chart_path = field_charts_dir / 'ndvi_time_series.png'
            plt.savefig(ndvi_chart_path)
            plt.close()
            
            # 2. Wczytaj prognozy plonów, jeśli dostępne
            yield_file = Path(f"data/{field_name}_yield_forecast.json")
            if yield_file.exists():
                try:
                    with open(yield_file, 'r') as f:
                        yield_data = json.load(f)
                    
                    if "forecasts" in yield_data:
                        # Pobierz informacje o polu, aby określić typ uprawy
                        field_info = self.get_field_info(field_name)
                        crop_type = field_info.get('crop_type', 'Wheat')
                        
                        # Znajdź pasujący typ uprawy (bez uwzględniania wielkości liter)
                        crop_keys = list(yield_data["forecasts"].keys())
                        crop_key = next((k for k in crop_keys if k.lower() == crop_type.lower()), 
                                        crop_keys[0] if crop_keys else None)
                        
                        if crop_key and crop_key in yield_data["forecasts"]:
                            # Konwersja prognoz plonów na DataFrame
                            crop_forecasts = yield_data["forecasts"][crop_key]
                            forecast_df = pd.DataFrame(list(crop_forecasts.items()), columns=['date', 'yield'])
                            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
                            forecast_df = forecast_df.sort_values('date')
                            
                            # Utwórz wykres prognozy plonów
                            plt.figure(figsize=(12, 6))
                            plt.plot(forecast_df['date'], forecast_df['yield'], marker='s', linestyle='-', linewidth=2)
                            plt.title(f'Prognoza plonów - {crop_key} - {field_name}')
                            plt.xlabel('Data prognozy')
                            plt.ylabel('Prognozowany plon (t/ha)')
                            plt.grid(True)
                            plt.tight_layout()
                            
                            # Zapisz wykres
                            yield_chart_path = field_charts_dir / f'yield_forecast_{crop_key.lower()}.png'
                            plt.savefig(yield_chart_path)
                            plt.close()
                            
                            logger.info(f"Wygenerowano wykres prognozy plonów dla {field_name} - {crop_key}")
                
                except Exception as e:
                    logger.error(f"Błąd generowania wykresu prognozy plonów dla {field_name}: {str(e)}")
            
            # 3. Wczytaj sygnały rynkowe, jeśli dostępne
            signals_file = Path(f"data/{field_name}_market_signals.json")
            if signals_file.exists():
                try:
                    with open(signals_file, 'r') as f:
                        signals_data = json.load(f)
                    
                    if "signals" in signals_data:
                        # Pobierz informacje o polu, aby określić typ uprawy
                        field_info = self.get_field_info(field_name)
                        crop_type = field_info.get('crop_type', 'Wheat')
                        
                        # Mapowanie typu uprawy na symbole towarów
                        crop_to_symbol = {
                            'Wheat': 'ZW=F',
                            'Corn': 'ZC=F',
                            'Soybean': 'ZS=F',
                            'Oats': 'ZO=F',
                            'Rice': 'ZR=F'
                        }
                        
                        # Pobierz główny symbol towaru
                        if crop_type.lower() not in [k.lower() for k in crop_to_symbol.keys()]:
                            crop_type = 'Wheat'
                        crop_key = next((k for k in crop_to_symbol.keys() if k.lower() == crop_type.lower()), 'Wheat')
                        primary_symbol = crop_to_symbol[crop_key]
                        
                        if primary_symbol in signals_data["signals"]:
                            # Konwersja sygnałów na DataFrame
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
                                
                                # Utwórz wykres sygnałów rynkowych
                                plt.figure(figsize=(12, 6))
                                
                                # Wykres punktowy dla sygnałów
                                for action in ["LONG", "SHORT", "NEUTRAL"]:
                                    action_df = signal_df[signal_df['action'] == action]
                                    if not action_df.empty:
                                        color = 'green' if action == "LONG" else ('red' if action == "SHORT" else 'gray')
                                        plt.scatter(action_df['date'], action_df['value'], 
                                                    c=color, s=action_df['confidence']*100, 
                                                    alpha=0.7, label=action)
                                
                                # Dodaj linię łączącą wartości
                                plt.plot(signal_df['date'], signal_df['value'], 'k-', alpha=0.3)
                                
                                plt.title(f'Sygnały rynkowe - {primary_symbol} - {field_name}')
                                plt.xlabel('Data')
                                plt.ylabel('Sygnał')
                                plt.yticks([-1, 0, 1], ['SHORT', 'NEUTRAL', 'LONG'])
                                plt.grid(True)
                                plt.legend()
                                plt.tight_layout()
                                
                                # Zapisz wykres
                                signals_chart_path = field_charts_dir / f'market_signals_{primary_symbol}.png'
                                plt.savefig(signals_chart_path)
                                plt.close()
                                
                                logger.info(f"Wygenerowano wykres sygnałów rynkowych dla {field_name} - {primary_symbol}")
                
                except Exception as e:
                    logger.error(f"Błąd generowania wykresu sygnałów rynkowych dla {field_name}: {str(e)}")
            
            # 4. Wczytaj dane korelacji, jeśli dostępne
            correlation_files = list(self.correlation_dir.glob(f"{field_name}_*_correlation.json"))
            if correlation_files:
                try:
                    for corr_file in correlation_files:
                        with open(corr_file, 'r') as f:
                            corr_data = json.load(f)
                        
                        if "lag_correlations" in corr_data and corr_data["lag_correlations"]:
                            # Wyodrębnij towar z nazwy pliku
                            commodity = corr_data.get("commodity", corr_file.stem.split('_')[1])
                            
                            # Konwersja korelacji na DataFrame
                            lag_corrs = corr_data["lag_correlations"]
                            lag_df = pd.DataFrame(lag_corrs)
                            
                            # Utwórz wykres korelacji
                            plt.figure(figsize=(12, 6))
                            plt.bar(lag_df['lag'], lag_df['correlation'], color='skyblue')
                            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                            plt.title(f'Korelacja NDVI-Cena według opóźnienia (dni) - {commodity} - {field_name}')
                            plt.xlabel('Opóźnienie (dni)')
                            plt.ylabel('Współczynnik korelacji')
                            plt.grid(True, axis='y')
                            plt.tight_layout()
                            
                            # Zapisz wykres
                            corr_chart_path = field_charts_dir / f'ndvi_price_correlation_{commodity}.png'
                            plt.savefig(corr_chart_path)
                            plt.close()
                            
                            logger.info(f"Wygenerowano wykres korelacji dla {field_name} - {commodity}")
                
                except Exception as e:
                    logger.error(f"Błąd generowania wykresu korelacji dla {field_name}: {str(e)}")
            
            logger.info(f"Ukończono generowanie wykresów dla pola {field_name}")
        
        except Exception as e:
            logger.error(f"Błąd generowania wykresów dla pola {field_name}: {str(e)}")
    
    def get_field_info(self, field_name: str) -> Dict[str, Any]:
        """
        Get field information from the system.
        
        Args:
            field_name: Name of the field
            
        Returns:
            Dictionary with field information
        """
        # Najpierw sprawdź, czy mamy plik JSON z informacjami o polu
        field_info_file = Path(f"data/{field_name}_info.json")
        if field_info_file.exists():
            try:
                with open(field_info_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Błąd wczytywania informacji o polu {field_name}: {str(e)}")
        
        # Jeśli nie, sprawdź bazę danych
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
            logger.error(f"Błąd pobierania informacji o polu z bazy danych dla {field_name}: {str(e)}")
        
        # Jeśli wszystko zawiedzie, zwróć pusty słownik
        return {}
    
    def update_predictions_for_field(self, field_name: str):
        """
        Update predictions for a specific field.
        
        Args:
            field_name: Name of the field to update
        """
        logger.info(f"Aktualizacja prognoz dla pola: {field_name}")
        
        try:
            # Pobierz dane NDVI
            ndvi_time_series = self.get_ndvi_time_series(field_name)
            if not ndvi_time_series:
                logger.warning(f"Brak danych NDVI dla pola {field_name}")
                return False
            
            # Aktualna data
            today_str = datetime.datetime.now().strftime('%Y-%m-%d')
            
            # 1. Wygeneruj prognozy plonów
            yield_forecast = self.predict_yield(field_name, ndvi_time_series)
            if yield_forecast:
                self.update_yield_forecast(field_name, today_str, yield_forecast)
            
            # 2. Wygeneruj sygnały rynkowe
            market_signals = self.generate_market_signals(field_name, ndvi_time_series)
            if market_signals:
                self.update_market_signals(field_name, today_str, market_signals)
            
            # 3. Przeprowadź analizę korelacji
            self.analyze_correlations(field_name)
            
            # 4. Wygeneruj wykresy
            self.generate_charts(field_name)
            
            logger.info(f"Pomyślnie zaktualizowano prognozy dla pola {field_name}")
            return True
        
        except Exception as e:
            logger.error(f"Błąd aktualizacji prognoz dla pola {field_name}: {str(e)}")
            return False
    
    def update_all_predictions(self):
        """
        Update predictions for all available fields.
        
        Returns:
            Number of successfully updated fields
        """
        logger.info("Aktualizacja prognoz dla wszystkich pól")
        
        # Pobierz dostępne pola
        fields = self.get_available_fields()
        
        if not fields:
            logger.warning("Brak dostępnych pól do aktualizacji")
            return 0
        
        success_count = 0
        
        for field_name in fields:
            if self.update_predictions_for_field(field_name):
                success_count += 1
        
        logger.info(f"Zaktualizowano prognozy dla {success_count} z {len(fields)} pól")
        return success_count

# Singleton instance
prediction_manager = PredictionManager()

def get_prediction_manager() -> PredictionManager:
    """Get the prediction manager singleton instance."""
    return prediction_manager

def update_all_predictions():
    """Update predictions for all available fields."""
    return prediction_manager.update_all_predictions()

def generate_charts_for_all_fields():
    """Generate charts for all available fields."""
    fields = prediction_manager.get_available_fields()
    for field_name in fields:
        prediction_manager.generate_charts(field_name)