"""
Insights Dashboard - Interaktywne porównania regionów, analiza trendów i prognozy ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple
import io
import base64
from scipy import stats
from utils.market_reports import MarketInsightsReportGenerator
import uuid
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import re

# Konfiguracja loggera
logger = logging.getLogger(__name__)

# Funkcja do ładowania danych z plików JSON
def load_data_from_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Błąd podczas ładowania danych z {file_path}: {str(e)}")
        return None

# Funkcja do wykrywania dostępnych pól
def load_available_fields():
    """Ładuje dostępne pola z katalogu danych"""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    
    # Pobierz wszystkie pliki JSON i GeoJSON
    json_files = list(data_dir.glob("*.json"))
    geojson_files = list(data_dir.glob("*.geojson"))
    tif_files = list(data_dir.glob("*.tif"))
    
    # Wyodrębnij nazwy pól
    field_names = set()
    
    # Z plików JSON
    for file in json_files:
        # Wyodrębnij nazwę pola z nazwy pliku (format: nazwa_pola_typ_danych.json)
        parts = file.stem.split('_')
        if len(parts) >= 1:
            field_names.add(parts[0])
    
    # Z plików GeoJSON
    for file in geojson_files:
        # Wyodrębnij nazwę pola z nazwy pliku (format: nazwa_pola.geojson)
        field_names.add(file.stem)
    
    # Z plików TIF
    for file in tif_files:
        # Wyodrębnij nazwę pola z nazwy pliku (format: nazwa_pola_indeks_sceneID.tif)
        parts = file.stem.split('_')
        if len(parts) >= 1:
            field_names.add(parts[0])
    
    return list(field_names)

# Funkcja do ładowania danych NDVI dla pola
def load_ndvi_data(field_name):
    file_path = Path(f"data/{field_name}_ndvi_time_series.json")
    if file_path.exists():
        return load_data_from_json(file_path)
    return None

# Funkcja do ładowania danych o prognozach plonów
def load_yield_forecast(field_name):
    file_path = Path(f"data/{field_name}_yield_forecast.json")
    if file_path.exists():
        return load_data_from_json(file_path)
    return None

# Funkcja do ładowania danych o sygnałach rynkowych
def load_market_signals(field_name):
    file_path = Path(f"data/{field_name}_market_signals.json")
    if file_path.exists():
        return load_data_from_json(file_path)
    return None

# Funkcja do przygotowania danych do porównania między regionami
def prepare_regions_comparison_data(field_names):
    comparison_data = {
        "region_names": field_names,
        "ndvi_values": {},
        "yield_forecasts": {},
        "market_signals": {}
    }
    
    # Pobieranie danych dla każdego pola
    for field_name in field_names:
        # NDVI
        ndvi_data = load_ndvi_data(field_name)
        if ndvi_data:
            comparison_data["ndvi_values"][field_name] = ndvi_data
        
        # Prognozy plonów
        yield_data = load_yield_forecast(field_name)
        if yield_data:
            comparison_data["yield_forecasts"][field_name] = yield_data
        
        # Sygnały rynkowe
        signals_data = load_market_signals(field_name)
        if signals_data:
            comparison_data["market_signals"][field_name] = signals_data
    
    return comparison_data

# Funkcja do tworzenia wykresu porównawczego NDVI
def create_ndvi_comparison_chart(ndvi_data, selected_fields):
    if not ndvi_data:
        return None
    
    # Tworzenie DataFrame z danych NDVI dla wybranych pól
    ndvi_df_data = []
    
    for field_name in selected_fields:
        if field_name in ndvi_data:
            field_ndvi = ndvi_data[field_name]
            for date, value in field_ndvi.items():
                ndvi_df_data.append({
                    "Region": field_name,
                    "Data": date,
                    "NDVI": value
                })
    
    if not ndvi_df_data:
        return None
    
    ndvi_df = pd.DataFrame(ndvi_df_data)
    
    # Konwersja dat na format datetime
    ndvi_df["Data"] = pd.to_datetime(ndvi_df["Data"])
    
    # Sortowanie po dacie
    ndvi_df = ndvi_df.sort_values("Data")
    
    # Tworzenie wykresu porównawczego
    fig = px.line(
        ndvi_df, 
        x="Data", 
        y="NDVI", 
        color="Region",
        title="Porównanie indeksu NDVI między regionami",
        labels={"Data": "Data", "NDVI": "Wartość NDVI", "Region": "Region"},
        markers=True,
        line_shape="linear"
    )
    
    # Dostosowanie układu wykresu
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Wartość NDVI",
        legend_title="Region",
        plot_bgcolor="white",
        hovermode="x unified"
    )
    
    return fig

# Funkcja do tworzenia wykresu porównawczego prognoz plonów
def create_yield_comparison_chart(yield_data, selected_fields, crop_type):
    if not yield_data:
        return None
    
    # Tworzenie DataFrame z danych o prognozach plonów dla wybranych pól
    yield_df_data = []
    
    for field_name in selected_fields:
        if field_name in yield_data:
            field_yield = yield_data[field_name]
            
            # Sprawdź, czy dane dotyczą wybranego typu uprawy
            if field_yield.get("crop_type") == crop_type:
                forecasted_yields = field_yield.get("forecasted_yields", {})
                
                for date, value in forecasted_yields.items():
                    yield_df_data.append({
                        "Region": field_name,
                        "Data": date,
                        "Prognoza plonów (t/ha)": value
                    })
    
    if not yield_df_data:
        return None
    
    yield_df = pd.DataFrame(yield_df_data)
    
    # Konwersja dat na format datetime
    yield_df["Data"] = pd.to_datetime(yield_df["Data"])
    
    # Sortowanie po dacie
    yield_df = yield_df.sort_values("Data")
    
    # Tworzenie wykresu porównawczego
    fig = px.line(
        yield_df, 
        x="Data", 
        y="Prognoza plonów (t/ha)", 
        color="Region",
        title=f"Porównanie prognoz plonów ({crop_type}) między regionami",
        labels={"Data": "Data", "Prognoza plonów (t/ha)": "Prognoza plonów (t/ha)", "Region": "Region"},
        markers=True,
        line_shape="linear"
    )
    
    # Dostosowanie układu wykresu
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Prognoza plonów (t/ha)",
        legend_title="Region",
        plot_bgcolor="white",
        hovermode="x unified"
    )
    
    return fig

# Funkcja do tworzenia wykresu porównawczego sygnałów rynkowych
def create_market_signals_comparison(signals_data, selected_fields, commodity):
    if not signals_data:
        return None
    
    # Przygotowanie danych o sygnałach rynkowych dla wybranych pól
    signals_df_data = []
    
    for field_name in selected_fields:
        if field_name in signals_data:
            field_signals = signals_data[field_name]
            signals = field_signals.get("signals", [])
            
            # Filtrowanie sygnałów dla wybranego towaru
            filtered_signals = [s for s in signals if s.get("commodity") == commodity]
            
            for signal in filtered_signals:
                signals_df_data.append({
                    "Region": field_name,
                    "Data": signal.get("signal_date", ""),
                    "Akcja": signal.get("action", ""),
                    "Pewność": signal.get("confidence", 0),
                    "Powód": signal.get("reason", "")
                })
    
    if not signals_df_data:
        return None
    
    signals_df = pd.DataFrame(signals_df_data)
    
    # Konwersja dat na format datetime
    signals_df["Data"] = pd.to_datetime(signals_df["Data"])
    
    # Sortowanie po dacie
    signals_df = signals_df.sort_values("Data")
    
    # Tworzenie mapy kolorów dla akcji
    color_map = {"LONG": "green", "SHORT": "red", "NEUTRAL": "blue"}
    
    # Tworzenie wykresu scatter
    fig = px.scatter(
        signals_df, 
        x="Data", 
        y="Pewność", 
        color="Akcja",
        symbol="Region",
        size="Pewność",
        hover_data=["Powód"],
        title=f"Porównanie sygnałów rynkowych dla {commodity} między regionami",
        color_discrete_map=color_map
    )
    
    # Dodanie linii trendu dla każdego regionu
    for field_name in selected_fields:
        field_df = signals_df[signals_df["Region"] == field_name]
        if not field_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=field_df["Data"],
                    y=field_df["Pewność"],
                    mode='lines',
                    line=dict(width=1, dash='dash'),
                    showlegend=False,
                    opacity=0.5,
                    name=f"{field_name} trend"
                )
            )
    
    # Dostosowanie układu wykresu
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Pewność sygnału",
        legend_title="Typ sygnału",
        plot_bgcolor="white",
        hovermode="closest"
    )
    
    return fig

# Funkcja do trenowania modelu ML dla prognozowania plonów
def train_advanced_yield_model(selected_fields, crop_type):
    # Zbieranie danych treningowych
    training_data = []
    
    for field_name in selected_fields:
        # Ładowanie danych NDVI
        ndvi_data = load_ndvi_data(field_name)
        
        # Ładowanie danych o prognozie plonów
        yield_data = load_yield_forecast(field_name)
        
        if ndvi_data and yield_data:
            # Konwersja danych NDVI na listę wartości
            ndvi_dates = sorted(ndvi_data.keys())
            ndvi_values = [ndvi_data[date] for date in ndvi_dates]
            
            # Konwersja dat na obiekty datetime
            ndvi_dates = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in ndvi_dates]
            
            # Obliczenie statystyk NDVI
            if len(ndvi_values) >= 3:
                ndvi_mean = np.mean(ndvi_values)
                ndvi_max = np.max(ndvi_values)
                ndvi_min = np.min(ndvi_values)
                ndvi_std = np.std(ndvi_values)
                ndvi_trend = np.polyfit(range(len(ndvi_values)), ndvi_values, 1)[0]
                
                # Obliczenie sezonowości
                month_values = {}
                for date, value in zip(ndvi_dates, ndvi_values):
                    month = date.month
                    if month not in month_values:
                        month_values[month] = []
                    month_values[month].append(value)
                
                month_averages = {month: np.mean(values) for month, values in month_values.items()}
                
                # Pobranie prognozowanych plonów dla tego pola i uprawy
                if yield_data.get("crop_type") == crop_type:
                    forecasted_yields = yield_data.get("forecasted_yields", {})
                    
                    if forecasted_yields:
                        # Dla każdej prognozy plonów dodaj wiersz danych treningowych
                        for yield_date, yield_value in forecasted_yields.items():
                            yield_datetime = datetime.datetime.strptime(yield_date, "%Y-%m-%d")
                            
                            # Dodanie cech do danych treningowych
                            training_row = {
                                "field_name": field_name,
                                "ndvi_mean": ndvi_mean,
                                "ndvi_max": ndvi_max,
                                "ndvi_min": ndvi_min,
                                "ndvi_std": ndvi_std,
                                "ndvi_trend": ndvi_trend,
                                "month": yield_datetime.month,
                                "day_of_year": yield_datetime.timetuple().tm_yday,
                                "year": yield_datetime.year
                            }
                            
                            # Dodanie średnich NDVI dla miesięcy
                            for month, avg in month_averages.items():
                                training_row[f"ndvi_month_{month}"] = avg
                            
                            # Dodanie ostatnich 3 wartości NDVI, jeśli dostępne
                            for i in range(min(3, len(ndvi_values))):
                                training_row[f"ndvi_last_{i+1}"] = ndvi_values[-(i+1)]
                            
                            # Dodanie wartości docelowej (prognoza plonów)
                            training_row["yield"] = yield_value
                            
                            training_data.append(training_row)
    
    if not training_data:
        return None, "Brak wystarczających danych treningowych dla wybranych pól i uprawy."
    
    # Tworzenie DataFrame z danych treningowych
    df = pd.DataFrame(training_data)
    
    # Wypełnienie brakujących wartości
    df = df.fillna(0)
    
    # Podział na cechy i wartość docelową
    X = df.drop(["yield", "field_name"], axis=1)
    y = df["yield"]
    
    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Trenowanie modelu Gradient Boosting
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Ocena modelu
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Ważność cech
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Zapisanie modelu
    model_path = f"data/advanced_yield_model_{crop_type}.joblib"
    joblib.dump(model, model_path)
    
    return {
        "model": model,
        "model_path": model_path,
        "metrics": {
            "mse": mse,
            "mae": mae,
            "r2": r2
        },
        "feature_importance": feature_importance.to_dict(),
        "training_size": len(X_train),
        "test_size": len(X_test)
    }, None

# Funkcja do obsługi poleceń głosowych
def process_voice_command(command_text):
    """
    Przetwarza polecenie głosowe i zwraca odpowiednią konfigurację.
    
    Args:
        command_text: Tekst polecenia głosowego
        
    Returns:
        Słownik z konfiguracją dla dashboardu
    """
    command_text = command_text.lower()
    
    # Inicjalizacja konfiguracji
    config = {
        "selected_fields": [],
        "selected_crop": None,
        "selected_commodity": None,
        "time_period": "all",
        "chart_type": "ndvi",
        "message": "Przetworzono polecenie głosowe."
    }
    
    # Słownik regionów
    regions = {
        "mazowsze": "Mazowsze",
        "podlasie": "Podlasie",
        "małopolska": "Malopolska",
        "wielkopolska": "Wielkopolska",
        "pomorze": "Pomorze"
    }
    
    # Słownik upraw
    crops = {
        "pszenica": "Wheat",
        "wheat": "Wheat",
        "kukurydza": "Corn",
        "corn": "Corn",
        "soja": "Soybean",
        "soybean": "Soybean",
        "owies": "Oats",
        "oats": "Oats",
        "ryż": "Rice",
        "rice": "Rice"
    }
    
    # Słownik towarów
    commodities = {
        "pszenica": "ZW=F",
        "wheat": "ZW=F",
        "kukurydza": "ZC=F",
        "corn": "ZC=F",
        "soja": "ZS=F",
        "soybean": "ZS=F",
        "owies": "ZO=F",
        "oats": "ZO=F",
        "ryż": "ZR=F",
        "rice": "ZR=F"
    }
    
    # Wykrywanie regionów
    for region_key, region_name in regions.items():
        if region_key in command_text:
            config["selected_fields"].append(region_name)
    
    # Wykrywanie upraw
    for crop_key, crop_name in crops.items():
        if crop_key in command_text:
            config["selected_crop"] = crop_name
            break
    
    # Wykrywanie towarów
    for commodity_key, commodity_symbol in commodities.items():
        if commodity_key in command_text:
            config["selected_commodity"] = commodity_symbol
            break
    
    # Wykrywanie okresów czasowych
    if "ostatni miesiąc" in command_text or "last month" in command_text:
        config["time_period"] = "month"
    elif "ostatni kwartał" in command_text or "last quarter" in command_text:
        config["time_period"] = "quarter"
    elif "ostatni rok" in command_text or "last year" in command_text:
        config["time_period"] = "year"
    
    # Wykrywanie typów wykresów
    if "ndvi" in command_text:
        config["chart_type"] = "ndvi"
    elif "plon" in command_text or "yield" in command_text:
        config["chart_type"] = "yield"
    elif "sygnał" in command_text or "signal" in command_text or "market" in command_text or "rynek" in command_text:
        config["chart_type"] = "market"
    elif "porównaj" in command_text or "compare" in command_text:
        config["chart_type"] = "compare"
    
    # Jeśli nie wykryto żadnych regionów, użyj wszystkich dostępnych
    if not config["selected_fields"]:
        available_fields = load_available_fields()
        config["selected_fields"] = available_fields[:3]  # Użyj pierwszych 3 dostępnych pól
        config["message"] += " Nie wykryto konkretnych regionów, używam domyślnych."
    
    # Jeśli nie wykryto uprawy, użyj domyślnej
    if not config["selected_crop"]:
        config["selected_crop"] = "Wheat"
        config["message"] += " Nie wykryto typu uprawy, używam pszenicy jako domyślnej."
    
    # Jeśli nie wykryto towaru, użyj odpowiadającego wybranej uprawie
    if not config["selected_commodity"]:
        crop_to_commodity = {
            "Wheat": "ZW=F",
            "Corn": "ZC=F",
            "Soybean": "ZS=F",
            "Oats": "ZO=F",
            "Rice": "ZR=F"
        }
        config["selected_commodity"] = crop_to_commodity.get(config["selected_crop"], "ZW=F")
    
    return config

# Inicjalizacja sesji Streamlit
if "initialized_insights" not in st.session_state:
    st.session_state.initialized_insights = True
    st.session_state.comparison_data = {}
    st.session_state.selected_fields = []
    st.session_state.selected_crop = "Wheat"
    st.session_state.selected_commodity = "ZW=F"
    st.session_state.time_period = "all"
    st.session_state.voice_command = ""
    st.session_state.last_training_result = None

# Tytuł strony
st.title("🔍 Insights Dashboard")
st.markdown("""
Interaktywna analiza i porównanie danych satelitarnych między regionami, zaawansowane modele predykcyjne i automatyczne raporty rynkowe.
""")

# Uproszczony interfejs wyboru danych
st.sidebar.header("Szybka konfiguracja")

# Ładowanie dostępnych pól
available_fields = load_available_fields()

# Sekcja dla zapytań głosowych - umieszczona na górze jako główny sposób interakcji
voice_command = st.sidebar.text_input(
    "💬 Wprowadź zapytanie głosowe",
    placeholder="np. 'Pokaż NDVI dla Mazowsze i Wielkopolska dla kukurydzy'",
    help="Szybki sposób na analizę - wprowadź naturalne zapytanie"
)

if voice_command:
    with st.sidebar:
        with st.spinner("Przetwarzanie..."):
            config = process_voice_command(voice_command)
            
            # Aktualizacja stanu sesji na podstawie konfiguracji
            st.session_state.selected_fields = config["selected_fields"]
            st.session_state.selected_crop = config["selected_crop"]
            st.session_state.selected_commodity = config["selected_commodity"]
            st.session_state.time_period = config["time_period"]
            st.session_state.chart_type = config["chart_type"]
            
            st.success(config["message"])

# Konfiguracja za pomocą kontrolek - uproszczona, umieszczona w jednym miejscu
with st.sidebar.expander("⚙️ Ręczna konfiguracja"):
    # Wybór pól do porównania - teraz w rozwijanym menu
    selected_fields = st.multiselect(
        "Regiony",
        options=available_fields,
        default=available_fields[:2] if len(available_fields) >= 2 else available_fields
    )
    
    # Umieszczenie kontrolek typu uprawy i towaru obok siebie
    col1, col2 = st.columns(2)
    
    with col1:
        # Wybór typu uprawy
        selected_crop = st.selectbox(
            "Uprawa",
            options=["Wheat", "Corn", "Soybean", "Oats", "Rice"],
            index=0
        )
    
    with col2:
        # Uproszczony wybór towaru, powiązany z uprawą
        crop_to_commodity = {
            "Wheat": "ZW=F",
            "Corn": "ZC=F",
            "Soybean": "ZS=F",
            "Oats": "ZO=F",
            "Rice": "ZR=F"
        }
        commodity_names = {
            "ZW=F": "Wheat (ZW=F)",
            "ZC=F": "Corn (ZC=F)",
            "ZS=F": "Soybean (ZS=F)",
            "ZO=F": "Oats (ZO=F)",
            "ZR=F": "Rice (ZR=F)"
        }
        selected_commodity = crop_to_commodity[selected_crop]
        selected_commodity_name = commodity_names[selected_commodity]
        st.text(selected_commodity_name)
    
    # Wybór okresu czasowego
    time_period = st.radio(
        "Okres czasowy",
        options=["Wszystkie dane", "Ostatni miesiąc", "Ostatni kwartał", "Ostatni rok"],
        horizontal=True
    )
    
    # Konwersja wyboru okresu na wartość do filtrowania
    time_period_value = "all"
    if time_period == "Ostatni miesiąc":
        time_period_value = "month"
    elif time_period == "Ostatni kwartał":
        time_period_value = "quarter"
    elif time_period == "Ostatni rok":
        time_period_value = "year"

# Centralny przycisk generowania analizy
if st.sidebar.button("🚀 Generuj analizę", use_container_width=True, type="primary"):
    if len(selected_fields) < 1:
        st.warning("Wybierz co najmniej jeden region do analizy.")
    else:
        # Aktualizacja stanu sesji
        st.session_state.selected_fields = selected_fields
        st.session_state.selected_crop = selected_crop
        st.session_state.selected_commodity = selected_commodity
        st.session_state.time_period = time_period_value
        
        # Przygotowanie danych do porównania
        with st.spinner("Przygotowywanie danych do porównania..."):
            comparison_data = prepare_regions_comparison_data(selected_fields)
            st.session_state.comparison_data = comparison_data

# Główna sekcja z porównaniem regionów
if st.session_state.selected_fields:
    st.header(f"Porównanie regionów: {', '.join(st.session_state.selected_fields)}")
    
    # Tworzenie zakładek dla różnych typów porównań
    tabs = st.tabs(["Porównanie NDVI", "Porównanie plonów", "Porównanie sygnałów rynkowych", "Zaawansowana analiza ML"])
    
    with tabs[0]:
        st.subheader("Porównanie indeksu NDVI między regionami")
        
        if "ndvi_values" in st.session_state.comparison_data and st.session_state.comparison_data["ndvi_values"]:
            # Tworzenie wykresu porównawczego NDVI
            ndvi_fig = create_ndvi_comparison_chart(
                st.session_state.comparison_data["ndvi_values"], 
                st.session_state.selected_fields
            )
            
            if ndvi_fig:
                st.plotly_chart(ndvi_fig, use_container_width=True)
                
                # Dodanie analizy statystycznej
                st.subheader("Analiza statystyczna NDVI")
                
                # Tworzenie tabeli z podstawowymi statystykami
                ndvi_stats = []
                
                for field_name in st.session_state.selected_fields:
                    if field_name in st.session_state.comparison_data["ndvi_values"]:
                        field_ndvi = st.session_state.comparison_data["ndvi_values"][field_name]
                        ndvi_values = list(field_ndvi.values())
                        
                        ndvi_stats.append({
                            "Region": field_name,
                            "Średnia NDVI": np.mean(ndvi_values),
                            "Min NDVI": np.min(ndvi_values),
                            "Max NDVI": np.max(ndvi_values),
                            "Odchylenie std.": np.std(ndvi_values),
                            "Ostatnia wartość": ndvi_values[-1] if ndvi_values else None
                        })
                
                if ndvi_stats:
                    st.dataframe(pd.DataFrame(ndvi_stats))
                    
                    # Dodanie korelacji między regionami
                    st.subheader("Korelacja NDVI między regionami")
                    
                    # Tworzenie DataFrame z danymi NDVI dla wszystkich regionów
                    ndvi_corr_data = {}
                    
                    for field_name in st.session_state.selected_fields:
                        if field_name in st.session_state.comparison_data["ndvi_values"]:
                            field_ndvi = st.session_state.comparison_data["ndvi_values"][field_name]
                            # Tworzenie serii czasowej
                            ndvi_corr_data[field_name] = pd.Series(field_ndvi)
                    
                    if ndvi_corr_data:
                        ndvi_corr_df = pd.DataFrame(ndvi_corr_data)
                        corr_matrix = ndvi_corr_df.corr()
                        
                        # Wyświetlenie macierzy korelacji
                        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
                        
                        # Wykres heatmap korelacji
                        fig = px.imshow(
                            corr_matrix, 
                            text_auto=True, 
                            color_continuous_scale='RdBu_r',
                            title="Heatmap korelacji NDVI między regionami"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Brak wystarczających danych NDVI dla wybranych regionów do utworzenia wykresu.")
        else:
            st.warning("Brak danych NDVI dla wybranych regionów.")
    
    with tabs[1]:
        st.subheader(f"Porównanie prognoz plonów ({st.session_state.selected_crop}) między regionami")
        
        if "yield_forecasts" in st.session_state.comparison_data and st.session_state.comparison_data["yield_forecasts"]:
            # Tworzenie wykresu porównawczego prognoz plonów
            yield_fig = create_yield_comparison_chart(
                st.session_state.comparison_data["yield_forecasts"], 
                st.session_state.selected_fields,
                st.session_state.selected_crop
            )
            
            if yield_fig:
                st.plotly_chart(yield_fig, use_container_width=True)
                
                # Dodanie analizy statystycznej
                st.subheader("Analiza statystyczna prognoz plonów")
                
                # Tworzenie tabeli z podstawowymi statystykami
                yield_stats = []
                
                for field_name in st.session_state.selected_fields:
                    if field_name in st.session_state.comparison_data["yield_forecasts"]:
                        field_yield = st.session_state.comparison_data["yield_forecasts"][field_name]
                        
                        if field_yield.get("crop_type") == st.session_state.selected_crop:
                            forecasted_yields = field_yield.get("forecasted_yields", {})
                            yield_values = list(forecasted_yields.values())
                            
                            if yield_values:
                                yield_stats.append({
                                    "Region": field_name,
                                    "Średnia prognoza (t/ha)": np.mean(yield_values),
                                    "Min prognoza (t/ha)": np.min(yield_values),
                                    "Max prognoza (t/ha)": np.max(yield_values),
                                    "Odchylenie std.": np.std(yield_values),
                                    "Ostatnia prognoza (t/ha)": yield_values[-1] if yield_values else None
                                })
                
                if yield_stats:
                    st.dataframe(pd.DataFrame(yield_stats))
                    
                    # Wykres słupkowy z ostatnimi prognozami
                    last_yield_df = pd.DataFrame(yield_stats)
                    
                    fig = px.bar(
                        last_yield_df,
                        x="Region",
                        y="Ostatnia prognoza (t/ha)",
                        title=f"Ostatnie prognozy plonów ({st.session_state.selected_crop}) dla wybranych regionów",
                        color="Region"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Brak prognoz plonów dla uprawy {st.session_state.selected_crop} w wybranych regionach.")
        else:
            st.warning("Brak danych o prognozach plonów dla wybranych regionów.")
    
    with tabs[2]:
        st.subheader(f"Porównanie sygnałów rynkowych ({selected_commodity_name}) między regionami")
        
        if "market_signals" in st.session_state.comparison_data and st.session_state.comparison_data["market_signals"]:
            # Tworzenie wykresu porównawczego sygnałów rynkowych
            signals_fig = create_market_signals_comparison(
                st.session_state.comparison_data["market_signals"], 
                st.session_state.selected_fields,
                st.session_state.selected_commodity
            )
            
            if signals_fig:
                st.plotly_chart(signals_fig, use_container_width=True)
                
                # Dodanie analizy sygnałów rynkowych
                st.subheader("Analiza sygnałów rynkowych")
                
                # Tworzenie tabeli z podsumowaniem sygnałów
                signals_summary = []
                
                for field_name in st.session_state.selected_fields:
                    if field_name in st.session_state.comparison_data["market_signals"]:
                        field_signals = st.session_state.comparison_data["market_signals"][field_name]
                        signals = field_signals.get("signals", [])
                        
                        # Filtrowanie sygnałów dla wybranego towaru
                        filtered_signals = [s for s in signals if s.get("commodity") == st.session_state.selected_commodity]
                        
                        if filtered_signals:
                            # Liczenie sygnałów LONG i SHORT
                            long_signals = len([s for s in filtered_signals if s.get("action") == "LONG"])
                            short_signals = len([s for s in filtered_signals if s.get("action") == "SHORT"])
                            neutral_signals = len([s for s in filtered_signals if s.get("action") == "NEUTRAL"])
                            
                            # Obliczenie średniej pewności
                            avg_confidence = np.mean([s.get("confidence", 0) for s in filtered_signals])
                            
                            # Określenie przeważającego sygnału
                            if long_signals > short_signals and long_signals > neutral_signals:
                                dominant_signal = "LONG"
                            elif short_signals > long_signals and short_signals > neutral_signals:
                                dominant_signal = "SHORT"
                            else:
                                dominant_signal = "NEUTRAL"
                            
                            signals_summary.append({
                                "Region": field_name,
                                "Liczba sygnałów": len(filtered_signals),
                                "Sygnały LONG": long_signals,
                                "Sygnały SHORT": short_signals,
                                "Sygnały NEUTRAL": neutral_signals,
                                "Średnia pewność": avg_confidence,
                                "Przeważający sygnał": dominant_signal
                            })
                
                if signals_summary:
                    st.dataframe(pd.DataFrame(signals_summary))
                    
                    # Wykres słupkowy porównujący sygnały LONG i SHORT
                    signals_df = pd.DataFrame(signals_summary)
                    
                    # Przygotowanie danych do wykresu
                    chart_data = []
                    for _, row in signals_df.iterrows():
                        chart_data.append({
                            "Region": row["Region"],
                            "Typ sygnału": "LONG",
                            "Liczba sygnałów": row["Sygnały LONG"]
                        })
                        chart_data.append({
                            "Region": row["Region"],
                            "Typ sygnału": "SHORT",
                            "Liczba sygnałów": row["Sygnały SHORT"]
                        })
                    
                    chart_df = pd.DataFrame(chart_data)
                    
                    fig = px.bar(
                        chart_df,
                        x="Region",
                        y="Liczba sygnałów",
                        color="Typ sygnału",
                        barmode="group",
                        title=f"Porównanie liczby sygnałów LONG i SHORT dla {selected_commodity_name}",
                        color_discrete_map={"LONG": "green", "SHORT": "red"}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Wykres trendu sygnałów w czasie
                    st.subheader("Trend sygnałów rynkowych w czasie")
                    
                    # Zbieranie danych o sygnałach w czasie
                    time_signals_data = []
                    
                    for field_name in st.session_state.selected_fields:
                        if field_name in st.session_state.comparison_data["market_signals"]:
                            field_signals = st.session_state.comparison_data["market_signals"][field_name]
                            signals = field_signals.get("signals", [])
                            
                            # Filtrowanie sygnałów dla wybranego towaru
                            filtered_signals = [s for s in signals if s.get("commodity") == st.session_state.selected_commodity]
                            
                            for signal in filtered_signals:
                                time_signals_data.append({
                                    "Region": field_name,
                                    "Data": signal.get("signal_date", ""),
                                    "Akcja": signal.get("action", ""),
                                    "Pewność": signal.get("confidence", 0)
                                })
                    
                    if time_signals_data:
                        time_signals_df = pd.DataFrame(time_signals_data)
                        time_signals_df["Data"] = pd.to_datetime(time_signals_df["Data"])
                        time_signals_df = time_signals_df.sort_values("Data")
                        
                        # Obliczanie trendu sygnałów (średnia krocząca)
                        time_signals_df["Sygnał wartość"] = time_signals_df["Akcja"].map({"LONG": 1, "NEUTRAL": 0, "SHORT": -1})
                        
                        # Grupowanie po regionie i dacie
                        grouped_signals = time_signals_df.groupby(["Region", pd.Grouper(key="Data", freq="W")])["Sygnał wartość"].mean().reset_index()
                        
                        # Wykres trendu sygnałów
                        fig = px.line(
                            grouped_signals,
                            x="Data",
                            y="Sygnał wartość",
                            color="Region",
                            title=f"Trend sygnałów rynkowych w czasie dla {selected_commodity_name}",
                            labels={"Sygnał wartość": "Wartość sygnału (1=LONG, 0=NEUTRAL, -1=SHORT)"}
                        )
                        
                        # Dodanie linii odniesienia na poziomie 0
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Brak sygnałów rynkowych dla towaru {selected_commodity_name} w wybranych regionach.")
        else:
            st.warning("Brak danych o sygnałach rynkowych dla wybranych regionów.")
    
    with tabs[3]:
        st.subheader("Zaawansowane analizy i automatyczne raporty")
        
        # Zakładki dla prostszej nawigacji
        ml_tabs = st.tabs(["Model predykcyjny", "Automatyczne raporty"])
        
        with ml_tabs[0]:
            # Uproszczona sekcja z modelem ML
            st.markdown("### 🤖 Model predykcyjny plonów")
            
            # Info box z wyjaśnieniem
            st.info("Model wykorzystuje dane satelitarne NDVI i historyczne prognozy do przewidywania przyszłych plonów.")
            
            # Prosty interfejs z jednym przyciskiem
            if st.button("Wytrenuj model dla wybranych regionów", type="primary", use_container_width=True):
                with st.spinner("Trenowanie zaawansowanego modelu predykcyjnego..."):
                    model_result, error = train_advanced_yield_model(
                        st.session_state.selected_fields,
                        st.session_state.selected_crop
                    )
                    
                    if error:
                        st.error(error)
                    elif model_result:
                        st.session_state.last_training_result = model_result
                        st.success(f"✅ Model ML wytrenowany! Dokładność (R²): {model_result['metrics']['r2']:.4f}")
            
            # Wyświetlanie wyniku trenowania, jeśli dostępny
            if st.session_state.last_training_result:
                model_result = st.session_state.last_training_result
                
                # Podsumowanie wyników w jednym miejscu
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{model_result['metrics']['r2']:.4f}")
                with col2:
                    st.metric("MSE", f"{model_result['metrics']['mse']:.4f}")
                with col3:
                    st.metric("MAE", f"{model_result['metrics']['mae']:.4f}")
                
                # Wykres ważności cech
                st.markdown("#### Najważniejsze czynniki wpływające na plony")
                
                # Konwersja słownika na DataFrame
                importance_data = []
                for i, feature in enumerate(model_result["feature_importance"]["feature"]):
                    importance_data.append({
                        "Cecha": feature,
                        "Ważność": model_result["feature_importance"]["importance"][i]
                    })
                
                importance_df = pd.DataFrame(importance_data)
                importance_df = importance_df.sort_values("Ważność", ascending=False)
                
                # Wyświetlenie wykresu ważności cech - tylko top 5 dla prostoty
                fig = px.bar(
                    importance_df.head(5),
                    x="Ważność",
                    y="Cecha",
                    orientation="h",
                    title="Top 5 najważniejszych czynników",
                    color="Ważność"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with ml_tabs[1]:
            st.markdown("### 📊 Automatyczne raporty rynkowe")
            
            # Uproszczona sekcja raportów automatycznych
            st.info("System automatycznie generuje raporty analizy rynkowej na podstawie aktualnych danych satelitarnych i trendów rynkowych.")
            
            # Jeden rząd przycisków
            col1, col2 = st.columns(2)
            
            with col1:
                report_format = st.radio(
                    "Format raportu",
                    options=["Markdown", "HTML"],
                    horizontal=True,
                    index=0
                )
            
            with col2:
                if st.session_state.selected_fields:
                    selected_field_for_report = st.selectbox(
                        "Region dla raportu",
                        options=st.session_state.selected_fields
                    )
                else:
                    selected_field_for_report = None
                    st.warning("Najpierw wybierz regiony do analizy")
            
            # Przycisk generowania raportu
            if st.button("Generuj automatyczny raport", type="primary", use_container_width=True):
                if not selected_field_for_report:
                    st.error("Wybierz co najmniej jeden region do analizy.")
                else:
                    with st.spinner("Generowanie automatycznego raportu..."):
                        # Inicjalizacja generatora raportów
                        report_generator = MarketInsightsReportGenerator()
                        
                        # Generowanie raportu
                        report_content, report_id = report_generator.generate_automated_report(
                            selected_field_for_report,
                            st.session_state.selected_crop,
                            report_format.lower()
                        )
                        
                        if report_id:
                            st.success(f"✅ Raport wygenerowany pomyślnie!")
                            
                            # Wyświetlenie raportu
                            st.markdown("#### Podgląd raportu")
                            
                            if report_format.lower() == "html":
                                try:
                                    from streamlit.components.v1 import html as st_html
                                    st_html(report_content, height=500, scrolling=True)
                                except:
                                    st.markdown(report_content)
                            else:
                                st.markdown(report_content)
                            
                            # Przycisk do pobrania raportu
                            st.download_button(
                                label=f"💾 Pobierz raport",
                                data=report_content,
                                file_name=f"raport_{selected_field_for_report}_{st.session_state.selected_crop}.{'html' if report_format.lower() == 'html' else 'md'}",
                                mime=f"text/{'html' if report_format.lower() == 'html' else 'markdown'}"
                            )
                        else:
                            st.error(f"Błąd podczas generowania raportu: {report_content}")
            
            # Uproszczone planowanie raportów
            with st.expander("📅 Planowanie automatycznych raportów"):
                st.write("Skonfiguruj automatyczne raporty i powiadomienia e-mail")
                
                # Uproszczone opcje
                frequency = st.select_slider(
                    "Częstotliwość",
                    options=["Codziennie", "Co tydzień", "Co miesiąc"]
                )
                
                emails = st.text_input(
                    "Adresy e-mail (oddzielone przecinkami)",
                    placeholder="email@example.com, drugi@example.com"
                )
                
                if st.button("Zaplanuj raporty automatyczne"):
                    if not st.session_state.selected_fields:
                        st.error("Wybierz co najmniej jeden region do analizy.")
                    elif not emails:
                        st.error("Wprowadź co najmniej jeden adres e-mail.")
                    else:
                        # Konwersja na wartość dla API
                        frequency_value = {
                            "Codziennie": "daily",
                            "Co tydzień": "weekly",
                            "Co miesiąc": "monthly"
                        }[frequency]
                        
                        with st.spinner("Konfigurowanie harmonogramu raportów..."):
                            report_generator = MarketInsightsReportGenerator()
                            emails_list = [email.strip() for email in emails.split(",")]
                            
                            schedule_info = report_generator.schedule_automated_reports(
                                st.session_state.selected_fields,
                                [st.session_state.selected_crop],
                                frequency_value,
                                emails_list
                            )
                            
                            st.success(f"✅ Zaplanowano automatyczne raporty")
                            st.info(f"Raporty będą wysyłane {frequency.lower()} na: {', '.join(emails_list)}")

# Dolna sekcja z podsumowaniem i dodatkowymi informacjami
st.markdown("---")

# Stopka
st.markdown("""
### Informacje o danych
Powyższe analizy wykorzystują dane satelitarne, prognozy pogodowe i dane rynkowe. Modele uczenia maszynowego są trenowane na zbiorach danych obejmujących historyczne wartości NDVI, prognozy plonów i sygnały rynkowe.

Interaktywne porównania regionów pozwalają na lepsze zrozumienie różnic w warunkach upraw między różnymi lokalizacjami, co może pomóc w podejmowaniu bardziej świadomych decyzji dotyczących zarządzania uprawami i strategii handlowych.
""")

# Wyświetlanie aktualnego stanu konfiguracji w bardziej zwartej i wizualnej formie
st.sidebar.markdown("---")

# Podsumowanie aktualnej konfiguracji w bardziej przejrzystej formie
if st.session_state.selected_fields:
    # Używamy kolorowych ikon i przejrzystego układu
    st.sidebar.markdown("### 📊 Aktualna konfiguracja")
    
    with st.sidebar:
        # Tworzymy przejrzysty interfejs kart
        columns = st.columns(2)
        with columns[0]:
            st.markdown("🌾 **Uprawa:**")
            st.markdown("🗺️ **Regiony:**")
            st.markdown("💹 **Kontrakt:**")
            st.markdown("📅 **Okres:**")
        
        with columns[1]:
            st.markdown(f"**{st.session_state.selected_crop}**")
            # Skracamy listę regionów jeśli jest ich wiele
            regions_text = ', '.join(st.session_state.selected_fields) 
            if len(regions_text) > 25:
                regions_text = regions_text[:22] + "..."
            st.markdown(f"**{regions_text}**")
            st.markdown(f"**{selected_commodity_name.split(' ')[0]}**")
            st.markdown(f"**{time_period}**")
    
    # Dodanie informacji o wytrenowanym modelu ML w bardziej atrakcyjnej formie
    if st.session_state.last_training_result:
        st.sidebar.markdown("### 🤖 Aktywny model ML")
        
        # Wyświetlenie dokładności w kolorowej ramce
        accuracy = st.session_state.last_training_result['metrics']['r2']
        color = "green" if accuracy > 0.7 else ("orange" if accuracy > 0.5 else "red")
        st.sidebar.markdown(
            f"""
            <div style="padding: 10px; border-radius: 5px; background-color: {color}; color: white; text-align: center;">
              <span style="font-size: 24px; font-weight: bold;">{accuracy:.2f}</span><br>
              <span>Dokładność R²</span>
            </div>
            """, 
            unsafe_allow_html=True
        )