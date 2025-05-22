"""
Satellite Monitor - Automatyczna aktualizacja prognoz i analiza korelacji danych satelitarnych
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import datetime
import json
import logging
from pathlib import Path
import os
import time
from threading import Thread

from utils.predictions import (
    get_prediction_manager,
    update_all_predictions,
    generate_charts_for_all_fields
)

# Konfiguracja loggera
logger = logging.getLogger(__name__)

# Ustawienia strony
st.set_page_config(
    page_title="Satellite Monitor - Agro Insight",
    page_icon="🛰️",
    layout="wide"
)

# Inicjalizacja stanu sesji
if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = None

# Funkcja do ładowania danych z plików
def load_data_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Błąd wczytywania danych z pliku {file_path}: {str(e)}")
        return None

# Funkcja do pobierania dostępnych pól
def get_available_fields():
    # Pobierz instancję managera prognoz
    prediction_manager = get_prediction_manager()
    # Pobierz dostępne pola
    return prediction_manager.get_available_fields()

# Funkcja do wykonania ręcznej aktualizacji
def manual_update():
    with st.spinner("Aktualizacja prognoz i generowanie wykresów..."):
        # Aktualizuj prognozy dla wszystkich pól
        updated_count = update_all_predictions()
        st.session_state.last_update_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Wygeneruj wykresy dla wszystkich pól
        generate_charts_for_all_fields()
        
        # Krótkie opóźnienie, żeby wykresy zdążyły się wygenerować
        time.sleep(1)
        
        # Odśwież stronę
        st.rerun()

# Funkcja do ładowania i wyświetlania wykresów
def load_and_display_charts(field_name):
    # Ścieżka do katalogu z wykresami
    charts_dir = Path(f"data/charts/{field_name}")
    
    if not charts_dir.exists():
        st.warning(f"Brak wygenerowanych wykresów dla pola {field_name}. Uruchom aktualizację, aby wygenerować wykresy.")
        return False
    
    # Znajdź wszystkie pliki PNG w katalogu
    chart_files = list(charts_dir.glob("*.png"))
    
    if not chart_files:
        st.warning(f"Brak wygenerowanych wykresów dla pola {field_name}. Uruchom aktualizację, aby wygenerować wykresy.")
        return False
    
    # Sortowanie wykresów według typu
    chart_files.sort()
    
    # Wyświetl wykresy w kolumnach
    col1, col2 = st.columns(2)
    
    for i, chart_file in enumerate(chart_files):
        chart_name = chart_file.stem.replace('_', ' ').title()
        
        # Odczytaj obraz
        try:
            with open(chart_file, "rb") as img_file:
                img_bytes = img_file.read()
            
            # Wyświetl w odpowiedniej kolumnie
            if i % 2 == 0:
                with col1:
                    st.subheader(chart_name)
                    st.image(img_bytes, use_column_width=True)
            else:
                with col2:
                    st.subheader(chart_name)
                    st.image(img_bytes, use_column_width=True)
        except Exception as e:
            st.error(f"Błąd wczytywania wykresu {chart_name}: {str(e)}")
    
    return True

# Funkcja do wyświetlania przeglądu korelacji
def display_correlation_overview():
    # Ścieżka do katalogu z wynikami korelacji
    correlation_dir = Path("data/correlations")
    
    if not correlation_dir.exists() or not list(correlation_dir.glob("*_correlation.json")):
        st.warning("Brak danych korelacji. Uruchom aktualizację, aby wygenerować analizy korelacji.")
        return
    
    # Znajdź wszystkie pliki JSON z korelacjami
    correlation_files = list(correlation_dir.glob("*_correlation.json"))
    
    # Przygotuj tabele korelacji
    correlation_data = []
    
    for corr_file in correlation_files:
        try:
            data = load_data_from_file(corr_file)
            if data and "field_name" in data and "commodity" in data and "overall_correlation" in data:
                # Znajdź najsilniejszą korelację z opóźnieniem
                strongest_lag = {"lag": 0, "correlation": 0}
                if "lag_correlations" in data and data["lag_correlations"]:
                    for lag_corr in data["lag_correlations"]:
                        if abs(lag_corr["correlation"]) > abs(strongest_lag["correlation"]):
                            strongest_lag = lag_corr
                
                correlation_data.append({
                    "Field": data["field_name"],
                    "Commodity": data["commodity"],
                    "Overall Correlation": round(data["overall_correlation"], 4),
                    "Strongest Lag": strongest_lag["lag"],
                    "Strongest Lag Correlation": round(strongest_lag["correlation"], 4),
                    "Date Analyzed": data.get("date_analyzed", "Unknown")
                })
        except Exception as e:
            logger.error(f"Błąd wczytywania danych korelacji z pliku {corr_file}: {str(e)}")
    
    if not correlation_data:
        st.warning("Nie udało się wczytać danych korelacji.")
        return
    
    # Stwórz ramkę danych z korelacjami
    correlations_df = pd.DataFrame(correlation_data)
    
    # Wyświetl tabele korelacji
    st.subheader("Przegląd korelacji NDVI z cenami towarów")
    st.dataframe(correlations_df, use_container_width=True)
    
    # Jeśli mamy wystarczająco danych, stwórz wykres
    if len(correlations_df) >= 3:
        # Stwórz wykres słupkowy z korelacjami ogólnymi
        fig = px.bar(
            correlations_df,
            x="Field",
            y="Overall Correlation",
            color="Commodity",
            barmode="group",
            title="Korelacja NDVI z cenami towarów",
            labels={"Overall Correlation": "Współczynnik korelacji", "Field": "Pole"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Stwórz wykres z korelacjami z opóźnieniem
        fig = px.scatter(
            correlations_df,
            x="Strongest Lag",
            y="Strongest Lag Correlation",
            color="Commodity",
            size=correlations_df["Strongest Lag Correlation"].abs() * 10,
            hover_name="Field",
            title="Najsilniejsze korelacje z opóźnieniem",
            labels={
                "Strongest Lag": "Opóźnienie (dni)",
                "Strongest Lag Correlation": "Współczynnik korelacji"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

# Funkcja do wyświetlania tabeli ostatnich prognoz dla wszystkich pól
def display_latest_predictions():
    # Pobierz dostępne pola
    fields = get_available_fields()
    
    if not fields:
        st.warning("Brak dostępnych pól do analizy.")
        return
    
    # Przygotuj tabele dla prognoz plonów i sygnałów rynkowych
    yield_data = []
    signal_data = []
    
    for field_name in fields:
        # Sprawdź prognozy plonów
        yield_file = Path(f"data/{field_name}_yield_forecast.json")
        if yield_file.exists():
            try:
                yield_forecast = load_data_from_file(yield_file)
                if yield_forecast and "forecasts" in yield_forecast:
                    for crop, forecasts in yield_forecast["forecasts"].items():
                        # Pobierz ostatnią prognozę
                        if forecasts:
                            dates = sorted(forecasts.keys())
                            latest_date = dates[-1]
                            latest_yield = forecasts[latest_date]
                            
                            yield_data.append({
                                "Field": field_name,
                                "Crop": crop,
                                "Forecast Date": latest_date,
                                "Predicted Yield (t/ha)": latest_yield,
                                "Last Updated": yield_forecast.get("date_updated", "Unknown")
                            })
            except Exception as e:
                logger.error(f"Błąd wczytywania prognoz plonów dla pola {field_name}: {str(e)}")
        
        # Sprawdź sygnały rynkowe
        signals_file = Path(f"data/{field_name}_market_signals.json")
        if signals_file.exists():
            try:
                market_signals = load_data_from_file(signals_file)
                if market_signals and "signals" in market_signals:
                    for commodity, signals in market_signals["signals"].items():
                        # Pobierz ostatni sygnał
                        if signals:
                            latest_signal = sorted(signals, key=lambda x: x.get("date", ""))[-1]
                            signal_data.append({
                                "Field": field_name,
                                "Commodity": commodity,
                                "Date": latest_signal.get("date", "Unknown"),
                                "Action": latest_signal.get("action", "NEUTRAL"),
                                "Confidence": latest_signal.get("confidence", 0),
                                "Last Updated": market_signals.get("date_updated", "Unknown")
                            })
            except Exception as e:
                logger.error(f"Błąd wczytywania sygnałów rynkowych dla pola {field_name}: {str(e)}")
    
    # Wyświetl tabele prognoz
    if yield_data:
        st.subheader("Najnowsze prognozy plonów")
        yield_df = pd.DataFrame(yield_data)
        st.dataframe(yield_df, use_container_width=True)
        
        # Stwórz wykres z prognozami plonów
        fig = px.bar(
            yield_df,
            x="Field",
            y="Predicted Yield (t/ha)",
            color="Crop",
            barmode="group",
            title="Najnowsze prognozy plonów",
            labels={"Predicted Yield (t/ha)": "Prognozowany plon (t/ha)", "Field": "Pole"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Brak danych prognoz plonów. Uruchom aktualizację, aby wygenerować prognozy.")
    
    if signal_data:
        st.subheader("Najnowsze sygnały rynkowe")
        signal_df = pd.DataFrame(signal_data)
        
        # Dodaj kolumnę z wartością numeryczną dla sygnału
        signal_df["Signal Value"] = signal_df["Action"].map({"LONG": 1, "NEUTRAL": 0, "SHORT": -1})
        
        # Wyświetl tabele bez stylowania (to eliminuje problemy z typami)
        st.dataframe(signal_df, use_container_width=True)
        
        # Dodaj prostą wizualizację sygnałów w formie tekstu
        st.subheader("Wizualizacja sygnałów")
        for _, row in signal_df.iterrows():
            if row["Action"] == "LONG":
                st.markdown(f"**{row['Field']} - {row['Commodity']}:** 🟢 LONG (Pewność: {row['Confidence']:.2f})")
            elif row["Action"] == "SHORT":
                st.markdown(f"**{row['Field']} - {row['Commodity']}:** 🔴 SHORT (Pewność: {row['Confidence']:.2f})")
            else:
                st.markdown(f"**{row['Field']} - {row['Commodity']}:** ⚪ NEUTRAL (Pewność: {row['Confidence']:.2f})")
        
        # Stwórz wykres z sygnałami rynkowymi
        fig = px.bar(
            signal_df,
            x="Field",
            y="Signal Value",
            color="Commodity",
            barmode="group",
            title="Najnowsze sygnały rynkowe",
            labels={"Signal Value": "Sygnał (1=LONG, 0=NEUTRAL, -1=SHORT)", "Field": "Pole"}
        )
        # Dodaj linię na poziomie 0
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Brak danych sygnałów rynkowych. Uruchom aktualizację, aby wygenerować sygnały.")

# Tytuł strony
st.title("🛰️ Satellite Monitor")
st.markdown("""
Automatyczna aktualizacja prognoz i analiza korelacji danych satelitarnych. System regularnie sprawdza dostępność nowych danych 
satelitarnych, generuje prognozy plonów, sygnały rynkowe i analizę ukrytych zależności.
""")

# Panel sterowania w sidebarze
st.sidebar.header("Aktualizacja prognoz")

# Przycisk do ręcznej aktualizacji
if st.sidebar.button("🔄 Aktualizuj prognozy i wykresy", type="primary"):
    manual_update()

if st.session_state.last_update_time:
    st.sidebar.markdown(f"**Ostatnia aktualizacja:** {st.session_state.last_update_time}")

# Wybór pola do wyświetlenia szczegółów
available_fields = get_available_fields()
if available_fields:
    selected_field = st.sidebar.selectbox(
        "Wybierz pole do wyświetlenia szczegółów", 
        options=available_fields
    )
else:
    st.warning("Brak dostępnych pól. Dodaj pola w zakładce Field Manager lub skonfiguruj połączenie z bazą danych.")
    st.stop()

# Główne zakładki
tab1, tab2, tab3 = st.tabs(["Aktualne prognozy", "Szczegóły pola", "Analiza korelacji"])

with tab1:
    st.header("Najnowsze prognozy i sygnały")
    st.markdown("""
    Ostatnie prognozy plonów i sygnały rynkowe dla wszystkich monitorowanych pól.
    Aby zaktualizować dane, kliknij przycisk 'Aktualizuj teraz' w panelu bocznym.
    """)
    
    display_latest_predictions()

with tab2:
    st.header(f"Szczegóły pola: {selected_field}")
    st.markdown("""
    Szczegółowa analiza wybranego pola z wykresami NDVI, prognoz plonów i sygnałów rynkowych.
    Wykresy są generowane po kliknięciu przycisku 'Aktualizuj prognozy i wykresy'.
    """)
    
    # Załaduj i wyświetl szczegółowe wykresy dla wybranego pola
    charts_loaded = load_and_display_charts(selected_field)
    
    if not charts_loaded:
        st.info("""
        Aby wygenerować wykresy, kliknij przycisk 'Aktualizuj prognozy i wykresy' w panelu bocznym.
        """)

with tab3:
    st.header("Analiza korelacji danych satelitarnych")
    st.markdown("""
    Analiza korelacji między danymi NDVI a cenami towarów rolnych.
    System automatycznie wykrywa ukryte zależności między wskaźnikami satelitarnymi a rynkiem.
    """)
    
    display_correlation_overview()

# Informacja o automatycznych aktualizacjach
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Informacja:** System automatycznie generuje:
- Prognozy plonów na podstawie danych NDVI
- Sygnały rynkowe dla towarów rolnych
- Analizę korelacji między danymi satelitarnymi a cenami
- Szczegółowe wykresy dla każdego pola
""")

# Uruchom monitorowanie przy pierwszym załadowaniu strony (opcjonalnie)
if "first_load" not in st.session_state:
    st.session_state.first_load = True
    # Nie uruchamiaj automatycznie, aby nie obciążać systemu
    # start_monitoring(interval_hours=monitoring_interval)
    # st.session_state.monitoring_status = True