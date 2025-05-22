"""
Expert Reasoning - Zaawansowana analiza z wykorzystaniem systemu agentów LLM
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from models.llm_agents import (
    get_agent_coordinator,
    analyze_field
)

from utils.predictions import get_prediction_manager

# Konfiguracja loggera
logger = logging.getLogger(__name__)

# Ustawienia strony
st.set_page_config(
    page_title="Expert Reasoning - Agro Insight",
    page_icon="🧠",
    layout="wide"
)

# Tytuł strony
st.title("🧠 Expert Reasoning")
st.markdown("""
Zaawansowana analiza z wykorzystaniem systemu agentów LLM. Ten moduł łączy analizę danych satelitarnych, 
prognozy plonów oraz analizę rynku w spójną narrację wspieraną przejrzystym procesem wnioskowania.
""")

# Pobierz koordynatora agentów
coordinator = get_agent_coordinator()

# Funkcja do pobierania dostępnych pól
def get_available_fields():
    prediction_manager = get_prediction_manager()
    return prediction_manager.get_available_fields()

# Funkcja do pobierania danych NDVI dla pola
def get_ndvi_data(field_name):
    prediction_manager = get_prediction_manager()
    return prediction_manager.get_ndvi_time_series(field_name)

# Funkcja do pobierania informacji o polu
def get_field_info(field_name):
    prediction_manager = get_prediction_manager()
    return prediction_manager.get_field_info(field_name)

# Funkcja do ładowania zapisanych analiz
def load_saved_analyses():
    results_dir = Path("data/agent_results")
    if not results_dir.exists():
        return []
    
    files = list(results_dir.glob("*_analysis_*.json"))
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Sortuj po czasie modyfikacji
    
    analyses = []
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            # Pobierz podstawowe informacje
            field_name = data.get("narrative", {}).get("field", "Nieznane pole")
            timestamp = data.get("narrative", {}).get("timestamp", "Nieznany czas")
            
            # Formatuj datę i czas
            try:
                dt = datetime.datetime.fromisoformat(timestamp)
                timestamp_formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                timestamp_formatted = timestamp
            
            analyses.append({
                "file_path": str(file),
                "field_name": field_name,
                "timestamp": timestamp_formatted,
                "analysis_id": file.stem
            })
        except Exception as e:
            logger.error(f"Błąd wczytywania analizy z pliku {file}: {str(e)}")
    
    return analyses

# Funkcja do wyświetlania strony
def display_reasoning_page():
    # Panel boczny z kontrolkami
    st.sidebar.header("Analiza pola")
    
    # Pobierz dostępne pola
    available_fields = get_available_fields()
    if not available_fields:
        st.sidebar.warning("Brak dostępnych pól. Dodaj pola w zakładce Field Manager.")
        return
    
    # Wybór pola
    selected_field = st.sidebar.selectbox(
        "Wybierz pole do analizy",
        options=available_fields
    )
    
    # Przycisk uruchamiający analizę
    run_analysis = st.sidebar.button("Przeprowadź analizę", type="primary")
    
    # Dodaj informację o procesie
    st.sidebar.markdown("""
    **Proces analizy:**
    1. Agent analizy satelitarnej ocenia stan uprawy
    2. Agent prognozy plonów przewiduje potencjalne zbiory
    3. Agent analizy rynku generuje sygnały handlowe
    4. Agent narracyjny tworzy spójne podsumowanie
    """)
    
    # Wyświetl zapisane analizy
    st.sidebar.markdown("---")
    st.sidebar.header("Zapisane analizy")
    
    saved_analyses = load_saved_analyses()
    if saved_analyses:
        analyses_df = pd.DataFrame(saved_analyses)
        selected_saved_analysis = st.sidebar.selectbox(
            "Wybierz zapisaną analizę",
            options=analyses_df["analysis_id"].tolist(),
            format_func=lambda x: f"{analyses_df[analyses_df['analysis_id']==x]['field_name'].iloc[0]} ({analyses_df[analyses_df['analysis_id']==x]['timestamp'].iloc[0]})"
        )
        
        load_saved = st.sidebar.button("Wczytaj analizę")
    else:
        st.sidebar.info("Brak zapisanych analiz.")
        load_saved = False
    
    # Główny panel z wynikami
    if run_analysis:
        with st.spinner("Trwa analiza pola..."):
            # Pobierz dane NDVI dla wybranego pola
            ndvi_data = get_ndvi_data(selected_field)
            if not ndvi_data:
                st.error(f"Brak danych NDVI dla pola {selected_field}. Nie można przeprowadzić analizy.")
                return
            
            # Pobierz informacje o polu
            field_info = get_field_info(selected_field)
            
            # Przeprowadź analizę
            results = analyze_field(selected_field, ndvi_data, field_info)
            
            # Wyświetl wyniki
            display_analysis_results(results)
    
    elif load_saved and saved_analyses:
        # Wczytaj zapisaną analizę
        selected_file = None
        for analysis in saved_analyses:
            if analysis["analysis_id"] == selected_saved_analysis:
                selected_file = analysis["file_path"]
                break
        
        if selected_file:
            with st.spinner("Wczytywanie zapisanej analizy..."):
                # Wczytaj analizę z pliku
                try:
                    with open(selected_file, 'r') as f:
                        results = json.load(f)
                    
                    # Wyświetl wyniki
                    display_analysis_results(results)
                except Exception as e:
                    st.error(f"Błąd wczytywania analizy z pliku: {str(e)}")
        else:
            st.warning("Nie znaleziono wybranej analizy.")
    
    else:
        # Wyświetl informacje o module
        st.info("""
        ### Jak to działa?
        
        Ten moduł wykorzystuje zestaw specjalistycznych agentów LLM, które współpracują ze sobą, 
        aby przeprowadzić wielowarstwową analizę. Każdy agent specjalizuje się w innym obszarze:
        
        1. **Agent analizy satelitarnej** - interpretuje dane NDVI i ocenia stan uprawy
        2. **Agent prognozy plonów** - przewiduje potencjalne plony na podstawie danych satelitarnych
        3. **Agent analizy rynku** - generuje sygnały handlowe i przewiduje trendy cenowe
        4. **Agent narracyjny** - tworzy spójne podsumowanie i rekomendacje
        
        System nie tylko generuje wyniki, ale także prezentuje pełny proces wnioskowania, 
        pozwalając zrozumieć, w jaki sposób wyprowadzono wnioski i rekomendacje.
        
        ### Rozpocznij analizę
        
        Wybierz pole z listy po lewej stronie i kliknij "Przeprowadź analizę", aby rozpocząć.
        """)

# Funkcja wyświetlająca wyniki analizy
def display_analysis_results(results):
    # Sprawdź, czy mamy wyniki
    if not results:
        st.warning("Brak wyników analizy.")
        return
    
    # Pobierz główne komponenty wyników
    satellite_analysis = results.get("satellite_analysis", {})
    yield_forecast = results.get("yield_forecast", {})
    market_analysis = results.get("market_analysis", {})
    narrative = results.get("narrative", {})
    
    # Sekcja z narracyjnym podsumowaniem
    st.header("Podsumowanie analizy")
    field_name = narrative.get("field", "nieznane pole")
    crop_type = narrative.get("crop_type", "nieznana uprawa")
    
    if narrative.get("status") == "success":
        st.markdown(f"### Pole: {field_name} ({crop_type})")
        
        # Wyświetl główną narrację
        st.markdown(narrative.get("narrative", "Brak narracji"))
        
        # Wyświetl rekomendacje dla rolnika
        recommendations = narrative.get("farmer_recommendations", [])
        if recommendations:
            st.subheader("Rekomendacje:")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
    else:
        st.error("Nie udało się wygenerować narracji dla analizy.")
    
    # Zakładki z szczegółowymi wynikami
    tab1, tab2, tab3, tab4 = st.tabs([
        "Analiza satelitarna", 
        "Prognoza plonów", 
        "Analiza rynku",
        "Proces wnioskowania"
    ])
    
    with tab1:
        st.subheader("Analiza danych satelitarnych")
        if satellite_analysis.get("status") == "success":
            # Podstawowe dane
            current_ndvi = satellite_analysis.get("current_ndvi", 0)
            health_status = satellite_analysis.get("health_status", "nieznany")
            trend_assessment = satellite_analysis.get("trend_assessment", "nieznany")
            
            # Wyświetl metryki w kolumnach
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Aktualne NDVI", value=f"{current_ndvi:.3f}")
            with col2:
                st.metric(label="Stan zdrowotny", value=health_status)
            with col3:
                st.metric(label="Trend", value=trend_assessment)
            with col4:
                min_ndvi = satellite_analysis.get("min_ndvi", 0)
                max_ndvi = satellite_analysis.get("max_ndvi", 0)
                st.metric(label="Zakres NDVI", value=f"{min_ndvi:.2f} - {max_ndvi:.2f}")
            
            # Wyświetl interpretację
            st.markdown("#### Interpretacja")
            st.markdown(satellite_analysis.get("interpretation", "Brak interpretacji"))
            
        else:
            st.error("Nie udało się przeprowadzić analizy satelitarnej.")
    
    with tab2:
        st.subheader("Prognoza plonów")
        if yield_forecast.get("status") == "success":
            # Pobierz dane prognozy
            crop_type = yield_forecast.get("crop_type", "nieznana uprawa")
            base_yield = yield_forecast.get("base_yield", 0)
            ndvi_factor = yield_forecast.get("ndvi_factor", 1)
            trend_factor = yield_forecast.get("trend_factor", 1)
            forecasts = yield_forecast.get("forecasts", {})
            
            # Wyświetl podstawowe informacje
            st.markdown(f"**Uprawa:** {crop_type}")
            
            # Wyświetl metryki w kolumnach
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Bazowy plon", value=f"{base_yield:.2f} t/ha")
            with col2:
                st.metric(label="Współczynnik NDVI", value=f"{ndvi_factor:.2f}")
            with col3:
                st.metric(label="Współczynnik trendu", value=f"{trend_factor:.2f}")
            
            # Wyświetl prognozy w tabeli
            if forecasts:
                st.markdown("#### Prognozy plonów")
                
                # Przygotuj dane do tabeli
                forecast_data = []
                for date, data in forecasts.items():
                    forecast_data.append({
                        "Data": date,
                        "Prognoza (t/ha)": data.get("yield", 0),
                        "Pewność": data.get("confidence", "nieznana"),
                        "Poziom niepewności": data.get("uncertainty_level", "nieznany")
                    })
                
                forecast_df = pd.DataFrame(forecast_data)
                st.table(forecast_df)
            
            # Wyświetl interpretację
            st.markdown("#### Interpretacja")
            st.markdown(yield_forecast.get("interpretation", "Brak interpretacji"))
            
        else:
            st.error("Nie udało się przeprowadzić prognozy plonów.")
    
    with tab3:
        st.subheader("Analiza rynku")
        if market_analysis.get("status") == "success":
            # Pobierz dane analizy rynku
            crop_type = market_analysis.get("crop_type", "nieznana uprawa")
            symbol = market_analysis.get("commodity_symbol", "nieznany")
            market_signal = market_analysis.get("market_signal", {})
            market_analysis_data = market_analysis.get("market_analysis", {})
            
            # Wyświetl podstawowe informacje
            st.markdown(f"**Uprawa:** {crop_type} | **Symbol towaru:** {symbol}")
            
            # Wyświetl sygnał rynkowy
            if market_signal:
                action = market_signal.get("action", "nieznany")
                confidence = market_signal.get("confidence", 0)
                reason = market_signal.get("reason", "nieznana przyczyna")
                
                # Wyświetl w atrakcyjny sposób
                signal_color = {
                    "LONG": "🟢",
                    "SHORT": "🔴",
                    "NEUTRAL": "⚪"
                }.get(action, "⚪")
                
                st.markdown(f"### Sygnał rynkowy: {signal_color} {action}")
                st.markdown(f"**Pewność:** {confidence:.2f}")
                st.markdown(f"**Uzasadnienie:** {reason}")
            
            # Wyświetl analizę rynku w różnych horyzontach czasowych
            st.markdown("#### Analiza rynku w różnych horyzontach czasowych")
            
            time_horizons = ["short_term", "medium_term", "long_term"]
            horizon_names = ["Krótki termin", "Średni termin", "Długi termin"]
            
            # Utwórz kolumny dla horyzontów czasowych
            cols = st.columns(3)
            
            for i, (horizon, name) in enumerate(zip(time_horizons, horizon_names)):
                with cols[i]:
                    if horizon in market_analysis_data:
                        horizon_data = market_analysis_data[horizon]
                        trend = horizon_data.get("trend", "nieznany")
                        confidence = horizon_data.get("confidence", 0)
                        
                        trend_color = {
                            "LONG": "🟢",
                            "SHORT": "🔴",
                            "NEUTRAL": "⚪"
                        }.get(trend, "⚪")
                        
                        st.markdown(f"**{name}:**")
                        st.markdown(f"{trend_color} {trend}")
                        st.markdown(f"Pewność: {confidence:.2f}")
                        st.markdown(horizon_data.get("reason", ""))
                    else:
                        st.markdown(f"**{name}:** Brak danych")
            
            # Wyświetl rekomendacje handlowe
            trade_recommendations = market_analysis.get("trade_recommendations", [])
            if trade_recommendations:
                st.markdown("#### Rekomendacje handlowe")
                for i, rec in enumerate(trade_recommendations, 1):
                    action = rec.get("action", "nieznana akcja")
                    rec_type = rec.get("type", "nieznany typ")
                    reasoning = rec.get("reasoning", "")
                    
                    st.markdown(f"{i}. **{action}** ({rec_type})")
                    st.markdown(f"   {reasoning}")
            
        else:
            st.error("Nie udało się przeprowadzić analizy rynku.")
    
    with tab4:
        st.subheader("Proces wnioskowania (reasoning)")
        
        # Wybór agenta do wyświetlenia procesu wnioskowania
        agents = ["Analiza satelitarna", "Prognoza plonów", "Analiza rynku", "Generator narracji"]
        agent_keys = ["satellite_analysis", "yield_forecast", "market_analysis", "narrative"]
        
        selected_agent = st.radio("Wybierz agenta:", agents, horizontal=True)
        agent_idx = agents.index(selected_agent)
        
        # Pobierz kroki reasoningu dla wybranego agenta
        agent_key = agent_keys[agent_idx]
        reasoning_steps = []
        
        if agent_key in results:
            agent_data = results[agent_key]
            reasoning_steps = agent_data.get("reasoning", [])
        
        # Wyświetl kroki reasoningu
        if reasoning_steps:
            for i, step in enumerate(reasoning_steps, 1):
                with st.expander(f"Krok {i}: {step.get('step', 'Krok wnioskowania')}"):
                    st.markdown(step.get("result", "Brak wyników"))
        else:
            st.info("Brak dostępnych kroków wnioskowania dla wybranego agenta.")

# Główna funkcja strony
def main():
    display_reasoning_page()

if __name__ == "__main__":
    main()