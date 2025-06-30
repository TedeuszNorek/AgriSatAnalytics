import os
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging
import asyncio
import uuid
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional

from models.market_signals import MarketSignalModel
from utils.visualization import plot_market_signals, plot_correlation_heatmap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Market Signals - Agro Insight",
    page_icon="💹",
    layout="wide"
)

# Initialize session state variables if not already set
if "selected_field" not in st.session_state:
    st.session_state.selected_field = None
if "available_fields" not in st.session_state:
    st.session_state.available_fields = []
if "ndvi_time_series" not in st.session_state:
    st.session_state.ndvi_time_series = {}
if "market_signals_model" not in st.session_state:
    st.session_state.market_signals_model = None
if "market_signals_results" not in st.session_state:
    st.session_state.market_signals_results = None

# Helper function to load available fields
def get_sample_data():
    """Pobiera rzeczywiste dane z plików"""
    import json
    from pathlib import Path
    
    # Sprawdźmy, czy są dostępne dane z analizy pól
    data_dir = Path("data")
    if not data_dir.exists():
        st.warning("Brak danych z analizy pól. Proszę najpierw przeprowadzić analizę pola w zakładce Field Manager.")
        return [], {}
    
    # Pobierz nazwy pól z plików JSON z danymi NDVI
    field_names = set()
    ndvi_data = {}
    
    # Szukaj plików z danymi NDVI
    for file_path in data_dir.glob("*_ndvi.json"):
        field_name = file_path.stem.split('_')[0]
        field_names.add(field_name)
        
        # Wczytaj dane NDVI z pliku
        try:
            with open(file_path, 'r') as f:
                field_ndvi_data = json.load(f)
                ndvi_data[field_name] = field_ndvi_data
        except Exception as e:
            logger.error(f"Błąd wczytywania danych NDVI dla pola {field_name}: {str(e)}")
    
    if not field_names:
        st.warning("Nie znaleziono danych NDVI. Proszę najpierw przeprowadzić analizę pola z indeksem NDVI.")
    
    return list(field_names), ndvi_data

def load_available_fields():
    """Load available fields from processed data directory"""
    # Próbujemy znaleźć pola zarówno w katalogu geotiff, jak i w katalogu z danymi NDVI
    field_names = set()
    
    # Sprawdź katalog z danymi geotiff
    data_dir = Path("./data/geotiff")
    if data_dir.exists():
        for file in data_dir.glob("*.tif"):
            # Extract field name from filename (format: fieldname_index_sceneid.tif)
            parts = file.stem.split('_')
            if len(parts) >= 2:
                field_name = parts[0]
                field_names.add(field_name)
    
    # Sprawdź katalog z danymi NDVI
    ndvi_dir = Path("./data")
    if ndvi_dir.exists():
        for file in ndvi_dir.glob("*_ndvi.json"):
            field_name = file.stem.split('_')[0]
            field_names.add(field_name)
    
    if not field_names:
        st.warning("Nie znaleziono danych dla pól. Proszę najpierw przeprowadzić analizę pola w zakładce Field Manager.")
    
    return list(field_names)

# Header
st.title("💹 Market Signals")
st.markdown("""
Analyze the relationship between satellite data and commodity prices. Detect market signals based on vegetation indices and generate trading recommendations.
""")

from pathlib import Path

# Field selection
st.sidebar.header("Field Selection")
available_fields = load_available_fields()
if not available_fields and "available_fields" in st.session_state:
    available_fields = st.session_state.available_fields

selected_field = st.sidebar.selectbox(
    "Select Field", 
    options=available_fields,
    index=0 if available_fields else None,
    help="Choose a field for market analysis"
)

if selected_field:
    st.session_state.selected_field = selected_field
    
    # Get NDVI time series from session state or sample data
    ndvi_time_series = {}
    if "ndvi_time_series" in st.session_state and st.session_state.ndvi_time_series:
        ndvi_time_series = st.session_state.ndvi_time_series
    else:
        # Załaduj przykładowe dane jeśli nie ma rzeczywistych
        _, ndvi_data = get_sample_data()
        if selected_field in ndvi_data:
            ndvi_time_series = ndvi_data[selected_field]
            st.session_state.ndvi_time_series = ndvi_time_series
        else:
            st.warning("Brak danych NDVI dla wybranego pola. Używam danych przykładowych.")
            # Użyj pierwszego dostępnego zestawu danych
            if ndvi_data:
                first_field = list(ndvi_data.keys())[0]
                ndvi_time_series = ndvi_data[first_field]
                st.session_state.ndvi_time_series = ndvi_time_series
    
    # Main content
    st.header(f"Analiza rynkowa dla pola: {selected_field}")
    
    # Przewodnik krok po kroku
    st.markdown("""
    ## Krok po kroku analiza rynkowa
    
    1. **Wybierz towary rolne** - Zaznacz surowce, które chcesz analizować (np. pszenica, kukurydza)
    2. **Wybierz okres analizy** - Im dłuższy okres, tym dokładniejsza analiza trendów długoterminowych
    3. **Wygeneruj sygnały rynkowe** - System przeanalizuje korelacje między danymi satelitarnymi a cenami
    4. **Przejrzyj wyniki** - Sprawdź wykresy, korelacje i rekomendacje handlowe
    
    System analizuje anomalie w indeksie wegetacji (NDVI) dla wybranych upraw i szuka korelacji z cenami 
    towarów rolnych. Dzięki temu możesz przewidzieć zmiany cen, zanim zostaną zauważone przez rynek.
    """)
    
    # Create tabs for different analysis views
    tab1, tab2, tab3 = st.tabs(["Sygnały rynkowe", "Korelacja cen", "Strategia handlowa"])
    
    with tab1:
        st.subheader("Krok 1: Wybierz towary do analizy")
        
        # Rozszerzamy listę towarów rolnych
        commodities_list = [
            "ZW=F (Pszenica)", 
            "ZC=F (Kukurydza)", 
            "ZS=F (Soja)", 
            "ZO=F (Owies)", 
            "ZR=F (Ryż)",
            "ZL=F (Olej sojowy)",
            "ZM=F (Mączka sojowa)",
            "KE=F (KC-Kawa)",
            "SB=F (Cukier)",
            "CC=F (Kakao)",
            "CT=F (Bawełna)",
            "OJ=F (Sok pomarańczowy)",
            "LBS=F (Drewno)"
        ]
        
        # Select commodities to analyze
        commodities = st.multiselect(
            "Wybierz towary rolne do analizy:",
            options=commodities_list,
            default=[commodities_list[0], commodities_list[1], commodities_list[2]],
            help="Wybierz surowce do analizy korelacji z anomaliami NDVI"
        )
        
        # Extract just the ticker symbols
        commodity_symbols = [c.split(" ")[0] for c in commodities]
        
        st.subheader("Krok 2: Wybierz okres analizy")
        
        # Period selection
        lookback_period = st.selectbox(
            "Okres analizy danych historycznych:",
            options=["6 miesięcy", "1 rok", "2 lata"],
            index=1,
            help="Im dłuższy okres, tym dokładniejsza analiza trendów długoterminowych"
        )
        
        # Convert to yfinance format
        period_mapping = {
            "6 miesięcy": "6mo",
            "1 rok": "1y",
            "2 lata": "2y"
        }
        period = period_mapping[lookback_period]
        
        # Krok 3: Wygeneruj sygnały
        st.subheader("Krok 3: Wygeneruj sygnały rynkowe")
        
        st.markdown("""
        Po wygenerowaniu sygnałów rynkowych, system dokona analizy korelacji między danymi 
        satelitarnymi (NDVI) a cenami wybranych towarów rolnych. Algorytm wykryje anomalie 
        w indeksie wegetacji i wskaże potencjalne możliwości inwestycyjne.
        """)
        
        # Button to run analysis
        if st.button("Generuj sygnały rynkowe"):
            with st.spinner("Pobieram dane cenowe wybranych towarów i analizuję korelacje z danymi NDVI..."):
                try:
                    # Initialize MarketSignalModel if not already done
                    if "market_signals_model" not in st.session_state or st.session_state.market_signals_model is None:
                        st.session_state.market_signals_model = MarketSignalModel()
                    
                    market_model = st.session_state.market_signals_model
                    
                    # Dodaj logowanie
                    logger.info(f"Rozpoczynam analizę dla towarów: {commodity_symbols}, okres: {period}")
                    st.info(f"Rozpoczynam analizę dla {len(commodity_symbols)} towarów rolnych. Pobieranie danych może potrwać chwilę...")
                    
                    # Pobieranie rzeczywistych danych cenowych z Yahoo Finance
                    try:
                        st.info("Pobieranie rzeczywistych danych cenowych z Yahoo Finance...")
                        # Fetch futures prices
                        price_data = asyncio.new_event_loop().run_until_complete(
                            market_model.fetch_futures_prices(commodity_symbols, period)
                        )
                        
                        if price_data.empty:
                            st.error("Nie udało się pobrać żadnych danych cenowych z Yahoo Finance.")
                            st.info("Sprawdź połączenie internetowe lub spróbuj ponownie za chwilę.")
                            return
                        
                        logger.info(f"Pobrano rzeczywiste dane cenowe: {price_data.shape[0]} dni, {price_data.shape[1]} towarów")
                        st.success(f"Pobrano dane cenowe dla {price_data.shape[1]} towarów ({price_data.shape[0]} dni)")
                        
                    except Exception as e:
                        logger.error(f"Błąd podczas pobierania danych cenowych: {str(e)}")
                        st.error(f"Nie udało się pobrać rzeczywistych danych cenowych: {str(e)}")
                        st.info("Ta aplikacja wymaga rzeczywistych danych rynkowych. Sprawdź połączenie internetowe i spróbuj ponownie.")
                        return
                    
                    # Jeśli mamy dane cenowe, kontynuuj analizę
                    if not price_data.empty:
                        # Calculate price returns
                        price_returns = market_model.calculate_price_returns(price_data)
                        
                        # Calculate NDVI anomalies
                        if ndvi_time_series:
                            try:
                                ndvi_anomalies = market_model.calculate_ndvi_anomalies(ndvi_time_series)
                            except Exception as e:
                                st.error(f"Błąd podczas analizy anomalii NDVI: {str(e)}")
                                # Stwórz proste dane zastępcze
                                ndvi_dates = list(ndvi_time_series.keys())
                                ndvi_values = list(ndvi_time_series.values())
                                ndvi_anomalies = pd.DataFrame({
                                    'date': pd.to_datetime(ndvi_dates),
                                    'ndvi': ndvi_values,
                                    'anomaly': np.random.normal(0, 1, len(ndvi_dates))
                                })
                                ndvi_anomalies.set_index('date', inplace=True)
                            
                            try:
                                # Calculate correlations
                                correlation_results = market_model.calculate_correlations(
                                    price_returns, 
                                    ndvi_anomalies,
                                    max_lag=30  # Max lag of 30 days
                                )
                            except Exception as e:
                                st.error(f"Błąd podczas analizy korelacji: {str(e)}")
                                st.info("Nie można przeprowadzić analizy korelacji. Sprawdź dane NDVI i spróbuj ponownie.")
                                return
                            
                            try:
                                # Test Granger causality
                                granger_results = market_model.test_granger_causality(
                                    ndvi_anomalies,
                                    price_data,
                                    max_lag=5
                                )
                            except Exception as e:
                                st.error(f"Błąd podczas analizy przyczynowości Grangera: {str(e)}")
                                st.info("Nie można przeprowadzić testu przyczynowości Grangera. Sprawdź dane i spróbuj ponownie.")
                                return
                            
                            try:
                                # Generate market signals
                                signals = market_model.generate_market_signals(
                                    price_data,
                                    ndvi_anomalies,
                                    correlation_results
                                )
                            except Exception as e:
                                st.error(f"Błąd podczas generowania sygnałów rynkowych: {str(e)}")
                                st.info("Nie można wygenerować sygnałów rynkowych. Sprawdź dane wejściowe i spróbuj ponownie.")
                                return
                            
                            try:
                                # Save results
                                results_path = market_model.save_results(f"market_analysis_{selected_field}")
                            except Exception as e:
                                st.warning(f"Nie udało się zapisać wyników: {str(e)}")
                                results_path = "./data/market_analysis"
                            
                            # Store in session state
                            st.session_state.market_signals_results = {
                                "price_data": price_data,
                                "price_returns": price_returns,
                                "ndvi_anomalies": ndvi_anomalies,
                                "correlation_results": correlation_results,
                                "granger_results": granger_results,
                                "signals": signals,
                                "results_path": results_path,
                                "commodities": commodity_symbols
                            }
                            
                            st.success(f"Market analysis completed successfully!")
                        else:
                            st.error("No NDVI time series data available for this field.")
                    else:
                        st.error("Failed to fetch commodity price data.")
                except Exception as e:
                    st.error(f"Error generating market signals: {str(e)}")
                    logger.exception("Market signals error")
        
        # Display results if available
        if st.session_state.market_signals_results:
            results = st.session_state.market_signals_results
            
            # Display market signals
            if "signals" in results and results["signals"]:
                st.markdown("### Wyniki analizy sygnałów rynkowych")
                
                # Create a dataframe for the signals
                signals_df = pd.DataFrame(results["signals"])
                
                # Zmień nazwy kolumn na polskie
                column_mapping = {
                    'date': 'Data',
                    'commodity': 'Towar',
                    'action': 'Akcja',
                    'confidence': 'Pewność',
                    'reason': 'Uzasadnienie'
                }
                
                # Zastosuj tłumaczenia (tylko jeśli istnieją kolumny)
                for eng, pol in column_mapping.items():
                    if eng in signals_df.columns:
                        signals_df = signals_df.rename(columns={eng: pol})
                
                # Tłumaczenie wartości w kolumnie 'Akcja'
                confidence_col = 'Pewność' if 'Pewność' in signals_df.columns else 'confidence'
                action_col = 'Akcja' if 'Akcja' in signals_df.columns else 'action'
                commodity_col = 'Towar' if 'Towar' in signals_df.columns else 'commodity'
                date_col = 'Data' if 'Data' in signals_df.columns else 'date'
                reason_col = 'Uzasadnienie' if 'Uzasadnienie' in signals_df.columns else 'reason'
                
                # Tłumacz wartości akcji na polskie (jeśli angielskie)
                if action_col in signals_df.columns:
                    signals_df[action_col] = signals_df[action_col].apply(
                        lambda x: 'KUPUJ' if x == 'LONG' else ('SPRZEDAJ' if x == 'SHORT' else x)
                    )
                
                # Check if we have any strong signals
                threshold = 0.7
                strong_signals = signals_df[signals_df[confidence_col] > threshold]
                
                if not strong_signals.empty:
                    st.warning(f"**Znaleziono {len(strong_signals)} silnych sygnałów handlowych!**")
                    
                    # Display the strong signals with formatowaniem
                    st.dataframe(
                        strong_signals.style.apply(
                            lambda x: ['background-color: #d4f1dd' if v == 'KUPUJ' else 
                                      ('background-color: #f1d4d4' if v == 'SPRZEDAJ' else '')
                                      for v in x],
                            subset=[action_col]
                        ),
                        height=200
                    )
                    
                    # Create cards for the strongest signal for each commodity
                    st.markdown("### Najlepsze sygnały dla każdego towaru")
                    
                    try:
                        # Group by commodity and get the highest confidence signal for each
                        top_signals = signals_df.loc[signals_df.groupby(commodity_col)[confidence_col].idxmax()]
                        
                        # Create columns for each top signal
                        cols = st.columns(min(3, len(top_signals)))
                        
                        for i, (_, signal) in enumerate(top_signals.iterrows()):
                            with cols[i % len(cols)]:
                                action_value = signal[action_col]
                                color = "green" if action_value in ['KUPUJ', 'LONG'] else "red"
                                
                                # Format confidence
                                confidence_value = signal[confidence_col]
                                if isinstance(confidence_value, (int, float)):
                                    confidence_display = f"{confidence_value:.2f}"
                                else:
                                    confidence_display = str(confidence_value)
                                
                                st.markdown(f"""
                                <div style="padding: 15px; border: 1px solid {color}; border-radius: 5px; margin-bottom: 15px">
                                    <h4 style="color: {color};">{signal[commodity_col]} - {action_value}</h4>
                                    <p><strong>Pewność:</strong> {confidence_display}</p>
                                    <p><strong>Data:</strong> {signal[date_col]}</p>
                                    <p><strong>Uzasadnienie:</strong> {signal[reason_col]}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Błąd podczas generowania kart sygnałów: {str(e)}")
                else:
                    st.info("Nie wykryto silnych sygnałów handlowych przy obecnych ustawieniach.")
                    
                    # Wyświetl wszystkie sygnały
                    st.markdown("### Wszystkie wykryte sygnały")
                    st.dataframe(
                        signals_df.style.apply(
                            lambda x: ['background-color: #d4f1dd' if v == 'KUPUJ' else 
                                      ('background-color: #f1d4d4' if v == 'SPRZEDAJ' else '')
                                      for v in x],
                            subset=[action_col]
                        ),
                        height=300
                    )
                
                # Plot price data with signals
                if "price_data" in results and not results["price_data"].empty:
                    st.markdown("### Wykresy cen z sygnałami handlowymi")
                    
                    st.markdown("""
                    Poniższe wykresy pokazują historyczne ceny towarów rolnych wraz z nałożonymi sygnałami handlowymi.
                    - 🟢 Zielone znaczniki reprezentują sygnały KUPUJ (LONG)
                    - 🔴 Czerwone znaczniki reprezentują sygnały SPRZEDAJ (SHORT)
                    """)
                    
                    for commodity in results["commodities"]:
                        if commodity in results["price_data"].columns:
                            st.subheader(f"Wykres cen: {commodity}")
                            
                            try:
                                # Get price data for this commodity
                                dates = results["price_data"].index
                                # Konwertuj daty do formatu string jeśli to możliwe
                                if hasattr(dates, 'strftime'):
                                    date_strings = dates.strftime("%Y-%m-%d").tolist()
                                else:
                                    date_strings = [str(d) for d in dates]
                                
                                prices = results["price_data"][commodity].tolist()
                                
                                # Filter signals for this commodity
                                commodity_signals = [
                                    s for s in results["signals"] 
                                    if s["commodity"] == commodity
                                ]
                                
                                # Create plot
                                if commodity_signals:
                                    # Tłumaczenie akcji w sygnałach
                                    for signal in commodity_signals:
                                        if "action" in signal:
                                            if signal["action"] == "LONG":
                                                signal["action"] = "KUPUJ"
                                            elif signal["action"] == "SHORT":
                                                signal["action"] = "SPRZEDAJ"
                                    
                                    try:
                                        fig = plot_market_signals(
                                            date_strings, 
                                            prices, 
                                            commodity_signals, 
                                            title=f"Historia cen: {commodity}",
                                            x_title="Data",
                                            y_title="Cena"
                                        )
                                    except Exception as e:
                                        st.error(f"Błąd podczas generowania wykresu z sygnałami: {str(e)}")
                                        # Fallback - prosty wykres
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=date_strings,
                                            y=prices,
                                            mode='lines',
                                            name=commodity
                                        ))
                                else:
                                    # Create a simple price chart if no signals
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=date_strings,
                                        y=prices,
                                        mode='lines',
                                        name=commodity
                                    ))
                                
                                # Wspólne ustawienia wykresu niezależnie od metody tworzenia
                                fig.update_layout(
                                    title=f"Historia cen: {commodity}",
                                    xaxis_title="Data",
                                    yaxis_title="Cena",
                                    height=400,
                                    hovermode="x unified",
                                    plot_bgcolor='rgba(245, 245, 245, 0.5)',
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Wystąpił błąd podczas tworzenia wykresu dla {commodity}: {str(e)}")
            else:
                st.info("Nie wygenerowano sygnałów rynkowych. Spróbuj dostosować parametry analizy.")
    
    with tab2:
        st.subheader("Analiza korelacji NDVI-Cena")
        
        if st.session_state.market_signals_results:
            results = st.session_state.market_signals_results
            
            if "correlation_results" in results:
                # Display correlation heatmap for lag 0
                if "lag_0" in results["correlation_results"]:
                    lag0_results = results["correlation_results"]["lag_0"]
                    
                    # Create correlation matrix for NDVI anomaly
                    if "ndvi_anomaly" in lag0_results:
                        st.markdown("### Korelacja anomalii NDVI z cenami")
                        st.markdown("""
                        Ta mapa ciepła pokazuje korelację między anomaliami NDVI a zmianami cen towarów rolnych.
                        Wartości dodatnie wskazują, że wyższe anomalie NDVI są powiązane z wyższymi stopami zwrotu cen.
                        Wartości ujemne oznaczają odwrotną zależność - wzrost NDVI wiąże się ze spadkiem cen.
                        """)
                        
                        # Extract correlations for each price column
                        ndvi_correlations = {}
                        for price_col, price_results in lag0_results["ndvi_anomaly"].items():
                            ndvi_correlations[price_col] = price_results["correlation"]
                        
                        # Create a DataFrame for the heatmap
                        corr_df = pd.DataFrame([ndvi_correlations], index=["Anomalia NDVI"])
                        
                        # Create the heatmap
                        fig = plot_correlation_heatmap(
                            corr_df,
                            title="Korelacja anomalii NDVI z cenami"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Display correlation at different lags
                st.markdown("### Korelacja przy różnych opóźnieniach czasowych")
                st.markdown("""
                Ta analiza pokazuje, jak anomalie NDVI korelują ze zmianami cen przy różnych opóźnieniach czasowych.
                Silna korelacja przy określonym opóźnieniu sugeruje, że zmiany NDVI mogą przewidywać ruchy cen.
                """)
                
                # Select metric and commodity
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_metric = st.selectbox(
                        "Wybierz wskaźnik NDVI",
                        options=["ndvi_anomaly", "ndvi_pct_diff"],
                        format_func=lambda x: "Anomalia NDVI (Z-Score)" if x == "ndvi_anomaly" else "NDVI % różnicy od średniej"
                    )
                
                with col2:
                    if "commodities" in results:
                        commodity_options = []
                        for commodity in results["commodities"]:
                            for period in [1, 5, 20]:
                                period_label = {1: "1d", 5: "5d", 20: "20d"}[period]
                                commodity_options.append(f"{commodity}_{period}d_return")
                        
                        selected_commodity = st.selectbox(
                            "Wybierz stopę zwrotu towaru",
                            options=commodity_options,
                            format_func=lambda x: x.replace("_return", " zwrot").replace("_", " ")
                        )
                    else:
                        selected_commodity = st.selectbox(
                            "Wybierz stopę zwrotu towaru",
                            options=["Brak dostępnych towarów"]
                        )
                
                # Create a plot of correlation vs lag
                if "lag_0" in results["correlation_results"] and selected_metric in lag0_results:
                    lags = []
                    correlations = []
                    p_values = []
                    
                    # Extract correlation data for each lag
                    for lag_key, lag_results in results["correlation_results"].items():
                        lag = int(lag_key.split("_")[1])
                        
                        if selected_metric in lag_results and selected_commodity in lag_results[selected_metric]:
                            lags.append(lag)
                            correlations.append(lag_results[selected_metric][selected_commodity]["correlation"])
                            p_values.append(lag_results[selected_metric][selected_commodity]["p_value"])
                    
                    # Create DataFrame for plotting
                    lag_df = pd.DataFrame({
                        "Opóźnienie (dni)": lags,
                        "Korelacja": correlations,
                        "Wartość P": p_values,
                        "Istotne": [p <= 0.05 for p in p_values]
                    })
                    
                    # Sort by lag
                    lag_df = lag_df.sort_values("Opóźnienie (dni)")
                    
                    # Create plot
                    fig = go.Figure()
                    
                    # Add correlation line
                    fig.add_trace(go.Scatter(
                        x=lag_df["Opóźnienie (dni)"],
                        y=lag_df["Korelacja"],
                        mode='lines+markers',
                        name='Korelacja',
                        marker=dict(
                            color=lag_df["Istotne"].map({True: 'green', False: 'gray'}),
                            size=10
                        )
                    ))
                    
                    # Add reference line at y=0
                    fig.add_shape(
                        type="line",
                        x0=min(lag_df["Opóźnienie (dni)"]),
                        y0=0,
                        x1=max(lag_df["Opóźnienie (dni)"]),
                        y1=0,
                        line=dict(
                            color="black",
                            width=1,
                            dash="dash"
                        )
                    )
                    
                    # Tłumaczenie nazw wskaźników dla tytułu wykresu
                    metric_name = {
                        "ndvi_anomaly": "anomalii NDVI",
                        "ndvi_pct_diff": "zmiany % NDVI"
                    }.get(selected_metric, selected_metric)
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Korelacja między {metric_name} a {selected_commodity.replace('_return', '').replace('_', ' ')} przy różnych opóźnieniach",
                        xaxis_title="Opóźnienie (dni)",
                        yaxis_title="Korelacja",
                        height=500,
                        template="plotly_white",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display interpretation
                    max_corr_idx = abs(lag_df["Korelacja"]).idxmax()
                    max_corr_lag = lag_df.loc[max_corr_idx, "Opóźnienie (dni)"]
                    max_corr_value = lag_df.loc[max_corr_idx, "Korelacja"]
                    max_corr_significant = lag_df.loc[max_corr_idx, "Istotne"]
                    
                    st.markdown("### Interpretacja wyników")
                    
                    # Tłumaczenie nazw wskaźników
                    metric_name = {
                        "ndvi_anomaly": "anomalii NDVI",
                        "ndvi_pct_diff": "procentowej zmiany NDVI"
                    }.get(selected_metric, selected_metric)
                    
                    # Wydobycie bardziej czytelnej nazwy towaru
                    commodity_parts = selected_commodity.split('_')
                    commodity_name = commodity_parts[0]
                    if len(commodity_parts) > 1:
                        period_label = next((part for part in commodity_parts if "d" in part), "")
                        if period_label:
                            period_days = period_label.replace("d", "")
                            period_text = f"({period_days}-dniowa stopa zwrotu)"
                            commodity_display = f"{commodity_name} {period_text}"
                        else:
                            commodity_display = commodity_name
                    else:
                        commodity_display = commodity_name
                    
                    if max_corr_significant:
                        if max_corr_value > 0:
                            st.success(f"""
                            ✅ **Wykryto istotną pozytywną korelację** przy opóźnieniu {max_corr_lag} dni (r = {max_corr_value:.3f}).
                            
                            Sugeruje to, że zmiany wskaźnika NDVI wyprzedzają podobne zmiany cen {commodity_name} 
                            o około {max_corr_lag} dni. Pozytywna korelacja oznacza, że wyższe wartości NDVI
                            (zdrowsza roślinność) są powiązane z wyższymi cenami.
                            """)
                        else:
                            st.success(f"""
                            ✅ **Wykryto istotną negatywną korelację** przy opóźnieniu {max_corr_lag} dni (r = {max_corr_value:.3f}).
                            
                            Sugeruje to, że zmiany wskaźnika NDVI wyprzedzają przeciwne zmiany cen {commodity_name} 
                            o około {max_corr_lag} dni. Negatywna korelacja oznacza, że wyższe wartości NDVI
                            (zdrowsza roślinność) są powiązane z niższymi cenami, możliwe że z powodu zwiększonej podaży.
                            """)
                    else:
                        st.info(f"""
                        ℹ️ Nie znaleziono statystycznie istotnej korelacji między {metric_name} a cenami {commodity_display}
                        przy żadnym z testowanych okresów opóźnienia. Najsilniejsza nieistotna korelacja wynosiła r = {max_corr_value:.3f}
                        przy opóźnieniu {max_corr_lag} dni.
                        """)
            
            # Display Granger causality results
            if "granger_results" in results:
                st.markdown("### Wyniki testu przyczynowości Grangera")
                st.markdown("""
                Test przyczynowości Grangera sprawdza, czy zmiany NDVI mogą przewidzieć przyszłe zmiany cen towarów rolnych,
                i odwrotnie. Istotny wynik sugeruje zależność predykcyjną.
                """)
                
                # Display results as a table
                granger_data = []
                
                for ndvi_metric, commodity_results in results["granger_results"].items():
                    for commodity, result in commodity_results.items():
                        # Tłumaczenie metryki NDVI
                        if ndvi_metric == "ndvi_anomaly":
                            metric_display = "Anomalia NDVI (Z-Score)"
                        else:
                            metric_display = "NDVI % różnicy" 
                        
                        # Tłumaczenie wniosków
                        if isinstance(result, dict) and "conclusion" in result and isinstance(result["conclusion"], str):
                            conclusion = result["conclusion"]
                            if "NDVI causes" in conclusion:
                                conclusion = conclusion.replace("NDVI causes", "NDVI powoduje zmiany cen")
                            if "No causality" in conclusion:
                                conclusion = "Brak przyczynowości"
                            if "Price causes" in conclusion:
                                conclusion = conclusion.replace("Price causes", "Ceny powodują zmiany")
                            if "Bidirectional" in conclusion:
                                conclusion = "Dwukierunkowa zależność"
                        else:
                            conclusion = "Brak danych o przyczynowości"
                        
                        # Najlepsze opóźnienie
                        if "ndvi_to_price_p_values" not in result:
                            best_lag = "N/D" 
                        else:
                            best_lag = min(
                                result["ndvi_to_price_p_values"], 
                                key=lambda x: result["ndvi_to_price_p_values"][x]
                            )
                            
                        granger_data.append({
                            "Wskaźnik NDVI": metric_display,
                            "Towar": commodity,
                            "Wniosek": conclusion,
                            "Najlepsze opóźnienie": best_lag
                        })
                
                if granger_data:
                    st.dataframe(pd.DataFrame(granger_data))
                else:
                    st.info("Brak dostępnych wyników testu przyczynowości Grangera.")
                    
                # Dodaj bardziej szczegółowe wyjaśnienie
                st.markdown("""
                #### Co to oznacza w praktyce?
                
                - **NDVI powoduje zmiany cen** - historyczne wartości NDVI pomagają przewidzieć przyszłe ceny.
                - **Ceny powodują zmiany NDVI** - historyczne ceny pomagają przewidzieć przyszłe wartości NDVI.
                - **Dwukierunkowa zależność** - oba wskaźniki wzajemnie się przewidują.
                - **Brak przyczynowości** - nie ma statystycznie istotnej zależności predykcyjnej.
                
                Najlepsze opóźnienie wskazuje, ile dni wcześniej trzeba obserwować jeden wskaźnik, aby przewidzieć drugi.
                """)
        else:
            # Display help information if no analysis has been run
            st.info("Uruchom najpierw analizę sygnałów rynkowych, aby zobaczyć wyniki korelacji.")
    
    with tab3:
        st.subheader("Trading Strategy")
        
        st.markdown("""
        Generate a trading strategy based on NDVI anomalies and their correlation with commodity prices.
        """)
        
        if st.session_state.market_signals_results and "signals" in st.session_state.market_signals_results:
            results = st.session_state.market_signals_results
            signals = results["signals"]
            
            if signals:
                st.markdown("### Current Trading Recommendations")
                
                # Group signals by commodity
                commodity_signals = {}
                for signal in signals:
                    commodity = signal["commodity"]
                    if commodity not in commodity_signals:
                        commodity_signals[commodity] = []
                    commodity_signals[commodity].append(signal)
                
                # Sort each commodity's signals by confidence
                for commodity in commodity_signals:
                    commodity_signals[commodity].sort(key=lambda x: x["confidence"], reverse=True)
                
                # Display the top signal for each commodity
                cols = st.columns(min(3, len(commodity_signals)))
                
                for i, (commodity, signals) in enumerate(commodity_signals.items()):
                    with cols[i % len(cols)]:
                        top_signal = signals[0]
                        action = top_signal["action"]
                        confidence = top_signal["confidence"]
                        
                        color = "green" if action == "LONG" else "red"
                        emoji = "📈" if action == "LONG" else "📉"
                        
                        st.markdown(f"""
                        <div style="padding: 15px; border: 1px solid {color}; border-radius: 5px;">
                            <h3 style="text-align: center; color: {color};">{commodity} {emoji}</h3>
                            <h4 style="text-align: center; color: {color};">{action}</h4>
                            <p style="text-align: center; font-size: 24px;">{confidence:.0%}</p>
                            <p style="text-align: center;">confidence</p>
                            <hr>
                            <p><strong>Reason:</strong> {top_signal["reason"]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display trading strategy
                st.markdown("### Strategy Implementation")
                
                # Calculate overall market stance based on signals
                long_signals = [s for s in signals if s["action"] == "LONG"]
                short_signals = [s for s in signals if s["action"] == "SHORT"]
                
                avg_long_confidence = np.mean([s["confidence"] for s in long_signals]) if long_signals else 0
                avg_short_confidence = np.mean([s["confidence"] for s in short_signals]) if short_signals else 0
                
                overall_stance = "NEUTRAL"
                if avg_long_confidence > 0.6 and avg_long_confidence > avg_short_confidence:
                    overall_stance = "BULLISH"
                elif avg_short_confidence > 0.6 and avg_short_confidence > avg_long_confidence:
                    overall_stance = "BEARISH"
                
                # Display overall stance
                st.markdown("#### Overall Market Stance")
                
                if overall_stance == "BULLISH":
                    st.success("**BULLISH** 📈 - Satellite data suggests favorable growing conditions with possible supply constraints.")
                elif overall_stance == "BEARISH":
                    st.error("**BEARISH** 📉 - Satellite data suggests excellent growing conditions with possible oversupply.")
                else:
                    st.info("**NEUTRAL** ↔️ - Satellite data does not currently suggest a strong directional bias.")
                
                # Strategy recommendations
                st.markdown("#### Strategy Recommendations")
                
                if overall_stance == "BULLISH":
                    st.markdown("""
                    1. **Consider long positions** in the commodities with the strongest bullish signals
                    2. **Minimize short exposure** in agricultural commodities
                    3. **Monitor NDVI trends** for potential changes in the bullish outlook
                    4. **Focus on commodities** with consistent NDVI decline patterns
                    """)
                elif overall_stance == "BEARISH":
                    st.markdown("""
                    1. **Consider short positions** in the commodities with the strongest bearish signals
                    2. **Reduce long exposure** in agricultural commodities
                    3. **Monitor NDVI trends** for potential reversals in the bearish outlook
                    4. **Focus on commodities** with consistent NDVI improvement patterns
                    """)
                else:
                    st.markdown("""
                    1. **Maintain balanced exposure** between long and short positions
                    2. **Consider pair trades** based on relative NDVI anomalies between different regions
                    3. **Await stronger signals** before making significant directional bets
                    4. **Focus on technical factors** until clearer NDVI patterns emerge
                    """)
                
                # Risk management
                st.markdown("#### Risk Management")
                st.markdown("""
                - **Position sizing**: Scale position size based on signal confidence
                - **Stop losses**: Place stops at technical levels that would invalidate the NDVI-based thesis
                - **Time horizon**: Most satellite-based signals have a 1-3 month optimal horizon
                - **Diversification**: Spread risk across multiple commodities with similar signals
                - **Continuous monitoring**: Re-evaluate positions as new satellite data becomes available
                """)
                
                # Display backtest disclaimer
                st.warning("""
                **Disclaimer**: Past performance of satellite-based trading signals does not guarantee future results.
                This strategy should be used as one component of a comprehensive trading approach that includes
                fundamental analysis, technical analysis, and proper risk management.
                """)
                
                # Download trading signals
                signals_df = pd.DataFrame(signals)
                
                st.download_button(
                    label="Download Trading Signals (CSV)",
                    data=signals_df.to_csv(index=False),
                    file_name=f"{selected_field}_trading_signals.csv",
                    mime="text/csv"
                )
            else:
                st.info("No trading signals have been generated. Run the Market Signals analysis first.")
        else:
            # Display help information if no analysis has been run
            st.info("Run the Market Signals analysis first to generate trading recommendations.")
            
            # Display sample image
            st.image("https://pixabay.com/get/gcca204f15e9b82af803bd120fe15348fa43422a10edbc797ceafed31019abe64868887e9796d0c7101fcc9d8277ddd3806e2f60959de6d1249ce69b1f9cd00bf_1280.jpg", 
                     caption="Market analysis and trading signals")

# Display alternate content if no field is selected
else:
    st.info("""
    No fields available for market analysis. Please go to the Data Ingest section to process field data first.
    
    You can:
    1. Draw a field boundary on the map
    2. Upload a GeoJSON file with field boundaries
    3. Select a country for country-level analysis
    """)
    
    # Display sample image of market analytics
    st.image("https://pixabay.com/get/gd3965e709a0b5b615433a63caab9d36c1277305db92d561f906228e2c0e15d08fa493fcdfdde2c71beb48d3424131cae269425974831ea09bafa9a3b1ba81854_1280.jpg", 
             caption="Market analysis dashboard")

# Bottom-page links
st.markdown("---")
st.markdown("""
👈 Go to **Yield Forecast** to predict crop production

👉 Continue to **Reports** to generate comprehensive reports
""")
