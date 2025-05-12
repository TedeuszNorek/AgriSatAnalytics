import os
import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
import logging
import uuid
import io
import base64
from pathlib import Path
import jinja2
import markdown
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from streamlit.components.v1 import html as st_html

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Reports - Agro Insight",
    page_icon="📝",
    layout="wide"
)

# Initialize session state variables if not already set
if "selected_field" not in st.session_state:
    st.session_state.selected_field = None
if "available_fields" not in st.session_state:
    st.session_state.available_fields = []
if "ndvi_time_series" not in st.session_state:
    st.session_state.ndvi_time_series = {}
if "yield_forecast_results" not in st.session_state:
    st.session_state.yield_forecast_results = None
if "market_signals_results" not in st.session_state:
    st.session_state.market_signals_results = None
if "generated_reports" not in st.session_state:
    st.session_state.generated_reports = {}

# Helper function to load available fields
def load_available_fields():
    """Load available fields from processed data directory"""
    data_dir = Path("./data/geotiff")
    if not data_dir.exists():
        return []
    
    # Get unique field names from filenames
    field_names = set()
    for file in data_dir.glob("*.tif"):
        # Extract field name from filename (format: fieldname_index_sceneid.tif)
        parts = file.stem.split('_')
        if len(parts) >= 2:
            field_name = parts[0]
            field_names.add(field_name)
    
    return list(field_names)

# Function to create a plot and return it as a base64 image
def plot_to_base64(fig):
    """Convert a matplotlib figure to a base64 encoded string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

# Function to get Plotly figure as HTML
def plotly_fig_to_html(fig):
    """Convert a Plotly figure to HTML string"""
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# Function to generate markdown report
def generate_markdown_report(field_name, data):
    """Generate a markdown report for the field"""
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    # Start with report header
    md_content = f"""# Agricultural Insights Report: {field_name}

**Generated on:** {today}

## Executive Summary

"""
    
    # Add executive summary based on available data
    if "ndvi_time_series" in data and data["ndvi_time_series"]:
        ndvi_values = list(data["ndvi_time_series"].values())
        latest_ndvi = ndvi_values[-1] if ndvi_values else None
        
        if latest_ndvi is not None:
            if latest_ndvi > 0.7:
                summary = "The field shows excellent vegetation health with high NDVI values, indicating optimal growing conditions."
            elif latest_ndvi > 0.5:
                summary = "The field shows good vegetation health with moderate NDVI values, indicating favorable growing conditions."
            elif latest_ndvi > 0.3:
                summary = "The field shows fair vegetation health with moderate NDVI values, which may indicate some stress or early growth stage."
            else:
                summary = "The field shows low NDVI values, which may indicate crop stress, early growth stage, or recently harvested conditions."
            
            md_content += summary + "\n\n"
    
    # Add yield forecast if available
    if "yield_forecast" in data and data["yield_forecast"]:
        forecast = data["yield_forecast"]
        
        md_content += "### Yield Forecast\n\n"
        md_content += f"**Crop Type:** {forecast.get('crop_type', 'Unknown')}\n\n"
        
        # Add forecast dates and values
        if "forecasted_yields" in forecast:
            md_content += "**Forecasted Yields:**\n\n"
            for date, yield_val in forecast["forecasted_yields"].items():
                md_content += f"- {date}: {yield_val:.2f} t/ha\n"
            md_content += "\n"
        
        # Add trend information
        if "forecasted_yields" in forecast and len(forecast["forecasted_yields"]) > 1:
            dates = list(forecast["forecasted_yields"].keys())
            first_date = dates[0]
            last_date = dates[-1]
            
            first_yield = forecast["forecasted_yields"][first_date]
            last_yield = forecast["forecasted_yields"][last_date]
            change = last_yield - first_yield
            
            if change > 0:
                md_content += f"The yield forecast shows an **improving trend** with an increase of {change:.2f} t/ha over the forecast period.\n\n"
            elif change < 0:
                md_content += f"The yield forecast shows a **declining trend** with a decrease of {abs(change):.2f} t/ha over the forecast period.\n\n"
            else:
                md_content += "The yield forecast shows a **stable trend** with no significant change over the forecast period.\n\n"
    
    # Add market signals if available
    if "market_signals" in data and data["market_signals"]:
        signals = data["market_signals"].get("signals", [])
        
        if signals:
            md_content += "### Market Signals\n\n"
            
            # Group signals by commodity
            commodity_signals = {}
            for signal in signals:
                commodity = signal["commodity"]
                if commodity not in commodity_signals:
                    commodity_signals[commodity] = []
                commodity_signals[commodity].append(signal)
            
            # Get the top signal for each commodity
            for commodity, c_signals in commodity_signals.items():
                # Sort by confidence
                c_signals.sort(key=lambda x: x["confidence"], reverse=True)
                top_signal = c_signals[0]
                
                md_content += f"**{commodity}:** {top_signal['action']} with {top_signal['confidence']:.0%} confidence\n\n"
                md_content += f"*Reason:* {top_signal['reason']}\n\n"
            
            # Add overall market stance
            long_signals = [s for s in signals if s["action"] == "LONG"]
            short_signals = [s for s in signals if s["action"] == "SHORT"]
            
            avg_long_confidence = np.mean([s["confidence"] for s in long_signals]) if long_signals else 0
            avg_short_confidence = np.mean([s["confidence"] for s in short_signals]) if short_signals else 0
            
            overall_stance = "NEUTRAL"
            if avg_long_confidence > 0.6 and avg_long_confidence > avg_short_confidence:
                overall_stance = "BULLISH"
            elif avg_short_confidence > 0.6 and avg_short_confidence > avg_long_confidence:
                overall_stance = "BEARISH"
            
            md_content += f"**Overall Market Stance:** {overall_stance}\n\n"
    
    # Add NDVI analysis if available
    if "ndvi_time_series" in data and data["ndvi_time_series"]:
        md_content += "## Vegetation Health Analysis\n\n"
        
        ndvi_dates = list(data["ndvi_time_series"].keys())
        ndvi_values = list(data["ndvi_time_series"].values())
        
        # Calculate average NDVI
        avg_ndvi = np.mean(ndvi_values) if ndvi_values else 0
        
        md_content += f"**Average NDVI:** {avg_ndvi:.3f}\n\n"
        
        # Calculate latest vs historical comparison
        if len(ndvi_values) > 1:
            latest_ndvi = ndvi_values[-1]
            historical_avg = np.mean(ndvi_values[:-1])
            diff_pct = (latest_ndvi - historical_avg) / historical_avg * 100 if historical_avg != 0 else 0
            
            if diff_pct > 10:
                md_content += f"The latest NDVI value ({latest_ndvi:.3f}) is **{diff_pct:.1f}% higher** than the historical average, indicating improved vegetation health.\n\n"
            elif diff_pct < -10:
                md_content += f"The latest NDVI value ({latest_ndvi:.3f}) is **{abs(diff_pct):.1f}% lower** than the historical average, indicating potential vegetation stress.\n\n"
            else:
                md_content += f"The latest NDVI value ({latest_ndvi:.3f}) is **within normal range** compared to the historical average.\n\n"
        
        # Add anomaly information if available
        if "anomalies" in data and data["anomalies"]:
            anomalies = data["anomalies"]
            if anomalies:
                md_content += f"**Detected {len(anomalies)} anomalies** in the NDVI time series. These anomalies may indicate significant events like weather impacts or management changes.\n\n"
        
        # Add drought information if available
        if "drought_events" in data and data["drought_events"]:
            drought_events = data["drought_events"]
            if drought_events:
                md_content += f"**Detected {len(drought_events)} potential drought periods:**\n\n"
                
                for event in drought_events:
                    md_content += f"- Period: {event['start_date']} to {event['end_date']}, Severity: {event['severity']:.2f}\n"
                md_content += "\n"
    
    # Add recommendations section
    md_content += "## Recommendations\n\n"
    
    # Yield optimization recommendations
    md_content += "### Yield Optimization\n\n"
    
    if "yield_forecast" in data and data["yield_forecast"]:
        forecast = data["yield_forecast"]
        
        if "forecasted_yields" in forecast:
            dates = list(forecast["forecasted_yields"].keys())
            if dates:
                last_date = dates[-1]
                last_yield = forecast["forecasted_yields"][last_date]
                
                if last_yield < 5.0:
                    md_content += "- Consider soil testing to identify potential nutrient deficiencies\n"
                    md_content += "- Evaluate irrigation systems if drought stress is visible in NDVI patterns\n"
                    md_content += "- Review fertilization program to optimize nutrient availability\n"
                else:
                    md_content += "- Maintain current management practices as yield forecasts are favorable\n"
                    md_content += "- Monitor for potential late-season stresses that could impact final yield\n"
    else:
        md_content += "- Establish baseline yield goals based on field history and regional benchmarks\n"
        md_content += "- Implement regular satellite monitoring to track vegetation health throughout the season\n"
    
    # Market strategy recommendations
    md_content += "\n### Market Strategy\n\n"
    
    if "market_signals" in data and data["market_signals"]:
        signals = data["market_signals"].get("signals", [])
        
        if signals:
            # Determine overall stance
            long_signals = [s for s in signals if s["action"] == "LONG"]
            short_signals = [s for s in signals if s["action"] == "SHORT"]
            
            avg_long_confidence = np.mean([s["confidence"] for s in long_signals]) if long_signals else 0
            avg_short_confidence = np.mean([s["confidence"] for s in short_signals]) if short_signals else 0
            
            if avg_long_confidence > avg_short_confidence:
                md_content += "- Consider forward contracting or hedging to lock in prices as yields improve\n"
                md_content += "- Monitor basis levels for optimal timing of cash sales\n"
                md_content += "- Evaluate storage options to potentially capture seasonal price improvements\n"
            else:
                md_content += "- Consider accelerating sales as satellite data suggests potential market weakness\n"
                md_content += "- Evaluate put options to protect against further price declines\n"
                md_content += "- Delay major input purchases if possible, as commodity prices may trend lower\n"
    else:
        md_content += "- Develop a balanced marketing plan that aligns with your risk tolerance\n"
        md_content += "- Utilize satellite data to identify early trends that may impact regional production\n"
        md_content += "- Stay informed on global crop conditions through satellite vegetation monitoring\n"
    
    # Field management recommendations
    md_content += "\n### Field Management\n\n"
    
    if "field_variance" in data:
        variance = data["field_variance"]
        if variance > 0.02:
            md_content += "- Field shows significant variability that may benefit from zone management\n"
            md_content += "- Consider variable rate applications based on satellite-identified zones\n"
            md_content += "- Investigate areas of consistently low NDVI for potential soil or drainage issues\n"
        else:
            md_content += "- Field shows good uniformity, indicating effective management practices\n"
            md_content += "- Continue monitoring for changes in field patterns during critical growth stages\n"
    else:
        md_content += "- Establish management zones based on satellite vegetation patterns\n"
        md_content += "- Compare satellite data with yield maps to identify yield-limiting factors\n"
        md_content += "- Utilize historical NDVI patterns to inform variable rate applications\n"
    
    # Add data sources section
    md_content += "\n## Data Sources\n\n"
    md_content += "- Satellite Data: Sentinel-2 (European Space Agency)\n"
    md_content += "- Weather Data: Open-Meteo API\n"
    md_content += "- Market Data: Yahoo Finance API\n"
    
    # Add timestamp and footer
    md_content += f"\n\n---\n\nReport generated by Agro Insight on {today}\n"
    
    return md_content

# Function to convert markdown to HTML
def markdown_to_html(md_content):
    """Convert markdown content to HTML"""
    return markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

# Funkcja generująca ekspercki raport z branży handlu plonami rolnymi
def generate_expert_commodity_report(field_name, crop_type, data, time_period="Krótkoterminowa"):
    """
    Generuje ekspercki raport analizy rynkowej dla danego pola i typu uprawy.
    
    Args:
        field_name: Nazwa pola
        crop_type: Typ uprawy (np. "Wheat", "Corn", "Soybean")
        data: Słownik z danymi do raportu
        time_period: Okres prognozy ("Krótkoterminowa", "Średnioterminowa", "Długoterminowa")
        
    Returns:
        String zawierający raport w formacie markdown
    """
    # Data generowania raportu
    today = datetime.date.today()
    current_year = today.year
    
    # Określenie horyzontu czasowego na podstawie parametru
    if time_period == "Krótkoterminowa":
        forecast_end_date = today + datetime.timedelta(days=30)
        time_description = f"30 dni (do {forecast_end_date.strftime('%d.%m.%Y')})"
    elif time_period == "Średnioterminowa":
        forecast_end_date = today + datetime.timedelta(days=90)
        time_description = f"90 dni (do {forecast_end_date.strftime('%d.%m.%Y')})"
    else:  # Długoterminowa
        days_to_next_year = (datetime.date(current_year+1, 1, 1) - today).days
        forecast_end_date = today + datetime.timedelta(days=days_to_next_year + 30)
        time_description = f"do {forecast_end_date.strftime('%d.%m.%Y')}"
        
    # Tłumaczenie nazwy uprawy na polski
    crop_translations = {
        "Wheat": "Pszenica",
        "Corn": "Kukurydza",
        "Soybean": "Soja",
        "Barley": "Jęczmień",
        "Oats": "Owies",
        "Rice": "Ryż",
        "Rye": "Żyto"
    }
    
    crop_pl = crop_translations.get(crop_type, crop_type)
    
    # Symbole kontraktów na giełdzie
    commodity_symbols = {
        "Wheat": "ZW=F",  # Pszenica
        "Corn": "ZC=F",   # Kukurydza
        "Soybean": "ZS=F",  # Soja
        "Oats": "ZO=F",   # Owies
        "Rice": "ZR=F"    # Ryż
    }
    
    # Ceny aktualne i historyczne (przykładowe)
    commodity_prices = {
        "Wheat": {"current": 228.50, "last_month": 232.75, "last_year": 220.25},
        "Corn": {"current": 187.25, "last_month": 185.50, "last_year": 193.75},
        "Soybean": {"current": 430.75, "last_month": 424.50, "last_year": 445.25},
        "Oats": {"current": 284.25, "last_month": 280.75, "last_year": 271.50},
        "Rice": {"current": 363.00, "last_month": 355.25, "last_year": 342.75}
    }
    
    # Pobierz wartości NDVI z danych, jeśli dostępne
    ndvi_trend = "stabilną"  # domyślna wartość
    if "ndvi_time_series" in data and data["ndvi_time_series"]:
        ndvi_values = list(data["ndvi_time_series"].values())
        if len(ndvi_values) >= 2:
            if ndvi_values[-1] > ndvi_values[-2] * 1.05:
                ndvi_trend = "rosnącą"
            elif ndvi_values[-1] < ndvi_values[-2] * 0.95:
                ndvi_trend = "malejącą"
    
    # Ceny i zmiany procentowe
    current_price = commodity_prices.get(crop_type, {}).get("current", 0)
    last_month_price = commodity_prices.get(crop_type, {}).get("last_month", 0)
    last_year_price = commodity_prices.get(crop_type, {}).get("last_year", 0)
    
    monthly_change = ((current_price - last_month_price) / last_month_price * 100) if last_month_price else 0
    yearly_change = ((current_price - last_year_price) / last_year_price * 100) if last_year_price else 0
    
    # Generowanie prognozy cenowej na podstawie trendu NDVI i aktualnych cen
    if ndvi_trend == "rosnącą":
        price_forecast = round(current_price * 0.95, 2)  # prognoza spadku cen o 5%
        forecast_direction = "spadek"
        market_recommendation = "Rozważ sprzedaż kontraktów terminowych teraz - dobre zbiory mogą prowadzić do spadku cen."
    elif ndvi_trend == "malejącą":
        price_forecast = round(current_price * 1.07, 2)  # prognoza wzrostu cen o 7%
        forecast_direction = "wzrost"
        market_recommendation = "Rozważ zakup kontraktów terminowych - słabsze zbiory mogą prowadzić do wzrostu cen."
    else:
        price_forecast = round(current_price * 1.02, 2)  # prognoza niewielkiego wzrostu o 2%
        forecast_direction = "stabilizację z lekkim wzrostem"
        market_recommendation = "Monitoruj rynek - brak wyraźnych sygnałów do agresywnych działań."
    
    # Nagłówek raportu
    report = f"""# Ekspercki Raport Rynkowy: {crop_pl}

**Wygenerowano dnia:** {today.strftime('%d.%m.%Y')}  
**Dotyczy obszaru:** {field_name}  
**Horyzont prognozy:** {time_description}

## Podsumowanie rynkowe

{crop_pl} wykazuje {ndvi_trend} tendencję wzrostu na badanym obszarze, co sugeruje **{forecast_direction}** cen w analizowanym okresie.

Aktualna cena kontraktów terminowych ({commodity_symbols.get(crop_type, "N/D")}): **{current_price:.2f} EUR/t**

* Zmiana miesięczna: **{monthly_change:.2f}%** ({last_month_price:.2f} EUR/t)
* Zmiana roczna: **{yearly_change:.2f}%** ({last_year_price:.2f} EUR/t)

## Analiza rynkowa

### Czynniki wpływające na rynek {crop_pl}

1. **Kondycja upraw** - Wskaźnik NDVI pokazuje {ndvi_trend} tendencję w ostatnim okresie, co wskazuje na {ndvi_trend} dynamikę wzrostu roślin.

2. **Warunki pogodowe** - Ostatnie dane meteorologiczne wskazują na {
    "korzystne" if ndvi_trend == "rosnącą" else 
    "niekorzystne" if ndvi_trend == "malejącą" else 
    "umiarkowane"
} warunki dla rozwoju {crop_pl}.

3. **Globalne zapasy** - Światowe zapasy {crop_pl} są obecnie na {"wysokim" if ndvi_trend == "rosnącą" else "niskim" if ndvi_trend == "malejącą" else "przeciętnym"} poziomie.

4. **Tendencje eksportowe** - {"Zwiększony" if ndvi_trend == "malejącą" else "Zmniejszony" if ndvi_trend == "rosnącą" else "Stabilny"} popyt eksportowy z kluczowych regionów importujących.

### Prognoza cenowa

Spodziewana cena {crop_pl} na koniec okresu prognozy: **{price_forecast:.2f} EUR/t**

Uzasadnienie: {
    f"Dobre warunki wzrostu sugerują wyższe zbiory, co może prowadzić do zwiększonej podaży i spadku cen o około 5%." if ndvi_trend == "rosnącą" else
    f"Gorsze warunki wzrostu mogą skutkować niższymi zbiorami, prowadząc do ograniczonej podaży i wzrostu cen o około 7%." if ndvi_trend == "malejącą" else
    f"Obecne warunki nie wskazują na znaczące zmiany w zbiorach, spodziewamy się lekkiego wzrostu cen o 2% zgodnie z ogólną inflacją w sektorze rolnym."
}

## Rekomendacje handlowe

{market_recommendation}

### Sugerowane działania:

{"- Rozważ sprzedaż kontraktów na {price_forecast:.2f} EUR/t\n- Zabezpiecz co najmniej 30% przewidywanych zbiorów\n- Monitoruj prognozy meteorologiczne pod kątem zmian" if ndvi_trend == "rosnącą" else
 "- Rozważ zakup kontraktów na {current_price:.2f} EUR/t\n- Monitoruj sytuację podażową w innych regionach\n- Śledź raporty o stanie upraw w głównych krajach producenckich" if ndvi_trend == "malejącą" else
 "- Rozłóż sprzedaż w czasie zamiast jednorazowej transakcji\n- Monitoruj kluczowe wskaźniki rynkowe jak NDVI, stan magazynów i raporty USDA\n- Przygotuj strategię na wypadek wzrostu zmienności"}

## Kluczowe terminy do obserwacji

1. **Raporty USDA WASDE** - najbliższy raport: {(today.replace(day=12) if today.day < 12 else today.replace(day=12, month=today.month+1 if today.month < 12 else 1, year=today.year+1 if today.month == 12 else today.year)).strftime('%d.%m.%Y')}
2. **Raport MARS UE** - publikacja: koniec miesiąca
3. **Termin żniw** - {f"lipiec-sierpień {current_year}" if crop_type in ["Wheat", "Barley"] else f"wrzesień-październik {current_year}" if crop_type in ["Corn", "Soybean"] else f"wrzesień {current_year}"}

---

*Raport wygenerowany przez Agro Insight Trading Expert System - {today.strftime('%d.%m.%Y')}, {datetime.datetime.now().strftime('%H:%M')}*
"""

    return report

# Header
st.title("📝 Reports")
st.markdown("""
Generate comprehensive reports with agricultural insights, yield forecasts, and market recommendations based on satellite data analysis.
""")

# Field selection
st.sidebar.header("Field Selection")
available_fields = load_available_fields()
if not available_fields and "available_fields" in st.session_state:
    available_fields = st.session_state.available_fields

selected_field = st.sidebar.selectbox(
    "Select Field", 
    options=available_fields,
    index=0 if available_fields else None,
    help="Choose a field for reporting"
)

if selected_field:
    st.session_state.selected_field = selected_field
    
    # Main content
    st.header(f"Generate Report for {selected_field}")
    
    # Create tabs for report options and view reports
    tab1, tab2, tab3 = st.tabs(["Generate Report", "Expert Trading Report", "View Reports"])
    
    with tab1:
        st.subheader("Report Options")
        
        # Report type selection
        report_type = st.selectbox(
            "Report Type",
            options=["Executive Summary", "Comprehensive Analysis", "Market Outlook"],
            index=1,
            help="Select the type of report to generate"
        )
        
        # Report format selection
        report_format = st.selectbox(
            "Report Format",
            options=["Markdown", "HTML", "PDF"],
            index=0,
            help="Select the format for the report"
        )
        
        # Collect data for the report
        report_data = {}
        
        # Include NDVI time series if available
        if st.session_state.ndvi_time_series:
            report_data["ndvi_time_series"] = st.session_state.ndvi_time_series
        
        # Include yield forecast if available
        if st.session_state.yield_forecast_results:
            report_data["yield_forecast"] = st.session_state.yield_forecast_results
        
        # Include market signals if available
        if st.session_state.market_signals_results:
            report_data["market_signals"] = st.session_state.market_signals_results
        
        # Sections to include
        st.markdown("### Report Sections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_summary = st.checkbox("Executive Summary", value=True)
            include_ndvi = st.checkbox("Vegetation Health Analysis", value=True)
            include_yield = st.checkbox("Yield Forecast", value="yield_forecast" in report_data)
        
        with col2:
            include_market = st.checkbox("Market Signals", value="market_signals" in report_data)
            include_recommendations = st.checkbox("Recommendations", value=True)
            include_maps = st.checkbox("Field Maps", value=True)
        
        # Generate report button
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                try:
                    # Generate a report ID
                    report_id = str(uuid.uuid4())
                    
                    # Generate markdown report
                    md_content = generate_markdown_report(selected_field, report_data)
                    
                    # Convert to selected format
                    if report_format == "HTML":
                        html_content = markdown_to_html(md_content)
                        # In a real app, we would handle this properly
                        final_content = f"""
                        <html>
                        <head>
                            <title>Report: {selected_field}</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
                                h1 {{ color: #2c3e50; }}
                                h2 {{ color: #27ae60; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                                h3 {{ color: #2980b9; }}
                                table {{ border-collapse: collapse; width: 100%; }}
                                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                                th {{ background-color: #f2f2f2; }}
                                .report-date {{ color: #7f8c8d; font-style: italic; }}
                                .highlight {{ background-color: #ffffcc; padding: 5px; }}
                            </style>
                        </head>
                        <body>
                            {html_content}
                        </body>
                        </html>
                        """
                    elif report_format == "PDF":
                        # In a real app, we would use weasyprint to convert the HTML to PDF
                        final_content = md_content
                        st.warning("PDF generation would use WeasyPrint in a full implementation")
                    else:
                        final_content = md_content
                    
                    # Save report in session state
                    if selected_field not in st.session_state.generated_reports:
                        st.session_state.generated_reports[selected_field] = {}
                    
                    st.session_state.generated_reports[selected_field][report_id] = {
                        "id": report_id,
                        "type": report_type,
                        "format": report_format,
                        "content": final_content,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
                    st.success(f"Report generated successfully! View it in the 'View Reports' tab.")
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    logger.exception("Report generation error")
        
        # Data availability warnings
        if not report_data:
            st.warning("No data available for reporting. Please process field data in the previous sections first.")
        else:
            if "ndvi_time_series" not in report_data:
                st.warning("No NDVI time series available. The report will be limited.")
            if "yield_forecast" not in report_data:
                st.info("No yield forecast available. Run the Yield Forecast analysis to include this in the report.")
            if "market_signals" not in report_data:
                st.info("No market signals available. Run the Market Signals analysis to include this in the report.")
    
    with tab2:
        st.subheader("Expert Trading Report")
        
        # Wybór uprawy
        crop_type = st.selectbox(
            "Rodzaj uprawy",
            options=["Wheat", "Corn", "Soybean", "Oats", "Rice"],
            help="Wybierz rodzaj uprawy dla analizy rynkowej"
        )
        
        # Wybór okresu prognozy
        forecast_period = st.selectbox(
            "Horyzont prognozy",
            options=["Krótkoterminowa", "Średnioterminowa", "Długoterminowa"],
            help="Wybierz horyzont czasowy dla prognozy rynkowej"
        )
        
        # Collect data for the expert report
        expert_report_data = {}
        
        # Include NDVI time series if available
        if "ndvi_time_series" in st.session_state and st.session_state.ndvi_time_series:
            expert_report_data["ndvi_time_series"] = st.session_state.ndvi_time_series
        
        # Include yield forecast if available
        if "yield_forecast_results" in st.session_state and st.session_state.yield_forecast_results:
            expert_report_data["yield_forecast"] = st.session_state.yield_forecast_results
        
        # Include market signals if available
        if "market_signals_results" in st.session_state and st.session_state.market_signals_results:
            expert_report_data["market_signals"] = st.session_state.market_signals_results
        
        # Report format selection
        expert_report_format = st.selectbox(
            "Format raportu",
            options=["Markdown", "HTML"],
            key="expert_report_format",
            help="Wybierz format dla raportu eksperckiego"
        )
        
        # Generate report button
        if st.button("Generuj raport ekspercki"):
            with st.spinner("Generowanie raportu eksperckiego..."):
                try:
                    # Generate a report ID
                    report_id = str(uuid.uuid4())
                    
                    # Generate expert commodity report
                    md_content = generate_expert_commodity_report(
                        selected_field, 
                        crop_type, 
                        expert_report_data, 
                        time_period=forecast_period
                    )
                    
                    # Convert to selected format
                    if expert_report_format == "HTML":
                        html_content = markdown_to_html(md_content)
                        # Create HTML document
                        final_content = f"""
                        <html>
                        <head>
                            <title>Ekspercki Raport: {selected_field} - {crop_type}</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
                                h1 {{ color: #2c3e50; }}
                                h2 {{ color: #27ae60; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                                h3 {{ color: #2980b9; }}
                                table {{ border-collapse: collapse; width: 100%; }}
                                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                                th {{ background-color: #f2f2f2; }}
                                .report-date {{ color: #7f8c8d; font-style: italic; }}
                                .highlight {{ background-color: #ffffcc; padding: 5px; }}
                            </style>
                        </head>
                        <body>
                            {html_content}
                        </body>
                        </html>
                        """
                    else:
                        final_content = md_content
                    
                    # Save report in session state
                    if selected_field not in st.session_state.generated_reports:
                        st.session_state.generated_reports[selected_field] = {}
                    
                    st.session_state.generated_reports[selected_field][report_id] = {
                        "id": report_id,
                        "type": f"Expert Trading Report - {crop_type}",
                        "format": expert_report_format,
                        "content": final_content,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
                    # Show success message
                    st.success("Raport ekspercki wygenerowany pomyślnie!")
                    
                    # Preview the report
                    st.subheader("Podgląd raportu")
                    if expert_report_format == "HTML":
                        try:
                            st_html(final_content, height=600, scrolling=True)
                        except:
                            st.warning("HTML components not available. Displaying raw content.")
                            st.text_area("HTML Content", final_content, height=300)
                    else:
                        st.markdown(md_content)
                    
                    # Download button
                    st.download_button(
                        label="Pobierz raport",
                        data=final_content,
                        file_name=f"raport_ekspercki_{selected_field}_{crop_type}.{'html' if expert_report_format == 'HTML' else 'md'}",
                        mime="text/html" if expert_report_format == "HTML" else "text/markdown"
                    )
                except Exception as e:
                    st.error(f"Błąd podczas generowania raportu: {str(e)}")
                    st.exception(e)
                    
    with tab3:
        st.subheader("View Reports")
        
        # Get reports for this field
        field_reports = st.session_state.generated_reports.get(selected_field, {})
        
        if field_reports:
            # Display list of reports
            reports_list = []
            for report_id, report in field_reports.items():
                reports_list.append({
                    "ID": report_id[:8] + "...",
                    "Type": report["type"],
                    "Format": report["format"],
                    "Date": datetime.datetime.fromisoformat(report["timestamp"]).strftime("%Y-%m-%d %H:%M")
                })
            
            reports_df = pd.DataFrame(reports_list)
            st.dataframe(reports_df, use_container_width=True)
            
            # Report selection
            report_ids = list(field_reports.keys())
            selected_report_id = st.selectbox(
                "Select Report to View",
                options=report_ids,
                format_func=lambda x: f"{field_reports[x]['type']} - {datetime.datetime.fromisoformat(field_reports[x]['timestamp']).strftime('%Y-%m-%d %H:%M')}",
                help="Choose a report to view"
            )
            
            if selected_report_id:
                report = field_reports[selected_report_id]
                
                # Display report
                st.markdown("### Report Preview")
                
                if report["format"] == "HTML":
                    st_html(report["content"], height=600, scrolling=True)
                elif report["format"] == "PDF":
                    st.warning("PDF preview not available. Use the download button to view the PDF.")
                else:
                    st.markdown(report["content"])
                
                # Download button
                filename = f"{selected_field}_report_{report['type'].replace(' ', '_')}_{datetime.datetime.fromisoformat(report['timestamp']).strftime('%Y%m%d')}"
                
                if report["format"] == "Markdown":
                    st.download_button(
                        label="Download Report (Markdown)",
                        data=report["content"],
                        file_name=f"{filename}.md",
                        mime="text/markdown"
                    )
                elif report["format"] == "HTML":
                    st.download_button(
                        label="Download Report (HTML)",
                        data=report["content"],
                        file_name=f"{filename}.html",
                        mime="text/html"
                    )
                else:
                    # In a real app, we would generate a PDF file
                    st.download_button(
                        label="Download Report (Text)",
                        data=report["content"],
                        file_name=f"{filename}.txt",
                        mime="text/plain"
                    )
        else:
            st.info("No reports have been generated for this field yet. Go to the 'Generate Report' tab to create a report.")
            
            # Sample report image
            st.image("https://pixabay.com/get/gd3965e709a0b5b615433a63caab9d36c1277305db92d561f906228e2c0e15d08fa493fcdfdde2c71beb48d3424131cae269425974831ea09bafa9a3b1ba81854_1280.jpg", 
                     caption="Sample report dashboard")

# Display alternate content if no field is selected
else:
    st.info("""
    No fields available for reporting. Please go to the Data Ingest section to process field data first.
    
    You can:
    1. Draw a field boundary on the map
    2. Upload a GeoJSON file with field boundaries
    3. Select a country for country-level analysis
    """)
    
    # Display sample image
    st.image("https://pixabay.com/get/g6bb50ef33a3aad66af13194b25f2700de9dcd4f5bddef1614e89eb634e0daa6297d7518d25e52ef85258015e7b9f6c10a0641be21f0925330270b5ab623b733d_1280.jpg", 
             caption="Generate reports with satellite insights")

# Bottom-page links
st.markdown("---")
st.markdown("""
👈 Go to **Market Signals** to analyze market implications

👉 Continue to **Debug Dashboard** to monitor system performance
""")
