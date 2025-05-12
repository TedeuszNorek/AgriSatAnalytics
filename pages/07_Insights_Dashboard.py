"""
Insights Dashboard - Spersonalizowany panel analityczny z odznakami osiągnięć
"""
import os
import json
import logging
import datetime
import traceback
from typing import Dict, List, Tuple, Any, Optional
import time

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import folium
from streamlit_folium import folium_static
from sqlalchemy import func

from database import get_db, Field, SatelliteImage, TimeSeries, YieldForecast, MarketSignal
from utils.visualization import (
    plot_field_statistics,
    plot_market_signals,
    create_insight_badge
)
from config import APP_NAME

# Konfiguracja loggera
logger = logging.getLogger(__name__)

# Tytuł strony
st.set_page_config(page_title=f"{APP_NAME} - Insights Dashboard", layout="wide")
st.title("Insights Dashboard")
st.markdown("Spersonalizowany panel analityczny z odznakami osiągnięć")

# Funkcje pomocnicze
def load_user_achievements() -> Dict[str, Any]:
    """Ładuje osiągnięcia użytkownika z bazy danych lub pliku"""
    achievements_path = os.path.join("data", "user", "achievements.json")
    
    # Domyślne osiągnięcia, jeśli plik nie istnieje
    default_achievements = {
        "field_explorer": {
            "level": 0,
            "progress": 0,
            "max": 3,
            "title": "Eksplorator pól",
            "description": "Dodaj {max} pól do analizy",
            "icon": "🌱",
            "rewards": ["Lepsze wizualizacje", "Porównywanie wielu pól"]
        },
        "satellite_analyst": {
            "level": 0,
            "progress": 0,
            "max": 5,
            "title": "Analityk satelitarny",
            "description": "Wykonaj {max} analiz danych satelitarnych",
            "icon": "🛰️",
            "rewards": ["Zaawansowane analizy trendu", "Eksport danych analitycznych"]
        },
        "yield_forecaster": {
            "level": 0,
            "progress": 0,
            "max": 3,
            "title": "Prognosta plonów",
            "description": "Stwórz {max} prognoz plonów",
            "icon": "📊",
            "rewards": ["Wyższe dokładności prognoz", "Porównanie historycznych przewidywań"]
        },
        "market_investor": {
            "level": 0,
            "progress": 0,
            "max": 3,
            "title": "Inwestor rynkowy",
            "description": "Wygeneruj {max} sygnałów rynkowych",
            "icon": "📈",
            "rewards": ["Wskaźniki dokładności sygnałów", "Rozszerzone dane rynkowe"]
        },
        "data_master": {
            "level": 0,
            "progress": 0,
            "max": 10,
            "title": "Mistrz danych",
            "description": "Utwórz łącznie {max} różnych analiz",
            "icon": "🏆",
            "rewards": ["Priorytetowe przetwarzanie", "Dostęp do zaawansowanych modeli"]
        }
    }
    
    # Sprawdź, czy plik z osiągnięciami istnieje
    if os.path.exists(achievements_path):
        try:
            with open(achievements_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Błąd podczas ładowania osiągnięć: {str(e)}")
            return default_achievements
    else:
        # Upewnij się, że katalog istnieje
        os.makedirs(os.path.dirname(achievements_path), exist_ok=True)
        
        # Zapisz domyślne osiągnięcia
        try:
            with open(achievements_path, 'w') as f:
                json.dump(default_achievements, f)
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania domyślnych osiągnięć: {str(e)}")
        
        return default_achievements

def save_user_achievements(achievements: Dict[str, Any]) -> bool:
    """Zapisuje osiągnięcia użytkownika do pliku"""
    achievements_path = os.path.join("data", "user", "achievements.json")
    
    try:
        os.makedirs(os.path.dirname(achievements_path), exist_ok=True)
        with open(achievements_path, 'w') as f:
            json.dump(achievements, f)
        return True
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania osiągnięć: {str(e)}")
        return False

def update_achievements_from_db():
    """Aktualizuje osiągnięcia na podstawie danych z bazy"""
    achievements = load_user_achievements()
    
    try:
        db = next(get_db())
        
        # Zliczanie pól
        field_count = db.query(func.count(Field.id)).scalar() or 0
        achievements["field_explorer"]["progress"] = field_count
        achievements["field_explorer"]["level"] = min(3, field_count // achievements["field_explorer"]["max"])
        
        # Zliczanie analiz satelitarnych
        sat_image_count = db.query(func.count(SatelliteImage.id)).scalar() or 0
        achievements["satellite_analyst"]["progress"] = sat_image_count
        achievements["satellite_analyst"]["level"] = min(3, sat_image_count // achievements["satellite_analyst"]["max"])
        
        # Zliczanie prognoz plonów
        forecast_count = db.query(func.count(YieldForecast.id)).scalar() or 0
        achievements["yield_forecaster"]["progress"] = forecast_count
        achievements["yield_forecaster"]["level"] = min(3, forecast_count // achievements["yield_forecaster"]["max"])
        
        # Zliczanie sygnałów rynkowych
        signal_count = db.query(func.count(MarketSignal.id)).scalar() or 0
        achievements["market_investor"]["progress"] = signal_count
        achievements["market_investor"]["level"] = min(3, signal_count // achievements["market_investor"]["max"])
        
        # Zliczanie wszystkich analiz
        total_analyses = sat_image_count + forecast_count + signal_count
        achievements["data_master"]["progress"] = total_analyses
        achievements["data_master"]["level"] = min(3, total_analyses // achievements["data_master"]["max"])
        
        # Zapisz zaktualizowane osiągnięcia
        save_user_achievements(achievements)
        
    except Exception as e:
        logger.error(f"Błąd podczas aktualizacji osiągnięć: {str(e)}")
        st.error(f"Nie udało się zaktualizować osiągnięć: {str(e)}")

def get_user_insights() -> Dict[str, Any]:
    """Pobiera wglądy użytkownika na podstawie danych w bazie"""
    insights = {
        "fields_count": 0,
        "total_area_ha": 0,
        "avg_ndvi": 0,
        "anomaly_count": 0,
        "forecast_accuracy": 0,
        "market_signal_success_rate": 0,
        "last_analysis_date": None,
        "most_productive_field": None,
        "health_trend": "stable",  # "improving", "stable", "declining"
        "recent_activities": []
    }
    
    try:
        db = next(get_db())
        
        # Podstawowe statystyki
        fields = db.query(Field).all()
        insights["fields_count"] = len(fields)
        
        if fields:
            # Całkowita powierzchnia
            insights["total_area_ha"] = sum(field.area_hectares or 0 for field in fields)
            
            # Średni NDVI
            ndvi_series = db.query(TimeSeries).filter(TimeSeries.series_type == "NDVI").all()
            if ndvi_series:
                all_ndvi_values = []
                for series in ndvi_series:
                    if series.data and isinstance(series.data, dict):
                        all_ndvi_values.extend(series.data.values())
                
                if all_ndvi_values:
                    insights["avg_ndvi"] = sum(all_ndvi_values) / len(all_ndvi_values)
            
            # Najbardziej produktywne pole
            if ndvi_series:
                field_avg_ndvi = {}
                for series in ndvi_series:
                    if series.data and isinstance(series.data, dict) and series.data.values():
                        field_id = series.field_id
                        field_avg_ndvi[field_id] = field_avg_ndvi.get(field_id, []) + list(series.data.values())
                
                best_field_id = None
                best_ndvi = -1
                
                for field_id, values in field_avg_ndvi.items():
                    if values:
                        avg = sum(values) / len(values)
                        if avg > best_ndvi:
                            best_ndvi = avg
                            best_field_id = field_id
                
                if best_field_id:
                    best_field = db.query(Field).filter(Field.id == best_field_id).first()
                    if best_field:
                        insights["most_productive_field"] = best_field.name
            
            # Niedawne aktywności
            recent_images = db.query(SatelliteImage).order_by(SatelliteImage.created_at.desc()).limit(5).all()
            recent_forecasts = db.query(YieldForecast).order_by(YieldForecast.created_at.desc()).limit(5).all()
            recent_signals = db.query(MarketSignal).order_by(MarketSignal.created_at.desc()).limit(5).all()
            
            # Wszystkie niedawne aktywności z datami
            all_activities = []
            
            for img in recent_images:
                field = db.query(Field).filter(Field.id == img.field_id).first()
                field_name = field.name if field else "Nieznane pole"
                all_activities.append({
                    "date": img.created_at,
                    "type": "image",
                    "description": f"Nowy obraz satelitarny: {img.image_type} dla {field_name}"
                })
            
            for forecast in recent_forecasts:
                field = db.query(Field).filter(Field.id == forecast.field_id).first()
                field_name = field.name if field else "Nieznane pole"
                all_activities.append({
                    "date": forecast.created_at,
                    "type": "forecast",
                    "description": f"Prognoza plonów dla {field_name}: {forecast.crop_type}"
                })
            
            for signal in recent_signals:
                field = db.query(Field).filter(Field.id == signal.field_id).first()
                field_name = field.name if field else "Nieznane pole"
                all_activities.append({
                    "date": signal.created_at,
                    "type": "signal",
                    "description": f"Sygnał rynkowy dla {field_name}: {signal.action} w {signal.commodity}"
                })
            
            # Sortuj według daty i weź 10 najnowszych
            all_activities.sort(key=lambda x: x["date"], reverse=True)
            insights["recent_activities"] = all_activities[:10]
            
            # Ostatnia data analizy
            if all_activities:
                insights["last_analysis_date"] = all_activities[0]["date"]
            
            # Określ trend zdrowia roślin (prosty algorytm)
            if ndvi_series:
                recent_ndvi_values = []
                for series in ndvi_series:
                    if series.data and isinstance(series.data, dict):
                        # Konwertuj daty stringowe na obiekty datetime i sortuj
                        sorted_data = sorted(
                            [(datetime.datetime.fromisoformat(date), value) for date, value in series.data.items()],
                            key=lambda x: x[0]
                        )
                        
                        # Weź ostatnie 5 wartości jeśli są dostępne
                        if len(sorted_data) >= 5:
                            recent_ndvi_values.extend([value for _, value in sorted_data[-5:]])
                
                if len(recent_ndvi_values) >= 5:
                    # Oblicz trend na podstawie nachylenia linii trendu
                    y = np.array(recent_ndvi_values)
                    x = np.arange(len(y))
                    slope = np.polyfit(x, y, 1)[0]
                    
                    if slope > 0.01:
                        insights["health_trend"] = "improving"
                    elif slope < -0.01:
                        insights["health_trend"] = "declining"
                    else:
                        insights["health_trend"] = "stable"
    
    except Exception as e:
        logger.error(f"Błąd podczas pobierania wglądów: {str(e)}")
        logger.error(traceback.format_exc())
    
    return insights

def display_achievements(achievements: Dict[str, Any]):
    """Wyświetla odznaki osiągnięć użytkownika"""
    st.markdown("## Twoje odznaki osiągnięć")
    
    # Stwórz układ w rzędach po 3 odznaki
    achievement_list = list(achievements.items())
    
    # Wyświetl odznaki w rzędach
    for i in range(0, len(achievement_list), 3):
        cols = st.columns(min(3, len(achievement_list) - i))
        
        for j in range(min(3, len(achievement_list) - i)):
            achievement_key, achievement = achievement_list[i + j]
            with cols[j]:
                # Stwórz odznakę za pomocą funkcji pomocniczej
                badge_html = create_insight_badge(
                    title=achievement["title"],
                    icon=achievement["icon"],
                    level=achievement["level"],
                    progress=achievement["progress"],
                    max_progress=achievement["max"],
                    description=achievement["description"].format(max=achievement["max"])
                )
                st.markdown(badge_html, unsafe_allow_html=True)
                
                # Wyświetl nagrody dla odblokowanych poziomów
                if achievement["level"] > 0:
                    rewards = achievement["rewards"][:achievement["level"]]
                    rewards_text = "\n".join([f"- {reward}" for reward in rewards])
                    st.markdown(f"**Odblokowane korzyści:**\n{rewards_text}")

def display_insights_dashboard(insights: Dict[str, Any], achievements: Dict[str, Any]):
    """Wyświetla główny panel wglądów użytkownika"""
    
    # Podsumowanie na górze
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("Liczba pól", insights["fields_count"])
        st.metric("Całkowita powierzchnia", f"{insights['total_area_ha']:.2f} ha")
    
    with summary_col2:
        st.metric("Średni NDVI", f"{insights['avg_ndvi']:.2f}")
        
        # Wskaźnik trendu zdrowia roślin
        health_trend = insights["health_trend"]
        if health_trend == "improving":
            st.metric("Trend zdrowia roślin", "Poprawa", delta="↗")
        elif health_trend == "declining":
            st.metric("Trend zdrowia roślin", "Pogorszenie", delta="↘")
        else:
            st.metric("Trend zdrowia roślin", "Stabilny", delta="→")
    
    with summary_col3:
        if insights["most_productive_field"]:
            st.metric("Najbardziej produktywne pole", insights["most_productive_field"])
        
        if insights["last_analysis_date"]:
            st.metric("Ostatnia analiza", insights["last_analysis_date"].strftime("%Y-%m-%d"))
    
    # Sekcja wizualizacji i osiągnięć
    viz_col, achievements_col = st.columns([2, 1])
    
    with viz_col:
        st.markdown("## Wizualizacje i trendy")
        
        # Jeśli są dane, wygeneruj wykresy
        if insights["fields_count"] > 0:
            # Wykres aktywności
            if insights["recent_activities"]:
                st.markdown("### Ostatnie aktywności")
                
                # Przygotuj dane do wykresu
                activities_df = pd.DataFrame(insights["recent_activities"])
                activities_df["date"] = pd.to_datetime(activities_df["date"])
                
                # Pogrupuj aktywności według typu i daty
                activities_by_type = activities_df.groupby([pd.Grouper(key="date", freq="D"), "type"]).size().unstack(fill_value=0)
                
                # Stwórz wykres
                fig, ax = plt.subplots(figsize=(10, 4))
                activities_by_type.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
                ax.set_xlabel("Data")
                ax.set_ylabel("Liczba aktywności")
                ax.set_title("Aktywności w czasie")
                plt.tight_layout()
                
                st.pyplot(fig)
            
            # Tabela ostatnich aktywności
            if insights["recent_activities"]:
                st.markdown("### Szczegóły ostatnich aktywności")
                
                # Stwórz tabelę aktywności
                activities_table = []
                for activity in insights["recent_activities"]:
                    activities_table.append({
                        "Data": activity["date"].strftime("%Y-%m-%d %H:%M"),
                        "Typ": activity["type"],
                        "Opis": activity["description"]
                    })
                
                st.table(pd.DataFrame(activities_table))
        else:
            st.info("Dodaj pola i przeprowadź analizy, aby zobaczyć tutaj więcej danych.")
    
    with achievements_col:
        # Oblicz całkowity postęp osiągnięć
        total_progress = sum(ach["progress"] for ach in achievements.values())
        total_max = sum(ach["max"] for ach in achievements.values())
        total_percentage = (total_progress / total_max) * 100 if total_max > 0 else 0
        
        # Wyświetl całkowity postęp
        st.markdown("## Twój postęp")
        st.progress(total_percentage / 100)
        st.markdown(f"**Postęp całkowity:** {total_progress}/{total_max} ({total_percentage:.1f}%)")
        
        # Wyświetl postęp w każdym osiągnięciu
        for key, achievement in achievements.items():
            progress_pct = (achievement["progress"] / achievement["max"]) * 100 if achievement["max"] > 0 else 0
            st.markdown(f"**{achievement['title']}:** {achievement['progress']}/{achievement['max']} ({progress_pct:.1f}%)")
            st.progress(progress_pct / 100)

def create_pdf_report(insights: Dict[str, Any], achievements: Dict[str, Any]) -> Optional[str]:
    """Generuje raport PDF z osiągnięciami i wglądami użytkownika"""
    try:
        import matplotlib.backends.backend_pdf
        import matplotlib.pyplot as plt
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        import io
        
        # Ścieżka do pliku raportu
        report_path = os.path.join("data", "user", "insights_report.pdf")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Stwórz dokument PDF
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Elementy dokumentu
        elements = []
        
        # Tytuł
        elements.append(Paragraph(f"Raport wglądów i osiągnięć - {datetime.datetime.now().strftime('%Y-%m-%d')}", styles["Title"]))
        elements.append(Spacer(1, 12))
        
        # Podsumowanie
        elements.append(Paragraph("Podsumowanie pól", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        
        summary_data = [
            ["Metryka", "Wartość"],
            ["Liczba pól", str(insights["fields_count"])],
            ["Całkowita powierzchnia", f"{insights['total_area_ha']:.2f} ha"],
            ["Średni NDVI", f"{insights['avg_ndvi']:.2f}"],
            ["Trend zdrowia roślin", insights["health_trend"].capitalize()],
            ["Najbardziej produktywne pole", insights["most_productive_field"] or "Brak danych"],
            ["Ostatnia analiza", insights["last_analysis_date"].strftime("%Y-%m-%d") if insights["last_analysis_date"] else "Brak danych"]
        ]
        
        summary_table = Table(summary_data, colWidths=[200, 200])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (1, 0), 12),
            ('BACKGROUND', (0, 1), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 12))
        
        # Osiągnięcia
        elements.append(Paragraph("Osiągnięcia", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        
        achievements_data = [["Osiągnięcie", "Poziom", "Postęp"]]
        for key, achievement in achievements.items():
            achievements_data.append([
                achievement["title"],
                f"{achievement['level']}/3",
                f"{achievement['progress']}/{achievement['max']}"
            ])
        
        achievements_table = Table(achievements_data, colWidths=[200, 100, 100])
        achievements_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (2, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (2, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (2, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (2, 0), 12),
            ('BOTTOMPADDING', (0, 0), (2, 0), 12),
            ('BACKGROUND', (0, 1), (2, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(achievements_table)
        elements.append(Spacer(1, 12))
        
        # Ostatnie aktywności
        if insights["recent_activities"]:
            elements.append(Paragraph("Ostatnie aktywności", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            
            activities_data = [["Data", "Typ", "Opis"]]
            for activity in insights["recent_activities"]:
                activities_data.append([
                    activity["date"].strftime("%Y-%m-%d %H:%M"),
                    activity["type"].capitalize(),
                    activity["description"]
                ])
            
            activities_table = Table(activities_data, colWidths=[100, 100, 300])
            activities_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (2, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (2, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (2, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (2, 0), 12),
                ('BOTTOMPADDING', (0, 0), (2, 0), 12),
                ('BACKGROUND', (0, 1), (2, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(activities_table)
            elements.append(Spacer(1, 12))
        
        # Zbuduj dokument
        doc.build(elements)
        
        return report_path
    
    except Exception as e:
        logger.error(f"Błąd podczas generowania raportu PDF: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Główny kod strony
try:
    # Zaktualizuj osiągnięcia z bazy danych
    update_achievements_from_db()
    
    # Pobierz aktualnie zapisane osiągnięcia
    achievements = load_user_achievements()
    
    try:
        # Pobierz wglądy użytkownika
        insights = get_user_insights()
        
        # Wyświetl odznaki osiągnięć
        display_achievements(achievements)
        
        # Wyświetl główny panel wglądów
        display_insights_dashboard(insights, achievements)
        
        # Generowanie raportu jest tymczasowo wyłączone z powodu braku zależności
        st.info("Generowanie PDF zostało tymczasowo wyłączone. W przyszłych wersjach będzie dostępne.")
    except NameError as e:
        if "SatellateImage" in str(e):
            st.error("Błąd w kodzie: SatellateImage powinno być SatelliteImage. Naprawimy to wkrótce.")
            logger.error(f"Błąd literówki w kodzie: {str(e)}")
        else:
            st.error(f"Błąd w kodzie: {str(e)}")
            logger.error(f"Błąd w kodzie: {str(e)}")
    
except Exception as e:
    st.error(f"Wystąpił błąd podczas ładowania panelu wglądów: {str(e)}")
    logger.error(f"Błąd podczas ładowania panelu wglądów: {str(e)}")
    logger.error(traceback.format_exc())