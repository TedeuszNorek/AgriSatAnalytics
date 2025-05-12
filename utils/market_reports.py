"""
Moduł do automatycznego generowania raportów rynkowych.
Zawiera funkcje do analizy danych rynkowych, generowania raportów i planowania ich automatycznego wysyłania.
"""

import datetime
import logging
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from pathlib import Path

# Konfiguracja loggera
logger = logging.getLogger(__name__)

# Klasa do automatycznego generowania raportów rynkowych
class MarketInsightsReportGenerator:
    """
    Klasa odpowiedzialna za automatyczne generowanie raportów analizy rynkowej.
    Generuje raporty na podstawie danych satelitarnych, prognoz pogody i danych rynkowych.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Inicjalizacja generatora raportów.
        
        Args:
            data_dir: Ścieżka do katalogu z danymi
        """
        self.data_dir = Path(data_dir)
        self.reports_dir = self.data_dir / "reports"
        
        # Tworzenie katalogu na raporty, jeśli nie istnieje
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Słownik mapujący uprawy na ich symbole giełdowe
        self.commodity_symbols = {
            "Wheat": "ZW=F",  # Pszenica
            "Corn": "ZC=F",   # Kukurydza
            "Soybean": "ZS=F", # Soja
            "Oats": "ZO=F",   # Owies
            "Rice": "ZR=F"    # Ryż
        }
        
        # Słownik tłumaczący nazwy upraw na polski
        self.crop_translations = {
            "Wheat": "Pszenica",
            "Corn": "Kukurydza",
            "Soybean": "Soja",
            "Barley": "Jęczmień",
            "Oats": "Owies",
            "Rice": "Ryż",
            "Rye": "Żyto"
        }
        
        # Inicjalizacja słownika przechowującego dane o generowanych raportach
        self.generated_reports = {}
        
        logger.info("Inicjalizacja generatora raportów rynkowych zakończona")
    
    def load_field_data(self, field_name: str) -> Dict[str, Any]:
        """
        Ładuje dane dotyczące wybranego pola.
        
        Args:
            field_name: Nazwa pola
            
        Returns:
            Słownik z danymi pola
        """
        try:
            field_data = {}
            
            # Ładowanie danych NDVI
            ndvi_path = self.data_dir / f"{field_name}_ndvi_time_series.json"
            if ndvi_path.exists():
                with open(ndvi_path, "r") as f:
                    field_data["ndvi_time_series"] = json.load(f)
                logger.info(f"Załadowano dane NDVI dla pola {field_name}")
            
            # Ładowanie danych o prognozie plonów
            yield_path = self.data_dir / f"{field_name}_yield_forecast.json"
            if yield_path.exists():
                with open(yield_path, "r") as f:
                    field_data["yield_forecast"] = json.load(f)
                logger.info(f"Załadowano prognozę plonów dla pola {field_name}")
            
            # Ładowanie danych o sygnałach rynkowych
            signals_path = self.data_dir / f"{field_name}_market_signals.json"
            if signals_path.exists():
                with open(signals_path, "r") as f:
                    field_data["market_signals"] = json.load(f)
                logger.info(f"Załadowano sygnały rynkowe dla pola {field_name}")
            
            return field_data
        except Exception as e:
            logger.error(f"Błąd podczas ładowania danych dla pola {field_name}: {str(e)}")
            return {}
    
    def analyze_market_trends(self, field_data: Dict[str, Any], crop_type: str) -> Dict[str, Any]:
        """
        Analizuje trendy rynkowe na podstawie danych pola i typu uprawy.
        
        Args:
            field_data: Słownik z danymi pola
            crop_type: Typ uprawy
            
        Returns:
            Słownik z wynikami analizy trendów rynkowych
        """
        try:
            analysis_results = {
                "crop_type": crop_type,
                "crop_pl": self.crop_translations.get(crop_type, crop_type),
                "commodity_symbol": self.commodity_symbols.get(crop_type, "N/A"),
                "analysis_date": datetime.datetime.now().isoformat(),
                "ndvi_trend": "stabilna",  # domyślna wartość
                "price_trend": "stabilna",  # domyślna wartość
                "correlation_strength": 0.0,
                "market_signals": [],
                "forecast": {}
            }
            
            # Analiza trendu NDVI, jeśli dane są dostępne
            if "ndvi_time_series" in field_data and field_data["ndvi_time_series"]:
                ndvi_data = field_data["ndvi_time_series"]
                ndvi_values = list(ndvi_data.values())
                
                if len(ndvi_values) >= 3:
                    # Obliczanie średniej zmian NDVI z ostatnich 3 odczytów
                    recent_changes = []
                    for i in range(1, min(4, len(ndvi_values))):
                        change = (ndvi_values[-i] - ndvi_values[-(i+1)]) / max(0.01, ndvi_values[-(i+1)]) * 100
                        recent_changes.append(change)
                    
                    avg_change = sum(recent_changes) / len(recent_changes)
                    
                    # Określanie trendu na podstawie średniej zmian
                    if avg_change > 3.0:
                        analysis_results["ndvi_trend"] = "rosnąca"
                    elif avg_change < -3.0:
                        analysis_results["ndvi_trend"] = "malejąca"
                
                # Dodanie wartości NDVI
                analysis_results["ndvi_values"] = ndvi_values
                analysis_results["ndvi_latest"] = ndvi_values[-1] if ndvi_values else None
                
                logger.info(f"Zidentyfikowano trend NDVI: {analysis_results['ndvi_trend']}")
            
            # Analiza sygnałów rynkowych, jeśli dane są dostępne
            if "market_signals" in field_data and field_data["market_signals"]:
                signals = field_data["market_signals"].get("signals", [])
                
                if signals:
                    # Filtrowanie sygnałów dla wybranej uprawy
                    crop_signals = [s for s in signals if s.get("commodity") == self.commodity_symbols.get(crop_type)]
                    
                    if crop_signals:
                        # Sortowanie sygnałów według daty
                        crop_signals = sorted(crop_signals, key=lambda x: x.get("signal_date", ""), reverse=True)
                        
                        # Analiza ostatnich sygnałów
                        recent_signals = crop_signals[:3]
                        
                        # Zliczanie sygnałów LONG i SHORT
                        long_signals = [s for s in recent_signals if s.get("action") == "LONG"]
                        short_signals = [s for s in recent_signals if s.get("action") == "SHORT"]
                        
                        # Określanie trendu cenowego na podstawie przeważających sygnałów
                        if len(long_signals) > len(short_signals):
                            analysis_results["price_trend"] = "rosnąca"
                        elif len(short_signals) > len(long_signals):
                            analysis_results["price_trend"] = "malejąca"
                        
                        # Dodanie sygnałów do wyników
                        analysis_results["market_signals"] = recent_signals
                        
                        # Obliczenie średniej siły sygnału (confidence)
                        if recent_signals:
                            avg_confidence = sum(s.get("confidence", 0) for s in recent_signals) / len(recent_signals)
                            analysis_results["correlation_strength"] = avg_confidence
                        
                        logger.info(f"Zidentyfikowano trend cenowy: {analysis_results['price_trend']}")
            
            # Prognozowanie przyszłych cen na podstawie trendów NDVI i cenowych
            price_forecast = self.generate_price_forecast(analysis_results)
            analysis_results["forecast"] = price_forecast
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Błąd podczas analizy trendów rynkowych: {str(e)}")
            return {
                "crop_type": crop_type,
                "crop_pl": self.crop_translations.get(crop_type, crop_type),
                "error": str(e)
            }
    
    def generate_price_forecast(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generuje prognozę cenową na podstawie wyników analizy.
        
        Args:
            analysis_results: Słownik z wynikami analizy
            
        Returns:
            Słownik z prognozą cenową
        """
        # Przykładowe aktualne ceny dla różnych upraw (EUR/t)
        current_prices = {
            "Wheat": 228.50,
            "Corn": 187.25,
            "Soybean": 430.75,
            "Oats": 284.25,
            "Rice": 363.00
        }
        
        crop_type = analysis_results["crop_type"]
        ndvi_trend = analysis_results["ndvi_trend"]
        price_trend = analysis_results["price_trend"]
        current_price = current_prices.get(crop_type, 200.0)
        
        # Inicjalizacja prognozy
        forecast = {
            "current_price": current_price,
            "short_term": {"price": current_price, "change_pct": 0.0},  # 1 miesiąc
            "medium_term": {"price": current_price, "change_pct": 0.0},  # 3 miesiące
            "long_term": {"price": current_price, "change_pct": 0.0},    # 6 miesięcy
            "forecast_rationale": "Stabilne warunki rynkowe bez wyraźnych sygnałów do zmiany"
        }
        
        # Generowanie prognozy na podstawie trendów NDVI i cenowych
        # Dla prognozy krótkoterminowej większy wpływ ma trend cenowy
        # Dla prognozy długoterminowej większy wpływ ma trend NDVI
        
        # Modyfikatory procentowe dla różnych trendów
        modifiers = {
            "short_term": {
                "ndvi": {"rosnąca": -0.02, "stabilna": 0.0, "malejąca": 0.03},
                "price": {"rosnąca": 0.04, "stabilna": 0.01, "malejąca": -0.02}
            },
            "medium_term": {
                "ndvi": {"rosnąca": -0.05, "stabilna": 0.0, "malejąca": 0.07},
                "price": {"rosnąca": 0.03, "stabilna": 0.02, "malejąca": -0.03}
            },
            "long_term": {
                "ndvi": {"rosnąca": -0.08, "stabilna": 0.0, "malejąca": 0.12},
                "price": {"rosnąca": 0.02, "stabilna": 0.03, "malejąca": -0.02}
            }
        }
        
        # Obliczanie prognoz dla różnych okresów
        for term in ["short_term", "medium_term", "long_term"]:
            ndvi_modifier = modifiers[term]["ndvi"].get(ndvi_trend, 0.0)
            price_modifier = modifiers[term]["price"].get(price_trend, 0.0)
            
            # Łączny modyfikator (wagi różnią się w zależności od okresu)
            if term == "short_term":
                total_modifier = ndvi_modifier * 0.3 + price_modifier * 0.7
            elif term == "medium_term":
                total_modifier = ndvi_modifier * 0.5 + price_modifier * 0.5
            else:  # long_term
                total_modifier = ndvi_modifier * 0.7 + price_modifier * 0.3
            
            # Obliczanie prognozowanej ceny
            forecast[term]["change_pct"] = total_modifier * 100  # zamiana na procenty
            forecast[term]["price"] = round(current_price * (1 + total_modifier), 2)
        
        # Generowanie uzasadnienia prognozy
        if ndvi_trend == "rosnąca" and price_trend == "malejąca":
            forecast["forecast_rationale"] = "Rosnący indeks NDVI sugeruje dobre zbiory, co w połączeniu z malejącym trendem cenowym wskazuje na prawdopodobny spadek cen w przyszłości."
        elif ndvi_trend == "malejąca" and price_trend == "rosnąca":
            forecast["forecast_rationale"] = "Malejący indeks NDVI sugeruje gorsze zbiory, co w połączeniu z rosnącym trendem cenowym wskazuje na prawdopodobny wzrost cen w przyszłości."
        elif ndvi_trend == "rosnąca" and price_trend == "rosnąca":
            forecast["forecast_rationale"] = "Wzrost NDVI sugeruje dobre zbiory, jednak aktualny wzrost cen może być spowodowany innymi czynnikami rynkowymi. Oczekuje się stabilizacji cen w dłuższym okresie."
        elif ndvi_trend == "malejąca" and price_trend == "malejąca":
            forecast["forecast_rationale"] = "Spadek NDVI sugeruje gorsze zbiory, jednak aktualne spadki cen mogą być tymczasowe. Oczekuje się odwrócenia trendu i wzrostu cen w dłuższym okresie."
        else:
            forecast["forecast_rationale"] = "Stabilne warunki wzrostu i umiarkowane sygnały rynkowe sugerują niewielkie zmiany cenowe w najbliższym czasie."
        
        return forecast
    
    def generate_automated_report(self, 
                                 field_name: str, 
                                 crop_type: str, 
                                 report_format: str = "markdown") -> Tuple[str, str]:
        """
        Generuje automatyczny raport rynkowy dla wybranego pola i uprawy.
        
        Args:
            field_name: Nazwa pola
            crop_type: Typ uprawy
            report_format: Format raportu (markdown lub html)
            
        Returns:
            Tuple zawierający treść raportu i jego identyfikator
        """
        try:
            # Ładowanie danych pola
            field_data = self.load_field_data(field_name)
            
            if not field_data:
                error_message = f"Brak danych dla pola {field_name}"
                logger.error(error_message)
                return error_message, ""
            
            # Analiza trendów rynkowych
            analysis_results = self.analyze_market_trends(field_data, crop_type)
            
            # Generowanie raportu w formie tekstowej
            report_id = f"auto_report_{field_name}_{crop_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            report_content = self._format_report(field_name, analysis_results, report_format)
            
            # Zapisywanie raportu
            report_filename = f"{report_id}.{'html' if report_format == 'html' else 'md'}"
            report_path = self.reports_dir / report_filename
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            # Zapisywanie informacji o raporcie
            self.generated_reports[report_id] = {
                "id": report_id,
                "field_name": field_name,
                "crop_type": crop_type,
                "report_format": report_format,
                "report_path": str(report_path),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            logger.info(f"Wygenerowano automatyczny raport: {report_id}")
            
            return report_content, report_id
            
        except Exception as e:
            error_message = f"Błąd podczas generowania automatycznego raportu: {str(e)}"
            logger.error(error_message)
            return error_message, ""
    
    def _format_report(self, field_name: str, analysis_results: Dict[str, Any], report_format: str) -> str:
        """
        Formatuje wyniki analizy do postaci raportu.
        
        Args:
            field_name: Nazwa pola
            analysis_results: Wyniki analizy
            report_format: Format raportu (markdown lub html)
            
        Returns:
            Sformatowany raport
        """
        crop_type = analysis_results["crop_type"]
        crop_pl = analysis_results["crop_pl"]
        ndvi_trend = analysis_results["ndvi_trend"]
        price_trend = analysis_results["price_trend"]
        forecast = analysis_results["forecast"]
        
        # Utworzenie obecnej daty i dat dla różnych okresów
        today = datetime.datetime.now()
        short_term_date = (today + datetime.timedelta(days=30)).strftime("%d.%m.%Y")
        medium_term_date = (today + datetime.timedelta(days=90)).strftime("%d.%m.%Y")
        long_term_date = (today + datetime.timedelta(days=180)).strftime("%d.%m.%Y")
        
        # Generowanie treści raportu w formacie Markdown
        md_content = f"""# Automatyczny Raport Analizy Rynkowej

**Wygenerowano:** {today.strftime("%d.%m.%Y, %H:%M")}  
**Pole:** {field_name}  
**Uprawa:** {crop_pl} ({crop_type})

## Podsumowanie Analizy

Analiza satelitarna wskazuje na **{ndvi_trend.upper()}** tendencję indeksu wegetacji (NDVI), 
co sugeruje {
    "dobre warunki wzrostu i potencjalnie wyższe zbiory." if ndvi_trend == "rosnąca" else
    "gorsze warunki wzrostu i potencjalnie niższe zbiory." if ndvi_trend == "malejąca" else
    "stabilne warunki wzrostu."
}

Jednocześnie analiza sygnałów rynkowych wskazuje na **{price_trend.upper()}** tendencję cen kontraktów terminowych, 
co może prowadzić do {
    "wzrostu cen w najbliższym okresie." if price_trend == "rosnąca" else
    "spadku cen w najbliższym okresie." if price_trend == "malejąca" else
    "stabilizacji cen w najbliższym okresie."
}

## Prognozy Cenowe

### Aktualna cena
{forecast['current_price']:.2f} EUR/t

### Prognoza krótkoterminowa (do {short_term_date})
**Spodziewana cena:** {forecast['short_term']['price']:.2f} EUR/t  
**Zmiana:** {forecast['short_term']['change_pct']:.2f}%

### Prognoza średnioterminowa (do {medium_term_date})
**Spodziewana cena:** {forecast['medium_term']['price']:.2f} EUR/t  
**Zmiana:** {forecast['medium_term']['change_pct']:.2f}%

### Prognoza długoterminowa (do {long_term_date})
**Spodziewana cena:** {forecast['long_term']['price']:.2f} EUR/t  
**Zmiana:** {forecast['long_term']['change_pct']:.2f}%

## Uzasadnienie Prognozy

{forecast['forecast_rationale']}

## Rekomendacje Handlowe

{
    "- Rozważ sprzedaż kontraktów na poziomie aktualnych cen\n- Zabezpiecz co najmniej 30% przewidywanych zbiorów\n- Monitoruj prognozy meteorologiczne pod kątem zmian" 
    if ndvi_trend == "rosnąca" and price_trend != "rosnąca" else
    
    "- Wstrzymaj się ze sprzedażą kontraktów\n- Monitoruj sytuację podażową w innych regionach\n- Rozważ zakup kontraktów po spadku cen"
    if ndvi_trend == "malejąca" and price_trend != "malejąca" else
    
    "- Rozłóż sprzedaż kontraktów w czasie\n- Zabezpiecz część zbiorów kontraktami terminowymi\n- Przygotuj strategię na wypadek wzrostu zmienności"
}

## Dane Wykorzystane w Analizie

- Dane satelitarne indeksu NDVI
- Historyczne ceny kontraktów {analysis_results['commodity_symbol']}
- Sygnały rynkowe oparte na korelacji danych satelitarnych i historycznych cen

---

*Raport wygenerowany automatycznie przez system Agro Insight*
"""
        
        # Jeśli format to HTML, konwertuj z Markdown
        if report_format.lower() == "html":
            import markdown
            html_content = markdown.markdown(md_content, extensions=['tables', 'nl2br'])
            
            html_report = f"""
            <html>
            <head>
                <title>Automatyczny Raport - {field_name} - {crop_type}</title>
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
            return html_report
        
        return md_content
    
    def schedule_automated_reports(self, 
                                  field_names: List[str], 
                                  crop_types: List[str], 
                                  frequency: str = "daily", 
                                  delivery_emails: List[str] = None) -> Dict[str, Any]:
        """
        Planuje automatyczne generowanie raportów.
        
        Args:
            field_names: Lista nazw pól
            crop_types: Lista typów upraw
            frequency: Częstotliwość generowania (daily, weekly, monthly)
            delivery_emails: Lista adresów e-mail do dostarczenia raportów
            
        Returns:
            Słownik z informacjami o zaplanowanych raportach
        """
        # W rzeczywistej implementacji tutaj byłaby logika planowania zadań
        # oraz integracja z systemem wysyłania e-maili
        
        schedule_id = f"schedule_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        schedule_info = {
            "id": schedule_id,
            "field_names": field_names,
            "crop_types": crop_types,
            "frequency": frequency,
            "delivery_emails": delivery_emails or [],
            "creation_date": datetime.datetime.now().isoformat(),
            "status": "active"
        }
        
        logger.info(f"Zaplanowano automatyczne generowanie raportów: {schedule_id}")
        
        return schedule_info