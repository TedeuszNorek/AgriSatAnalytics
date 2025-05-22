"""
Moduł agentów LLM do zaawansowanego reasoningu i prowadzenia narracji w systemie Agro Insight.
System oparty na komunikujących się partycjach LLM, które współpracują 
w celu generowania spójnych i uzasadnionych analiz danych satelitarnych.
"""
import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np

# Inicjalizacja loggera
logger = logging.getLogger(__name__)

class LLMAgent:
    """Bazowa klasa agenta LLM z funkcjami reasoningu."""
    
    def __init__(self, agent_name: str, expertise: str):
        """
        Inicjalizacja agenta LLM.
        
        Args:
            agent_name: Nazwa agenta
            expertise: Obszar ekspertyzy agenta
        """
        self.agent_name = agent_name
        self.expertise = expertise
        self.memory = []
        self.reasoning_chain = []
        
        # Katalog do przechowywania danych agenta
        self.data_dir = Path(f"data/llm_agents/{agent_name}")
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Agent LLM '{agent_name}' utworzony z ekspertyzą: {expertise}")
    
    def think(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Przeprowadź proces reasoningu na podstawie danych wejściowych.
        
        Args:
            input_data: Dane wejściowe dla procesu reasoningu
            
        Returns:
            Wyniki procesu reasoningu
        """
        # Zapisz dane wejściowe w pamięci
        self.memory.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "input": input_data
        })
        
        # Podstawowy proces reasoningu (może być rozszerzony w podklasach)
        reasoning_steps = [
            {"step": "Analiza danych wejściowych", "result": f"Otrzymano dane do analizy: {list(input_data.keys())}"},
            {"step": "Wnioskowanie", "result": f"Agent {self.agent_name} przeprowadza wnioskowanie w obszarze {self.expertise}"},
            {"step": "Wygenerowanie odpowiedzi", "result": "Wygenerowano odpowiedź na podstawie dostępnych danych"}
        ]
        
        # Zapisz kroki reasoningu
        self.reasoning_chain.extend(reasoning_steps)
        
        # W wersji podstawowej zwracamy dane wejściowe z dodatkową interpretacją
        result = {
            "agent": self.agent_name,
            "expertise": self.expertise,
            "timestamp": datetime.datetime.now().isoformat(),
            "interpretation": f"Analiza danych w zakresie {self.expertise}",
            "confidence": 0.75,
            "reasoning": reasoning_steps
        }
        
        return result
    
    def save_reasoning(self, filename: str = None):
        """
        Zapisz łańcuch reasoningu do pliku.
        
        Args:
            filename: Nazwa pliku (opcjonalna)
        """
        if filename is None:
            filename = f"{self.agent_name}_reasoning_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        file_path = self.data_dir / filename
        
        data = {
            "agent": self.agent_name,
            "expertise": self.expertise,
            "timestamp": datetime.datetime.now().isoformat(),
            "reasoning_chain": self.reasoning_chain,
            "memory_size": len(self.memory)
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Zapisano reasoning agenta {self.agent_name} do pliku {file_path}")
        return str(file_path)
    
    def load_reasoning(self, filepath: str):
        """
        Wczytaj zapisany łańcuch reasoningu z pliku.
        
        Args:
            filepath: Ścieżka do pliku z zapisanym reasoningiem
        """
        file_path = Path(filepath)
        if not file_path.exists():
            logger.error(f"Nie znaleziono pliku reasoningu: {file_path}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if "reasoning_chain" in data:
                self.reasoning_chain.extend(data["reasoning_chain"])
                logger.info(f"Wczytano reasoning z pliku {file_path}")
                return True
            else:
                logger.error(f"Nieprawidłowy format pliku reasoningu: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Błąd wczytywania reasoningu z pliku {file_path}: {str(e)}")
            return False


class SatelliteAnalysisAgent(LLMAgent):
    """Agent LLM specjalizujący się w analizie danych satelitarnych."""
    
    def __init__(self):
        """Inicjalizacja agenta analizy satelitarnej."""
        super().__init__("SatelliteAnalyst", "analiza danych satelitarnych i wskaźników wegetacji")
    
    def analyze_ndvi_data(self, field_name: str, ndvi_time_series: Dict[str, float]) -> Dict[str, Any]:
        """
        Analizuj dane NDVI dla pola i wyprowadź wnioski.
        
        Args:
            field_name: Nazwa pola
            ndvi_time_series: Szereg czasowy NDVI (data -> wartość)
            
        Returns:
            Wyniki analizy
        """
        if not ndvi_time_series:
            logger.warning(f"Brak danych NDVI dla pola {field_name}")
            return {
                "agent": self.agent_name,
                "field": field_name,
                "status": "error",
                "message": "Brak danych NDVI do analizy"
            }
        
        # Konwersja dat na format datetime dla analizy
        dated_values = []
        for date_str, value in ndvi_time_series.items():
            try:
                date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
                dated_values.append((date, value))
            except Exception:
                continue
        
        dated_values.sort(key=lambda x: x[0])
        
        if len(dated_values) < 2:
            return {
                "agent": self.agent_name,
                "field": field_name,
                "status": "insufficient_data",
                "message": "Zbyt mało punktów danych do analizy trendu"
            }
        
        # Podstawowa analiza danych NDVI
        dates = [d for d, _ in dated_values]
        values = [v for _, v in dated_values]
        
        # Oblicz statystyki
        current_ndvi = values[-1]
        avg_ndvi = sum(values) / len(values)
        min_ndvi = min(values)
        max_ndvi = max(values)
        
        # Oblicz trend krótkoterminowy i długoterminowy
        if len(values) >= 3:
            short_term_trend = values[-1] - values[-3]  # Zmiana w ostatnich 3 punktach
        else:
            short_term_trend = values[-1] - values[0]
        
        long_term_trend = values[-1] - values[0]  # Zmiana od początku do końca
        
        # Proces reasoningu dla interpretacji danych
        reasoning_steps = [
            {"step": "Analiza danych NDVI", 
             "result": f"Analizowano {len(values)} punktów danych NDVI dla pola {field_name}"},
            {"step": "Obliczenie statystyk", 
             "result": f"Aktualne NDVI: {current_ndvi:.3f}, Średnie: {avg_ndvi:.3f}, Min: {min_ndvi:.3f}, Max: {max_ndvi:.3f}"}
        ]
        
        # Interpretacja trendu
        interpretation = ""
        trend_assessment = ""
        health_status = ""
        
        if current_ndvi > 0.7:
            health_status = "bardzo dobry"
            interpretation += "Wysokie wartości NDVI wskazują na gęstą, zdrową roślinność z wysoką aktywnością fotosyntezy. "
            reasoning_steps.append({"step": "Interpretacja wartości NDVI", 
                                  "result": "Wysokie NDVI (>0.7) wskazuje na bardzo dobrą kondycję roślinności"})
        elif current_ndvi > 0.5:
            health_status = "dobry"
            interpretation += "Dobre wartości NDVI wskazują na zdrową roślinność. "
            reasoning_steps.append({"step": "Interpretacja wartości NDVI", 
                                  "result": "Dobre NDVI (>0.5) wskazuje na dobrą kondycję roślinności"})
        elif current_ndvi > 0.3:
            health_status = "umiarkowany"
            interpretation += "Umiarkowane wartości NDVI wskazują na mniej gęstą roślinność lub początkowe stadia rozwoju. "
            reasoning_steps.append({"step": "Interpretacja wartości NDVI", 
                                  "result": "Umiarkowane NDVI (>0.3) wskazuje na średnią kondycję roślinności"})
        else:
            health_status = "słaby"
            interpretation += "Niskie wartości NDVI wskazują na rzadką roślinność, suszę lub okres spoczynku. "
            reasoning_steps.append({"step": "Interpretacja wartości NDVI", 
                                  "result": "Niskie NDVI (<0.3) wskazuje na słabą kondycję roślinności"})
        
        if short_term_trend > 0.05:
            trend_assessment = "silnie rosnący"
            interpretation += "Obserwujemy silny wzrost wskaźnika NDVI w ostatnim okresie, co wskazuje na szybki rozwój roślinności. "
            reasoning_steps.append({"step": "Analiza trendu krótkoterminowego", 
                                  "result": f"Silny wzrost NDVI (+{short_term_trend:.3f}) wskazuje na szybki rozwój roślinności"})
        elif short_term_trend > 0.02:
            trend_assessment = "rosnący"
            interpretation += "Obserwujemy wzrost wskaźnika NDVI, co wskazuje na rozwój roślinności. "
            reasoning_steps.append({"step": "Analiza trendu krótkoterminowego", 
                                  "result": f"Wzrost NDVI (+{short_term_trend:.3f}) wskazuje na rozwój roślinności"})
        elif short_term_trend < -0.05:
            trend_assessment = "silnie spadający"
            interpretation += "Obserwujemy silny spadek wskaźnika NDVI, co może wskazywać na problemy z kondycją upraw (susza, choroby, szkodniki). "
            reasoning_steps.append({"step": "Analiza trendu krótkoterminowego", 
                                  "result": f"Silny spadek NDVI ({short_term_trend:.3f}) może wskazywać na problemy z uprawami"})
        elif short_term_trend < -0.02:
            trend_assessment = "spadający"
            interpretation += "Obserwujemy spadek wskaźnika NDVI, co może wskazywać na pogorszenie kondycji upraw. "
            reasoning_steps.append({"step": "Analiza trendu krótkoterminowego", 
                                  "result": f"Spadek NDVI ({short_term_trend:.3f}) może wskazywać na pogorszenie kondycji upraw"})
        else:
            trend_assessment = "stabilny"
            interpretation += "Wskaźnik NDVI jest stabilny, co wskazuje na utrzymującą się kondycję roślinności. "
            reasoning_steps.append({"step": "Analiza trendu krótkoterminowego", 
                                  "result": f"Stabilny NDVI ({short_term_trend:.3f}) wskazuje na utrzymującą się kondycję roślinności"})
        
        # Zapisz kroki reasoningu
        self.reasoning_chain.extend(reasoning_steps)
        
        # Podsumowanie analizy
        analysis_result = {
            "agent": self.agent_name,
            "field": field_name,
            "status": "success",
            "current_ndvi": float(current_ndvi),
            "avg_ndvi": float(avg_ndvi),
            "min_ndvi": float(min_ndvi),
            "max_ndvi": float(max_ndvi),
            "short_term_trend": float(short_term_trend),
            "long_term_trend": float(long_term_trend),
            "health_status": health_status,
            "trend_assessment": trend_assessment,
            "interpretation": interpretation,
            "reasoning": reasoning_steps,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return analysis_result


class YieldForecastAgent(LLMAgent):
    """Agent LLM specjalizujący się w prognozowaniu plonów."""
    
    def __init__(self):
        """Inicjalizacja agenta prognozowania plonów."""
        super().__init__("YieldForecaster", "prognozowanie plonów na podstawie danych satelitarnych")
    
    def forecast_yield(self, field_name: str, ndvi_analysis: Dict[str, Any], field_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prognozuj plony na podstawie analizy NDVI i informacji o polu.
        
        Args:
            field_name: Nazwa pola
            ndvi_analysis: Wyniki analizy NDVI
            field_info: Informacje o polu (typ uprawy, itp.)
            
        Returns:
            Prognoza plonów
        """
        # Sprawdź, czy mamy wymaganę dane
        if not ndvi_analysis or ndvi_analysis.get("status") != "success":
            logger.warning(f"Brak poprawnej analizy NDVI dla pola {field_name}")
            return {
                "agent": self.agent_name,
                "field": field_name,
                "status": "error",
                "message": "Brak poprawnej analizy NDVI do prognozowania plonów"
            }
        
        # Pobierz informacje o uprawie
        crop_type = field_info.get("crop_type", "unknown")
        
        # Bazowe plony dla różnych upraw (tony na hektar)
        base_yields = {
            'Wheat': 3.5,
            'Corn': 9.0,
            'Soybean': 3.0,
            'Barley': 3.2,
            'Oats': 2.5,
            'Rice': 4.5
        }
        
        # Domyślnie pszenica, jeśli typ uprawy jest nieznany
        if crop_type.lower() not in [k.lower() for k in base_yields.keys()]:
            crop_type = 'Wheat'
            reasoning_step = {"step": "Określenie typu uprawy", 
                            "result": f"Nieznany typ uprawy, przyjęto domyślnie: Wheat"}
        else:
            # Znajdź pasujący typ uprawy (bez uwzględniania wielkości liter)
            crop_key = next((k for k in base_yields.keys() if k.lower() == crop_type.lower()), 'Wheat')
            crop_type = crop_key
            reasoning_step = {"step": "Określenie typu uprawy", 
                            "result": f"Określono typ uprawy na podstawie danych: {crop_type}"}
        
        self.reasoning_chain.append(reasoning_step)
        
        # Pobierz bazowy plon dla danego typu uprawy
        base_yield = base_yields[crop_type]
        
        # Pobierz dane z analizy NDVI
        current_ndvi = ndvi_analysis.get("current_ndvi", 0.5)
        short_term_trend = ndvi_analysis.get("short_term_trend", 0)
        
        # Proces reasoningu dla prognozy plonów
        reasoning_steps = [
            {"step": "Analiza danych wejściowych", 
             "result": f"Analizowano dane NDVI dla pola {field_name}, typ uprawy: {crop_type}"},
            {"step": "Określenie bazowego plonu", 
             "result": f"Bazowy plon dla uprawy {crop_type}: {base_yield} t/ha"}
        ]
        
        # Oblicz wpływ NDVI na plony
        ndvi_factor = 1.0 + ((current_ndvi - 0.5) * 0.5)  # Wpływ NDVI na plon
        reasoning_steps.append({"step": "Obliczenie wpływu NDVI", 
                             "result": f"Współczynnik NDVI: {ndvi_factor:.3f} (obecne NDVI: {current_ndvi:.3f})"})
        
        # Oblicz wpływ trendu na plony
        trend_factor = 1.0 + (short_term_trend * 15)  # Wpływ trendu na plon
        reasoning_steps.append({"step": "Obliczenie wpływu trendu", 
                             "result": f"Współczynnik trendu: {trend_factor:.3f} (trend krótkoterminowy: {short_term_trend:.3f})"})
        
        # Przygotuj daty prognozy (następne 30, 60, 90 dni)
        today = datetime.datetime.now()
        forecast_dates = [
            (today + datetime.timedelta(days=30)).strftime('%Y-%m-%d'),
            (today + datetime.timedelta(days=60)).strftime('%Y-%m-%d'),
            (today + datetime.timedelta(days=90)).strftime('%Y-%m-%d')
        ]
        
        # Oblicz prognozy plonów dla każdej daty prognozy
        forecasts = {}
        for i, date in enumerate(forecast_dates):
            # Różne horyzonty prognozy mają różne niepewności
            horizon_factor = 1.0 - (i * 0.05)  # Zmniejsz pewność dla dalszych prognoz
            reasoning_steps.append({"step": f"Obliczenie prognozy na {date}", 
                                 "result": f"Współczynnik horyzontu czasowego: {horizon_factor:.3f}"})
            
            # Dodaj losową wariację, aby symulować niepewność
            random_factor = 1.0 + np.random.uniform(-0.05, 0.05)
            
            # Oblicz finalną prognozę plonu
            yield_prediction = base_yield * ndvi_factor * trend_factor * horizon_factor * random_factor
            
            # Dodaj kontekst niepewności
            if i == 0:
                confidence = "wysoka"
                uncertainty_level = "niski"
            elif i == 1:
                confidence = "średnia"
                uncertainty_level = "średni"
            else:
                confidence = "niska"
                uncertainty_level = "wysoki"
            
            forecasts[date] = {
                "yield": round(yield_prediction, 2),
                "unit": "t/ha",
                "confidence": confidence,
                "uncertainty_level": uncertainty_level
            }
        
        # Interpretacja prognozy
        if ndvi_factor > 1.1 and trend_factor > 1.05:
            interpretation = "Prognoza plonów jest bardzo optymistyczna ze względu na wysokie wartości NDVI i pozytywny trend rozwoju roślinności. "
            reasoning_steps.append({"step": "Interpretacja prognozy", 
                                 "result": "Bardzo optymistyczna prognoza plonów"})
        elif ndvi_factor > 1.0 and trend_factor > 1.0:
            interpretation = "Prognoza plonów jest optymistyczna - wartości NDVI i trend rozwoju roślinności są pozytywne. "
            reasoning_steps.append({"step": "Interpretacja prognozy", 
                                 "result": "Optymistyczna prognoza plonów"})
        elif ndvi_factor < 0.9 and trend_factor < 0.95:
            interpretation = "Prognoza plonów jest pesymistyczna ze względu na niskie wartości NDVI i negatywny trend rozwoju roślinności. "
            reasoning_steps.append({"step": "Interpretacja prognozy", 
                                 "result": "Pesymistyczna prognoza plonów"})
        else:
            interpretation = "Prognoza plonów jest umiarkowana - wartości NDVI i trend rozwoju roślinności są przeciętne. "
            reasoning_steps.append({"step": "Interpretacja prognozy", 
                                 "result": "Umiarkowana prognoza plonów"})
        
        # Dodaj rekomendacje
        if ndvi_factor < 0.9:
            interpretation += "Zalecane jest rozważenie dodatkowego nawożenia lub innych działań wspierających rozwój upraw. "
        
        # Zapisz kroki reasoningu
        self.reasoning_chain.extend(reasoning_steps)
        
        # Podsumowanie prognozy
        forecast_result = {
            "agent": self.agent_name,
            "field": field_name,
            "crop_type": crop_type,
            "status": "success",
            "base_yield": base_yield,
            "ndvi_factor": float(ndvi_factor),
            "trend_factor": float(trend_factor),
            "forecasts": forecasts,
            "interpretation": interpretation,
            "reasoning": reasoning_steps,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return forecast_result


class MarketAnalysisAgent(LLMAgent):
    """Agent LLM specjalizujący się w analizie rynku i prognozowaniu cen."""
    
    def __init__(self):
        """Inicjalizacja agenta analizy rynku."""
        super().__init__("MarketAnalyst", "analiza rynku i prognozowanie cen towarów")
        
        # Mapowanie typu uprawy na symbole towarów
        self.crop_to_symbol = {
            'Wheat': 'ZW=F',
            'Corn': 'ZC=F',
            'Soybean': 'ZS=F',
            'Oats': 'ZO=F',
            'Rice': 'ZR=F'
        }
    
    def analyze_market(self, field_name: str, crop_type: str, ndvi_analysis: Dict[str, Any], yield_forecast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizuj rynek i generuj sygnały handlowe na podstawie analizy NDVI i prognozy plonów.
        
        Args:
            field_name: Nazwa pola
            crop_type: Typ uprawy
            ndvi_analysis: Wyniki analizy NDVI
            yield_forecast: Prognoza plonów
            
        Returns:
            Analiza rynku i sygnały handlowe
        """
        # Sprawdź dane wejściowe
        if not ndvi_analysis or ndvi_analysis.get("status") != "success":
            logger.warning(f"Brak poprawnej analizy NDVI dla pola {field_name}")
            return {
                "agent": self.agent_name,
                "field": field_name,
                "status": "error",
                "message": "Brak poprawnej analizy NDVI do analizy rynku"
            }
        
        # Domyślnie pszenica, jeśli typ uprawy jest nieznany
        if crop_type.lower() not in [k.lower() for k in self.crop_to_symbol.keys()]:
            crop_type = 'Wheat'
            reasoning_step = {"step": "Określenie typu uprawy dla analizy rynku", 
                            "result": f"Nieznany typ uprawy, przyjęto domyślnie: Wheat"}
        else:
            # Znajdź pasujący typ uprawy (bez uwzględniania wielkości liter)
            crop_key = next((k for k in self.crop_to_symbol.keys() if k.lower() == crop_type.lower()), 'Wheat')
            crop_type = crop_key
            reasoning_step = {"step": "Określenie typu uprawy dla analizy rynku", 
                            "result": f"Określono typ uprawy na podstawie danych: {crop_type}"}
        
        self.reasoning_chain.append(reasoning_step)
        
        # Pobierz symbol dla danego typu uprawy
        primary_symbol = self.crop_to_symbol[crop_type]
        
        # Pobierz dane z analizy NDVI
        current_ndvi = ndvi_analysis.get("current_ndvi", 0.5)
        short_term_trend = ndvi_analysis.get("short_term_trend", 0)
        
        # Pobierz dane z prognozy plonów
        yield_trend = "neutral"
        if yield_forecast and yield_forecast.get("status") == "success":
            forecasts = yield_forecast.get("forecasts", {})
            if forecasts:
                forecast_dates = sorted(forecasts.keys())
                first_forecast = forecasts[forecast_dates[0]].get("yield", 0)
                last_forecast = forecasts[forecast_dates[-1]].get("yield", 0)
                
                if last_forecast > first_forecast * 1.05:
                    yield_trend = "increasing"
                    reasoning_step = {"step": "Analiza trendu prognozy plonów", 
                                    "result": f"Rosnący trend prognozy plonów: {first_forecast} -> {last_forecast}"}
                elif last_forecast < first_forecast * 0.95:
                    yield_trend = "decreasing"
                    reasoning_step = {"step": "Analiza trendu prognozy plonów", 
                                    "result": f"Spadający trend prognozy plonów: {first_forecast} -> {last_forecast}"}
                else:
                    reasoning_step = {"step": "Analiza trendu prognozy plonów", 
                                    "result": f"Stabilny trend prognozy plonów: {first_forecast} -> {last_forecast}"}
                
                self.reasoning_chain.append(reasoning_step)
        
        # Proces reasoningu dla analizy rynku
        reasoning_steps = [
            {"step": "Analiza danych wejściowych", 
             "result": f"Analizowano dane dla pola {field_name}, typ uprawy: {crop_type}, symbol: {primary_symbol}"},
            {"step": "Analiza wskaźników NDVI", 
             "result": f"NDVI: {current_ndvi:.3f}, trend krótkoterminowy: {short_term_trend:.3f}"}
        ]
        
        # Generuj sygnały rynkowe dla głównego towaru
        market_signal = {}
        
        # Logika generowania sygnałów rynkowych
        if short_term_trend > 0.03:  # Silny pozytywny trend NDVI
            action = "SHORT"  # Oczekuj spadku cen ze względu na dobre warunki upraw
            confidence = min(0.8, 0.6 + (short_term_trend * 5))
            reason = f"Silny pozytywny trend NDVI (+{short_term_trend:.4f}) sugeruje dobre warunki upraw, co potencjalnie prowadzi do zwiększonej podaży i niższych cen."
            reasoning_steps.append({"step": "Generowanie sygnału rynkowego", 
                                 "result": f"Pozytywny trend NDVI -> Sygnał SHORT (zwiększona podaż -> niższe ceny)"})
        elif short_term_trend < -0.03:  # Silny negatywny trend NDVI
            action = "LONG"  # Oczekuj wzrostu cen ze względu na złe warunki upraw
            confidence = min(0.8, 0.6 + (abs(short_term_trend) * 5))
            reason = f"Silny negatywny trend NDVI ({short_term_trend:.4f}) sugeruje złe warunki upraw, co potencjalnie prowadzi do zmniejszonej podaży i wyższych cen."
            reasoning_steps.append({"step": "Generowanie sygnału rynkowego", 
                                 "result": f"Negatywny trend NDVI -> Sygnał LONG (zmniejszona podaż -> wyższe ceny)"})
        else:  # Neutralny trend NDVI
            action = "NEUTRAL"
            confidence = 0.5
            reason = f"Neutralny trend NDVI ({short_term_trend:.4f}) sugeruje stabilne warunki upraw bez silnego sygnału rynkowego."
            reasoning_steps.append({"step": "Generowanie sygnału rynkowego", 
                                 "result": f"Neutralny trend NDVI -> Brak wyraźnego sygnału rynkowego"})
        
        # Dostosuj pewność na podstawie zgodności różnych wskaźników
        if (action == "SHORT" and yield_trend == "increasing") or (action == "LONG" and yield_trend == "decreasing"):
            confidence += 0.1  # Zwiększ pewność, jeśli sygnał zgodny z trendem prognozy plonów
            reasoning_steps.append({"step": "Dostosowanie pewności sygnału", 
                                 "result": f"Zwiększono pewność sygnału ze względu na zgodność z trendem prognozy plonów"})
        
        # Utwórz sygnał
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        market_signal = {
            "date": today,
            "symbol": primary_symbol,
            "crop_type": crop_type,
            "action": action,
            "confidence": round(min(0.9, confidence), 2),  # Ogranicz do 0.9
            "reason": reason,
            "time_horizon": "short_term"  # Krótkoterminowa prognoza (domyślnie)
        }
        
        # Dodaj analizę rynku dla medium i long term
        market_analysis = {
            "short_term": {
                "trend": market_signal["action"],
                "confidence": market_signal["confidence"],
                "reason": market_signal["reason"]
            },
            "medium_term": {},
            "long_term": {}
        }
        
        # Generuj analizę średnioterminową (bazując na NDVI i prognozie plonów)
        if yield_forecast and "forecasts" in yield_forecast:
            # W średnim terminie bardziej opieramy się na prognozie plonów
            if yield_trend == "increasing":
                medium_action = "SHORT"
                medium_confidence = 0.65
                medium_reason = "Rosnący trend prognozy plonów wskazuje na potencjalnie większe zbiory, co może prowadzić do spadku cen w średnim terminie."
                reasoning_steps.append({"step": "Generowanie analizy średnioterminowej", 
                                     "result": f"Rosnący trend prognozy plonów -> Spadek cen w średnim terminie"})
            elif yield_trend == "decreasing":
                medium_action = "LONG"
                medium_confidence = 0.65
                medium_reason = "Spadający trend prognozy plonów wskazuje na potencjalnie mniejsze zbiory, co może prowadzić do wzrostu cen w średnim terminie."
                reasoning_steps.append({"step": "Generowanie analizy średnioterminowej", 
                                     "result": f"Spadający trend prognozy plonów -> Wzrost cen w średnim terminie"})
            else:
                medium_action = "NEUTRAL"
                medium_confidence = 0.5
                medium_reason = "Stabilny trend prognozy plonów nie daje wyraźnych sygnałów zmian cen w średnim terminie."
                reasoning_steps.append({"step": "Generowanie analizy średnioterminowej", 
                                     "result": f"Stabilny trend prognozy plonów -> Brak wyraźnych sygnałów cenowych"})
            
            market_analysis["medium_term"] = {
                "trend": medium_action,
                "confidence": medium_confidence,
                "reason": medium_reason
            }
            
            # Dodaj prostą analizę długoterminową
            # W długim terminie opieramy się głównie na kondycji upraw i szerszych trendach
            long_term_analysis = {
                "trend": "NEUTRAL",
                "confidence": 0.4,
                "reason": "Długoterminowa prognoza wymaga uwzględnienia dodatkowych czynników rynkowych i globalnych."
            }
            market_analysis["long_term"] = long_term_analysis
        
        # Dodaj rekomendacje handlowe
        trade_recommendations = []
        
        if market_signal["action"] == "LONG" and market_signal["confidence"] > 0.7:
            trade_recommendations.append({
                "type": "entry",
                "action": "BUY",
                "reasoning": "Silny sygnał wzrostowy z wysoką pewnością wskazuje na dobry moment do zajęcia pozycji długiej (BUY)."
            })
        elif market_signal["action"] == "SHORT" and market_signal["confidence"] > 0.7:
            trade_recommendations.append({
                "type": "entry",
                "action": "SELL",
                "reasoning": "Silny sygnał spadkowy z wysoką pewnością wskazuje na dobry moment do zajęcia pozycji krótkiej (SELL)."
            })
        
        if market_analysis["medium_term"].get("trend") != market_signal["action"]:
            trade_recommendations.append({
                "type": "risk_management",
                "action": "HEDGE",
                "reasoning": "Różnica między krótko- i średnioterminową prognozą sugeruje rozważenie strategii zabezpieczającej (HEDGE)."
            })
        
        # Zapisz kroki reasoningu
        self.reasoning_chain.extend(reasoning_steps)
        
        # Podsumowanie analizy rynku
        market_result = {
            "agent": self.agent_name,
            "field": field_name,
            "crop_type": crop_type,
            "commodity_symbol": primary_symbol,
            "status": "success",
            "market_signal": market_signal,
            "market_analysis": market_analysis,
            "trade_recommendations": trade_recommendations,
            "reasoning": reasoning_steps,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return market_result


class NarrativeGenerator(LLMAgent):
    """Agent LLM generujący narrację i podsumowania w języku naturalnym."""
    
    def __init__(self):
        """Inicjalizacja agenta generującego narrację."""
        super().__init__("NarrativeGenerator", "generowanie narracji i podsumowań w języku naturalnym")
    
    def generate_narrative(self, field_name: str, satellite_analysis: Dict[str, Any], 
                         yield_forecast: Dict[str, Any], market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generuj narrację na podstawie analiz innych agentów.
        
        Args:
            field_name: Nazwa pola
            satellite_analysis: Wyniki analizy danych satelitarnych
            yield_forecast: Prognoza plonów
            market_analysis: Analiza rynku
            
        Returns:
            Podsumowanie narracyjne analiz
        """
        # Sprawdź dane wejściowe
        if not satellite_analysis or satellite_analysis.get("status") != "success":
            logger.warning(f"Brak poprawnej analizy satelitarnej dla pola {field_name}")
            return {
                "agent": self.agent_name,
                "field": field_name,
                "status": "error",
                "message": "Brak poprawnej analizy satelitarnej do generowania narracji"
            }
        
        # Pobierz istotne dane z analiz
        crop_type = yield_forecast.get("crop_type", "nieznana uprawa") if yield_forecast else "nieznana uprawa"
        health_status = satellite_analysis.get("health_status", "nieznany")
        trend_assessment = satellite_analysis.get("trend_assessment", "nieznany")
        
        # Proces reasoningu dla narracji
        reasoning_steps = [
            {"step": "Analiza danych wejściowych", 
             "result": f"Analizowano dane dla pola {field_name}, typ uprawy: {crop_type}"},
            {"step": "Identyfikacja głównych wniosków", 
             "result": f"Stan uprawy: {health_status}, trend: {trend_assessment}"}
        ]
        
        # Generuj podsumowanie stanu pola
        field_summary = f"Pole {field_name} z uprawą {crop_type} jest obecnie w {health_status} stanie zdrowotnym"
        if trend_assessment:
            field_summary += f" z {trend_assessment} trendem rozwoju."
        else:
            field_summary += "."
        
        reasoning_steps.append({"step": "Generowanie podsumowania stanu pola", 
                             "result": field_summary})
        
        # Generuj narrację o prognozie plonów
        yield_narrative = ""
        if yield_forecast and yield_forecast.get("status") == "success":
            forecasts = yield_forecast.get("forecasts", {})
            if forecasts:
                forecast_dates = sorted(forecasts.keys())
                nearest_forecast = forecasts[forecast_dates[0]].get("yield", 0)
                furthest_forecast = forecasts[forecast_dates[-1]].get("yield", 0)
                
                yield_narrative = f"Prognoza plonów dla {crop_type} wynosi {nearest_forecast} t/ha w krótkim terminie"
                
                if furthest_forecast > nearest_forecast * 1.05:
                    yield_narrative += f" i wzrasta do {furthest_forecast} t/ha w dalszej perspektywie, co wskazuje na pozytywny trend rozwojowy."
                    reasoning_steps.append({"step": "Generowanie narracji o prognozie plonów", 
                                         "result": "Zidentyfikowano wzrostowy trend prognozy plonów"})
                elif furthest_forecast < nearest_forecast * 0.95:
                    yield_narrative += f" i spada do {furthest_forecast} t/ha w dalszej perspektywie, co może wskazywać na ryzyko dla rozwoju uprawy."
                    reasoning_steps.append({"step": "Generowanie narracji o prognozie plonów", 
                                         "result": "Zidentyfikowano spadkowy trend prognozy plonów"})
                else:
                    yield_narrative += f" i utrzymuje się na podobnym poziomie ({furthest_forecast} t/ha) w dalszej perspektywie."
                    reasoning_steps.append({"step": "Generowanie narracji o prognozie plonów", 
                                         "result": "Zidentyfikowano stabilny trend prognozy plonów"})
        else:
            yield_narrative = "Brak wystarczających danych do wiarygodnej prognozy plonów."
            reasoning_steps.append({"step": "Generowanie narracji o prognozie plonów", 
                                 "result": "Brak danych prognozy plonów"})
        
        # Generuj narrację o rynku
        market_narrative = ""
        if market_analysis and market_analysis.get("status") == "success":
            market_signal = market_analysis.get("market_signal", {})
            market_analysis_data = market_analysis.get("market_analysis", {})
            
            if market_signal:
                action = market_signal.get("action")
                confidence = market_signal.get("confidence", 0)
                symbol = market_signal.get("symbol", "nieznany")
                
                market_narrative = f"Analiza rynku dla towaru {symbol} wskazuje na sygnał: {action} "
                
                if action == "LONG":
                    market_narrative += f"z {int(confidence*100)}% pewnością, co sugeruje potencjalny wzrost cen w najbliższym czasie. "
                    reasoning_steps.append({"step": "Generowanie narracji o rynku", 
                                         "result": "Zidentyfikowano sygnał wzrostowy (LONG)"})
                elif action == "SHORT":
                    market_narrative += f"z {int(confidence*100)}% pewnością, co sugeruje potencjalny spadek cen w najbliższym czasie. "
                    reasoning_steps.append({"step": "Generowanie narracji o rynku", 
                                         "result": "Zidentyfikowano sygnał spadkowy (SHORT)"})
                else:
                    market_narrative += "co sugeruje stabilność cen w najbliższym czasie. "
                    reasoning_steps.append({"step": "Generowanie narracji o rynku", 
                                         "result": "Zidentyfikowano sygnał neutralny (NEUTRAL)"})
                
                # Dodaj perspektywę średnioterminową
                if "medium_term" in market_analysis_data:
                    medium_trend = market_analysis_data["medium_term"].get("trend")
                    if medium_trend and medium_trend != action:
                        market_narrative += f"W średnim terminie przewidywany jest sygnał {medium_trend}, co może wskazywać na zmianę trendu. "
                        reasoning_steps.append({"step": "Generowanie narracji o perspektywie średnioterminowej", 
                                             "result": f"Zidentyfikowano zmianę trendu w średnim terminie: {medium_trend}"})
            else:
                market_narrative = "Brak jednoznacznych sygnałów rynkowych na podstawie dostępnych danych. "
                reasoning_steps.append({"step": "Generowanie narracji o rynku", 
                                     "result": "Brak jednoznacznych sygnałów rynkowych"})
            
            # Dodaj rekomendacje handlowe
            trade_recommendations = market_analysis.get("trade_recommendations", [])
            if trade_recommendations:
                market_narrative += "Rekomendacje handlowe: "
                for rec in trade_recommendations:
                    market_narrative += f"{rec.get('action')} ({rec.get('type')}). "
                
                reasoning_steps.append({"step": "Dodanie rekomendacji handlowych", 
                                     "result": f"Dodano {len(trade_recommendations)} rekomendacji handlowych"})
        else:
            market_narrative = "Brak wystarczających danych do wiarygodnej analizy rynku. "
            reasoning_steps.append({"step": "Generowanie narracji o rynku", 
                                 "result": "Brak danych analizy rynku"})
        
        # Generuj rekomendacje działań dla rolnika
        farmer_recommendations = []
        
        # Rekomendacje na podstawie stanu uprawy (NDVI)
        if health_status == "słaby":
            farmer_recommendations.append("Zalecamy przeprowadzenie dokładnej inspekcji pola w celu identyfikacji problemów z uprawą.")
            farmer_recommendations.append("Rozważ dodatkowe nawożenie lub zastosowanie środków ochrony roślin odpowiednich dla zidentyfikowanych problemów.")
            reasoning_steps.append({"step": "Generowanie rekomendacji dla rolnika", 
                                 "result": "Dodano rekomendacje związane ze słabym stanem uprawy"})
        elif health_status == "umiarkowany":
            farmer_recommendations.append("Zalecamy regularne monitorowanie stanu uprawy i rozważenie dodatkowego nawożenia, jeśli stan nie ulegnie poprawie.")
            reasoning_steps.append({"step": "Generowanie rekomendacji dla rolnika", 
                                 "result": "Dodano rekomendacje związane z umiarkowanym stanem uprawy"})
        
        # Rekomendacje na podstawie trendu rozwoju uprawy
        if trend_assessment == "spadający" or trend_assessment == "silnie spadający":
            farmer_recommendations.append("Spadający trend rozwoju uprawy może wskazywać na problemy z dostępnością wody lub składników odżywczych. Zalecamy przeprowadzenie analizy gleby.")
            reasoning_steps.append({"step": "Generowanie rekomendacji dla rolnika", 
                                 "result": "Dodano rekomendacje związane ze spadającym trendem rozwoju uprawy"})
        
        # Rekomendacje na podstawie prognozy plonów
        if yield_forecast and yield_forecast.get("status") == "success":
            interpretation = yield_forecast.get("interpretation", "")
            if "pesymistyczna" in interpretation.lower():
                farmer_recommendations.append("Prognoza plonów jest pesymistyczna. Rozważ zastosowanie dodatkowych zabiegów agrotechnicznych, które mogą poprawić potencjał plonowania.")
                reasoning_steps.append({"step": "Generowanie rekomendacji dla rolnika", 
                                     "result": "Dodano rekomendacje związane z pesymistyczną prognozą plonów"})
        
        # Rekomendacje na podstawie analizy rynku
        if market_analysis and market_analysis.get("status") == "success":
            market_signal = market_analysis.get("market_signal", {})
            if market_signal:
                action = market_signal.get("action")
                confidence = market_signal.get("confidence", 0)
                
                if action == "LONG" and confidence > 0.7:
                    farmer_recommendations.append("Analiza rynku wskazuje na potencjalny wzrost cen. Rozważ wstrzymanie się ze sprzedażą plonów, jeśli to możliwe.")
                    reasoning_steps.append({"step": "Generowanie rekomendacji dla rolnika", 
                                         "result": "Dodano rekomendacje związane z prognozą wzrostu cen"})
                elif action == "SHORT" and confidence > 0.7:
                    farmer_recommendations.append("Analiza rynku wskazuje na potencjalny spadek cen. Rozważ zabezpieczenie cen plonów poprzez kontrakty terminowe lub wcześniejszą sprzedaż.")
                    reasoning_steps.append({"step": "Generowanie rekomendacji dla rolnika", 
                                         "result": "Dodano rekomendacje związane z prognozą spadku cen"})
        
        # Zapisz kroki reasoningu
        self.reasoning_chain.extend(reasoning_steps)
        
        # Utwórz pełną narrację
        narrative = f"{field_summary} {yield_narrative} {market_narrative}"
        
        # Podsumowanie narracji
        narrative_result = {
            "agent": self.agent_name,
            "field": field_name,
            "crop_type": crop_type,
            "status": "success",
            "narrative": narrative,
            "field_summary": field_summary,
            "yield_narrative": yield_narrative,
            "market_narrative": market_narrative,
            "farmer_recommendations": farmer_recommendations,
            "reasoning": reasoning_steps,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return narrative_result


class AgentCoordinator:
    """Koordynator agentów LLM, zarządzający przepływem informacji między nimi."""
    
    def __init__(self):
        """Inicjalizacja koordynatora agentów."""
        # Inicjalizacja agentów
        self.satellite_agent = SatelliteAnalysisAgent()
        self.yield_agent = YieldForecastAgent()
        self.market_agent = MarketAnalysisAgent()
        self.narrative_agent = NarrativeGenerator()
        
        # Lista wszystkich agentów
        self.agents = [
            self.satellite_agent,
            self.yield_agent,
            self.market_agent,
            self.narrative_agent
        ]
        
        # Katalog do przechowywania wyników
        self.results_dir = Path("data/agent_results")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info("Koordynator agentów LLM zainicjalizowany")
    
    def analyze_field(self, field_name: str, ndvi_time_series: Dict[str, float], field_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Przeprowadź pełną analizę pola z wykorzystaniem wszystkich agentów.
        
        Args:
            field_name: Nazwa pola
            ndvi_time_series: Szereg czasowy NDVI (data -> wartość)
            field_info: Informacje o polu
            
        Returns:
            Wyniki analizy ze wszystkich agentów
        """
        logger.info(f"Rozpoczynanie pełnej analizy pola {field_name}")
        
        results = {}
        
        # 1. Analiza danych satelitarnych
        satellite_analysis = self.satellite_agent.analyze_ndvi_data(field_name, ndvi_time_series)
        results["satellite_analysis"] = satellite_analysis
        
        # 2. Prognoza plonów na podstawie analizy satelitarnej
        if satellite_analysis.get("status") == "success":
            yield_forecast = self.yield_agent.forecast_yield(field_name, satellite_analysis, field_info)
            results["yield_forecast"] = yield_forecast
        else:
            logger.warning(f"Nie można przeprowadzić prognozy plonów dla pola {field_name} - brak poprawnej analizy satelitarnej")
            results["yield_forecast"] = {"status": "error", "message": "Brak poprawnej analizy satelitarnej"}
        
        # 3. Analiza rynku na podstawie analizy satelitarnej i prognozy plonów
        if satellite_analysis.get("status") == "success":
            crop_type = field_info.get("crop_type", "Wheat")
            market_analysis = self.market_agent.analyze_market(field_name, crop_type, 
                                                            satellite_analysis,
                                                            results.get("yield_forecast", {}))
            results["market_analysis"] = market_analysis
        else:
            logger.warning(f"Nie można przeprowadzić analizy rynku dla pola {field_name} - brak poprawnej analizy satelitarnej")
            results["market_analysis"] = {"status": "error", "message": "Brak poprawnej analizy satelitarnej"}
        
        # 4. Generowanie narracji na podstawie wszystkich analiz
        if satellite_analysis.get("status") == "success":
            narrative = self.narrative_agent.generate_narrative(field_name, 
                                                              satellite_analysis,
                                                              results.get("yield_forecast", {}),
                                                              results.get("market_analysis", {}))
            results["narrative"] = narrative
        else:
            logger.warning(f"Nie można wygenerować narracji dla pola {field_name} - brak poprawnej analizy satelitarnej")
            results["narrative"] = {"status": "error", "message": "Brak poprawnej analizy satelitarnej"}
        
        # Zapisz wyniki
        self.save_results(field_name, results)
        
        logger.info(f"Zakończono pełną analizę pola {field_name}")
        return results
    
    def save_results(self, field_name: str, results: Dict[str, Any]) -> str:
        """
        Zapisz wyniki analizy do pliku.
        
        Args:
            field_name: Nazwa pola
            results: Wyniki analizy ze wszystkich agentów
            
        Returns:
            Ścieżka do zapisanego pliku
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = self.results_dir / f"{field_name}_analysis_{timestamp}.json"
        
        # Zapisz wyniki do pliku
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Zapisano wyniki analizy dla pola {field_name} do pliku {file_path}")
        return str(file_path)
    
    def save_all_reasoning(self) -> List[str]:
        """
        Zapisz łańcuchy reasoningu wszystkich agentów.
        
        Returns:
            Lista ścieżek do zapisanych plików
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = []
        
        for agent in self.agents:
            filename = f"{agent.agent_name}_reasoning_{timestamp}.json"
            file_path = agent.save_reasoning(filename)
            saved_files.append(file_path)
        
        return saved_files
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Wczytaj zapisane wyniki analizy z pliku.
        
        Args:
            filepath: Ścieżka do pliku z zapisanymi wynikami
            
        Returns:
            Wyniki analizy lub pusty słownik w przypadku błędu
        """
        file_path = Path(filepath)
        if not file_path.exists():
            logger.error(f"Nie znaleziono pliku wyników: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Wczytano wyniki analizy z pliku {file_path}")
            return results
        except Exception as e:
            logger.error(f"Błąd wczytywania wyników z pliku {file_path}: {str(e)}")
            return {}


# Singleton instance
coordinator = AgentCoordinator()

def get_agent_coordinator() -> AgentCoordinator:
    """Pobierz singleton instancji koordynatora agentów."""
    return coordinator

def analyze_field(field_name: str, ndvi_time_series: Dict[str, float], field_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Przeprowadź pełną analizę pola z wykorzystaniem systemu agentów LLM.
    
    Args:
        field_name: Nazwa pola
        ndvi_time_series: Szereg czasowy NDVI (data -> wartość)
        field_info: Informacje o polu (opcjonalne)
        
    Returns:
        Wyniki analizy ze wszystkich agentów
    """
    if field_info is None:
        field_info = {"crop_type": "Wheat"}
    
    return coordinator.analyze_field(field_name, ndvi_time_series, field_info)