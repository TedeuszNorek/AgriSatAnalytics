"""
Funkcje do generowania przykładowych danych do celów testowania i demonstracji.
"""
import os
import json
import datetime
import random
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

def generate_mock_ndvi_time_series(
    start_date: datetime.datetime = None,
    days: int = 60,
    base_value: float = 0.65,
    noise_level: float = 0.1,
    trend: float = 0.001,
    include_anomalies: bool = True,
    anomaly_count: int = 3
) -> Dict[str, float]:
    """
    Generuje przykładowy szereg czasowy NDVI.
    
    Args:
        start_date: Data początkowa (domyślnie 60 dni temu)
        days: Liczba dni do wygenerowania
        base_value: Podstawowa wartość NDVI
        noise_level: Poziom szumu
        trend: Trend wzrostowy/spadkowy (dodatni/ujemny)
        include_anomalies: Czy dołączyć anomalie
        anomaly_count: Liczba anomalii do wygenerowania
        
    Returns:
        Słownik mapujący daty (jako stringi) na wartości NDVI
    """
    if start_date is None:
        start_date = datetime.datetime.now() - datetime.timedelta(days=days)
    
    # Generuj daty
    dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
    date_strings = [d.strftime("%Y-%m-%d") for d in dates]
    
    # Generuj wartości NDVI z trendem i szumem
    values = []
    for i in range(days):
        # Dodajemy trend
        trend_value = base_value + trend * i
        
        # Dodajemy sezonowość (sinusoidalny wzorzec)
        seasonal = 0.05 * np.sin(2 * np.pi * i / 30)  # 30-dniowy cykl
        
        # Dodajemy losowy szum
        noise = random.uniform(-noise_level, noise_level)
        
        # Łączymy wszystkie komponenty
        value = max(0, min(1, trend_value + seasonal + noise))
        values.append(value)
    
    # Tworzymy słownik
    time_series = dict(zip(date_strings, values))
    
    # Dodaj anomalie
    if include_anomalies and anomaly_count > 0:
        anomaly_indices = random.sample(range(days), anomaly_count)
        for idx in anomaly_indices:
            # Wyraźna negatywna anomalia
            date_key = date_strings[idx]
            time_series[date_key] = max(0, time_series[date_key] - random.uniform(0.2, 0.4))
    
    return time_series

def generate_mock_satellite_image(
    width: int = 100,
    height: int = 100,
    image_type: str = "ndvi",
    base_value: float = 0.65,
    add_features: bool = True
) -> np.ndarray:
    """
    Generuje przykładowy obraz satelitarny.
    
    Args:
        width: Szerokość obrazu
        height: Wysokość obrazu
        image_type: Typ obrazu (ndvi, evi, itp.)
        base_value: Podstawowa wartość
        add_features: Czy dodać cechy krajobrazu
        
    Returns:
        Tablica NumPy reprezentująca obraz
    """
    # Utwórz podstawową tablicę z losowym szumem
    image = np.random.normal(base_value, 0.05, (height, width))
    
    # Określ zakres wartości w zależności od typu obrazu
    if image_type.lower() == "ndvi" or image_type.lower() == "evi":
        # NDVI i EVI są zwykle w zakresie [-1, 1], ale głównie [0, 1]
        image = np.clip(image, 0, 1)
    elif image_type.lower() == "moisture":
        # Wilgotność gleby w zakresie [0, 1]
        image = np.clip(image, 0, 1)
    elif image_type.lower() == "temperature":
        # Temperatura w stopniach Celsjusza (np. 10-40)
        image = image * 10 + 20
    
    # Dodaj cechy krajobrazu
    if add_features:
        # Dodaj "pola" jako prostokąty o różnych wartościach
        for _ in range(3):
            x1 = random.randint(0, width - 20)
            y1 = random.randint(0, height - 20)
            w = random.randint(10, min(width - x1, 40))
            h = random.randint(10, min(height - y1, 40))
            value = random.uniform(0.4, 0.9)
            
            image[y1:y1+h, x1:x1+w] = value + np.random.normal(0, 0.02, (h, w))
        
        # Dodaj "drogę" jako linię
        x1 = random.randint(0, width - 1)
        y1 = random.randint(0, height - 1)
        x2 = random.randint(0, width - 1)
        y2 = random.randint(0, height - 1)
        
        # Narysuj linię używając algorytmu Bresenhama
        steep = abs(y2 - y1) > abs(x2 - x1)
        if steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        
        dx = x2 - x1
        dy = abs(y2 - y1)
        error = dx // 2
        y = y1
        y_step = 1 if y1 < y2 else -1
        
        for x in range(x1, x2 + 1):
            if steep:
                # Dodaj szerokość drogi
                for offset in range(-1, 2):
                    if 0 <= y + offset < height and 0 <= x < width:
                        image[y + offset, x] = 0.2  # Drogi mają niższą wartość NDVI
            else:
                # Dodaj szerokość drogi
                for offset in range(-1, 2):
                    if 0 <= y < height and 0 <= x + offset < width:
                        image[y, x + offset] = 0.2  # Drogi mają niższą wartość NDVI
            
            error -= dy
            if error < 0:
                y += y_step
                error += dx
    
    return image

def generate_mock_weather_data(
    start_date: datetime.datetime = None,
    days: int = 60
) -> Dict[str, Dict[str, float]]:
    """
    Generuje przykładowe dane pogodowe.
    
    Args:
        start_date: Data początkowa (domyślnie 60 dni temu)
        days: Liczba dni do wygenerowania
        
    Returns:
        Słownik mapujący daty na dane pogodowe
    """
    if start_date is None:
        start_date = datetime.datetime.now() - datetime.timedelta(days=days)
    
    # Generuj daty
    dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
    date_strings = [d.strftime("%Y-%m-%d") for d in dates]
    
    # Początkowe wartości
    base_temp = 20  # stopnie Celsjusza
    base_precip = 2  # mm
    base_humidity = 60  # procent
    
    # Generuj dane pogodowe
    weather_data = {}
    for i, date in enumerate(date_strings):
        # Dodajemy sezonowość
        seasonal_temp = 5 * np.sin(2 * np.pi * i / 30)  # 30-dniowy cykl
        
        # Dodajemy losowy szum
        temp_noise = random.uniform(-3, 3)
        precip_noise = max(0, random.gauss(0, 1.5))
        humidity_noise = random.uniform(-10, 10)
        
        # Łączymy wszystkie komponenty
        temp = base_temp + seasonal_temp + temp_noise
        
        # Opady są zwykle skorelowane z wilgotnością
        humidity = min(100, max(30, base_humidity + humidity_noise))
        precip_prob = (humidity - 30) / 70  # 30% wilgotność = 0 prawdopodobieństwo, 100% wilgotność = 1 prawdopodobieństwo
        
        # Dodatkowa losowość dla opadów (większość dni bez opadów)
        if random.random() < 0.3 * precip_prob:
            precip = base_precip + precip_noise
        else:
            precip = 0
        
        weather_data[date] = {
            "temperature": round(temp, 1),
            "precipitation": round(precip, 1),
            "humidity": round(humidity, 1),
            "wind_speed": round(random.uniform(0, 20), 1)
        }
    
    return weather_data

def generate_mock_yield_forecast(
    start_date: datetime.datetime = None,
    forecast_days: int = 90,
    crop_type: str = "wheat",
    current_yield: float = 5.0,
    trend: float = 0.01
) -> Dict[str, float]:
    """
    Generuje przykładową prognozę plonów.
    
    Args:
        start_date: Data początkowa (domyślnie dzisiaj)
        forecast_days: Liczba dni do prognozowania
        crop_type: Typ uprawy
        current_yield: Aktualna wydajność w t/ha
        trend: Trend wzrostowy/spadkowy
        
    Returns:
        Słownik mapujący daty na prognozowane plony
    """
    if start_date is None:
        start_date = datetime.datetime.now()
    
    # Generuj daty
    dates = [start_date + datetime.timedelta(days=i) for i in range(forecast_days)]
    date_strings = [d.strftime("%Y-%m-%d") for d in dates]
    
    # Generuj prognozy
    forecasts = {}
    for i, date in enumerate(date_strings):
        # Uwzględniamy trend i niepewność rosnącą z czasem
        uncertainty = 0.01 * i  # Niepewność rośnie z czasem
        forecast_value = current_yield + trend * i + random.uniform(-uncertainty, uncertainty)
        forecasts[date] = round(max(0, forecast_value), 2)
    
    return forecasts

def generate_mock_market_signals(
    num_signals: int = 5,
    days_back: int = 90,
    commodities: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Generuje przykładowe sygnały rynkowe.
    
    Args:
        num_signals: Liczba sygnałów do wygenerowania
        days_back: Maksymalna liczba dni wstecz dla sygnałów
        commodities: Lista towarów (domyślnie: pszenica, kukurydza, soja)
        
    Returns:
        Lista słowników reprezentujących sygnały rynkowe
    """
    if commodities is None:
        commodities = ["ZW=F", "ZC=F", "ZS=F"]  # Pszenica, kukurydza, soja
    
    now = datetime.datetime.now()
    signals = []
    
    for _ in range(num_signals):
        days_ago = random.randint(0, days_back)
        signal_date = now - datetime.timedelta(days=days_ago)
        commodity = random.choice(commodities)
        action = random.choice(["LONG", "SHORT"])
        confidence = round(random.uniform(0.5, 0.95), 2)
        
        # Wygeneruj powód na podstawie akcji
        if action == "LONG":
            reason = random.choice([
                "Pozytywne anomalie NDVI wskazują na lepsze niż oczekiwano zbiory",
                "Negatywne warunki pogodowe w regionach produkcyjnych mogą ograniczyć podaż",
                "Historyczne korelacje wskazują na wyższe ceny w nadchodzącym okresie"
            ])
        else:  # SHORT
            reason = random.choice([
                "Negatywne anomalie NDVI wskazują na gorsze niż oczekiwano zbiory",
                "Korzystne warunki pogodowe w głównych regionach produkcyjnych zwiększą podaż",
                "Historyczne korelacje wskazują na niższe ceny w nadchodzącym okresie"
            ])
        
        signal = {
            "date": signal_date.strftime("%Y-%m-%d"),
            "commodity": commodity,
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "key_levels": {
                "support": round(random.uniform(500, 700), 2),
                "resistance": round(random.uniform(700, 900), 2)
            }
        }
        
        signals.append(signal)
    
    # Sortuj według daty (od najnowszych)
    signals.sort(key=lambda x: x["date"], reverse=True)
    
    return signals

def generate_mock_price_data(
    start_date: datetime.datetime = None,
    days: int = 120,
    commodities: List[str] = None
) -> pd.DataFrame:
    """
    Generuje przykładowe dane cenowe dla towarów.
    
    Args:
        start_date: Data początkowa (domyślnie 120 dni temu)
        days: Liczba dni do wygenerowania
        commodities: Lista towarów (domyślnie: pszenica, kukurydza, soja)
        
    Returns:
        DataFrame z cenami towarów
    """
    if commodities is None:
        commodities = ["ZW=F", "ZC=F", "ZS=F"]  # Pszenica, kukurydza, soja
    
    if start_date is None:
        start_date = datetime.datetime.now() - datetime.timedelta(days=days)
    
    # Generuj daty
    dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
    
    # Inicjuj początkowe ceny
    initial_prices = {
        "ZW=F": 650,  # Pszenica (centy za buszel)
        "ZC=F": 450,  # Kukurydza (centy za buszel)
        "ZS=F": 1200  # Soja (centy za buszel)
    }
    
    # Parametry dla symulacji cen
    volatility = {
        "ZW=F": 0.015,
        "ZC=F": 0.012,
        "ZS=F": 0.018
    }
    
    # Generuj ceny używając geometrycznego ruchu Browna
    price_data = {commodity: [] for commodity in commodities}
    
    for commodity in commodities:
        price = initial_prices.get(commodity, 500)
        vol = volatility.get(commodity, 0.01)
        
        prices = [price]
        for i in range(1, days):
            # Dodajemy trend sezonowy
            seasonal = 0.05 * np.sin(2 * np.pi * i / 90)  # 90-dniowy cykl
            
            # Symulacja ceny
            daily_return = np.random.normal(0.0002 + seasonal, vol)
            price = price * (1 + daily_return)
            prices.append(price)
        
        price_data[commodity] = prices
    
    # Tworzymy DataFrame
    df = pd.DataFrame(price_data, index=dates)
    
    return df

def generate_mock_field_data() -> Dict[str, Any]:
    """
    Generuje pełny zestaw przykładowych danych dla jednego pola.
    
    Returns:
        Słownik z danymi dla pola
    """
    now = datetime.datetime.now()
    start_date = now - datetime.timedelta(days=60)
    
    # Generuj komponenty danych
    ndvi_time_series = generate_mock_ndvi_time_series(start_date=start_date)
    weather_data = generate_mock_weather_data(start_date=start_date)
    yield_forecast = generate_mock_yield_forecast(start_date=now, crop_type="wheat")
    market_signals = generate_mock_market_signals(num_signals=3)
    
    # Generuj losowe anomalie na podstawie szeregu NDVI
    ndvi_values = list(ndvi_time_series.values())
    ndvi_mean = sum(ndvi_values) / len(ndvi_values)
    ndvi_std = np.std(ndvi_values)
    
    anomalies = {}
    for date, value in ndvi_time_series.items():
        if (value - ndvi_mean) / ndvi_std < -1.5:
            # Wyraźna negatywna anomalia
            anomalies[date] = abs((value - ndvi_mean) / ndvi_std)
    
    # Przygotuj pełny zestaw danych
    field_data = {
        "ndvi_time_series": ndvi_time_series,
        "weather_data": weather_data,
        "yield_forecast": yield_forecast,
        "market_signals": market_signals,
        "anomalies": anomalies,
        "satellite_images": {}
    }
    
    # Dodaj 3 przykładowe obrazy satelitarne z różnych dat
    for i in range(3):
        date = (now - datetime.timedelta(days=20*i)).strftime("%Y-%m-%d")
        field_data["satellite_images"][date] = {
            "ndvi": generate_mock_satellite_image(image_type="ndvi"),
            "rgb": None,  # Prawdziwy obraz RGB wymagałby dodatkowego kodu
            "metadata": {
                "acquisition_date": date,
                "cloud_cover": random.uniform(0, 20),
                "satellite": "Sentinel-2"
            }
        }
    
    return field_data

def get_mock_field_boundary(
    center_lat: float, 
    center_lon: float, 
    area_hectares: float = 10.0
) -> Dict[str, Any]:
    """
    Generuje przykładowe granice pola jako GeoJSON.
    
    Args:
        center_lat: Szerokość geograficzna środka
        center_lon: Długość geograficzna środka
        area_hectares: Powierzchnia pola w hektarach
        
    Returns:
        Słownik GeoJSON reprezentujący granice pola
    """
    # Aproksymacja przeliczenia hektarów na stopnie (bardzo uproszczone)
    # 1 hektar to około 0.0001 stopnia kwadratowego w średnich szerokościach
    side_length = np.sqrt(area_hectares * 0.0001)
    
    # Tworzymy prostokąt
    half_side = side_length / 2
    polygon_coords = [
        [center_lon - half_side, center_lat - half_side],
        [center_lon + half_side, center_lat - half_side],
        [center_lon + half_side, center_lat + half_side],
        [center_lon - half_side, center_lat + half_side],
        [center_lon - half_side, center_lat - half_side]  # Zamykamy wielokąt
    ]
    
    # Dodajmy trochę losowości do kształtu
    for i in range(4):  # Nie modyfikujemy ostatniego punktu, który jest kopią pierwszego
        polygon_coords[i][0] += random.uniform(-half_side/5, half_side/5)
        polygon_coords[i][1] += random.uniform(-half_side/5, half_side/5)
    
    # Upewniamy się, że ostatni punkt jest kopią pierwszego
    polygon_coords[4] = polygon_coords[0].copy()
    
    # Tworzymy GeoJSON
    geojson = {
        "type": "Feature",
        "properties": {
            "name": "Mock Field Boundary",
            "area_hectares": area_hectares
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [polygon_coords]
        }
    }
    
    return geojson

def save_mock_data(field_name: str, data: Dict[str, Any], directory: str = "data/mock") -> str:
    """
    Zapisuje przykładowe dane do pliku.
    
    Args:
        field_name: Nazwa pola
        data: Dane do zapisania
        directory: Katalog do zapisania danych
        
    Returns:
        Ścieżka do zapisanego pliku
    """
    # Upewnij się, że katalog istnieje
    os.makedirs(directory, exist_ok=True)
    
    # Przygotuj ścieżkę pliku
    safe_name = field_name.replace(" ", "_").lower()
    file_path = os.path.join(directory, f"{safe_name}.json")
    
    # Konwertuj typy danych niesprawdzalne przez JSON
    for date, image_data in data.get("satellite_images", {}).items():
        for img_type, img in image_data.items():
            if isinstance(img, np.ndarray):
                # Konwertuj NumPy array do listy list
                data["satellite_images"][date][img_type] = img.tolist() if img is not None else None
    
    # Zapisz dane
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return file_path

def load_mock_data(field_name: str, directory: str = "data/mock") -> Optional[Dict[str, Any]]:
    """
    Wczytuje przykładowe dane z pliku.
    
    Args:
        field_name: Nazwa pola
        directory: Katalog z danymi
        
    Returns:
        Słownik z danymi lub None, jeśli plik nie istnieje
    """
    # Przygotuj ścieżkę pliku
    safe_name = field_name.replace(" ", "_").lower()
    file_path = os.path.join(directory, f"{safe_name}.json")
    
    # Sprawdź czy plik istnieje
    if not os.path.exists(file_path):
        return None
    
    # Wczytaj dane
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Konwertuj listy list z powrotem do NumPy arrays
    for date, image_data in data.get("satellite_images", {}).items():
        for img_type, img in image_data.items():
            if isinstance(img, list):
                # Konwertuj listę list do NumPy array
                data["satellite_images"][date][img_type] = np.array(img) if img is not None else None
    
    return data

def generate_and_save_mock_fields(
    num_fields: int = 3,
    base_location: Tuple[float, float] = (50.0, 19.0),
    directory: str = "data/mock"
) -> List[Dict[str, Any]]:
    """
    Generuje i zapisuje przykładowe dane dla kilku pól.
    
    Args:
        num_fields: Liczba pól do wygenerowania
        base_location: Podstawowa lokalizacja (szerokość, długość)
        directory: Katalog do zapisania danych
        
    Returns:
        Lista słowników z metadanymi pól
    """
    field_metadata = []
    
    for i in range(num_fields):
        # Generuj losową lokalizację wokół podstawowej lokalizacji
        lat = base_location[0] + random.uniform(-0.1, 0.1)
        lon = base_location[1] + random.uniform(-0.1, 0.1)
        area = random.uniform(5, 20)
        
        # Generuj nazwę pola
        field_name = f"Pole testowe {i+1}"
        
        # Generuj granice pola
        geojson = get_mock_field_boundary(lat, lon, area)
        
        # Generuj dane pola
        field_data = generate_mock_field_data()
        
        # Dodaj granice do danych
        field_data["geojson"] = geojson
        
        # Zapisz dane
        file_path = save_mock_data(field_name, field_data, directory)
        
        # Zapisz metadane
        metadata = {
            "name": field_name,
            "center_lat": lat,
            "center_lon": lon,
            "area_hectares": area,
            "crop_type": random.choice(["wheat", "corn", "soybean", "barley", "canola"]),
            "file_path": file_path
        }
        
        field_metadata.append(metadata)
    
    return field_metadata