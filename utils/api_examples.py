"""
API Examples - Przykłady wykorzystania API danych satelitarnych
"""
import os
import logging
import streamlit as st
import requests
import json
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def display_sentinel_hub_examples():
    """Wyświetla przykłady użycia Sentinel Hub API"""
    st.markdown("""
    ## Przykłady użycia Sentinel Hub API
    
    Sentinel Hub używa uwierzytelniania OAuth2 do dostępu do API. Poniżej znajdziesz przykłady wywołań API
    używając różnych metod.
    """)
    
    # Zakładki dla różnych przykładów
    example_tab1, example_tab2, example_tab3 = st.tabs([
        "Uwierzytelnianie (CURL)", 
        "Zapytanie o dane (CURL)",
        "Python SDK"
    ])
    
    with example_tab1:
        st.markdown("""
        ### Uzyskanie tokenu OAuth2 z Sentinel Hub API
        
        Do uwierzytelnienia z Sentinel Hub potrzebujesz client_id i client_secret. Token musisz odnowić co kilka godzin.
        """)
        
        st.code("""
# Za pomocą CURL:
curl -X POST \\
  https://services.sentinel-hub.com/oauth/token \\
  -H 'content-type: application/x-www-form-urlencoded' \\
  -d 'grant_type=client_credentials&client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET'

# Odpowiedź (przykład):
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "scope": "..."
}
        """, language="bash")

    with example_tab2:
        st.markdown("""
        ### Zapytanie o sceny Sentinel-2 dla wybranego obszaru
        
        Ten przykład pokazuje jak pobrać listę dostępnych scen Sentinel-2 dla danego obszaru i okresu czasu.
        """)
        
        st.code("""
# Najpierw uzyskaj token OAuth2 jak pokazano wcześniej
ACCESS_TOKEN="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."

# Za pomocą CURL - Zapytanie o dostępne sceny:
curl -X POST \\
  'https://services.sentinel-hub.com/api/v1/catalog/search' \\
  -H 'Authorization: Bearer $ACCESS_TOKEN' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "bbox": [13.822174, 45.85, 14.55, 46.65],
    "datetime": "2020-05-01T00:00:00Z/2020-06-01T23:59:59Z",
    "collections": ["sentinel-2-l2a"],
    "limit": 5
}'
        """, language="bash")
        
        st.markdown("""
        ### Pobieranie obrazu NDVI z Sentinel Hub Processing API
        
        Ten przykład pokazuje jak napisać skrypt do obliczenia i pobrania obrazu NDVI.
        """)
        
        st.code("""
# Za pomocą CURL - Pobieranie przetworzonego obrazu NDVI:
curl -X POST \\
  'https://services.sentinel-hub.com/api/v1/process' \\
  -H 'Authorization: Bearer $ACCESS_TOKEN' \\
  -H 'Content-Type: application/json' \\
  -d '{
  "input": {
    "bounds": {
      "bbox": [13.822174, 45.85, 14.55, 46.65],
      "properties": {
        "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
      }
    },
    "data": [
      {
        "dataFilter": {
          "timeRange": {
            "from": "2020-05-01T00:00:00Z",
            "to": "2020-05-30T23:59:59Z"
          },
          "maxCloudCoverage": 20
        },
        "type": "sentinel-2-l2a"
      }
    ]
  },
  "output": {
    "width": 512,
    "height": 512,
    "responses": [
      {
        "identifier": "default",
        "format": {
          "type": "image/png"
        }
      }
    ]
  },
  "evalscript": "//VERSION=3\\nfunction setup() {\\n  return {\\n    input: [\\\"B04\\\", \\\"B08\\\", \\\"dataMask\\\"],\\n    output: { bands: 3 }\\n  };\\n}\\n\\nfunction evaluatePixel(sample) {\\n  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);\\n  \\n  if (sample.dataMask == 0) {\\n    return [0, 0, 0];\\n  }\\n  \\n  return [0.1 + 0.8 * ndvi, 0.1 + 0.8 * ndvi, 0.1 + 0.8 * ndvi];\\n}"
}'
        """, language="bash")
    
    with example_tab3:
        st.markdown("""
        ### Używanie Sentinel Hub Python SDK
        
        Sentinel Hub dostarcza bibliotekę Python, która ułatwia integrację z ich API.
        """)
        
        st.code("""
# Instalacja:
# pip install sentinelhub

from sentinelhub import SHConfig, BBox, CRS, DataCollection
from sentinelhub import SentinelHubRequest, MimeType, bbox_to_dimensions
import matplotlib.pyplot as plt
import numpy as np

# Konfiguracja uwierzytelniania
config = SHConfig()
config.sh_client_id = "TWÓJ_CLIENT_ID"
config.sh_client_secret = "TWÓJ_CLIENT_SECRET"

# Zdefiniuj obszar zainteresowania (AOI)
bbox = BBox(bbox=[13.822174, 45.85, 14.55, 46.65], crs=CRS.WGS84)
resolution = 10  # 10m na piksel
dims = bbox_to_dimensions(bbox, resolution=resolution)

# Zdefiniuj skrypt ewaluacji do obliczenia NDVI
evalscript = '''
//VERSION=3
function setup() {
  return {
    input: ["B04", "B08", "dataMask"],
    output: { bands: 1 }
  };
}

function evaluatePixel(sample) {
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  
  if (sample.dataMask == 0) {
    return [0];
  }
  
  return [ndvi];
}
'''

# Utwórz zapytanie
request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=('2020-05-01', '2020-05-30'),
            mosaicking_order='leastCC'
        )
    ],
    responses=[
        {"identifier": "default", "format": {"type": "image/tiff"}}
    ],
    bbox=bbox,
    size=dims,
    config=config
)

# Wykonaj zapytanie
response = request.get_data()

# Wyświetl wynik
ndvi_img = response[0]
plt.figure(figsize=(10, 10))
plt.imshow(ndvi_img, cmap='RdYlGn', vmin=-1, vmax=1)
plt.colorbar(label='NDVI')
plt.title('NDVI z Sentinel-2')
plt.axis('off')
plt.show()
        """, language="python")


def display_planet_api_examples():
    """Wyświetla przykłady użycia Planet API"""
    st.markdown("""
    ## Przykłady użycia Planet API
    
    Planet API używa klucza API do uwierzytelniania. Poniżej znajdziesz przykłady wywołań API
    używając różnych metod.
    """)
    
    # Zakładki dla różnych przykładów
    example_tab1, example_tab2, example_tab3 = st.tabs([
        "Uwierzytelnianie (CURL)", 
        "Zapytanie o dane (CURL)",
        "Python SDK"
    ])
    
    with example_tab1:
        st.markdown("""
        ### Uwierzytelnianie do Planet API
        
        Planet API używa uwierzytelniania Basic Auth, gdzie kluczem API jest nazwa użytkownika, a hasło pozostaje puste.
        """)
        
        st.code("""
# Za pomocą CURL (Basic Auth):
curl -i -u 'PLANET_API_KEY:' https://api.planet.com/data/v1/quick-search

# Za pomocą CURL (Authorization Header):
curl -i -H "Authorization: api-key PLANET_API_KEY" https://api.planet.com/data/v1/quick-search

# Za pomocą CURL (parametr URL):
curl -i "https://api.planet.com/basemaps/v1/mosaics?api_key=PLANET_API_KEY"
        """, language="bash")
    
    with example_tab2:
        st.markdown("""
        ### Wyszukiwanie zdjęć z Planet API
        
        Ten przykład pokazuje jak wyszukać obrazy dla określonego obszaru i okresu czasu.
        """)
        
        st.code("""
# Za pomocą CURL - Wyszukiwanie obrazów:
curl -X POST \\
  https://api.planet.com/data/v1/quick-search \\
  -H 'Content-Type: application/json' \\
  -u 'PLANET_API_KEY:' \\
  -d '{
  "item_types": ["PSScene"],
  "filter": {
    "type": "AndFilter",
    "config": [
      {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": {
          "type": "Polygon",
          "coordinates": [
            [
              [-122.54, 37.81],
              [-122.38, 37.81],
              [-122.38, 37.69],
              [-122.54, 37.69],
              [-122.54, 37.81]
            ]
          ]
        }
      },
      {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {
          "gte": "2020-01-01T00:00:00Z",
          "lte": "2020-05-01T00:00:00Z"
        }
      },
      {
        "type": "RangeFilter",
        "field_name": "cloud_cover",
        "config": {
          "lte": 0.1
        }
      }
    ]
  }
}'
        """, language="bash")
        
        st.markdown("""
        ### Pobieranie miniatur obrazów
        
        Pobieranie miniatur obrazów znalezionych w wyszukiwaniu.
        """)
        
        st.code("""
# Za pomocą CURL - Pobieranie miniatury obrazu:
ITEM_ID="TWÓJ_ITEM_ID"
curl -L -u 'PLANET_API_KEY:' "https://api.planet.com/data/v1/item-types/PSScene/items/$ITEM_ID/thumb" > thumb.png
        """, language="bash")
    
    with example_tab3:
        st.markdown("""
        ### Używanie Planet Python SDK
        
        Planet dostarcza bibliotekę Python, która ułatwia integrację z ich API.
        """)
        
        st.code("""
# Instalacja:
# pip install planet

import os
import json
from planet import Api
from planet.data.filter import and_filter, range_filter, geom_filter

# Inicjalizacja klienta API
api = Api(api_key=os.getenv('PLANET_API_KEY'))

# Zdefiniuj obszar zainteresowania (AOI)
geometry = {
  "type": "Polygon",
  "coordinates": [
    [
      [-122.54, 37.81],
      [-122.38, 37.81],
      [-122.38, 37.69],
      [-122.54, 37.69],
      [-122.54, 37.81]
    ]
  ]
}

# Utwórz filtr
query = and_filter(
    geom_filter(geometry),
    range_filter('cloud_cover', lt=0.1),
    range_filter('acquired', gt='2020-01-01T00:00:00Z', lt='2020-05-01T00:00:00Z')
)

# Wykonaj wyszukiwanie
item_types = ['PSScene']
request = api.filters.build_search_request(query, item_types)
results = api.filters.search(request)

# Wyświetl wyniki
for item in results.items_iter(limit=5):
    print(json.dumps(item, indent=2))
    
    # Pobierz miniaturę obrazu
    assets = api.get_assets(item).get()
    thumbnail = assets.get('thumbnail')
    
    if thumbnail:
        # Pobierz URL miniatury
        thumbnail_url = thumbnail.get('location')
        print(f"Miniatura URL: {thumbnail_url}")
        
        # Pobierz miniatury za pomocą requests
        # import requests
        # response = requests.get(thumbnail_url, auth=(os.getenv('PLANET_API_KEY'), ''))
        # with open(f"{item['id']}_thumb.png", 'wb') as f:
        #     f.write(response.content)
        """, language="python")


def show_api_examples():
    """Główna funkcja wyświetlająca przykłady API"""
    st.markdown("# Przykłady integracji z API danych satelitarnych")
    
    tab1, tab2 = st.tabs(["Sentinel Hub API", "Planet API"])
    
    with tab1:
        display_sentinel_hub_examples()
    
    with tab2:
        display_planet_api_examples()