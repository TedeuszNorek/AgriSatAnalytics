"""
Data Ingest - Import and manage field data
"""
import os
import json
import logging
import datetime
import traceback
from typing import Dict, List, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import folium
import geopandas as gpd
from streamlit_folium import folium_static
from shapely.geometry import Polygon, shape

from database import get_db, Field
from utils.data_access import (
    get_sentinel_hub_config, 
    parse_geojson, 
    get_bbox_from_polygon,
    get_country_boundary
)
from config import (
    SENTINEL_HUB_CLIENT_ID,
    SENTINEL_HUB_CLIENT_SECRET,
    CROP_TYPES,
    APP_NAME,
    APP_DESCRIPTION
)

# Configure page
st.set_page_config(
    page_title=f"{APP_NAME} - Data Ingest",
    page_icon="🛰️",
    layout="wide"
)

# Configure logging
logger = logging.getLogger(__name__)

# Header
st.markdown("# Data Ingest")
st.markdown("Import and manage agricultural field data")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## Data Options")
    data_option = st.radio(
        "Choose Data Source",
        ["Upload GeoJSON", "Draw on Map", "Import from Database"]
    )
    
    st.markdown("## Field Management")
    manage_option = st.radio(
        "Manage Fields",
        ["Add New Field", "Edit Existing Field", "Delete Field"]
    )

# Main content
if data_option == "Upload GeoJSON" and manage_option == "Add New Field":
    st.markdown("## Upload Field Boundary")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload GeoJSON file", type=["geojson", "json"])
    
    if uploaded_file is not None:
        try:
            # Read file content
            geojson_content = uploaded_file.read().decode()
            geojson_data = json.loads(geojson_content)
            
            # Parse GeoJSON and extract polygon
            polygon, crs = parse_geojson(geojson_data)
            bbox = get_bbox_from_polygon(polygon)
            
            # Calculate center and area
            centroid = polygon.centroid
            center_lat, center_lon = centroid.y, centroid.x
            
            # Convert area to hectares (assuming coordinates are in degrees)
            # This is a rough approximation, more accurate calculations would require reprojection
            area_m2 = polygon.area * 111000 * 111000  # Approximate conversion from degrees to meters
            area_hectares = area_m2 / 10000  # Convert m² to hectares
            
            # Display map with the field boundary
            st.markdown("### Field Boundary")
            
            # Create map centered on the field
            m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
            
            # Add GeoJSON to map
            folium.GeoJson(
                geojson_data,
                name="Field Boundary",
                style_function=lambda x: {
                    'fillColor': '#28a745',
                    'color': '#28a745',
                    'weight': 2,
                    'fillOpacity': 0.4
                }
            ).add_to(m)
            
            # Add marker at center
            folium.Marker(
                [center_lat, center_lon],
                popup="Field Center",
                icon=folium.Icon(color="green", icon="leaf")
            ).add_to(m)
            
            # Display map
            folium_static(m, width=800, height=500)
            
            # Field details form
            st.markdown("### Field Details")
            
            with st.form("field_details_form"):
                field_name = st.text_input("Field Name", value=f"Field {datetime.datetime.now().strftime('%Y%m%d')}")
                crop_type = st.selectbox("Crop Type", options=[""] + CROP_TYPES)
                
                # Display calculated area and coordinates
                st.markdown(f"**Calculated Area:** {area_hectares:.2f} hectares")
                st.markdown(f"**Center Coordinates:** Lat: {center_lat:.6f}, Lon: {center_lon:.6f}")
                
                # Additional notes
                notes = st.text_area("Notes", placeholder="Enter any additional information about this field")
                
                # Submit button
                submit_button = st.form_submit_button("Save Field")
                
                if submit_button:
                    try:
                        # Create database session
                        db = next(get_db())
                        
                        # Check if field name already exists
                        existing_field = db.query(Field).filter(Field.name == field_name).first()
                        
                        if existing_field:
                            st.error(f"Field name '{field_name}' already exists. Please choose a different name.")
                        else:
                            # Create new field
                            new_field = Field(
                                name=field_name,
                                geojson=json.dumps(geojson_data),
                                center_lat=float(center_lat),
                                center_lon=float(center_lon),
                                area_hectares=float(area_hectares),
                                crop_type=crop_type if crop_type else None
                            )
                            
                            # Add and commit to database
                            db.add(new_field)
                            db.commit()
                            
                            st.success(f"Field '{field_name}' has been saved successfully!")
                            st.balloons()
                            
                    except Exception as e:
                        st.error(f"Error saving field: {str(e)}")
                        logger.error(f"Error saving field: {traceback.format_exc()}")
                        
        except Exception as e:
            st.error(f"Error processing GeoJSON file: {str(e)}")
            logger.error(f"Error processing GeoJSON file: {traceback.format_exc()}")

elif data_option == "Draw on Map" and manage_option == "Add New Field":
    st.markdown("## Draw Field Boundary")
    st.info("This feature is under development. Please use the GeoJSON upload option.")
    
    # Placeholder for future implementation
    st.markdown("""
    In a future version, you will be able to:
    1. Draw field boundaries directly on a map
    2. Edit the shape interactively
    3. Save the drawn field to the database
    """)

elif data_option == "Import from Database" or manage_option in ["Edit Existing Field", "Delete Field"]:
    st.markdown("## Manage Existing Fields")
    
    # Get fields from database
    fields = []
    try:
        db = next(get_db())
        fields = db.query(Field).all()
    except Exception as e:
        st.error(f"Error fetching fields from database: {str(e)}")
    
    if fields:
        # Create a dataframe for display
        field_data = []
        for field in fields:
            field_data.append({
                "ID": field.id,
                "Name": field.name,
                "Crop Type": field.crop_type or "Not specified",
                "Area (ha)": round(field.area_hectares, 2),
                "Created": field.created_at.strftime('%Y-%m-%d')
            })
        
        st.dataframe(pd.DataFrame(field_data), use_container_width=True)
        
        # Field selection
        field_names = [field.name for field in fields]
        selected_field_name = st.selectbox("Select Field", options=field_names)
        
        # Get the selected field
        selected_field = next((field for field in fields if field.name == selected_field_name), None)
        
        if selected_field:
            # Display field details
            st.markdown("### Field Details")
            
            # Create two columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display map with field boundary
                try:
                    # Parse GeoJSON
                    geojson_data = json.loads(selected_field.geojson) if isinstance(selected_field.geojson, str) else selected_field.geojson
                    
                    # Create map centered on the field
                    m = folium.Map(location=[selected_field.center_lat, selected_field.center_lon], zoom_start=13)
                    
                    # Add GeoJSON to map
                    folium.GeoJson(
                        geojson_data,
                        name="Field Boundary",
                        style_function=lambda x: {
                            'fillColor': '#28a745',
                            'color': '#28a745',
                            'weight': 2,
                            'fillOpacity': 0.4
                        }
                    ).add_to(m)
                    
                    # Add marker at center
                    folium.Marker(
                        [selected_field.center_lat, selected_field.center_lon],
                        popup=selected_field.name,
                        icon=folium.Icon(color="green", icon="leaf")
                    ).add_to(m)
                    
                    # Display map
                    folium_static(m, width=600, height=400)
                    
                except Exception as e:
                    st.error(f"Error displaying field boundary: {str(e)}")
            
            with col2:
                # Display field information
                st.markdown(f"**Name:** {selected_field.name}")
                st.markdown(f"**Crop Type:** {selected_field.crop_type or 'Not specified'}")
                st.markdown(f"**Area:** {selected_field.area_hectares:.2f} hectares")
                st.markdown(f"**Center:** Lat: {selected_field.center_lat:.6f}, Lon: {selected_field.center_lon:.6f}")
                st.markdown(f"**Created:** {selected_field.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Actions based on manage_option
                if manage_option == "Edit Existing Field":
                    with st.form("edit_field_form"):
                        st.markdown("### Edit Field")
                        
                        new_name = st.text_input("Field Name", value=selected_field.name)
                        new_crop_type = st.selectbox(
                            "Crop Type", 
                            options=[""] + CROP_TYPES, 
                            index=0 if not selected_field.crop_type else CROP_TYPES.index(selected_field.crop_type) + 1
                        )
                        
                        update_button = st.form_submit_button("Update Field")
                        
                        if update_button:
                            try:
                                # Update field in database
                                db = next(get_db())
                                field_to_update = db.query(Field).filter(Field.id == selected_field.id).first()
                                
                                if field_to_update:
                                    # Check if the new name already exists for another field
                                    if new_name != selected_field.name:
                                        existing_field = db.query(Field).filter(Field.name == new_name).first()
                                        if existing_field:
                                            st.error(f"Field name '{new_name}' already exists. Please choose a different name.")
                                            st.stop()
                                    
                                    # Update field properties
                                    field_to_update.name = new_name
                                    field_to_update.crop_type = new_crop_type if new_crop_type else None
                                    
                                    # Commit changes
                                    db.commit()
                                    st.success(f"Field '{new_name}' has been updated successfully!")
                                    st.rerun()
                                else:
                                    st.error("Field not found in database.")
                                    
                            except Exception as e:
                                st.error(f"Error updating field: {str(e)}")
                                logger.error(f"Error updating field: {traceback.format_exc()}")
                
                elif manage_option == "Delete Field":
                    st.markdown("### Delete Field")
                    st.warning(f"Are you sure you want to delete the field '{selected_field.name}'? This action cannot be undone.")
                    
                    if st.button("Delete Field", type="primary"):
                        try:
                            # Delete field from database
                            db = next(get_db())
                            field_to_delete = db.query(Field).filter(Field.id == selected_field.id).first()
                            
                            if field_to_delete:
                                db.delete(field_to_delete)
                                db.commit()
                                st.success(f"Field '{selected_field.name}' has been deleted.")
                                st.rerun()
                            else:
                                st.error("Field not found in database.")
                                
                        except Exception as e:
                            st.error(f"Error deleting field: {str(e)}")
                            logger.error(f"Error deleting field: {traceback.format_exc()}")
    else:
        st.info("No fields found in the database. Use the 'Add New Field' option to create fields.")

# Help information
with st.expander("Help & Information"):
    st.markdown("""
    ## How to add a field
    
    ### Option 1: Upload GeoJSON
    1. Prepare a GeoJSON file with your field boundary
    2. Select "Upload GeoJSON" from the sidebar
    3. Upload your file and fill in the field details
    4. Click "Save Field" to store it in the database
    
    ### Option 2: Draw on Map (Coming Soon)
    In a future update, you'll be able to draw field boundaries directly on the map.
    
    ## GeoJSON Format
    GeoJSON files should contain a polygon or multipolygon geometry representing your field boundary.
    
    Example:
    ```json
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [longitude1, latitude1],
            [longitude2, latitude2],
            ...
            [longitude1, latitude1]
          ]
        ]
      }
    }
    ```
    
    ## Managing Fields
    - **Edit Field**: Change a field's name or crop type
    - **Delete Field**: Permanently remove a field from the database
    """)