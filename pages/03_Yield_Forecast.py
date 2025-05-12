import os
import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
import logging
import asyncio
import uuid
import glob
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional

from models.yield_forecast import YieldForecastModel
from utils.visualization import plot_yield_forecast, plot_feature_importance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Yield Forecast - Agro Insight",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if not already set
if "selected_field" not in st.session_state:
    st.session_state.selected_field = None
if "available_fields" not in st.session_state:
    st.session_state.available_fields = []
if "ndvi_time_series" not in st.session_state:
    st.session_state.ndvi_time_series = {}
if "trained_yield_model" not in st.session_state:
    st.session_state.trained_yield_model = None
if "yield_forecast_results" not in st.session_state:
    st.session_state.yield_forecast_results = None

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

# Helper function to extract coordinates from field data
def extract_field_coordinates(field_name):
    """Extract coordinates for a field from its first available file"""
    data_dir = Path("./data/geotiff")
    
    # Find first file for this field
    field_files = list(data_dir.glob(f"{field_name}_*.tif"))
    
    if not field_files:
        return None, None
    
    try:
        import rasterio
        with rasterio.open(field_files[0]) as src:
            bounds = src.bounds
            center_lat = (bounds.bottom + bounds.top) / 2
            center_lon = (bounds.left + bounds.right) / 2
            return center_lat, center_lon
    except:
        return None, None

# Header
st.title("📊 Yield Forecast")
st.markdown("""
Use machine learning to forecast crop yields based on satellite data. Combine NDVI time series, weather data, and historical trends to predict future yields.
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
    help="Choose a field for yield forecasting"
)

if selected_field:
    st.session_state.selected_field = selected_field
    
    # Get NDVI time series from session state or reload it
    ndvi_time_series = {}
    if selected_field in st.session_state.ndvi_time_series:
        ndvi_time_series = st.session_state.ndvi_time_series
    
    # Get field coordinates
    field_lat, field_lon = extract_field_coordinates(selected_field)
    
    # Main content
    st.header(f"Yield Forecast for {selected_field}")
    
    # Create tabs for model training and forecasting
    tab1, tab2, tab3 = st.tabs(["Yield Forecast", "Model Training", "Model Performance"])
    
    with tab1:
        st.subheader("Crop Yield Forecast")
        
        # Input for crop type
        crop_type = st.selectbox(
            "Crop Type",
            options=["Wheat", "Corn", "Soybean", "Barley", "Canola", "Rice", "Other"],
            index=0,
            help="Select the crop type for yield forecasting"
        )
        
        # Choose forecast period
        st.markdown("### Forecast Period")
        forecast_months = st.slider(
            "Forecast Months Ahead",
            min_value=1,
            max_value=3,
            value=1,
            help="Number of months to forecast ahead"
        )
        
        # Generate forecast dates
        today = datetime.date.today()
        forecast_dates = []
        for i in range(1, forecast_months + 1):
            forecast_date = today + datetime.timedelta(days=i * 30)
            forecast_dates.append(forecast_date.strftime("%Y-%m-%d"))
        
        # Get weather data for the location if we have coordinates
        if field_lat is not None and field_lon is not None:
            # Create a forecast model instance
            if st.session_state.trained_yield_model is None:
                st.session_state.trained_yield_model = YieldForecastModel()
            
            yield_model = st.session_state.trained_yield_model
            
            # Show forecast button
            if st.button("Generate Yield Forecast"):
                # Show a spinner while processing
                with st.spinner("Fetching weather data and generating forecast..."):
                    try:
                        # Fetch weather data
                        # In a real app, we'd use asyncio.run for this, but Streamlit doesn't support it well
                        # So we'll run it synchronously for this demo
                        start_date = (today - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
                        end_date = today.strftime("%Y-%m-%d")
                        
                        weather_data = asyncio.new_event_loop().run_until_complete(
                            yield_model.fetch_weather_data(field_lat, field_lon, start_date, end_date)
                        )
                        
                        # Create a DataFrame from NDVI time series
                        if ndvi_time_series:
                            ndvi_df = pd.DataFrame(
                                [(date, value) for date, value in ndvi_time_series.items()],
                                columns=['date', 'ndvi']
                            )
                            
                            # Convert date strings to datetime
                            ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])
                            
                            # Add some historical yield data for demonstration
                            historical_yields = {
                                str(today.year - 2): 6.5,
                                str(today.year - 1): 7.1
                            }
                            
                            # Prepare training data
                            training_data = yield_model.prepare_training_data(ndvi_time_series, weather_data, historical_yields)
                            
                            # Train Prophet model for NDVI forecasting
                            prophet_results = yield_model.train_prophet_model(
                                training_data,
                                target_column='ndvi',
                                date_column='date',
                                forecast_periods=90  # 3 months
                            )
                            
                            # Train LightGBM model for yield prediction
                            lightgbm_results = yield_model.train_lightgbm_model(
                                training_data,
                                target_column='yield'
                            )
                            
                            # Generate yield forecast
                            forecasted_yields = yield_model.forecast_yield(
                                training_data,
                                forecast_dates
                            )
                            
                            # Store forecast results in session state
                            st.session_state.yield_forecast_results = {
                                "crop_type": crop_type,
                                "forecast_dates": forecast_dates,
                                "forecasted_yields": forecasted_yields,
                                "historical_yields": historical_yields,
                                "lightgbm_results": lightgbm_results,
                                "prophet_results": prophet_results
                            }
                            
                            # Show success message
                            st.success("Yield forecast generated successfully!")
                        else:
                            st.error("No NDVI time series data available for this field.")
                    except Exception as e:
                        st.error(f"Error generating yield forecast: {str(e)}")
                        logger.exception("Yield forecast error")
            
            # Display forecast results if available
            if st.session_state.yield_forecast_results:
                results = st.session_state.yield_forecast_results
                
                # Show forecast summary
                st.markdown("### Yield Forecast Summary")
                
                # Create columns for displaying results
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create a table with forecast dates and yields
                    forecast_data = []
                    for date in results["forecast_dates"]:
                        forecast_data.append({
                            "Forecast Date": date,
                            "Yield (t/ha)": results["forecasted_yields"].get(date, "N/A")
                        })
                    
                    forecast_df = pd.DataFrame(forecast_data)
                    st.table(forecast_df)
                    
                    # Add metrics
                    if results["forecast_dates"]:
                        first_date = results["forecast_dates"][0]
                        last_date = results["forecast_dates"][-1]
                        
                        if first_date in results["forecasted_yields"] and last_date in results["forecasted_yields"]:
                            st.metric(
                                "Yield Trend",
                                f"{results['forecasted_yields'][last_date]:.2f} t/ha",
                                f"{results['forecasted_yields'][last_date] - results['forecasted_yields'][first_date]:.2f} t/ha"
                            )
                
                with col2:
                    # Create a plot with historical and forecasted yields
                    # Extract historical data
                    historical_years = list(results["historical_yields"].keys())
                    historical_values = list(results["historical_yields"].values())
                    
                    # Create forecast plot
                    fig = plot_yield_forecast(
                        historical_years,
                        historical_values,
                        results["forecasted_yields"],
                        crop_type=results["crop_type"]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display feature importance if available
                if "lightgbm_results" in results and "feature_importance" in results["lightgbm_results"]:
                    st.markdown("### Key Factors Influencing Yield")
                    
                    # Extract feature importance
                    importance_df = results["lightgbm_results"]["feature_importance"]
                    feature_names = importance_df["Feature"].tolist()
                    importance_values = importance_df["Importance"].tolist()
                    
                    # Create feature importance plot
                    fig = plot_feature_importance(
                        feature_names,
                        importance_values,
                        title="Factors Influencing Yield Prediction"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation of features
                    st.markdown("#### Interpretation of Key Factors")
                    st.markdown("""
                    - **NDVI metrics**: Higher vegetation index values generally correlate with higher yield potential
                    - **Weather factors**: Temperature extremes and precipitation patterns significantly impact crop development
                    - **Temporal factors**: Time of year and growing degree days affect crop growth stages
                    """)
                
                # Download options
                st.markdown("### Download Forecast Data")
                
                # Create a complete DataFrame with all results
                forecast_df = pd.DataFrame(
                    [(date, yield_val) for date, yield_val in results["forecasted_yields"].items()],
                    columns=["date", "forecasted_yield_t_ha"]
                )
                
                # Add historical yields
                historical_df = pd.DataFrame(
                    [(year, yield_val) for year, yield_val in results["historical_yields"].items()],
                    columns=["year", "historical_yield_t_ha"]
                )
                
                st.download_button(
                    label="Download Forecast Data (CSV)",
                    data=forecast_df.to_csv(index=False),
                    file_name=f"{selected_field}_{crop_type}_yield_forecast.csv",
                    mime="text/csv"
                )
    
    with tab2:
        st.subheader("Train Custom Yield Model")
        st.markdown("""
        Train a custom yield prediction model by uploading historical yield data for your field.
        This will improve the accuracy of predictions by incorporating your specific field history.
        """)
        
        # Upload historical yield data
        uploaded_file = st.file_uploader("Upload Historical Yield Data (CSV)", type=["csv"])
        
        if uploaded_file:
            try:
                # Read the CSV file
                hist_data = pd.read_csv(uploaded_file)
                
                # Display the uploaded data
                st.dataframe(hist_data)
                
                # Check if the CSV has the required columns
                required_columns = ["year", "yield_t_ha"]
                missing_columns = [col for col in required_columns if col not in hist_data.columns]
                
                if missing_columns:
                    st.error(f"The CSV file is missing required columns: {', '.join(missing_columns)}")
                    st.markdown("""
                    The CSV file should have at least these columns:
                    - `year`: Year of the yield data (e.g., 2021)
                    - `yield_t_ha`: Yield in metric tons per hectare (e.g., 7.5)
                    """)
                else:
                    # Allow user to train the model
                    if st.button("Train Custom Model"):
                        with st.spinner("Training custom yield model..."):
                            try:
                                # Create a YieldForecastModel instance if not already created
                                if st.session_state.trained_yield_model is None:
                                    st.session_state.trained_yield_model = YieldForecastModel()
                                
                                yield_model = st.session_state.trained_yield_model
                                
                                # Convert historical yield data to dictionary format
                                historical_yields = {
                                    str(row['year']): row['yield_t_ha']
                                    for _, row in hist_data.iterrows()
                                }
                                
                                # Fetch weather data if we have coordinates
                                if field_lat is not None and field_lon is not None:
                                    start_date = datetime.date(min(hist_data['year']), 1, 1).strftime("%Y-%m-%d")
                                    end_date = datetime.date.today().strftime("%Y-%m-%d")
                                    
                                    weather_data = asyncio.new_event_loop().run_until_complete(
                                        yield_model.fetch_weather_data(field_lat, field_lon, start_date, end_date)
                                    )
                                    
                                    # Prepare training data
                                    if ndvi_time_series:
                                        training_data = yield_model.prepare_training_data(
                                            ndvi_time_series, 
                                            weather_data, 
                                            historical_yields
                                        )
                                        
                                        # Train models
                                        prophet_results = yield_model.train_prophet_model(
                                            training_data,
                                            target_column='ndvi',
                                            date_column='date',
                                            forecast_periods=90
                                        )
                                        
                                        lightgbm_results = yield_model.train_lightgbm_model(
                                            training_data,
                                            target_column='yield'
                                        )
                                        
                                        # Save model
                                        model_path = yield_model.save_model(f"custom_{selected_field}")
                                        
                                        # Store results in session state
                                        st.session_state.yield_forecast_results = {
                                            "crop_type": "Custom",
                                            "historical_yields": historical_yields,
                                            "lightgbm_results": lightgbm_results,
                                            "prophet_results": prophet_results,
                                            "model_path": model_path
                                        }
                                        
                                        st.success(f"Custom model trained successfully and saved to {model_path}")
                                    else:
                                        st.error("No NDVI time series data available for this field.")
                                else:
                                    st.error("Could not determine field coordinates for weather data.")
                            except Exception as e:
                                st.error(f"Error training custom model: {str(e)}")
                                logger.exception("Custom model training error")
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
        
        # Upload/download model section
        st.markdown("### Model Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Export Trained Model")
            
            if st.session_state.trained_yield_model is not None and st.session_state.yield_forecast_results:
                if "model_path" in st.session_state.yield_forecast_results:
                    model_path = st.session_state.yield_forecast_results["model_path"]
                    st.success(f"Model saved at: {model_path}")
                    
                    # Create a compressed version of the model folder for download
                    # Note: In a real app, you'd create a proper zip file here
                    st.markdown("""
                    In the full application, there would be a button here to download 
                    the trained model as a zip file for use in other systems.
                    """)
                else:
                    if st.button("Save Current Model"):
                        with st.spinner("Saving model..."):
                            try:
                                model_path = st.session_state.trained_yield_model.save_model(
                                    f"export_{selected_field}"
                                )
                                st.session_state.yield_forecast_results["model_path"] = model_path
                                st.success(f"Model saved at: {model_path}")
                            except Exception as e:
                                st.error(f"Error saving model: {str(e)}")
            else:
                st.info("Train a model first to enable export.")
        
        with col2:
            st.markdown("#### Import Trained Model")
            model_file = st.file_uploader("Upload Model Files (ZIP)", type=["zip"])
            
            if model_file:
                st.warning("Model import functionality would extract and load a model from a zip file.")
                
                if st.button("Import Model"):
                    st.info("In the full application, this would extract the zip file and load the model.")
                    
                    # In a real application, you would:
                    # 1. Extract the zip file to a temporary directory
                    # 2. Create a YieldForecastModel instance
                    # 3. Load the model using the load_model method
                    # 4. Update the session state
    
    with tab3:
        st.subheader("Model Performance")
        
        # Display model metrics if available
        if st.session_state.yield_forecast_results and "lightgbm_results" in st.session_state.yield_forecast_results:
            results = st.session_state.yield_forecast_results
            
            if "metrics" in results["lightgbm_results"]:
                metrics = results["lightgbm_results"]["metrics"]
                
                # Display metrics in cards
                st.markdown("### Model Evaluation Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Mean Absolute Error (MAE)",
                        f"{metrics.get('mae', 0):.3f} t/ha"
                    )
                
                with col2:
                    st.metric(
                        "Root Mean Squared Error (RMSE)",
                        f"{metrics.get('rmse', 0):.3f} t/ha"
                    )
                
                with col3:
                    st.metric(
                        "R² Score",
                        f"{metrics.get('r2', 0):.3f}"
                    )
                
                # Add interpretation
                st.markdown("### Interpretation of Model Metrics")
                
                r2 = metrics.get('r2', 0)
                if r2 > 0.7:
                    st.success("""
                    ✅ **Good Model Performance**: The model explains a significant portion of the variability 
                    in yield outcomes. Predictions should be reliable for decision-making.
                    """)
                elif r2 > 0.5:
                    st.info("""
                    ℹ️ **Moderate Model Performance**: The model captures some of the patterns in the data, 
                    but predictions should be used as general guidance rather than precise forecasts.
                    """)
                else:
                    st.warning("""
                    ⚠️ **Limited Model Performance**: The model has limited predictive power. Consider 
                    gathering more historical data or adding additional features to improve accuracy.
                    """)
                
                # Show feature importance again
                if "feature_importance" in results["lightgbm_results"]:
                    st.markdown("### Feature Importance")
                    
                    importance_df = results["lightgbm_results"]["feature_importance"]
                    feature_names = importance_df["Feature"].tolist()
                    importance_values = importance_df["Importance"].tolist()
                    
                    fig = plot_feature_importance(
                        feature_names,
                        importance_values,
                        title="Feature Contribution to Yield Prediction"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add suggestions for improvement
                st.markdown("### Suggestions for Model Improvement")
                st.markdown("""
                1. **Add more historical data**: Longer time series improve model training
                2. **Include soil data**: Soil type and quality significantly impact yield
                3. **Add management practices**: Fertilization, irrigation, and other field operations
                4. **Consider crop rotation**: Previous crops affect soil nutrients and pest pressure
                5. **Include more satellite indices**: Adding more vegetation indices can capture different aspects of crop health
                """)
            else:
                st.info("Model training did not produce evaluation metrics.")
        else:
            # Show sample image if no model is trained yet
            st.info("Train a model first to see performance metrics.")
            st.image("https://pixabay.com/get/g6bb50ef33a3aad66af13194b25f2700de9dcd4f5bddef1614e89eb634e0daa6297d7518d25e52ef85258015e7b9f6c10a0641be21f0925330270b5ab623b733d_1280.jpg", 
                     caption="Train models to forecast crop yields")

# Display alternate content if no field is selected
else:
    st.info("""
    No fields available for yield forecasting. Please go to the Data Ingest section to process field data first.
    
    You can:
    1. Draw a field boundary on the map
    2. Upload a GeoJSON file with field boundaries
    3. Select a country for country-level analysis
    """)
    
    # Display sample image
    st.image("https://pixabay.com/get/g0d06bf8b0e242c631e7c3b967cb7056beb183aa24ad0249767ee98d037c172d64a5c113f4eba58ab9b73dc2daf92b98e01b0027fd4acb848ace1fc64352c0639_1280.jpg", 
             caption="Yield forecasting for agricultural fields")

# Bottom-page links
st.markdown("---")
st.markdown("""
👈 Go to **Field Analysis** to analyze vegetation indices

👉 Continue to **Market Signals** to analyze market implications
""")
