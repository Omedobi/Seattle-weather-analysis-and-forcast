import streamlit as st
import logging
import os
import pandas as pd
import cmdstanpy
from prophet import Prophet
from prophet.plot import plot_plotly
from xgboost import XGBClassifier
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load pre-trained models
@st.cache_resource
def load_models():
    model_paths = {
        'Prophet_model': 'saved_model/Prophet_model.joblib',  
        'xgb_model': 'saved_model/xgb_model.pkl'
    }
    models = {}
    for key, path in model_paths.items():
        if os.path.exists(path):
            try:
                if key == 'Prophet_model':
                    # Use joblib for Prophet model
                    models[key] = joblib.load(path)
                else:
                    # Use pickle for XGBoost model
                    with open(path, 'rb') as file:
                        models[key] = pickle.load(file)
                logging.info(f"Model loaded successfully from {path}")
            except Exception as e:
                models[key] = None
                logging.error(f"Failed to load model from {path}: {e}")
        else:
            models[key] = None
            logging.error(f"Model file not found at {path}")
    return models['Prophet_model'], models['xgb_model']

Prophet_model, xgb_model = load_models()


# Function to load the data
@st.cache_data
def load_data():
    data = pd.read_parquet('data/seattle-weather.pq')
    forecast_data = pd.read_parquet('data/forecast-data.pq')
    if data['date'].dtype == 'object':
        data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    
    label_encoder = LabelEncoder()
    if 'weather_condition' in data.columns:
        data['weather_condition_encoded'] = label_encoder.fit_transform(data['weather_condition'])
    return data, forecast_data, label_encoder


# Function to preprocess data
def preprocess_data(data, month, year):
    return data[(data['month'] == month) & (data['year'] == year)]

#making the forecast model adaptive
def train_prophet_model(forecast_data):
    # Assuming model training needs explicit initiation and not re-training automatically
    model = Prophet()
    model.fit(forecast_data)
    return model

def display_forecast(Prophet_model, forecast_data):
    if 'ds' in forecast_data.columns and 'y' in forecast_data.columns:
        # Initialize or retrieve the model
        if 'Prophet_model' not in st.session_state or st.session_state.retrain_model:
            st.session_state.Prophet_model = train_prophet_model(forecast_data)
            st.success('Model trained and forecast displayed successfully')

        # Use the model to predict and plot
        Prophet_model = st.session_state.Prophet_model
        future = forecast_data[['ds']]
        forecast = Prophet_model.predict(future)
        
        fig = plot_plotly(Prophet_model, forecast)
        st.plotly_chart(fig)     

    else:
        st.error("Forecast data is not formatted correctly. It must include 'ds' and 'y' columns.")
    st.session_state.retrain_model = False  # Reset retraining flag after operation

# Streamlit controls to trigger model retraining
def model_controls():
    st.session_state.retrain_model = st.checkbox("Retrain model", value=False)


# Function to display XGBoost predictions
def display_predictions(xgb_model, data, label_encoder):
    required_features = ['precipitation','wind_speed','dew_point','mean_temp','humidity','year','month']
    if xgb_model:
        if set(required_features).issubset(data.columns):
            features = data[required_features]
            try:
                predicted_labels = xgb_model.predict(features)
                data['predicted_weather_condition'] = label_encoder.inverse_transform(predicted_labels)
                st.write('The Predicted weather condition is:')
                col1, col2, col3 = st.columns([10,6,2])
                with col1:
                    st.write(data[['date', 'predicted_weather_condition']])
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.error("Data does not contain the required features for predictions.")
    else:
        st.error("Prediction model is not loaded. Please check the model files.")


def set_background(image_path):
    """
    A function set a full background image in Streamlit.
    :param image_path: The path or URL to the image file.
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_path}");
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


#user interface
def main():
    # set the image URL or local path
    set_background('img/Seattle-Weather-forecasting.jpg')  
    
    st.title('Seattle Weather Forecast App')

    # Load data
    try:
        data, forecast_data, label_encoder = load_data()
    except ValueError as e:
        st.error(f"Failed to load data: {str(e)}")
        return  # Properly placed return to exit function if data loading fails

    # Check if data is loaded and has necessary columns
    if data.empty or 'year' not in data.columns:
        st.error("No data available or 'year' data is missing from the dataset.")
        return  # Exit function if data is empty or year column is missing

    # User inputs for year and month
    year = st.selectbox('Select Year', options=pd.unique(data['year']))
    month = st.slider('Select Month', min_value=min(data['month']), max_value=max(data['month']), value=min(data['month']))

    # Button to show predictions and forecasts
    if st.button('Show Forecast and Predictions'):
        filtered_data = preprocess_data(data, month, year)
        if not filtered_data.empty:
            display_forecast(Prophet_model, forecast_data)  # Ensure correct reference
            display_predictions(xgb_model, filtered_data, label_encoder)
        else:
            st.error("No data available for the selected month and year.")

    # Button to refresh data
    if st.button('Refresh Data'):
        st.rerun()


if __name__ == "__main__":
    main()
