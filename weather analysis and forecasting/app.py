import streamlit as st
import pandas as pd
import logging
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import xgboost as xgb
import plotly.express as px
import matplotlib.pyplot as plt
from joblib import load
from sklearn.preprocessing import LabelEncoder

#configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#load model
@st.cache_resource
def load_models():
    model_paths = [
        'saved_model/Prophet_model.joblib',
        'saved_model/xgb_model.joblib'
    ]
    models = []
    for path in model_paths:
        model_key = path.split('/')[-1].split('.')[0]
        try:
            models[model_key] = load(path)
            logging.info(f"Model {model_key} loaded successfully from {path}")
        except Exception as e:
            logging.error(f"Failed to load model {model_key} from {path}: {e}")
            st.error(f'Error loading model {model_key} from {path}: {e}')
            models[model_key] = None
    return models

@st.cache_data
#load data
def load_data():
    data = pd.read_parquet('data/seattle-weather.pq')
    forecast_data = pd.read_parquet('data/forecast-data.pq')
    return data , forecast_data

    
 #data preprocessing
def handle_data(data:pd.DataFrame) -> tuple:
    label_encoder = LabelEncoder()
    data_cols = data.select_dtypes(include=['object']).columns
    for col in data_cols:
        data[col] = label_encoder.fit_transform(data[col])
    return data, label_encoder
 
            
# prediction
def prophet_prediction(forecast_data, periods):
    
    """Fit the Prophet model and make predictions for the specified number of periods.

    Args:
        forecast_data (pd.DataFrame): DataFrame containing the data for forecasting. 
                                      Must contain 'ds' (datestamp) and 'y' (value) columns.
        periods (int): Number of future periods to forecast.

    Returns:
        Tuple[Prophet, pd.DataFrame]: The fitted Prophet model and the forecast DataFrame.
    """
    try:
        model = Prophet()
        model.fit(forecast_data)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return model, forecast
    except Exception as e:
        logging.error(f'Error in Prophet prediction: {e}')
        st.error(f'error during forecasting: {e}')
        return None, None

def xgb_prediction(data, label_encoder):
    try:
        X = data[['precipitation', 'wind_speed', 'dew_point', 'mean_temp', 'humidity', 'year', 'month']]
        models = load_models()
        xgb_model = models[1]
        if xgb_model is not None:
            xgb_pred = xgb_model.predict(X)
            preds = label_encoder.inverse_transform(xgb_pred.astype(int))
            logging.info(f"XGBoost predict completed")
            return preds
        else:
            st.error('XGBoost model is not loaded')
            return None
    except Exception as e:
        logging.error(f"Error in predicting weather condition using XGBoost: {e}")
        st.error(f'Error predicting weather condition: {e}')
        return None

#The user interface

st.title('Seattle Weather Forecast App')
if st.button('Load Data'):
    with st.spinner('Loading data...'):
        data, forecast_data = load_data()
        data, label_encoder = handle_data(data)
        st.success('Data loaded successfully!')

if data is not None:
    st.write('Weather data:', data)

st.header('Prophet Forecast')
periods = st.slider('Select number of days to forecast', 1, 365, 30)
model, forecast = prophet_prediction(forecast_data, periods)
if forecast is not None:
    st.write('Forecast data:', forecast)
    st.plotly_chart(plot_plotly(model,forecast))
    


st.header('XGBoost Weather Condition Predition')
predictions = xgb_prediction(data, label_encoder)
if predictions is not None:
    st.write('Predicted weather conditions:', predictions)

if st.button('Refreshing data...'):
    data, forecast = load_data()
    data, label_encoder = handle_data(data)
    st.write('Weather data:', data)