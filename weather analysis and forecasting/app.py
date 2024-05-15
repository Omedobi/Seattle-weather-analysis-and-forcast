import streamlit as st
import pandas as pd
import logging
from prophet import Prophet
from prophet.plot import plot_plotly
import xgboost as xgb
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_models():
    model_paths = {
        'Prophet_model': 'saved_model/Prophet_model.joblib',
        'xgb_model': 'saved_model/xgb_model.joblib'
    }
    models = {}
    for key, path in model_paths.items():
        try:
            models[key] = load(path)
            logging.info(f"Model {key} loaded successfully.")
        except Exception as e:
            models[key] = None
            logging.error(f"Failed to load model {key}: {e}")
            st.error(f"Failed to load {key}. Please check the logs.")
    return models

def load_data():
    try:
        data = pd.read_parquet('data/seattle-weather.pq')
        forecast_data = pd.read_parquet('data/forecast-data.pq')
        return data, forecast_data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        st.error("Failed to load data. Please check the logs.")
        return None, None

def handle_data(data):
    label_encoder = LabelEncoder()
    if data is not None:
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = label_encoder.fit_transform(data[col])
    return data, label_encoder

def main():
    st.title('Seattle Weather Forecast App')
    
    if 'data' not in st.session_state or 'models' not in st.session_state:
        st.session_state.data, st.session_state.forecast_data = load_data()
        st.session_state.models = load_models()
        st.session_state.data, st.session_state.label_encoder = handle_data(st.session_state.data)
    
    if st.button('Refresh Data'):
        st.session_state.data, st.session_state.forecast_data = load_data()
        st.session_state.data, st.session_state.label_encoder = handle_data(st.session_state.data)

    if st.session_state.data is not None:
        st.download_button("Download Data", data=st.session_state.data.to_csv(), file_name='weather_data.csv', mime='text/csv')

        periods = st.slider('Select number of days to forecast', 1, 365, 30)
        forecast(st.session_state.forecast_data, periods, st.session_state.models)
        predict_weather_conditions(st.session_state.data, st.session_state.label_encoder, st.session_state.models)

def forecast(forecast_data, periods, models):
    if forecast_data is not None and 'Prophet_model' in models:
        model = models['Prophet_model']
        if model:
            try:
                future = model.make_future_dataframe(periods=periods)
                forecast = model.predict(future)
                st.write('Forecast data:', forecast)
                st.plotly_chart(plot_plotly(model, forecast))
            except Exception as e:
                logging.error(f"Prophet prediction error: {e}")
                st.error("Error during forecasting with Prophet.")

def predict_weather_conditions(data, label_encoder, models):
    if data is not None and 'xgb_model' in models:
        model = models['xgb_model']
        if model:
            try:
                X = data[['precipitation', 'wind_speed', 'dew_point', 'mean_temp', 'humidity', 'year', 'month']]
                preds = model.predict(X)
                st.write('Predicted weather conditions:', label_encoder.inverse_transform(preds))
            except Exception as e:
                logging.error(f"XGBoost prediction error: {e}")
                st.error("Error predicting weather conditions.")

if __name__ == "__main__":
    main()
