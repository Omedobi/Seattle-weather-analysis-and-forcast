![seattle-weather](img/Seattle-Weather-forecasting.jpg)

# Seattle Weather Forecasting Project

This project analyzes and predicts weather conditions in Seattle using a dataset of historical weather data. It employs several data science techniques and models, including the Facebook Prophet model for time series forecasting and various machine learning classifiers from the scikit-learn and XGBoost libraries for predictive modeling.

## Features

- **Data Cleaning and Preprocessing**: The dataset is cleaned, manipulated to make more features. subsequently, preprocessed to fit the models' requirements.
  
- **Exploratory Data Analysis (EDA)**: Visualizations are generated using matplotlib.pyplot and plotly.express libraries. Line plots are created to visualize the humidity, wind speed, and mean temperature trends over a certain period. Scatter plots explore the relationship between temperature and humidity. Heatmaps and line plots analyze temperature changes over the years.
  
- **Forecasting**: Utilizing the Facebook Prophet model to forecast future weather conditions. The dataset is prepared by renaming columns and converting the date column to the required format. The Prophet model is trained on the data, and forecasts are made for future periods.
  
- **Classification**: Applying machine learning models to predict weather conditions based on historical data.
- **Feature Importance Analysis**: Analyzing the importance of various features in the prediction models.

## Installation

To run this project, follow these steps:

1. **Clone the Repository**

   ```
   git clone https://github.com/your-username/seattle-weather-forecasting.git
   cd seattle-weather-forecasting
   ```

2. **Set Up a Virtual Environment** (Optional but recommended)

   For Windows:
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

   For macOS and Linux:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Libraries**

   Make sure you have a `requirements.txt` file that lists all the necessary packages. Then run:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script to start the application:

```
streamlit run app.py
```

This will start a Streamlit application that you can interact with in your web browser at `http://localhost:8501`.

## Dataset

The dataset used in this project contains daily weather conditions in Seattle. It includes temperature, humidity, wind speed, and weather conditions, among others. The data is cleaned and preprocessed to handle missing values and transform features into a format suitable for modeling.

## Models

- **Facebook Prophet**: Used for forecasting future weather metrics.
- **XGBoost**, **Random Forest**, **Decision Trees**, **Extra Trees**, **K-Nearest Neighbors**: Used for predicting categorical weather conditions.

## Results

Visualizations and metrics such as accuracy, precision, recall, and F1-score are provided to evaluate the models' performances. The best-performing model's details are saved for later use or deployment.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your enhancements.

### Additional Recommendations

- **Contact Information**: [![Email](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:ikennaanyawuike@gmail.com) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anyawuike-ikenna)

