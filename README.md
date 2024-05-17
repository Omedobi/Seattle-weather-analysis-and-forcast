![](https://github.com/Omedobi/Seattle-weather-analysis-and-forcast/tree/481a39e8375ab3d1dc8a6d8b489b39a841a57189/img)

# Seattle Weather Data Analysis and Forecasting
This repository contains Python code for analyzing and forecasting weather data in Seattle. The code utilizes various libraries such as pandas, matplotlib.pyplot, seaborn, and plotly.express to perform data manipulation, visualization, and forecasting tasks.

## Dataset
The analysis is performed on a dataset named "seattle-weather.csv", which likely contains historical weather data for Seattle. The dataset includes information such as date, precipitation, temperature, wind speed, and weather conditions.

## Code Overview
### Data Loading and Exploration: 
The code loads the dataset into a pandas DataFrame and provides an overview of the data by displaying the first few rows, summary statistics, and information about the columns and their data types.

### Data Cleaning: 
Data cleaning operations are performed to ensure data integrity. Missing values are handled using the dropna() function, and columns are renamed for clarity.

### Data Transformation: 
The code performs data transformation by calculating additional columns such as mean temperature and dew point based on existing temperature data and a predefined relative humidity value.

### Data Visualization: 
Visualizations are generated using matplotlib.pyplot and plotly.express libraries. Line plots are created to visualize the humidity, wind speed, and mean temperature trends over a certain period. Scatter plots explore the relationship between temperature and humidity. Heatmaps and line plots analyze temperature changes over the years.

### Weather Forecasting: 
The code utilizes the Facebook Prophet model, a powerful time series forecasting technique, to forecast future weather conditions. The dataset is prepared by renaming columns and converting the date column to the required format. The Prophet model is trained on the data, and forecasts are made for future periods.

## Usage
To use this code, follow these steps:

Download the "seattle-weather.csv" dataset and place it in the same directory as the Python script.

Run the Python script to perform the weather data analysis and forecasting.

Explore the generated visualizations and results to gain insights into the weather patterns in Seattle and forecast future weather conditions.

License
This code is released under the MIT License. Feel free to use and modify it according to your needs.

Acknowledgements
This project was inspired by the desire to analyze and forecast weather data in Seattle. The code utilizes popular Python libraries and techniques for data analysis and forecasting. The dataset used in this project is sourced from an undisclosed weather data provider.
