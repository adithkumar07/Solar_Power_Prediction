import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and scaler
model = joblib.load('/Users/adithkumar/Desktop/untitled folder 4/untitled folder 2/gradient_boosting_model.pkl')
scaler = joblib.load('/Users/adithkumar/Desktop/untitled folder 4/untitled folder 2/scaler.pkl')

# Load the dataset
df = pd.read_csv('/Users/adithkumar/Desktop/untitled folder 4/untitled folder 2/solarpowergeneration.csv')
average_wind_speed_mean=df['average-wind-speed-(period)'].mean()
df['average-wind-speed-(period)'].fillna(average_wind_speed_mean,inplace=True)
df1=df

df1.rename(columns={'distance-to-solar-noon': 'distance_to_solar_noon','wind-direction':'wind_direction','wind-speed':'wind_speed','sky-cover':'sky_cover','average-wind-speed-(period)':'average_wind_speed','average-pressure-(period)':'average_pressure'}, inplace=True)
df=df1


# Function to predict power generation
def predict_power(distance_to_solar_noon, temperature, wind_direction, wind_speed,
                  sky_cover, visibility, humidity, average_wind_speed, average_pressure):
    input_data = pd.DataFrame({
        'distance_to_solar_noon': [distance_to_solar_noon],
        'temperature': [temperature],
        'wind_direction': [wind_direction],
        'wind_speed': [wind_speed],
        'sky_cover': [sky_cover],
        'visibility': [visibility],
        'humidity': [humidity],
        'average_wind_speed': [average_wind_speed],
        'average_pressure': [average_pressure]
    })
    input_data = input_data.reindex(columns=df.columns[:-1])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Streamlit app
st.title('Solar Power Generation Prediction and Data Visualization')

# Navigation Bar (Sidebar)
st.sidebar.title('Predict Power Generation')
st.sidebar.write("Adjust the sliders below to input the features for prediction:")

# Sidebar sliders for user inputs
distance_to_solar_noon = st.sidebar.slider('Distance to Solar Noon (degrees)', min_value=0.0, max_value=180.0, value=90.0)
temperature = st.sidebar.slider('Temperature (°C)', min_value=-10.0, max_value=70.0, value=25.0)
wind_direction = st.sidebar.slider('Wind Direction (°)', min_value=0.0, max_value=360.0, value=180.0)
wind_speed = st.sidebar.slider('Wind Speed (m/s)', min_value=0.0, max_value=20.0, value=10.0)
sky_cover = st.sidebar.slider('Sky Cover (0-1)', min_value=0.0, max_value=1.0, value=0.5)
visibility = st.sidebar.slider('Visibility (km)', min_value=0.0, max_value=20.0, value=10.0)
humidity = st.sidebar.slider('Humidity (%)', min_value=0.0, max_value=100.0, value=50.0)
average_wind_speed = st.sidebar.slider('Average Wind Speed (m/s)', min_value=0.0, max_value=20.0, value=10.0)
average_pressure = st.sidebar.slider('Average Pressure (hPa)', min_value=20.0, max_value=110.0, value=1000.0)

# Buttons side by side
col1, col2 = st.sidebar.columns(2)

# Predict button in the sidebar
with col1:
    if st.button('Predict'):
        prediction = predict_power(distance_to_solar_noon, temperature, wind_direction, wind_speed,
                                   sky_cover, visibility, humidity, average_wind_speed, average_pressure)
        st.sidebar.write(f'Predicted Power Generated: {prediction:.2f} kW')

# Reset button in the sidebar
with col2:
    if st.button('Reset'):
        st.experimental_rerun()

# Section 1: Dataset Overview
st.header('1. Dataset Overview')
st.write("Here is an overview of the first few rows of the dataset used for prediction:")
st.dataframe(df.head(), width=800, height=200)

# Section 2: Summary Statistics
st.header('2. Summary Statistics')
st.write("Below are the summary statistics of the dataset:")
st.write(df.describe())

# Section 3: Data Visualizations
st.header('3. Data Visualizations')

# Scatter Plot: Temperature vs Power Generated
st.subheader('3.1 Scatter Plot: Temperature vs Power Generated')
st.write("This scatter plot shows the relationship between Temperature and Power Generated, with Wind Speed as a color gradient.")
fig_scatter = px.scatter(df, x='temperature', y='power-generated', color='wind_speed',
                         title='Temperature vs Power Generated',
                         template='plotly_dark')  # Set dark theme
fig_scatter.update_layout(title_font=dict(size=20, color='white'), 
                          xaxis_title_font=dict(size=15, color='white'),
                          yaxis_title_font=dict(size=15, color='white'),
                          legend_title_font=dict(size=15, color='white'))
st.plotly_chart(fig_scatter)

# Box Plot: Feature Distribution
st.subheader('3.2 Box Plot: Feature Distribution')
st.write("This box plot displays the distribution of the numerical features in the dataset.")
fig_box = plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")  # Start with a white grid to customize it for dark theme
boxplot = sns.boxplot(data=df.drop('power-generated', axis=1))
plt.xticks(rotation=45, color='white')
plt.yticks(color='white')
plt.gca().set_facecolor('#111111')  # Dark background
fig_box.patch.set_facecolor('#111111')  # Dark background for the entire figure
boxplot.set_title('Feature Distribution', color='white')
boxplot.set_xlabel('Features', color='white')
boxplot.set_ylabel('Values', color='white')
st.pyplot(fig_box)

# Correlation Heatmap
st.subheader('3.3 Correlation Heatmap')
st.write("This heatmap visualizes the correlation between different features of the dataset.")
fig_heatmap = plt.figure(figsize=(10, 6))
sns.set(style="white")  # Use white style for better customization
heatmap = sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', annot_kws={'color': 'white'})
plt.gca().set_facecolor('#111111')  # Dark background
fig_heatmap.patch.set_facecolor('#111111')  # Dark background for the entire figure
plt.title('Correlation Heatmap', color='white')
plt.xticks(color='white')
plt.yticks(color='white')
st.pyplot(fig_heatmap)

# Histogram: Power Generated
st.subheader('3.4 Histogram: Power Generated')
st.write("This histogram shows the distribution of the Power Generated across the dataset.")
fig_hist, ax_hist = plt.subplots()
ax_hist.hist(df['power-generated'], bins=20, color='skyblue', edgecolor='black')
ax_hist.set_title('Histogram of Power Generated', color='white')
ax_hist.set_xlabel('Power Generated (kW)', color='white')
ax_hist.set_ylabel('Frequency', color='white')
ax_hist.set_facecolor('#111111')  # Dark background
fig_hist.patch.set_facecolor('#111111')  # Dark background for the entire figure
plt.xticks(color='white')
plt.yticks(color='white')
st.pyplot(fig_hist)
