import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('house_price_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title("House Price Prediction App")
st.write("Enter the details of the house to predict its price:")

# User inputs
income = st.number_input("Average Area Income", min_value=0.0, value=50000.0)
house_age = st.number_input("Average House Age", min_value=0.0, value=20.0)
rooms = st.number_input("Average Number of Rooms", min_value=0.0, value=6.0)
bedrooms = st.number_input("Average Number of Bedrooms", min_value=0.0, value=3.0)
population = st.number_input("Area Population", min_value=0.0, value=30000.0)

# Prepare input as NumPy array (order must match training)
input_array = np.array([[income, house_age, rooms, bedrooms, population]])

# Scale input
input_scaled = scaler.transform(input_array)

# Predict button
if st.button("Predict Price"):
    predicted_price = model.predict(input_scaled)
    st.success(f"Estimated House Price: ${predicted_price[0]:,.2f}")
