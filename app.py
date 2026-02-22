import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# ==============================
# Load Model Files Safely
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "xgboost_delivery_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler (1).pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

# ==============================
# App Title
# ==============================

st.title("🚚 Amazon Delivery Time Prediction System")
st.write("Enter order details to predict delivery time.")

# ==============================
# User Inputs
# ==============================

distance = st.number_input("Distance (KM)", min_value=0.0, value=5.0)
pickup_delay = st.number_input("Pickup Delay (Minutes)", min_value=0.0, value=10.0)
agent_age = st.number_input("Agent Age", min_value=18, max_value=60, value=30)
agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.0)
order_hour = st.slider("Order Hour", 0, 23, 12)

st.subheader("Traffic Conditions")
traffic_low = st.checkbox("Traffic Low")
traffic_medium = st.checkbox("Traffic Medium")
traffic_jam = st.checkbox("Traffic Jam")

st.subheader("Weather Conditions")
weather_sunny = st.checkbox("Weather Sunny")
weather_stormy = st.checkbox("Weather Stormy")
weather_fog = st.checkbox("Weather Fog")

is_weekend = st.checkbox("Weekend Order")

# ==============================
# Prepare Input Dictionary
# ==============================

input_dict = {col: 0 for col in feature_columns}

# Numeric Features
if "Distance_KM" in input_dict:
    input_dict["Distance_KM"] = distance

if "Pickup_Delay_Min" in input_dict:
    input_dict["Pickup_Delay_Min"] = pickup_delay

if "Agent_Age" in input_dict:
    input_dict["Agent_Age"] = agent_age

if "Agent_Rating" in input_dict:
    input_dict["Agent_Rating"] = agent_rating

if "order_hour" in input_dict:
    input_dict["order_hour"] = order_hour

if "Is_Weekend" in input_dict:
    input_dict["Is_Weekend"] = int(is_weekend)

# Traffic Features
if "Traffic_Low" in input_dict:
    input_dict["Traffic_Low"] = int(traffic_low)

if "Traffic_Medium" in input_dict:
    input_dict["Traffic_Medium"] = int(traffic_medium)

if "Traffic_Jam" in input_dict:
    input_dict["Traffic_Jam"] = int(traffic_jam)

# Weather Features
if "Weather_Sunny" in input_dict:
    input_dict["Weather_Sunny"] = int(weather_sunny)

if "Weather_Stormy" in input_dict:
    input_dict["Weather_Stormy"] = int(weather_stormy)

if "Weather_Fog" in input_dict:
    input_dict["Weather_Fog"] = int(weather_fog)

# ==============================
# Convert to DataFrame
# ==============================

input_df = pd.DataFrame([input_dict])

# Scale Input
input_scaled = scaler.transform(input_df)

# ==============================
# Prediction
# ==============================

if st.button("Predict Delivery Time"):
    prediction = model.predict(input_scaled)
    st.success(f"📦 Estimated Delivery Time: {round(prediction[0], 2)} minutes")
