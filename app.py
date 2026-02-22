import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import folium
from streamlit_folium import st_folium

# ==========================================
# Load Model
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "xgboost_delivery_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler (1).pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

st.set_page_config(page_title="Amazon Delivery Prediction", layout="wide")

st.title("🚚 Amazon Delivery Time Prediction System")

# ==========================================
# Inputs
# ==========================================

col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("Distance (KM)", min_value=0.0, value=5.0)
    pickup_delay = st.number_input("Pickup Delay (Minutes)", min_value=0.0, value=10.0)
    agent_age = st.number_input("Agent Age", min_value=18, max_value=60, value=30)
    agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.0)
    order_hour = st.slider("Order Hour", 0, 23, 12)

with col2:
    traffic_low = st.checkbox("Traffic Low")
    traffic_medium = st.checkbox("Traffic Medium")
    traffic_jam = st.checkbox("Traffic Jam")

    weather_sunny = st.checkbox("Weather Sunny")
    weather_stormy = st.checkbox("Weather Stormy")
    weather_fog = st.checkbox("Weather Fog")

    is_weekend = st.checkbox("Weekend Order")

st.subheader("📍 Delivery Route")

store_lat = st.number_input("Store Latitude", value=19.0760)
store_lon = st.number_input("Store Longitude", value=72.8777)

drop_lat = st.number_input("Drop Latitude", value=19.2183)
drop_lon = st.number_input("Drop Longitude", value=72.9781)

# ==========================================
# Create Real Map (Folium)
# ==========================================

m = folium.Map(location=[store_lat, store_lon], zoom_start=12)

# Store marker
folium.Marker(
    [store_lat, store_lon],
    tooltip="Store Location",
    icon=folium.Icon(color="green", icon="shopping-cart")
).add_to(m)

# Drop marker
folium.Marker(
    [drop_lat, drop_lon],
    tooltip="Delivery Location",
    icon=folium.Icon(color="red", icon="home")
).add_to(m)

# Draw route line
folium.PolyLine(
    [[store_lat, store_lon], [drop_lat, drop_lon]],
    tooltip="Delivery Route"
).add_to(m)

st_folium(m, width=800, height=500)

# ==========================================
# Prediction
# ==========================================

if st.button("Predict Delivery Time"):

    input_dict = {col: 0 for col in feature_columns}

    input_dict["Distance_KM"] = distance
    input_dict["Pickup_Delay_Min"] = pickup_delay
    input_dict["Agent_Age"] = agent_age
    input_dict["Agent_Rating"] = agent_rating
    input_dict["order_hour"] = order_hour
    input_dict["Is_Weekend"] = int(is_weekend)

    input_dict["Traffic_Low"] = int(traffic_low)
    input_dict["Traffic_Medium"] = int(traffic_medium)
    input_dict["Traffic_Jam"] = int(traffic_jam)

    input_dict["Weather_Sunny"] = int(weather_sunny)
    input_dict["Weather_Stormy"] = int(weather_stormy)
    input_dict["Weather_Fog"] = int(weather_fog)

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    st.success(f"📦 Estimated Delivery Time: {round(prediction[0], 2)} minutes")
