import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import requests
import folium
from streamlit_folium import st_folium

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Smart Delivery Prediction", layout="wide")

st.title("🚚 Smart Delivery Prediction System (Real Routing + ML)")
st.write("System automatically calculates best route and predicts delivery time.")

# ==========================================
# LOAD MODEL FILES
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "xgboost_delivery_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler (1).pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

# ==========================================
# GET API KEY FROM STREAMLIT SECRETS
# ==========================================
try:
    API_KEY = st.secrets["ORS_API_KEY"].strip()
except Exception:
    st.error("❌ API Key not found in Streamlit Secrets.")
    st.stop()

# ==========================================
# USER INPUT SECTION
# ==========================================
col1, col2 = st.columns(2)

with col1:
    store_lat = st.number_input("Store Latitude", value=19.0760)
    store_lon = st.number_input("Store Longitude", value=72.8777)

    drop_lat = st.number_input("Drop Latitude", value=19.2183)
    drop_lon = st.number_input("Drop Longitude", value=72.9781)

with col2:
    pickup_delay = st.number_input("Pickup Delay (Minutes)", value=10.0)
    agent_age = st.number_input("Agent Age", min_value=18, max_value=60, value=30)
    agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.0)
    is_weekend = st.checkbox("Weekend Order")

# ==========================================
# FUNCTION: GET REAL ROUTE FROM API
# ==========================================
def get_route_data(start_lat, start_lon, end_lat, end_lon):

    url = "https://api.openrouteservice.org/v2/directions/driving-car"

    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json"
    }

    body = {
        "coordinates": [
            [start_lon, start_lat],
            [end_lon, end_lat]
        ]
    }

    try:
        response = requests.post(url, json=body, headers=headers, timeout=20)
    except requests.exceptions.RequestException as e:
        raise Exception(f"Connection Error: {e}")

    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    data = response.json()

    if "routes" not in data:
        raise Exception("Invalid response from routing service.")

    distance_km = data["routes"][0]["summary"]["distance"] / 1000
    duration_min = data["routes"][0]["summary"]["duration"] / 60

    return distance_km, duration_min, data

# ==========================================
# MAIN BUTTON
# ==========================================
if st.button("🚀 Calculate Best Route & Predict"):

    try:
        # Get real route info
        distance_km, route_time, route_data = get_route_data(
            store_lat, store_lon, drop_lat, drop_lon
        )

        st.success(f"📍 Real Route Distance: {round(distance_km, 2)} KM")
        st.success(f"⏱ Estimated Travel Time: {round(route_time, 2)} minutes")

        # ==========================================
        # MAP DISPLAY
        # ==========================================
        m = folium.Map(location=[store_lat, store_lon], zoom_start=13)

        folium.Marker(
            [store_lat, store_lon],
            tooltip="Store",
            icon=folium.Icon(color="green")
        ).add_to(m)

        folium.Marker(
            [drop_lat, drop_lon],
            tooltip="Delivery Location",
            icon=folium.Icon(color="red")
        ).add_to(m)

        coords = route_data["routes"][0]["geometry"]["coordinates"]
        route_points = [(c[1], c[0]) for c in coords]

        folium.PolyLine(route_points).add_to(m)

        st_folium(m, width=900, height=500)

        # ==========================================
        # PREPARE ML INPUT
        # ==========================================
        input_df = pd.DataFrame(columns=feature_columns)
        row = dict.fromkeys(feature_columns, 0)

        # Only fill if feature exists in training
        if "Distance_KM" in row:
            row["Distance_KM"] = distance_km
        if "Pickup_Delay_Min" in row:
            row["Pickup_Delay_Min"] = pickup_delay
        if "Agent_Age" in row:
            row["Agent_Age"] = agent_age
        if "Agent_Rating" in row:
            row["Agent_Rating"] = agent_rating
        if "Is_Weekend" in row:
            row["Is_Weekend"] = int(is_weekend)

        input_df.loc[0] = row
        input_df = input_df[feature_columns]

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)

        st.success(f"📦 Final Predicted Delivery Time: {round(prediction[0], 2)} minutes")

    except Exception as e:
        st.error(f"❌ Route calculation failed: {e}")
