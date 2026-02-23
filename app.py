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

st.title("🚚 Smart Delivery Prediction System")
st.write("Real road routing + ML-based delivery time prediction")

# ==========================================
# LOAD MODEL FILES
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "xgboost_delivery_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler (1).pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

# ==========================================
# USER INPUT
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
# OSRM ROUTING FUNCTION
# ==========================================
def get_route_data(start_lat, start_lon, end_lat, end_lon):
    url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"OSRM Error {response.status_code}")

    data = response.json()

    if data["code"] != "Ok":
        raise Exception("No route found")

    distance_km = data["routes"][0]["distance"] / 1000
    duration_min = data["routes"][0]["duration"] / 60

    return distance_km, duration_min, data

# ==========================================
# SESSION STATE INIT
# ==========================================
if "route_data" not in st.session_state:
    st.session_state.route_data = None
    st.session_state.distance_km = None
    st.session_state.route_time = None
    st.session_state.prediction = None

# ==========================================
# SINGLE BUTTON (ONLY ONE!)
# ==========================================
if st.button("🚀 Calculate Route & Predict Delivery Time"):

    try:
        distance_km, route_time, route_data = get_route_data(
            store_lat, store_lon, drop_lat, drop_lon
        )

        st.session_state.route_data = route_data
        st.session_state.distance_km = distance_km
        st.session_state.route_time = route_time

        input_df = pd.DataFrame(columns=feature_columns)
        row = dict.fromkeys(feature_columns, 0)

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
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)

        st.session_state.prediction = prediction[0]

    except Exception as e:
        st.error(f"❌ Error: {e}")

# ==========================================
# DISPLAY RESULTS
# ==========================================
if st.session_state.route_data is not None:

    st.success(f"📍 Real Road Distance: {round(st.session_state.distance_km, 2)} KM")
    st.success(f"⏱ Estimated Driving Time: {round(st.session_state.route_time, 2)} minutes")
    st.success(f"📦 Final Predicted Delivery Time: {round(st.session_state.prediction, 2)} minutes")

    m = folium.Map(location=[store_lat, store_lon], zoom_start=12)

    folium.Marker([store_lat, store_lon], tooltip="Store",
                  icon=folium.Icon(color="green")).add_to(m)

    folium.Marker([drop_lat, drop_lon], tooltip="Delivery Location",
                  icon=folium.Icon(color="red")).add_to(m)

    coords = st.session_state.route_data["routes"][0]["geometry"]["coordinates"]
    route_points = [(c[1], c[0]) for c in coords]

    folium.PolyLine(route_points).add_to(m)

    st_folium(m, width=900, height=500)
