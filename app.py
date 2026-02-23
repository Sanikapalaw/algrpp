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
st.write("Multiple routes + ML-based delivery prediction")

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
# ROUTING FUNCTION (MULTIPLE ROUTES)
# ==========================================
def get_route_data(start_lat, start_lon, end_lat, end_lon):

    url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson&alternatives=true"

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"OSRM Error {response.status_code}")

    data = response.json()

    if data["code"] != "Ok":
        raise Exception("No route found")

    return data["routes"]

# ==========================================
# SESSION STATE INIT
# ==========================================
if "routes" not in st.session_state:
    st.session_state.routes = None

if "selected_route_index" not in st.session_state:
    st.session_state.selected_route_index = 0

if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ==========================================
# BUTTON
# ==========================================
if st.button("🚀 Calculate Routes"):

    try:
        routes = get_route_data(store_lat, store_lon, drop_lat, drop_lon)
        st.session_state.routes = routes
        st.session_state.selected_route_index = 0
        st.session_state.prediction = None

    except Exception as e:
        st.error(f"❌ Error: {e}")

# ==========================================
# DISPLAY ROUTES
# ==========================================
if st.session_state.routes is not None:

    routes = st.session_state.routes

    st.subheader("🛣 Available Routes")

    route_options = []

    for i, route in enumerate(routes):
        distance_km = route["distance"] / 1000
        duration_min = route["duration"] / 60
        route_options.append(
            f"Route {i+1} → {round(distance_km,2)} KM | {round(duration_min,2)} mins"
        )

    selected_option = st.selectbox(
        "Select Route for Prediction",
        route_options
    )

    selected_index = route_options.index(selected_option)
    st.session_state.selected_route_index = selected_index

    # ==========================================
    # MAP DISPLAY
    # ==========================================
    m = folium.Map(location=[store_lat, store_lon], zoom_start=12)
    colors = ["blue", "red", "purple", "orange"]

    for i, route in enumerate(routes):

        coords = route["geometry"]["coordinates"]
        route_points = [(c[1], c[0]) for c in coords]

        folium.PolyLine(
            route_points,
            color=colors[i % len(colors)],
            weight=6 if i == selected_index else 3,
            opacity=0.9 if i == selected_index else 0.4
        ).add_to(m)

    folium.Marker([store_lat, store_lon],
                  tooltip="Store",
                  icon=folium.Icon(color="green")).add_to(m)

    folium.Marker([drop_lat, drop_lon],
                  tooltip="Delivery Location",
                  icon=folium.Icon(color="black")).add_to(m)

    st_folium(m, width=900, height=500)

    # ==========================================
    # PREDICTION BUTTON
    # ==========================================
    if st.button("📦 Predict Delivery Time"):

        selected_route = routes[selected_index]
        distance_km = selected_route["distance"] / 1000

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

    # ==========================================
    # SHOW PREDICTION
    # ==========================================
    if st.session_state.prediction is not None:
        st.success(
            f"📦 Predicted Delivery Time (Selected Route): {round(st.session_state.prediction,2)} minutes"
        )
