import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.title("NYC Taxi Trip Duration — What-if")

with st.form("trip_form"):
    pickup_datetime = st.text_input("Pickup datetime (YYYY-MM-DD HH:MM:SS)", "2016-03-15 08:10:00")
    pickup_longitude = st.number_input("Pickup longitude", value=-73.982154)
    pickup_latitude = st.number_input("Pickup latitude", value=40.767937)
    dropoff_longitude = st.number_input("Dropoff longitude", value=-73.96463)
    dropoff_latitude = st.number_input("Dropoff latitude", value=40.765602)
    passenger_count = st.number_input("Passenger count", min_value=0, value=1)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "pickup_datetime": pickup_datetime,
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "passenger_count": passenger_count
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=5)
        if r.status_code == 200:
            data = r.json()
            st.success(f"Predicted duration (seconds): {data.get('predicted_duration_seconds'):.1f}")
        else:
            st.error(f"API error: {r.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")