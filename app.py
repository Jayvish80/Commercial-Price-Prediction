import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("commercial_price_model.pkl")

st.set_page_config(page_title="Commercial Price Prediction", layout="centered")

st.title("🏢 Commercial Property Price Prediction")
st.write("Enter property details below to predict price")

# Example input fields (Adjust based on your dataset)

area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1000)
location = st.number_input("Location Code", min_value=0, value=1)
floor = st.number_input("Floor Number", min_value=0, max_value=50, value=1)
parking = st.number_input("Parking Spaces", min_value=0, max_value=10, value=1)

# Predict Button
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "Area": [area],
        "Location": [location],
        "Floor": [floor],
        "Parking": [parking]
    })

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Commercial Price: ₹ {prediction[0]:,.2f}")
