import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Commercial Analytics",
    page_icon="🏢",
    layout="wide"
)

# ----------------------------------
# Load Model
# ----------------------------------
@st.cache_resource
def load_model():
    return joblib.load("commercial_price_model.pkl")

model = load_model()

# ----------------------------------
# Sidebar Navigation
# ----------------------------------
st.sidebar.title("🏢 Commercial Analytics Modules")

module = st.sidebar.selectbox(
    "Choose Module",
    ["Price Prediction", "Revenue Dashboard", "Location Insights"]
)

# ==================================
# 1️⃣ PRICE PREDICTION MODULE
# ==================================
if module == "Price Prediction":

    st.title("💰 Commercial Property Price Prediction")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area (sq ft)", 100, 10000, 1000)
        floor = st.number_input("Floor Number", 0, 50, 1)

    with col2:
        parking = st.number_input("Parking Spaces", 0, 10, 1)
        location_code = st.number_input("Location Code", 0, 20, 1)

    if st.button("Predict Price"):

        input_df = pd.DataFrame({
            "Area": [area],
            "Floor": [floor],
            "Parking": [parking],
            "Location": [location_code]
        })

        prediction = model.predict(input_df)[0]

        st.success(f"🏷️ Estimated Price: ₹ {prediction:,.2f}")

# ==================================
# 2️⃣ REVENUE DASHBOARD
# ==================================
elif module == "Revenue Dashboard":

    st.title("📊 Commercial Revenue Dashboard")

    df = pd.read_csv("New Commercial Master.csv")

    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Revenue Summary")

    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns

    if len(numeric_cols) > 0:
        revenue_col = st.selectbox("Select Revenue Column", numeric_cols)

        total_revenue = df[revenue_col].sum()
        avg_revenue = df[revenue_col].mean()

        col1, col2 = st.columns(2)
        col1.metric("Total Revenue", f"₹ {total_revenue:,.2f}")
        col2.metric("Average Revenue", f"₹ {avg_revenue:,.2f}")

        st.line_chart(df[revenue_col])

# ==================================
# 3️⃣ LOCATION INSIGHTS
# ==================================
elif module == "Location Insights":

    st.title("📍 Location Analysis")

    df = pd.read_csv("New Commercial Master.csv")

    categorical_cols = df.select_dtypes(include=["object"]).columns

    if len(categorical_cols) > 0:
        location_col = st.selectbox("Select Location Column", categorical_cols)

        top_locations = df[location_col].value_counts().head(10)

        st.bar_chart(top_locations)

        st.write("Top 10 Locations")
        st.write(top_locations)
