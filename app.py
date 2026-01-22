import streamlit as st
import joblib
import numpy as np
import os

# Page Config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Prediction System")
st.write("Name: ISHOLA OLUFEMI | Matric: 22H032024")

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'house_price_model.pkl')
    return joblib.load(model_path)

try:
    model = load_model()
    st.success("System Status: Model Loaded Successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

# User Inputs (6 Features)
st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    gr_liv_area = st.number_input("Living Area (sq ft)", min_value=300, value=1500)
    year_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=2000)

with col2:
    garage_cars = st.selectbox("Garage Capacity (Cars)", [0, 1, 2, 3, 4])
    full_bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4])
    total_bsmt_sf = st.number_input("Total Basement (sq ft)", min_value=0, value=800)

# Predict
if st.button("Predict Price"):
    features = np.array([[overall_qual, gr_liv_area, total_bsmt_sf, garage_cars, full_bath, year_built]])
    prediction = model.predict(features)[0]
    st.balloons()
    st.success(f"Estimated Price: ${prediction:,.2f}")
