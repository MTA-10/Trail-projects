import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load Model and Scaler from GitHub (Use Raw URLs)
model_url = "https://github.com/MTA-10/Trail-projects/raw/refs/heads/main/diabetes_model.pkl"
scaler_url = "https://github.com/MTA-10/Trail-projects/raw/refs/heads/main/preprocessor.pkl"

# Download and Load Model & Scaler
model = joblib.load(model_url)
scaler = joblib.load(scaler_url)

# Streamlit App
st.title("🩺 Diabetes Prediction AI Model")
st.write("Enter your details to check diabetes risk.")

# User Inputs
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin Level", 0, 900, 79)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

# Prepare Data for Prediction
input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
input_data = scaler.transform(input_data)  # Normalize Input

# Predict Button
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.success(f"🩺 Prediction: **{result}**")
