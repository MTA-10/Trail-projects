import streamlit as st
import pandas as pd
import joblib
import numpy as np
import urllib.request

# Load Model and Scaler from GitHub (First Download, Then Load)
model_url = "https://raw.githubusercontent.com/MTA-10/Trail-projects/main/diabetes_model.pkl"
scaler_url = "https://raw.githubusercontent.com/MTA-10/Trail-projects/main/preprocessor.pkl"

# Define local filenames
model_filename = "diabetes_model.pkl"
scaler_filename = "preprocessor.pkl"

# Download files first
urllib.request.urlretrieve(model_url, model_filename)
urllib.request.urlretrieve(scaler_url, scaler_filename)

# Load Model & Scaler from local files
model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

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
hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.7)  # Added missing feature

# Define expected feature names
feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
    "BMI", "DiabetesPedigreeFunction", "Age", "HbA1c_Level"
]

# Create DataFrame with feature names
input_df = pd.DataFrame([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age, hba1c]], 
                        columns=feature_names)

# Debugging: Check Feature Mismatch
st.write(f"🔍 Input data shape: {input_df.shape}")
st.write(f"🔍 Scaler expected input shape: {scaler.n_features_in_}")
st.write(f"🔍 Expected feature names: {scaler.feature_names_in_}")

# Ensure input data has correct features
if list(input_df.columns) != list(scaler.feature_names_in_):
    st.error("⚠️ Feature mismatch! Column names do not match the expected ones.")
else:
    # Normalize input
    input_transformed = scaler.transform(input_df)

    # Predict Button
    if st.button("Predict"):
        prediction = model.predict(input_transformed)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        st.success(f"🩺 Prediction: **{result}**")
