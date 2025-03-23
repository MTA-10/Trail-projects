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

# User Inputs (Updated Features)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 1, 120, 30)
hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
smoking_history = st.selectbox("Smoking History", ["Never", "Former Smoker", "Current Smoker"])
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.7)
blood_glucose = st.number_input("Blood Glucose Level", 50, 300, 100)

# Convert categorical values to numerical (if needed)
gender = 1 if gender == "Male" else 0  # Example encoding
smoking_dict = {"Never": 0, "Former Smoker": 1, "Current Smoker": 2}
smoking_history = smoking_dict[smoking_history]

# Define expected feature names
feature_names = ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]

# Create DataFrame
input_df = pd.DataFrame([[gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, blood_glucose]], 
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
