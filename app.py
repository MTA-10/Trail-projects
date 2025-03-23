import streamlit as st
import pandas as pd
import joblib
import numpy as np
import urllib.request

# Load Model and Scaler from GitHub
model_url = "https://raw.githubusercontent.com/MTA-10/Trail-projects/main/diabetes_model.pkl"
scaler_url = "https://raw.githubusercontent.com/MTA-10/Trail-projects/main/preprocessor.pkl"

model_filename = "diabetes_model.pkl"
scaler_filename = "preprocessor.pkl"

urllib.request.urlretrieve(model_url, model_filename)
urllib.request.urlretrieve(scaler_url, scaler_filename)

model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Streamlit App
st.title("🩺 Diabetes Prediction AI Model")
st.write("Enter your details to check diabetes risk.")

# User Inputs
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
age = st.number_input("Age", 1, 120, 30)
hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
smoking_history = st.selectbox("Smoking History", ["Never", "Former Smoker", "Current Smoker"])
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.7)
blood_glucose = st.number_input("Blood Glucose Level", 50, 300, 100)

# 🔹 Convert categorical values to match training data
smoking_dict = {"Never": 0, "Former Smoker": 1, "Current Smoker": 2}
smoking_history = smoking_dict.get(smoking_history, 0)  # Default to 0 if unexpected value

# 🔹 One-Hot Encode Gender (ensure correct format)
gender_male = 1 if gender == "Male" else 0
gender_other = 1 if gender == "Other" else 0

# 🔹 Define feature names & match model's input order
feature_names = ['age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 
                 'HbA1c_level', 'blood_glucose_level', 'gender_Male', 'gender_Other']

# 🔹 Create DataFrame with the correct format
input_df = pd.DataFrame([[age, hypertension, heart_disease, smoking_history, bmi, hba1c, 
                          blood_glucose, gender_male, gender_other]], 
                        columns=feature_names)

# 🔹 Convert all to numeric (fix NaN issues)
input_df = input_df.apply(pd.to_numeric, errors='coerce')

# 🔹 Check for NaN values (which can cause the isnan error)
if input_df.isnull().values.any():
    st.error("⚠️ Input contains missing or invalid values. Please check your inputs!")

# 🔹 Debugging: Check Feature Mismatch
st.write(f"🔍 Input data shape: {input_df.shape}")
st.write(f"🔍 Expected feature names: {scaler.feature_names_in_}")

# **Transform input & Predict**
try:
    input_transformed = scaler.transform(input_df)

    if st.button("Predict"):
        prediction = model.predict(input_transformed)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        st.success(f"🩺 Prediction: **{result}**")

except Exception as e:
    st.error(f"🚨 Transformation Error: {str(e)}")
