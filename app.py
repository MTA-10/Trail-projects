import streamlit as st
import pandas as pd

# GitHub raw file URL (Replace with your actual link)
url = "https://raw.githubusercontent.com/MTA-10/Trail-projects/refs/heads/main/diabetes_prediction_dataset.csv"

# Function to load dataset
@st.cache_data
def load_data():
    return pd.read_csv(url)

df = load_data()

# Streamlit App
st.title("Dataset Viewer")
st.write("Displaying data from GitHub:")
st.dataframe(df)
