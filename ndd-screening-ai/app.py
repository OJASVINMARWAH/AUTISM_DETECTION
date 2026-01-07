# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# -------------------------------
# Load your trained model
# -------------------------------
MODEL_PATH = os.path.join("models", "autism_model.pkl")  # adjust if your model is elsewhere

@st.cache_data
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.set_page_config(page_title="Autism/NDD Screening App", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload Data", "Predict", "About"])

# -------------------------------
# Home Page
# -------------------------------
if page == "Home":
    st.title("Autism / NDD Screening App 🧠")
    st.markdown("""
    Welcome! This tool helps in **screening for Neurodevelopmental Disorders (NDDs)** using AI-based analysis.  
    Navigate through the sidebar to upload data, make predictions, and visualize results.
    """)
    st.image("assets/home_image.png", use_column_width=True)  # optional: put a relevant image in assets

# -------------------------------
# Upload Data Page
# -------------------------------
elif page == "Upload Data":
    st.title("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV file with patient data", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Data Uploaded Successfully!")
        st.dataframe(df.head())
        
        if st.checkbox("Show Summary Statistics"):
            st.write(df.describe())

# -------------------------------
# Prediction Page
# -------------------------------
elif page == "Predict":
    st.title("Predict NDD Risk")
    
    st.markdown("Fill in patient details below:")
    
    # Example input fields (customize based on your dataset features)
    age = st.number_input("Age", min_value=1, max_value=100, value=5)
    gender = st.selectbox("Gender", ["Male", "Female"])
    score_1 = st.number_input("Feature 1 Score", min_value=0, max_value=10, value=5)
    score_2 = st.number_input("Feature 2 Score", min_value=0, max_value=10, value=5)
    
    # Convert inputs to model-ready format
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [1 if gender=="Male" else 0],
        "score_1": [score_1],
        "score_2": [score_2]
    })
    
    if st.button("Predict"):
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:,1]  # if classification model
        
        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error(f"⚠️ High risk of NDD detected! (Probability: {probability[0]*100:.2f}%)")
        else:
            st.success(f"✅ Low risk of NDD detected. (Probability: {probability[0]*100:.2f}%)")

# -------------------------------
# About Page
# -------------------------------
elif page == "About":
    st.title("About This Project")
    st.markdown("""
    - **Project:** AI-based Autism/NDD Screening  
    - **Tech Stack:** Python, Streamlit, Scikit-learn / XGBoost  
    - **Purpose:** Assist early detection and screening of neurodevelopmental disorders.  
    - **GitHub:** [Your Repo Link](https://github.com/OJASVINMARWAH/AUTISM_DETECTION)
    """)
