import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title
st.title("👔 Churn Prediction")
st.write("App uses a saved Naive Bayes model to predict Customer Churn based on measurements.")

# Load CSV data
df = pd.read_excel('churn_dataset.xlsx')

# Show dataset preview
st.subheader("📊 Dataset Preview")
st.write(df.head())

# Load saved model
model = joblib.load("model_naive_bayes.pkl")

# Maping Sex Attribute to be 0 for Male and 1 for Female
gender_mapping = {
    'Male': 0,
    'Female': 1,
    'M': 0,  # Handle alternate forms
    'F': 1,
    '1': 1,  # Handle numeric representations
    '0': 0
}

df['Sex'] = df['Sex'].map(gender_mapping)

# Manual mapping for Chrun labels
Churn = {
    0: 'No',
    1: 'Yes',
}

# --- User Input Section ---
st.sidebar.header("🔍 Input Churn Features")

# Age
Age = st.sidebar.slider(
    "Age",
    int(df["Age"].min()),
    int(df["Age"].max()),
    int(df["Age"].mean())
)

# Tenure
Tenure = st.sidebar.slider(
    "Tenure",
    int(df["Tenure"].min()),
    int(df["Tenure"].max()),
    int(df["Tenure"].mean())
)

# Sex
Sex = st.sidebar.slider(
    "Sex",
    int(df["Sex"].min()),
    int(df["Sex"].max()),
    int(df["Sex"].mean())
)

# --- Prediction Section ---
# Collect input into array
input_data = np.array([[Age,Tenure,Sex]])

# # Predict species
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]

# # Show prediction
st.subheader("🎯 Prediction Result")
st.write(f"**Predicted Churn:** {Churn[prediction]}")

# # Show prediction probabilities
st.subheader("📈 Prediction Probabilities")
st.write(f"Yes: {prediction_proba[1]:.2%}")
st.write(f"No: {prediction_proba[0]:.2%}")


