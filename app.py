import streamlit as st
import pickle
import numpy as np

# Load the trained models and scaler
with open("ridge_model.pkl", "rb") as f:
    ridge_model = pickle.load(f)

with open("lasso_model.pkl", "rb") as f:
    lasso_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Function to make predictions
def predict_diabetes(features):
    # Convert input to numpy array and reshape for model
    features = np.array(features).reshape(1, -1)
    
    # Scale the input using the same scaler used during training
    features_scaled = scaler.transform(features)
    
    # Get predictions from both models
    ridge_pred = ridge_model.predict(features_scaled)[0]
    lasso_pred = lasso_model.predict(features_scaled)[0]

    return ridge_pred, lasso_pred

# Streamlit UI
st.title("Diabetes Progression Prediction")

# User Inputs
age = st.number_input("Age", min_value=20, max_value=80, value=50)
sex = st.radio("Sex", ["Female", "Male"])
bmi = st.number_input("BMI", min_value=15.0, max_value=45.0, value=25.0)
bp = st.number_input("Blood Pressure", min_value=80, max_value=180, value=120)
s1 = st.number_input("Total Serum Cholesterol", min_value=100, max_value=300, value=200)
s2 = st.number_input("LDL Cholesterol", min_value=50, max_value=200, value=100)
s3 = st.number_input("HDL Cholesterol", min_value=20, max_value=100, value=50)
s4 = st.number_input("Cholesterol/HDL Ratio", min_value=1.5, max_value=7.0, value=4.0)
s5 = st.number_input("Serum Triglycerides", min_value=50, max_value=500, value=150)
s6 = st.number_input("Blood Sugar Level", min_value=50, max_value=300, value=100)

# Convert sex to numeric
sex_value = 1 if sex == "Male" else 0

# Predict button
if st.button("Predict"):
    features = [age, sex_value, bmi, bp, s1, s2, s3, s4, s5, s6]
    ridge_pred, lasso_pred = predict_diabetes(features)

    st.subheader("Predictions:")
    st.write(f"🔹 **Ridge Regression Prediction:** {ridge_pred:.2f}")
    st.write(f"🔹 **Lasso Regression Prediction:** {lasso_pred:.2f}")