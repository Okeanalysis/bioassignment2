import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load('svc_top_features_model.pkl')

# Feature names
top_features = [
    'mean concave points',
    'worst texture',
    'radius error',
    'worst symmetry',
    'area error',
    'mean compactness',
    'worst concavity',
    'mean concavity',
    'fractal dimension error',
    'worst radius'
]

# Streamlit App
st.title("🧬 Breast Cancer Prediction App")
st.write("Enter the values for the top features to predict whether the tumor is **Malignant (0)** or **Benign (1)**.")

# Create input fields
inputs = []
for feature in top_features:
    value = st.number_input(f"{feature}", min_value=0.0, format="%.4f")
    inputs.append(value)

# Predict button
if st.button("Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    result = "Benign (1)" if prediction == 1 else "Malignant (0)"
    
    st.subheader("🔍 Prediction Result")
    st.success(f"The model predicts: **{result}**")
