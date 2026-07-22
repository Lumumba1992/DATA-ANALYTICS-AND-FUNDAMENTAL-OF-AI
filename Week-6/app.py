import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
model = joblib.load('best_heart_disease_model.pkl')

st.title("🫀 Heart Disease Prediction App")
st.write("Enter the patient's clinical details to predict the likelihood of heart disease.")

# Create user input fields
st.header("Patient Metrics")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    chest_pain = st.selectbox("Chest Pain Type", ['typical', 'asymptomatic', 'nonanginal', 'nontypical'])
    rest_bp = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    
with col2:
    rest_ecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    max_hr = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=220, value=150)
    ex_ang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=7.0, value=1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [1, 2, 3])
    ca = st.selectbox("Number of Major Vessels Colored (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ['normal', 'fixed', 'reversable'])

# Predict button
if st.button("Predict Heart Disease"):
    # Create a dataframe from user input
    input_data = pd.DataFrame({
        'Age': [age], 'Sex': [sex], 'ChestPain': [chest_pain], 'RestBP': [rest_bp],
        'Chol': [chol], 'Fbs': [fbs], 'RestECG': [rest_ecg], 'MaxHR': [max_hr],
        'ExAng': [ex_ang], 'Oldpeak': [oldpeak], 'Slope': [slope], 'Ca': [ca], 'Thal': [thal]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # Display result
    if prediction == 1:
        st.error(f"🚨 High Risk: The model predicts the presence of heart disease. (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk: The model predicts no heart disease. (Probability of disease: {probability:.2%})")