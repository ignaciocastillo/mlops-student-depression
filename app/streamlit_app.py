import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("model/modelo_depresion.pkl")

st.set_page_config(page_title="Student Depression Predictor", layout="centered")
st.title("🧠 Student Depression Predictor")

st.markdown("Fill out the form below to get a prediction.")

# === Input form ===
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.slider("Age", 15, 40, step=1)
academic_pressure = st.slider("Academic Pressure", 1, 5, step=1)
work_pressure = st.slider("Work Pressure", 1, 5, step=1)
cgpa = st.slider("CGPA", 0, 10, step=1)
study_satisfaction = st.slider("Study Satisfaction", 1, 5, step=1)
work_hours = st.slider("Daily Study/Work Hours", 0, 18, step=1)
sleep_duration = st.selectbox("Sleep Duration", ["4.0", "5.0", "7.0", "9.0"])
dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
suicidal_thoughts = st.selectbox("Have you had suicidal thoughts?", ["No", "Yes"])
financial_stress = st.selectbox("Financial Stress (1-5)", ["1", "2", "3", "4", "5"])
family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])


# === Encode inputs ===
def encode_inputs():
    data = [
        1 if gender == "Male" else 0,
        age,
        academic_pressure,
        work_pressure,
        cgpa,
        study_satisfaction,
        float(sleep_duration),
        ["Healthy", "Moderate", "Unhealthy"].index(dietary_habits),
        1 if suicidal_thoughts == "Yes" else 0,
        work_hours,
        int(financial_stress),
        1 if family_history == "Yes" else 0
    ]
    return np.array([data])


# === Prediction ===
if st.button("🔍 Predict"):
    user_input = encode_inputs()
    prediction = model.predict(user_input)

    if prediction[0] == 1:
        st.error("Prediction: Student is likely **depressed** 😔")
    else:
        st.success("Prediction: Student is likely **not depressed** 🤙")
