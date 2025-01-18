from utils import db_connect
engine = db_connect()

# your code here
import streamlit as st
from pickle import load

with open("../models/tree_classifier_crit-entro_maxdepth-5_minleaf-4_minsplit2_42.sav", "rb") as f:
    model = load(f)

st.title("Decision Tree - 1st try")

Pregnancies = st.slider("Pregnancies", min_value = 0.0, max_value = 5.0, step = 1.0)
Glucose = st.slider("Glucose", min_value = 100.0, max_value = 145.0, step = 1.0)
BloodPressure = st.slider("BloodPressure", min_value = 60.0, max_value = 80.0, step = 1.0)
Insulin = st.slider("Insulin", min_value = 0.0, max_value = 180.0, step = 5.0)
BMI = st.slider("BMI", min_value = 26.0, max_value = 48.0, step = 1.0)
DiabetesPedigreeFunction = st.slider("DiabetesPedigreeFunction", min_value = 0.2, max_value = 0.8, step = 0.1)
Age = st.slider("Age", min_value = 22.0, max_value = 42.0, step = 1.0)

if st.button("Predict"):
    data_a_predecir = [[Pregnancies, Glucose, BloodPressure, Insulin, BMI, DiabetesPedigreeFunction, Age]]
    prediction = model.predict(data_a_predecir)[0]
    st.write("Prediction", prediction)