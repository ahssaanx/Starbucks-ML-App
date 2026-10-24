
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Starbucks Coffee Predictor",
    page_icon="☕"
)

st.title("Starbucks Coffee Calorie Predictor")

model_path = os.path.join(
    os.path.dirname(__file__),
    "starbucks_model.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

calories = st.number_input(
    "Enter coffee calories",
    min_value=0,
    max_value=1000,
    value=120
)

if st.button("Predict"):
    data = pd.DataFrame({"Calories": [calories]})
    result = model.predict(data)[0]
    st.success(f"Predicted category: {result}")
