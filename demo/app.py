# demo/app.py

import sys
import os

# Thêm thư mục gốc của project vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.predict import load_model, predict_single

st.title("🔮 Customer Churn Prediction App")

model = load_model()

st.write("Điền thông tin khách hàng:")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("SeniorCitizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=10)
phone = st.selectbox("PhoneService", ["Yes", "No"])
internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("PaperlessBilling", ["Yes", "No"])
payment = st.selectbox("PaymentMethod", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

monthly = st.number_input("MonthlyCharges", min_value=0.0, value=70.0)
total = st.number_input("TotalCharges", min_value=0.0, value=500.0)


if st.button("Predict"):
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    pred, prob = predict_single(model, customer)

    if pred == 1:
        st.error(f"⚠️ KHÁCH HÀNG SẼ CHURN (p = {prob:.2f})")
    else:
        st.success(f"✔ KHÁCH HÀNG KHÔNG CHURN (p = {prob:.2f})")
