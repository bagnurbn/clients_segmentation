import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Card Segmentation", layout="centered")
st.title("KMeans Segmentation")


FEATURES = [
    "BALANCE",
    "BALANCE_FREQUENCY",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "CASH_ADVANCE_TRX",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT",
    "TENURE",
]

SEGMENT_NAME = {
    0: "Regular customer",
    1: "Cautios customer",
}

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('kmeans_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

kmeans, scaler = load_model_and_scaler()

c1, c2, c3 = st.columns(3)

with c1:
    BALANCE = st.number_input("Balance", min_value=0.0, value=0.0, step=100.0, format="%.2f")
    BALANCE_FREQUENCY = st.slider("Balance Frequency", 0.0, 1.0, 0.50, 0.01)
    PURCHASES = st.number_input("Purchases", min_value=0.0, value=0.0, step=100.0, format="%.2f")
    PURCHASES_FREQUENCY = st.slider("Purchases Frequency", 0.0, 1.0, 0.50, 0.01)
    CASH_ADVANCE_TRX = st.number_input("Cash Advance Transactions", min_value=0, value=0, step=1)
    CREDIT_LIMIT = st.number_input("Credit Limit", min_value=0.0, value=0.0, step=100.0, format="%.2f")

with c2:
    ONEOFF_PURCHASES = st.number_input("One-off Purchases", min_value=0.0, value=0.0, step=100.0, format="%.2f")
    ONEOFF_PURCHASES_FREQUENCY = st.slider("One-off Purchases Frequency", 0.0, 1.0, 0.50, 0.01)
    CASH_ADVANCE = st.number_input("Cash Advance", min_value=0.0, value=0.0, step=100.0, format="%.2f")
    CASH_ADVANCE_FREQUENCY = st.slider("Cash Advance Frequency", 0.0, 1.0, 0.50, 0.01)
    PURCHASES_TRX = st.number_input("Purchases Transactions", min_value=0, value=0, step=1)
    PAYMENTS = st.number_input("Payments", min_value=0.0, value=0.0, step=100.0, format="%.2f")

with c3:
    INSTALLMENTS_PURCHASES = st.number_input("Installments Purchases", min_value=0.0, value=0.0, step=100.0, format="%.2f")
    PURCHASES_INSTALLMENTS_FREQUENCY = st.slider("Purchases Installments Frequency", 0.0, 1.0, 0.50, 0.01)
    MINIMUM_PAYMENTS = st.number_input("Minimum Payments", min_value=0.0, value=0.0, step=50.0, format="%.2f")
    PRC_FULL_PAYMENT = st.slider("Percent of Full Payment", 0.0, 1.0, 0.50, 0.01)
    TENURE = st.number_input("Tenure (months)", min_value=0, value=0, step=1)

row = [
    BALANCE,
    BALANCE_FREQUENCY,
    PURCHASES,
    ONEOFF_PURCHASES,
    INSTALLMENTS_PURCHASES,
    CASH_ADVANCE,
    PURCHASES_FREQUENCY,
    ONEOFF_PURCHASES_FREQUENCY,
    PURCHASES_INSTALLMENTS_FREQUENCY,
    CASH_ADVANCE_FREQUENCY,
    CASH_ADVANCE_TRX,
    PURCHASES_TRX,
    CREDIT_LIMIT,
    PAYMENTS,
    MINIMUM_PAYMENTS,
    PRC_FULL_PAYMENT,
    TENURE,
]
X = pd.DataFrame([row], columns=FEATURES)

st.markdown("---")
if st.button("Predict Cluster"):
    try:
        X_scaled = scaler.transform(X)
        seg = int(kmeans.predict(X_scaled)[0])
        name = SEGMENT_NAME.get(seg, "Unknown")
        st.success(f"Predicted Cluster: {seg} — **{name}**")
    except ValueError as e:
        st.error(f"Input/feature mismatch: {e}")