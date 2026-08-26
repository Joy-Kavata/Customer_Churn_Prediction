import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔮")

st.title("🔮 Telco Customer Churn Predictor")
st.write("Adjust customer attributes to test real-time churn predictions.")

@st.cache_resource
def load_assets():
    model = joblib.load('churn_random_forest_model.pkl')
    scaler = joblib.load('churn_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# Sidebar Inputs
st.sidebar.header("Customer Profile")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", value=65.0)
total_charges = st.sidebar.number_input("Total Charges ($)", value=780.0)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

if st.button("Calculate Churn Risk"):
    input_dict = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
        'InternetService_No': 1 if internet == "No" else 0,
        'PaymentMethod_Electronic check': 1 if payment == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment == "Mailed check" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment == "Credit card" else 0,
    }

    input_df = pd.DataFrame([input_dict])

    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model.feature_names_in_]

    probability = model.predict_proba(input_df)[0][1] * 100

    st.markdown("---")
    st.subheader("Prediction Outcome")
    if probability >= 70:
        st.error(f"⚠️ **High Churn Risk: {probability:.1f}%**\n\nRecommendation: Offer retention incentive immediately.")
    elif probability >= 35:
        st.warning(f"⚡ **Medium Churn Risk: {probability:.1f}%**\n\nRecommendation: Send customer satisfaction survey.")
    else:
        st.success(f"✅ **Low Churn Risk: {probability:.1f}%**\n\nRecommendation: Standard lifecycle communications.")
