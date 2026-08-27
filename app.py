import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Wide layout configuration
st.set_page_config(page_title="Telco Churn Predictor", page_icon="🔮", layout="wide")

st.title("Telco Customer Churn Predictor")
st.write("Adjust customer attributes in the sidebar to test real-time churn predictions and evaluate risk drivers.")

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

# Layout setup for main view
st.markdown("---")

if st.button("Calculate Churn Risk", type="primary"):
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

    # Reorder/pad columns to match trained model exact alignment
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model.feature_names_in_]

    # Calculate Probability
    probability = model.predict_proba(input_df)[0][1] * 100

    # 1. Top KPI Summary Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Churn Probability", value=f"{probability:.1f}%")

    with col2:
        if probability >= 70:
            st.metric(label="Risk Category", value="🔴 High Risk")
        elif probability >= 35:
            st.metric(label="Risk Category", value="🟡 Medium Risk")
        else:
            st.metric(label="Risk Category", value="🟢 Low Risk")

    with col3:
        annual_val = monthly_charges * 12
        st.metric(label="Est. Annual Value", value=f"${annual_val:,.2f}")

    st.write("### Churn Probability Score")
    st.progress(int(probability))

    # 2. Recommendation Callout Box
    if probability >= 70:
        st.error("⚠️ **High Churn Risk:** Recommendation: Offer targeted retention incentive or long-term contract discount immediately.")
    elif probability >= 35:
        st.warning("⚡ **Medium Churn Risk:** Recommendation: Send customer satisfaction survey and offer basic add-on perks.")
    else:
        st.success("✅ **Low Churn Risk:** Recommendation: Standard automated lifecycle communications.")

    # 3. Model Feature Importance Visual
    st.markdown("---")
    st.write("### Top Risk Factor Importance")
    
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'Feature': model.feature_names_in_,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig = px.bar(
            importances.tail(8), 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title="Relative Feature Weighting in Churn Decision",
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
