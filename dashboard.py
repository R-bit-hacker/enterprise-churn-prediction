import streamlit as st
import joblib
import pandas as pd
import time

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

# Load model and columns
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Predefine categorical options
categorical_options = {
    "gender": ["Male", "Female"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ]
}

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]

# User input
st.header("Enter Customer Data")

input_data = {}
cols = st.columns(3)  # 3-column layout

i = 0
for col_name in categorical_options.keys():
    input_data[col_name] = cols[i % 3].selectbox(col_name, categorical_options[col_name])
    i += 1

# Numeric fields
for num_col in numeric_features:
    input_data[num_col] = cols[i % 3].number_input(num_col, value=0.0)
    i += 1

# Prediction
if st.button("Predict", type="primary"):
    with st.spinner("Analyzing customer data..."):
        progress = st.progress(0)
        for percent in range(0, 101, 20):
            time.sleep(0.1)
            progress.progress(percent)

        df = pd.DataFrame([input_data])
        df = pd.get_dummies(df)

        # Align with model columns
        missing_cols = set(model_columns) - set(df.columns)
        for c in missing_cols:
            df[c] = 0
        df = df[model_columns]

        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0, 1]

    # Display results
    st.subheader("Prediction Results")
    st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
    st.info(f"Churn Probability: {proba:.2%}")

    # Risk level indicator
    if proba >= 0.7:
        st.error("⚠ High Risk of Churn — Immediate Action Recommended")
    elif proba >= 0.4:
        st.warning("🟠 Medium Risk of Churn — Monitor Closely")
    else:
        st.success("🟢 Low Risk of Churn")
