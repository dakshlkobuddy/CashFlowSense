import streamlit as st
import pandas as pd
from utils.preprocessing import preprocess
from utils.scoring import generate_scores

st.set_page_config(page_title="CashFlowSense", layout="wide")

st.title("💳 CashFlowSense — AI Based Credit Risk Engine")

st.markdown("Upload bank transaction data to generate AI-based credit risk decisions.")

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    X_scaled, processed_df = preprocess(df)
    results = generate_scores(X_scaled, processed_df)

    st.subheader("📊 Risk Assessment Results")
    st.dataframe(results[[
        "amount",
        "Risk_Probability",
        "Risk_Score",
        "Credit_Decision"
    ]])

    # Metrics
    avg_score = results["Risk_Score"].mean()
    high_risk = len(results[results["Risk_Score"] < 50])

    col1, col2 = st.columns(2)

    col1.metric("Average Risk Score", f"{avg_score:.2f}")
    col2.metric("High Risk Transactions", high_risk)

    st.subheader("📈 Risk Score Distribution")
    st.bar_chart(results["Risk_Score"])