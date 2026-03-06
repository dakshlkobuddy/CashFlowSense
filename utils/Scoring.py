import joblib
import pandas as pd

model = joblib.load("model/AI-Based Cash Flow Credit Risk Model.pkl")

def generate_scores(X_scaled, original_df):

    probabilities = model.predict_proba(X_scaled)[:, 1]

    original_df["Risk_Probability"] = probabilities
    original_df["Risk_Score"] = (probabilities * 100).round(2)

    def decision(score):
        if score >= 80:
            return "Approve"
        elif score >= 50:
            return "Review"
        else:
            return "Reject"

    original_df["Credit_Decision"] = original_df["Risk_Score"].apply(decision)

    return original_df