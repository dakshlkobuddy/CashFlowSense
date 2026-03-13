from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "model/AI-Based Cash Flow Credit Risk Model.pkl"


@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(MODEL_PATH)


def _decision(score, approve_at, review_at):
    if score >= approve_at:
        return "Approve"
    if score >= review_at:
        return "Review"
    return "Reject"


def _risk_label(probability):
    if probability >= 0.75:
        return "High Risk"
    if probability >= 0.40:
        return "Moderate Risk"
    return "Low Risk"


def _stability_score(df):
    liquidity_component = np.clip(df["liquidity_ratio"] / 3.0, 0, 1)
    recovery_component = np.clip(df["balance_recovery_ratio"], 0, 1)
    gap_component = 1 - np.clip(df["total_balance_gap"] / (df["amount"].abs() + 1), 0, 1)
    score = (0.4 * liquidity_component + 0.35 * recovery_component + 0.25 * gap_component) * 100
    return score.round(2)


def _suggest_credit_limit(df, stability_score):
    liquidity_buffer = np.clip(df["oldbalanceOrg"], a_min=0, a_max=None) * 0.35
    transaction_capacity = df["amount"].abs() * (0.20 + (stability_score / 100.0) * 0.45)
    suggestion = np.minimum(liquidity_buffer, transaction_capacity)
    return suggestion.round(2)


def _suggest_tenure(stability_score):
    if stability_score >= 80:
        return "12 months"
    if stability_score >= 65:
        return "9 months"
    if stability_score >= 50:
        return "6 months"
    return "3 months"


def _explanation(row):
    points = []

    if row["liquidity_ratio"] >= 1.5:
        points.append("strong pre-transaction liquidity")
    elif row["liquidity_ratio"] < 0.5:
        points.append("thin liquidity relative to transaction size")

    if row["balance_recovery_ratio"] >= 0.5:
        points.append("healthy post-transaction balance recovery")
    elif row["balance_recovery_ratio"] <= 0.1:
        points.append("post-transaction balance drops close to zero")

    if row["total_balance_gap"] <= max(row["amount"] * 0.05, 1):
        points.append("balance trail is internally consistent")
    else:
        points.append("balance trail shows reconciliation gaps")

    if row["type"] == "TRANSFER":
        points.append("transfer behavior carries elevated monitoring priority")

    return "; ".join(points[:3]).capitalize() + "."


def generate_scores(X_scaled, original_df, approve_at=80.0, review_at=50.0):
    model = _load_model()

    probabilities = model.predict_proba(X_scaled)[:, 1]

    scored_df = pd.DataFrame(index=original_df.index)
    scored_df["type"] = original_df["type"]
    scored_df["amount"] = original_df["amount"].astype(float)
    scored_df["balance"] = original_df["newbalanceOrig"].astype(float)
    scored_df["Risk_Probability"] = probabilities.round(4)
    scored_df["Risk_Score"] = ((1 - probabilities) * 100).round(2)
    scored_df["Risk_Label"] = pd.Series(probabilities, index=original_df.index).apply(_risk_label)
    scored_df["Stability_Score"] = _stability_score(original_df)
    scored_df["Credit_Decision"] = scored_df["Risk_Score"].apply(
        lambda s: _decision(s, approve_at, review_at)
    )
    scored_df["Suggested_Credit_Limit"] = _suggest_credit_limit(
        original_df, scored_df["Stability_Score"]
    )
    scored_df["Suggested_Tenure"] = scored_df["Stability_Score"].apply(_suggest_tenure)
    scored_df["AI_Explanation"] = original_df.apply(_explanation, axis=1)

    return scored_df
