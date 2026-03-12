from functools import lru_cache

import joblib
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


def generate_scores(X_scaled, original_df, approve_at=80.0, review_at=50.0):
    model = _load_model()

    probabilities = model.predict_proba(X_scaled)[:, 1]

    scored_df = original_df.copy()
    scored_df["Risk_Probability"] = probabilities
    scored_df["Risk_Score"] = (probabilities * 100).round(2)
    scored_df["Credit_Decision"] = scored_df["Risk_Score"].apply(
        lambda s: _decision(s, approve_at, review_at)
    )

    return scored_df