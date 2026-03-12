from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

SCALER_PATH = "model/scaler.pkl"

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "origBalance_inacc",
    "destBalance_inacc",
]

REQUIRED_COLUMNS = set(
    [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
)


@lru_cache(maxsize=1)
def _load_scaler():
    return joblib.load(SCALER_PATH)


def preprocess(df):
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    raw_count = len(df)

    # Remove irrelevant transaction types
    df = df[df["type"].isin(["CASH_OUT", "TRANSFER"])].copy()

    # Remove zero/negative amount rows
    df = df[df["amount"] > 0]

    # Balance inaccuracy features
    df["origBalance_inacc"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    df["destBalance_inacc"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]

    if df.empty:
        raise ValueError("No valid rows left after filtering transaction types and amounts.")

    X = df[FEATURES].astype(float)

    scaler = _load_scaler()
    X_scaled = scaler.transform(X)

    quality = {
        "rows_in": int(raw_count),
        "rows_out": int(len(df)),
        "rows_filtered": int(raw_count - len(df)),
        "filter_rate_pct": round((raw_count - len(df)) / max(raw_count, 1) * 100, 2),
    }

    return X_scaled, df, quality