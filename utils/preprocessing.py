from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

SCALER_PATH = "model/scaler.pkl"

FEATURES = [
    "amount_boxcox",
    "oldBalanceOrig_boxcox",
    "newbalanceOrig_boxcox",
    "oldBalanceDest_boxcox",
    "newBalanceDest_boxcox",
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

COLUMN_ALIASES = {
    "oldBalanceOrig": "oldbalanceOrg",
    "newBalanceOrig": "newbalanceOrig",
    "oldBalanceDest": "oldbalanceDest",
    "newBalanceDest": "newbalanceDest",
    "oldbalanceOrig": "oldbalanceOrg",
    "newbalanceDest": "newbalanceDest",
    "newbalanceOrig": "newbalanceOrig",
}


@lru_cache(maxsize=1)
def _load_scaler():
    return joblib.load(SCALER_PATH)


def _standardize_columns(df):
    renamed = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns}).copy()

    for column in renamed.columns:
        if isinstance(column, str):
            normalized = column.strip()
            if normalized != column:
                renamed = renamed.rename(columns={column: normalized})

    return renamed


def _coerce_numeric(df, columns):
    converted = df.copy()
    for column in columns:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")
    return converted


def _build_model_features(df):
    raw_features = pd.DataFrame(
        {
            "amount_boxcox": df["amount"].astype(float),
            "oldBalanceOrig_boxcox": df["oldbalanceOrg"].astype(float),
            "newbalanceOrig_boxcox": df["newbalanceOrig"].astype(float),
            "oldBalanceDest_boxcox": df["oldbalanceDest"].astype(float),
            "newBalanceDest_boxcox": df["newbalanceDest"].astype(float),
        },
        index=df.index,
    )

    # The persisted model expects five positive "boxcox"-named features.
    # The original fitted power-transformer is not present in the repo,
    # so use log1p as a stable monotonic transform that also handles zeros.
    return np.log1p(raw_features)


def preprocess(df):
    df = _standardize_columns(df)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    raw_count = len(df)
    df = _coerce_numeric(
        df,
        ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"],
    )

    numeric_valid_mask = df[
        ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    ].notna().all(axis=1)
    invalid_numeric_rows = int((~numeric_valid_mask).sum())
    df = df[numeric_valid_mask].copy()

    # Remove irrelevant transaction types
    df["type"] = df["type"].astype(str).str.strip().str.upper()
    df = df[df["type"].isin(["CASH_OUT", "TRANSFER"])].copy()

    # Remove zero/negative amount rows
    df = df[df["amount"] > 0]

    # Balance inaccuracy features
    df["origBalance_inacc"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    df["destBalance_inacc"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
    df["originator_expected_balance"] = df["oldbalanceOrg"] - df["amount"]
    df["destination_expected_balance"] = df["oldbalanceDest"] + df["amount"]
    df["originator_balance_gap"] = (df["newbalanceOrig"] - df["originator_expected_balance"]).abs()
    df["destination_balance_gap"] = (
        df["newbalanceDest"] - df["destination_expected_balance"]
    ).abs()
    df["total_balance_gap"] = df["originator_balance_gap"] + df["destination_balance_gap"]
    df["balance_recovery_ratio"] = np.where(
        df["amount"] > 0,
        np.clip(df["newbalanceOrig"], a_min=0, a_max=None) / df["amount"],
        0.0,
    )
    df["liquidity_ratio"] = np.where(
        df["amount"] > 0,
        np.clip(df["oldbalanceOrg"], a_min=0, a_max=None) / df["amount"],
        0.0,
    )

    if df.empty:
        raise ValueError("No valid rows left after filtering transaction types and amounts.")

    X = _build_model_features(df)[FEATURES]

    scaler = _load_scaler()
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURES, index=X.index)

    quality = {
        "rows_in": int(raw_count),
        "rows_out": int(len(df)),
        "rows_filtered": int(raw_count - len(df)),
        "filter_rate_pct": round((raw_count - len(df)) / max(raw_count, 1) * 100, 2),
        "invalid_numeric_rows": invalid_numeric_rows,
    }

    return X_scaled, df, quality
