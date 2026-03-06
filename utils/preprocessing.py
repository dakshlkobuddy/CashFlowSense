import pandas as pd
import joblib
import numpy as np

scaler = joblib.load("model/scaler.pkl")

def preprocess(df):
    
    # Remove irrelevant transaction types
    df = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])]

    # Remove zero amount rows
    df = df[df['amount'] > 0]

    # Balance inaccuracy features
    df['origBalance_inacc'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
    df['destBalance_inacc'] = df['newbalanceDest'] - df['oldbalanceDest'] - df['amount']

    # Select final features used in training
    features = [
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest',
        'origBalance_inacc',
        'destBalance_inacc'
    ]

    X = df[features]

    # Scaling
    X_scaled = scaler.transform(X)

    return X_scaled, df