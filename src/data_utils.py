"""Load the cleaned Week 1 dataset with the correct dtypes (and verify it)."""
import hashlib

import pandas as pd

from config import CLEAN_DATA_PATH, CLEAN_DATA_SHA256

CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
CATEGORICAL_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "Churn",
]
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
ADDON_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]


def load_clean_data(path=CLEAN_DATA_PATH, verify=True):
    """Read the cleaned CSV, check its SHA-256 and restore categorical dtypes."""
    if verify:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != CLEAN_DATA_SHA256:
            raise ValueError("Cleaned data does not match the Week 1 output (checksum mismatch).")
    df = pd.read_csv(path)
    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype("category")
    df["Contract"] = pd.Categorical(df["Contract"], categories=CONTRACT_ORDER, ordered=True)
    return df
