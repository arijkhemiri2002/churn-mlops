"""Data loading, cleaning and feature preprocessing for churn prediction."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMN = "customerID"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning. Does not learn anything from the data (no leakage)."""
    df = df.copy()
    df = df.drop(columns=[ID_COLUMN], errors="ignore")

    # TotalCharges is read as text because some rows contain a blank string.
    # Those customers have tenure 0 (never billed yet), so 0 is the logical value.
    # This is a fixed business rule, not learned from data, so it cannot leak.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"].eq(0) & df["TotalCharges"].isna(), "TotalCharges"] = 0.0
    # Any other missing value is handled by the imputer inside the pipeline.

    if TARGET in df.columns:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def split_data(df: pd.DataFrame, test_size: float, seed: int):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Imputation + scaling + one-hot encoding, fitted only on training data."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, numeric_cols), ("cat", categorical, categorical_cols)]
    )