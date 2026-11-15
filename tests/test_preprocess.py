import pandas as pd

from src.features.preprocess import build_preprocessor, clean_data


def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["A", "B", "C", "D"],
            "gender": ["Male", "Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0, 0],
            "tenure": [1, 24, 0, 60],
            "MonthlyCharges": [29.9, 70.5, 50.0, 99.0],
            "TotalCharges": ["29.9", "1692.0", " ", "5940.0"],  # blank like the real data
            "Churn": ["Yes", "No", "No", "No"],
        }
    )


def test_clean_drops_id_and_fixes_total_charges():
    df = clean_data(sample_df())
    assert "customerID" not in df.columns
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    # the blank row has tenure 0 -> TotalCharges must become 0, not NaN
    assert df["TotalCharges"].isna().sum() == 0
    assert df.loc[df["tenure"] == 0, "TotalCharges"].iloc[0] == 0.0


def test_target_is_binary():
    df = clean_data(sample_df())
    assert set(df["Churn"].unique()) <= {0, 1}
    assert df["Churn"].tolist() == [1, 0, 0, 0]


def test_preprocessor_handles_missing_values():
    df = clean_data(sample_df())
    X = df.drop(columns=["Churn"])
    out = build_preprocessor(X).fit_transform(X)
    assert not pd.isna(out).any()