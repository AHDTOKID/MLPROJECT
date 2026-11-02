"""Feature engineering + preprocessing pipeline (self-contained, no training-serving skew)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import (
    ALL_NUMERIC, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
)


def clamp_numerical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clamps extreme numerical outliers and removes uninformative constants (Risk Mitigation §5)."""
    df = df.copy()
    # Remove uninformative constants if present (§1.6)
    for col in ["EmployeeCount", "StandardHours", "Over18"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Outlier clamping bounds
    bounds = {
        "Age": (18, 75),
        "MonthlyIncome": (1000.0, 50000.0),
        "TotalWorkingYears": (0.0, 50.0),
        "YearsAtCompany": (0.0, 45.0),
        "YearsInCurrentRole": (0.0, 30.0),
        "DistanceFromHome": (0.0, 100.0),
        "AbsenteeismDays": (0.0, 60.0),
        "ProjectsCompleted": (0.0, 50.0),
        "Certifications": (0.0, 30.0),
        "DeadlineAdherence": (0.0, 1.0),
        "EngagementScore": (1.0, 5.0),
        "WorkLifeBalance": (1.0, 5.0),
        "PeerFeedbackScore": (1.0, 5.0),
        "ManagerRatingPrev": (1.0, 5.0),
    }
    for col, (low, high) in bounds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=low, upper=high)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes domain-specific composite metrics (§2.5)."""
    df = clamp_numerical_outliers(df)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Safe defaults if missing during single inference
    years_at_co = df.get("YearsAtCompany", pd.Series(0, index=df.index)).fillna(0)
    total_years = df.get("TotalWorkingYears", pd.Series(0, index=df.index)).fillna(0)
    promotions = df.get("PromotionsLast5Y", pd.Series(0, index=df.index)).fillna(0)
    certs = df.get("Certifications", pd.Series(0, index=df.index)).fillna(0)
    eng = df.get("EngagementScore", pd.Series(3, index=df.index)).fillna(3)
    peer = df.get("PeerFeedbackScore", pd.Series(3, index=df.index)).fillna(3)
    mgr = df.get("ManagerRatingPrev", pd.Series(3, index=df.index)).fillna(3)

    df["TenureRatio"] = years_at_co / (total_years + 1.0)
    df["PromotionRate"] = promotions / 5.0
    df["TrainingPerYear"] = certs / (years_at_co + 1.0)
    df["EngagementIndex"] = (eng + peer + mgr) / 3.0
    df["ManagerPeerGap"] = mgr - peer
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, ALL_NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline([
        ("feature_engineering", FunctionTransformer(add_engineered_features, validate=False)),
        ("preprocessor", build_preprocessor()),
        ("classifier", model),
    ])
