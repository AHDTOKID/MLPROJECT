"""Sanity checks for the ML pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.config import ALL_FEATURES, CLASS_NAMES
from src.preprocessing import add_engineered_features


def test_feature_engineering_creates_all_derived_columns():
    df = pd.DataFrame([{
        "Age": 30, "DistanceFromHome": 5, "Education": 3, "YearsInCurrentRole": 2,
        "MonthlyIncome": 6500, "TotalWorkingYears": 8, "YearsAtCompany": 4,
        "NumCompaniesWorked": 2, "PromotionsLast5Y": 1, "EngagementScore": 4,
        "WorkLifeBalance": 3, "PeerFeedbackScore": 4, "ManagerRatingPrev": 5,
        "ProjectsCompleted": 8, "DeadlineAdherence": 0.85, "AbsenteeismDays": 4,
        "Certifications": 3, "Gender": "Male", "MaritalStatus": "Married",
        "Department": "Sales", "JobRole": "Manager", "OverTime": "No",
    }])
    out = add_engineered_features(df)
    for col in ["TenureRatio", "PromotionRate", "TrainingPerYear",
                "EngagementIndex", "ManagerPeerGap"]:
        assert col in out.columns


def test_predictor_outputs_valid_class(tmp_path):
    from src.predict import PerformancePredictor
    from src.config import MODEL_DIR
    if not (MODEL_DIR / "performance_model.pkl").exists():
        pytest.skip("Train the model first: python -m src.train")
    p = PerformancePredictor()
    row = {
        "Age": 34, "Gender": "Female", "MaritalStatus": "Married",
        "DistanceFromHome": 6, "Education": 4, "Department": "Research & Development",
        "JobRole": "Developer", "YearsInCurrentRole": 3, "OverTime": "No",
        "MonthlyIncome": 8000, "TotalWorkingYears": 10, "YearsAtCompany": 5,
        "NumCompaniesWorked": 2, "PromotionsLast5Y": 1, "EngagementScore": 4,
        "WorkLifeBalance": 3, "PeerFeedbackScore": 4, "ManagerRatingPrev": 5,
        "ProjectsCompleted": 9, "DeadlineAdherence": 0.9,
        "AbsenteeismDays": 3, "Certifications": 4,
    }
    res = p.predict_one(row)
    assert res["predicted_class"] in CLASS_NAMES
    assert abs(sum(res["probabilities"].values()) - 1.0) < 1e-6