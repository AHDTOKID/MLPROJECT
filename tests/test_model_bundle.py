"""Tests for ModelBundle serialization and inference interface (§4.4)."""
import pickle
import numpy as np
import pandas as pd
import pytest

from src.config import CLASS_NAMES, MODEL_DIR
from src.model_bundle import ModelBundle


@pytest.fixture
def sample_employee():
    return {
        "Age": 34, "Gender": "Female", "MaritalStatus": "Married",
        "DistanceFromHome": 6, "Education": 4, "Department": "Research & Development",
        "JobRole": "Developer", "YearsInCurrentRole": 3, "OverTime": "No",
        "MonthlyIncome": 8000.0, "TotalWorkingYears": 10, "YearsAtCompany": 5,
        "NumCompaniesWorked": 2, "PromotionsLast5Y": 1, "EngagementScore": 4,
        "WorkLifeBalance": 3, "PeerFeedbackScore": 4, "ManagerRatingPrev": 5,
        "ProjectsCompleted": 9, "DeadlineAdherence": 0.9,
        "AbsenteeismDays": 3, "Certifications": 4,
    }


def test_pickle_load_model_bundle():
    pkl_path = MODEL_DIR / "performance_model.pkl"
    assert pkl_path.exists(), "Model pickle file must exist"

    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    assert isinstance(bundle, ModelBundle)
    assert bundle.class_names == CLASS_NAMES
    assert hasattr(bundle, "predict")
    assert hasattr(bundle, "predict_proba")
    assert hasattr(bundle, "predict_one")
    assert hasattr(bundle, "predict_many")


def test_predict_one_output_format(sample_employee):
    pkl_path = MODEL_DIR / "performance_model.pkl"
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    res = bundle.predict_one(sample_employee)
    assert "predicted_class" in res
    assert res["predicted_class"] in CLASS_NAMES
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "probabilities" in res
    assert len(res["probabilities"]) == len(CLASS_NAMES)
    assert abs(sum(res["probabilities"].values()) - 1.0) < 1e-5


def test_predict_many_batch(sample_employee):
    pkl_path = MODEL_DIR / "performance_model.pkl"
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    df = pd.DataFrame([sample_employee, sample_employee])
    results = bundle.predict_many(df)

    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert r["predicted_class"] in CLASS_NAMES
        assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-5
