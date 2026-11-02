"""Tests for SHAP explainability and automated risk/strength identification (§2.3 Module 6)."""
import pytest

from src.explain import identify_strengths_and_risks


def test_burnout_risk_detection():
    # Burnout: OverTime == Yes and WorkLifeBalance <= 2
    features = {
        "OverTime": "Yes",
        "WorkLifeBalance": 1,
        "AbsenteeismDays": 10,
        "YearsInCurrentRole": 2,
        "PromotionsLast5Y": 1,
        "EngagementScore": 3,
        "PeerFeedbackScore": 3,
        "ManagerRatingPrev": 3,
        "DeadlineAdherence": 0.85,
        "ProjectsCompleted": 8,
        "Certifications": 2,
    }
    analysis = identify_strengths_and_risks(features)
    risk_codes = [r["code"] for r in analysis["risks"]]
    assert "BURNOUT_RISK" in risk_codes


def test_career_stagnation_risk_detection():
    # Stagnation: YearsInCurrentRole >= 4 and PromotionsLast5Y == 0
    features = {
        "OverTime": "No",
        "WorkLifeBalance": 4,
        "AbsenteeismDays": 2,
        "YearsInCurrentRole": 6,
        "PromotionsLast5Y": 0,
        "EngagementScore": 3,
        "PeerFeedbackScore": 3,
        "ManagerRatingPrev": 3,
        "DeadlineAdherence": 0.85,
        "ProjectsCompleted": 7,
        "Certifications": 1,
    }
    analysis = identify_strengths_and_risks(features)
    risk_codes = [r["code"] for r in analysis["risks"]]
    assert "CAREER_STAGNATION" in risk_codes


def test_employee_strengths_identification():
    features = {
        "OverTime": "No",
        "WorkLifeBalance": 4,
        "AbsenteeismDays": 1,
        "YearsInCurrentRole": 2,
        "PromotionsLast5Y": 1,
        "EngagementScore": 5,
        "PeerFeedbackScore": 5,
        "ManagerRatingPrev": 5,
        "DeadlineAdherence": 0.98,
        "ProjectsCompleted": 12,
        "Certifications": 5,
    }
    analysis = identify_strengths_and_risks(features)
    assert len(analysis["strengths"]) >= 3
    strengths_text = " ".join(analysis["strengths"])
    assert "Deadline Adherence" in strengths_text
    assert "High-Volume Delivery" in strengths_text
    assert "Continuous Upskilling" in strengths_text
