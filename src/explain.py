"""SHapley Additive exPlanations (SHAP) + automated strengths & operational risk engine (§2.3 Module 6).

Implements:
- Local SHAP attribution for single-prediction transparent audit trails
- Global feature importance extraction
- Automated identification of Strengths and Operational Risk Factors:
    * Burnout Risk (Overtime + Low WLB + Absenteeism)
    * Career Stagnation Risk (High tenure in role without promotion)
    * Retention / Flight Risk (Low engagement & manager-peer delta)
    * Delivery & Upskilling Strengths
"""
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import shap

from .config import CLASS_NAMES, MODEL_DIR
from .preprocessing import add_engineered_features


def _split_pipeline(pipeline):
    return pipeline.named_steps["preprocessor"], pipeline.named_steps["classifier"]


def _normalize_shap(values, n_classes):
    if isinstance(values, list):
        return np.stack(values, axis=-1)
    arr = np.asarray(values)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        return np.stack([-arr, arr], axis=-1) if n_classes == 2 else arr[..., None]
    raise ValueError(f"Unexpected SHAP shape: {arr.shape}")


def build_explainer(pipeline, background_df: pd.DataFrame):
    """Builds and caches the SHAP explainer, feature column names, and preprocessor."""
    prep, model = _split_pipeline(pipeline)
    bg = prep.transform(add_engineered_features(background_df))
    feature_names = list(prep.get_feature_names_out())

    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, bg)
    else:
        explainer = shap.TreeExplainer(model)
    return explainer, feature_names, pipeline


def identify_strengths_and_risks(features: dict) -> dict:
    """Automated operational risk and strength detection (§1.5 & §2.3 Module 6)."""
    strengths: List[str] = []
    risks: List[Dict[str, str]] = []

    ot = str(features.get("OverTime", "No")).capitalize()
    wlb = float(features.get("WorkLifeBalance", 3))
    absent = float(features.get("AbsenteeismDays", 5))
    tenure_role = float(features.get("YearsInCurrentRole", 2))
    promos = float(features.get("PromotionsLast5Y", 1))
    eng = float(features.get("EngagementScore", 3))
    peer = float(features.get("PeerFeedbackScore", 3))
    mgr = float(features.get("ManagerRatingPrev", 3))
    deadline = float(features.get("DeadlineAdherence", 0.85))
    projects = float(features.get("ProjectsCompleted", 8))
    certs = float(features.get("Certifications", 2))

    # --- Operational Risk Rules ---
    # 1. Burnout Risk
    if ot == "Yes" and (wlb <= 2 or absent >= 8):
        risks.append({
            "code": "BURNOUT_RISK",
            "level": "High",
            "title": "Burnout Alert",
            "description": "Employee logs regular overtime alongside compromised work-life balance or elevated absenteeism.",
        })
    elif ot == "Yes" and wlb <= 3:
        risks.append({
            "code": "WORKLOAD_STRAIN",
            "level": "Medium",
            "title": "Workload Strain",
            "description": "Frequent overtime observed; review distribution of assignments to prevent fatigue.",
        })

    # 2. Career Stagnation Risk
    if tenure_role >= 4 and promos == 0:
        risks.append({
            "code": "CAREER_STAGNATION",
            "level": "High" if tenure_role >= 6 else "Medium",
            "title": "Career Stagnation Risk",
            "description": f"Tenured {int(tenure_role)} years in current role with 0 promotions over the last 5 years.",
        })

    # 3. Retention / Flight Risk
    if eng <= 2 or (mgr - peer) <= -2:
        risks.append({
            "code": "ATTRITION_RISK",
            "level": "High",
            "title": "Retention / Flight Risk",
            "description": "Low engagement signals or significant divergence between supervisor evaluation and peer feedback.",
        })

    # 4. Chronic Absenteeism
    if absent > 12:
        risks.append({
            "code": "CHRONIC_ABSENTEEISM",
            "level": "High",
            "title": "Chronic Absenteeism",
            "description": f"Recorded {int(absent)} absent days, exceeding organization threshold.",
        })

    # --- Employee Strengths ---
    if deadline >= 0.90:
        strengths.append(f"Exemplary Deadline Adherence ({deadline*100:.1f}%)")
    if projects >= 10:
        strengths.append(f"High-Volume Delivery ({int(projects)} Projects Completed)")
    if certs >= 3:
        strengths.append(f"Active Continuous Upskilling ({int(certs)} Professional Certifications)")
    if eng >= 4:
        strengths.append("High Cultural Engagement & Morale (Score >= 4/5)")
    if peer >= 4:
        strengths.append("Strong Peer Trust & Collaboration (Score >= 4/5)")
    if wlb >= 4 and absent <= 3:
        strengths.append("Sustainable Work Rhythm & Attendance Stability")

    return {
        "strengths": strengths,
        "risks": risks,
    }


def explain_single(features: dict, explainer, feature_names, pipeline, top_k: int = 8) -> dict:
    """Computes local SHAP attributions and blends with rule-based operational diagnostics."""
    prep, _ = _split_pipeline(pipeline)
    row = pd.DataFrame([features])
    x = prep.transform(add_engineered_features(row))

    shap_raw = explainer.shap_values(x)
    shap_values = _normalize_shap(shap_raw, len(CLASS_NAMES))
    proba = pipeline.predict_proba(row)[0]
    pred_idx = int(np.argmax(proba))
    contribs = shap_values[0, :, pred_idx]

    order = np.argsort(np.abs(contribs))[::-1][:top_k]
    shap_factors = [
        {
            "feature": feature_names[i].replace("num__", "").replace("cat__", ""),
            "shap_value": round(float(contribs[i]), 4),
            "impact": "positive" if contribs[i] > 0 else "negative",
            "direction": "increases" if contribs[i] > 0 else "decreases",
        }
        for i in order
    ]

    analysis = identify_strengths_and_risks(features)

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": round(float(proba[pred_idx]), 4),
        "top_contributions": shap_factors,
        "strengths": analysis["strengths"],
        "risks": analysis["risks"],
    }
