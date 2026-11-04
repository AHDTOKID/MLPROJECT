"""ModelBundle container for self-contained model persistence (§4.4).

Supports direct execution after unpickling via standard `pickle.load()`:
- `bundle.predict(X)`
- `bundle.predict_proba(X)`
- `bundle.predict_one(dict)`
- `bundle.predict_many(DataFrame or list of dicts)`
"""
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

from .config import CLASS_NAMES

COLUMN_ALIASES = {
    "age": "Age",
    "gender": "Gender",
    "sex": "Gender",
    "maritalstatus": "MaritalStatus",
    "marital_status": "MaritalStatus",
    "marital": "MaritalStatus",
    "distancefromhome": "DistanceFromHome",
    "distance_from_home": "DistanceFromHome",
    "distance": "DistanceFromHome",
    "education": "Education",
    "educationlevel": "Education",
    "education_level": "Education",
    "department": "Department",
    "dept": "Department",
    "jobrole": "JobRole",
    "job_role": "JobRole",
    "role": "JobRole",
    "yearsincurrentrole": "YearsInCurrentRole",
    "years_in_current_role": "YearsInCurrentRole",
    "role_years": "YearsInCurrentRole",
    "overtime": "OverTime",
    "over_time": "OverTime",
    "monthlyincome": "MonthlyIncome",
    "monthly_income": "MonthlyIncome",
    "income": "MonthlyIncome",
    "salary": "MonthlyIncome",
    "totalworkingyears": "TotalWorkingYears",
    "total_working_years": "TotalWorkingYears",
    "total_experience": "TotalWorkingYears",
    "yearsatcompany": "YearsAtCompany",
    "years_at_company": "YearsAtCompany",
    "tenure": "YearsAtCompany",
    "numcompaniesworked": "NumCompaniesWorked",
    "num_companies_worked": "NumCompaniesWorked",
    "companies_worked": "NumCompaniesWorked",
    "promotionslast5y": "PromotionsLast5Y",
    "promotions_last_5y": "PromotionsLast5Y",
    "promotions": "PromotionsLast5Y",
    "engagementscore": "EngagementScore",
    "engagement_score": "EngagementScore",
    "engagement": "EngagementScore",
    "worklifebalance": "WorkLifeBalance",
    "work_life_balance": "WorkLifeBalance",
    "wlb": "WorkLifeBalance",
    "peerfeedbackscore": "PeerFeedbackScore",
    "peer_feedback_score": "PeerFeedbackScore",
    "peer_feedback": "PeerFeedbackScore",
    "managerratingprev": "ManagerRatingPrev",
    "manager_rating_prev": "ManagerRatingPrev",
    "manager_rating": "ManagerRatingPrev",
    "projectscompleted": "ProjectsCompleted",
    "projects_completed": "ProjectsCompleted",
    "projects": "ProjectsCompleted",
    "deadlineadherence": "DeadlineAdherence",
    "deadline_adherence": "DeadlineAdherence",
    "deadline": "DeadlineAdherence",
    "absenteeismdays": "AbsenteeismDays",
    "absenteeism_days": "AbsenteeismDays",
    "absenteeism": "AbsenteeismDays",
    "certifications": "Certifications",
}

DEFAULT_VALUES = {
    "Age": 35.0,
    "Gender": "Male",
    "MaritalStatus": "Married",
    "DistanceFromHome": 8.0,
    "Education": 3.0,
    "Department": "Research & Development",
    "JobRole": "Developer",
    "YearsInCurrentRole": 3.0,
    "OverTime": "No",
    "MonthlyIncome": 6500.0,
    "TotalWorkingYears": 10.0,
    "YearsAtCompany": 5.0,
    "NumCompaniesWorked": 2.0,
    "PromotionsLast5Y": 1.0,
    "EngagementScore": 3.0,
    "WorkLifeBalance": 3.0,
    "PeerFeedbackScore": 3.0,
    "ManagerRatingPrev": 3.0,
    "ProjectsCompleted": 8.0,
    "DeadlineAdherence": 0.85,
    "AbsenteeismDays": 4.0,
    "Certifications": 2.0,
}


def normalize_columns_and_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans column headers and fills missing schema columns with sensible defaults."""
    df = df.copy()
    # Strip whitespace and UTF-8 BOM
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    rename_map = {}
    for col in df.columns:
        norm = col.lower().replace(" ", "").replace("_", "").replace("-", "")
        if norm in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[norm]
        elif col.lower() in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[col.lower()]
    df = df.rename(columns=rename_map)

    # Impute any missing expected columns
    for col, default_val in DEFAULT_VALUES.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            # Handle NaN / None inside existing columns
            df[col] = df[col].fillna(default_val)

    return df


class ModelBundle:
    """Production ModelBundle conforming to Section 4.4 specifications."""

    def __init__(self, pipeline: Any, class_names: List[str] = None, metadata: Dict[str, Any] = None):
        self.pipeline = pipeline
        self.class_names = list(class_names or CLASS_NAMES)
        self.metadata = metadata or {}

    def _ensure_dataframe(self, X: Union[pd.DataFrame, dict, list]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        if "PerformanceRating" in df.columns:
            df = df.drop(columns=["PerformanceRating"])

        df = normalize_columns_and_defaults(df)
        return df

    def predict(self, X: Union[pd.DataFrame, dict, list]) -> np.ndarray:
        """Standard scikit-learn compatible predict."""
        df = self._ensure_dataframe(X)
        return self.pipeline.predict(df)

    def predict_proba(self, X: Union[pd.DataFrame, dict, list]) -> np.ndarray:
        """Standard scikit-learn compatible predict_proba."""
        df = self._ensure_dataframe(X)
        return self.pipeline.predict_proba(df)

    def predict_one(self, features: dict) -> dict:
        """Single employee inference returning predicted class, confidence, and probability map."""
        df = self._ensure_dataframe(features)
        proba = self.pipeline.predict_proba(df)[0]
        model_classes = list(getattr(self.pipeline, "classes_", self.class_names))
        prob_map = {cls_name: float(p) for cls_name, p in zip(model_classes, proba)}
        pred_label = max(prob_map, key=prob_map.get)
        return {
            "predicted_class": pred_label,
            "confidence": float(prob_map[pred_label]),
            "probabilities": {name: float(prob_map.get(name, 0.0)) for name in self.class_names},
        }

    def predict_many(self, records: Union[pd.DataFrame, list]) -> list:
        """Batch employee inference returning structured list of predictions."""
        df = self._ensure_dataframe(records)
        probas = self.pipeline.predict_proba(df)
        model_classes = list(getattr(self.pipeline, "classes_", self.class_names))
        results = []
        for proba in probas:
            prob_map = {cls_name: float(p) for cls_name, p in zip(model_classes, proba)}
            pred_label = max(prob_map, key=prob_map.get)
            results.append({
                "predicted_class": pred_label,
                "confidence": float(prob_map[pred_label]),
                "probabilities": {name: float(prob_map.get(name, 0.0)) for name in self.class_names},
            })
        return results
