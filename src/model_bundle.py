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
        idx = int(np.argmax(proba))
        pred_label = self.class_names[idx]
        return {
            "predicted_class": pred_label,
            "confidence": float(proba[idx]),
            "probabilities": {name: float(proba[i]) for i, name in enumerate(self.class_names)},
        }

    def predict_many(self, records: Union[pd.DataFrame, list]) -> list:
        """Batch employee inference returning structured list of predictions."""
        df = self._ensure_dataframe(records)
        probas = self.pipeline.predict_proba(df)
        results = []
        for proba in probas:
            idx = int(np.argmax(proba))
            pred_label = self.class_names[idx]
            results.append({
                "predicted_class": pred_label,
                "confidence": float(proba[idx]),
                "probabilities": {name: float(proba[i]) for i, name in enumerate(self.class_names)},
            })
        return results
