"""Thin inference wrapper around the persisted ModelBundle and scikit-learn pipeline."""
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union
import joblib
import pandas as pd

from .config import CLASS_NAMES, MODEL_DIR
from .model_bundle import ModelBundle


class PerformancePredictor:
    """Production inference interface (§2.3 Module 5)."""

    def __init__(self, model_path: Union[str, Path] = None):
        target = Path(model_path) if model_path else (MODEL_DIR / "performance_model.pkl")
        if not target.exists():
            raise FileNotFoundError(
                f"Trained model not found at {target}. Please execute 'python -m src.train' first."
            )

        with open(target, "rb") as f:
            loaded = pickle.load(f)

        if isinstance(loaded, ModelBundle):
            self.bundle = loaded
            self.pipeline = loaded.pipeline
        else:
            self.pipeline = loaded
            self.bundle = ModelBundle(pipeline=self.pipeline, class_names=CLASS_NAMES)

    def predict_one(self, features: dict) -> dict:
        return self.bundle.predict_one(features)

    def predict_many(self, df_or_records: Union[pd.DataFrame, list]) -> list:
        return self.bundle.predict_many(df_or_records)
