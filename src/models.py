"""
Chronos: Model Architectures Module
Implements dual-task predictive maintenance algorithms:
1. Continuous RUL Regression
2. Early-Warning Failure Classification
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression

class RULRegressionModel:
    def __init__(self, model_type: str = "random_forest", **kwargs):
        self.model_type = model_type
        if model_type == "random_forest":
            params = {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5, "random_state": 42, "n_jobs": -1}
            params.update(kwargs)
            self.model = RandomForestRegressor(**params)
        elif model_type == "gradient_boosting":
            params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.08, "random_state": 42}
            params.update(kwargs)
            self.model = GradientBoostingRegressor(**params)
        elif model_type == "ridge":
            params = {"alpha": 10.0, "random_state": 42}
            params.update(kwargs)
            self.model = Ridge(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.maximum(preds, 0)  # RUL cannot be negative

class FailureClassificationModel:
    def __init__(self, model_type: str = "random_forest", **kwargs):
        self.model_type = model_type
        if model_type == "random_forest":
            params = {"n_estimators": 100, "max_depth": 12, "min_samples_split": 4, 
                      "class_weight": "balanced", "random_state": 42, "n_jobs": -1}
            params.update(kwargs)
            self.model = RandomForestClassifier(**params)
        elif model_type == "logistic_regression":
            params = {"max_iter": 1000, "class_weight": "balanced", "random_state": 42}
            params.update(kwargs)
            self.model = LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

def save_model(model_obj, filepath: str):
    """Saves serialized model artifact using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model_obj, filepath)

def load_model(filepath: str):
    """Loads serialized model artifact."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    return joblib.load(filepath)
