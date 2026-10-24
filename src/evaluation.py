"""
Chronos: Evaluation Metrics Module
Includes NASA C-MAPSS Asymmetric Scoring Function and standard diagnostic metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

def compute_nasa_scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the official NASA PHM Challenge asymmetric penalty score.
    Penalizes late predictions (overestimating remaining life, risking catastrophic failure)
    significantly more than early maintenance alerts.
    
    d_i = y_pred - y_true
    s_i = exp(-d_i / 13) - 1  if d_i < 0 (early)
    s_i = exp(d_i / 10) - 1   if d_i >= 0 (late)
    Score = sum(s_i)
    """
    diff = y_pred - y_true
    score = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(score))

def evaluate_rul_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluates RUL regression model with RMSE, MAE, R2, and NASA Score.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nasa_score = compute_nasa_scoring_function(y_true, y_pred)
    
    return {
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "R2_Score": round(float(r2), 4),
        "NASA_Score": round(float(nasa_score), 2)
    }

def evaluate_risk_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """
    Evaluates binary maintenance risk classifier.
    """
    metrics = {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred)), 4),
        "Recall": round(float(recall_score(y_true, y_pred)), 4),
        "F1_Score": round(float(f1_score(y_true, y_pred)), 4)
    }
    if y_prob is not None:
        metrics["ROC_AUC"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        
    return metrics
