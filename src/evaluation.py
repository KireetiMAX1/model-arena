"""
Evaluation metrics + cross-validation utilities.
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold


def evaluate_model(model, X_test, y_test, task: str) -> dict:
    """Compute test-set metrics."""
    y_pred = model.predict(X_test)

    if task == "classification":
        n_classes = len(np.unique(y_test))
        avg = "binary" if n_classes == 2 else "macro"
        metrics = {
            "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
            "f1":        round(float(f1_score(y_test, y_pred, average=avg)), 4),
            "precision": round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4),
        }
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
                if n_classes == 2:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob[:, 1])), 4)
                    metrics["avg_precision"] = round(float(average_precision_score(y_test, y_prob[:, 1])), 4)
                else:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")), 4)
            except Exception:
                metrics["roc_auc"] = np.nan
        return metrics

    elif task == "regression":
        return {
            "r2":   round(float(r2_score(y_test, y_pred)), 4),
            "mae":  round(float(mean_absolute_error(y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        }


def cross_validate_model(model, X_train, y_train, task: str, cv: int = 5) -> dict:
    """Stratified k-fold CV; returns mean + std for the primary metric."""
    if task == "classification":
        scoring = "roc_auc" if len(np.unique(y_train)) == 2 else "roc_auc_ovr_weighted"
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        key = "roc_auc"
    else:
        scoring = "r2"
        splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
        key = "r2"

    try:
        scores = cross_val_score(model, X_train, y_train, cv=splitter, scoring=scoring, n_jobs=-1)
        return {
            f"{key}_mean": round(float(scores.mean()), 4),
            f"{key}_std":  round(float(scores.std()), 4),
        }
    except Exception:
        return {f"{key}_mean": np.nan, f"{key}_std": np.nan}
