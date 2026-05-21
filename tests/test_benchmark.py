"""Tests for benchmark suite components."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from data_loader import load_dataset, DATASETS
from models import get_all_models, get_classification_models, get_regression_models
from evaluation import evaluate_model, cross_validate_model


# ── Data loader tests ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", list(DATASETS.keys()))
def test_load_dataset(name):
    X_train, X_test, y_train, y_test, feature_names, target_names, task = load_dataset(name)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(feature_names) == X_train.shape[1]
    assert task in {"classification", "regression"}


def test_load_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_dataset("not_a_real_dataset")


def test_data_is_standardized():
    X_train, _, _, _, _, _, _ = load_dataset("breast_cancer")
    # StandardScaler makes mean ≈ 0, std ≈ 1 on training data
    assert abs(X_train.mean()) < 0.1
    assert 0.8 < X_train.std() < 1.2


# ── Model registry tests ──────────────────────────────────────────────────────

def test_classification_models_load():
    models = get_classification_models()
    assert len(models) >= 8  # at least the core sklearn ones
    for name, factory in models.items():
        model = factory()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


def test_regression_models_load():
    models = get_regression_models()
    assert len(models) >= 8
    for name, factory in models.items():
        model = factory()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


def test_quick_mode_skips_slow_models():
    full = get_classification_models(quick=False)
    quick = get_classification_models(quick=True)
    assert len(quick) < len(full)


def test_get_all_models_routes_by_task():
    cls_models = get_all_models("classification")
    reg_models = get_all_models("regression")
    assert "Logistic Regression" in cls_models
    assert "Ridge Regression" in reg_models


# ── Evaluation tests ──────────────────────────────────────────────────────────

def test_evaluate_classification():
    from sklearn.linear_model import LogisticRegression
    X_train, X_test, y_train, y_test, *_ = load_dataset("breast_cancer")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, task="classification")
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_evaluate_regression():
    from sklearn.linear_model import Ridge
    X_train, X_test, y_train, y_test, *_ = load_dataset("diabetes")
    model = Ridge()
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, task="regression")
    assert "r2" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics


def test_cross_validate_returns_mean_and_std():
    from sklearn.linear_model import LogisticRegression
    X_train, _, y_train, _, *_ = load_dataset("breast_cancer")
    cv = cross_validate_model(LogisticRegression(max_iter=1000), X_train, y_train, task="classification")
    assert "roc_auc_mean" in cv
    assert "roc_auc_std" in cv


# ── End-to-end smoke test ─────────────────────────────────────────────────────

def test_smoke_run_single_model():
    """Run a single model end to end to make sure the full pipeline works."""
    from sklearn.linear_model import LogisticRegression
    X_train, X_test, y_train, y_test, *_ = load_dataset("iris")
    model = LogisticRegression(max_iter=1000)
    cv = cross_validate_model(model, X_train, y_train, task="classification")
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, task="classification")
    assert metrics["accuracy"] > 0.8  # iris is easy, should always pass
