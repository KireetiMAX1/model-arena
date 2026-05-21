"""
Dataset loader — bundled sklearn datasets for zero-network CI runs.
"""

from __future__ import annotations
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

DATASETS = {
    "breast_cancer": {"task": "classification", "loader": datasets.load_breast_cancer},
    "wine":          {"task": "classification", "loader": datasets.load_wine},
    "iris":          {"task": "classification", "loader": datasets.load_iris},
    "diabetes":      {"task": "regression",     "loader": datasets.load_diabetes},
    "digits":        {"task": "classification", "loader": datasets.load_digits},
}


def load_dataset(name: str, test_size: float = 0.2, seed: int = 42):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    config = DATASETS[name]
    data = config["loader"]()
    X, y = data.data, data.target
    feature_names = list(data.feature_names) if hasattr(data, "feature_names") else [f"f{i}" for i in range(X.shape[1])]
    target_names = list(data.target_names) if hasattr(data, "target_names") else None

    stratify = y if config["task"] == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=seed
    )

    # Standardize for fair model comparison
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names, target_names, config["task"]
