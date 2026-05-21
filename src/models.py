"""
Model registry — all models benchmarked by the suite.

Add new models here and they're automatically included in the leaderboard.
"""

from __future__ import annotations
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

SEED = 42


def get_classification_models(quick: bool = False) -> dict:
    models = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree":       lambda: DecisionTreeClassifier(max_depth=10, random_state=SEED),
        "Random Forest":       lambda: RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1),
        "Gradient Boosting":   lambda: GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=SEED),
        "Extra Trees":         lambda: ExtraTreesClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "K-Nearest Neighbors": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Naive Bayes":         lambda: GaussianNB(),
        "SVM (RBF)":           lambda: SVC(kernel="rbf", probability=True, random_state=SEED),
        "Neural Network":      lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED),
    }

    # Optional XGBoost / LightGBM (skip if not installed)
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = lambda: XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            eval_metric="logloss", random_state=SEED, n_jobs=-1, verbosity=0,
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = lambda: LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=SEED, n_jobs=1, verbosity=-1,
        )
    except ImportError:
        pass

    if quick:
        slow_models = {"SVM (RBF)", "Neural Network"}
        models = {k: v for k, v in models.items() if k not in slow_models}

    return models


def get_regression_models(quick: bool = False) -> dict:
    models = {
        "Ridge Regression":    lambda: Ridge(alpha=1.0, random_state=SEED),
        "ElasticNet":          lambda: ElasticNet(alpha=0.1, random_state=SEED),
        "Decision Tree":       lambda: DecisionTreeRegressor(max_depth=10, random_state=SEED),
        "Random Forest":       lambda: RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1),
        "Gradient Boosting":   lambda: GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=SEED),
        "Extra Trees":         lambda: ExtraTreesRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
        "K-Nearest Neighbors": lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "SVM (RBF)":           lambda: SVR(kernel="rbf"),
        "Neural Network":      lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED),
    }

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = lambda: XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=SEED, n_jobs=-1, verbosity=0,
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = lambda: LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=SEED, n_jobs=1, verbosity=-1,
        )
    except ImportError:
        pass

    if quick:
        slow_models = {"SVM (RBF)", "Neural Network"}
        models = {k: v for k, v in models.items() if k not in slow_models}

    return models


def get_all_models(task: str, quick: bool = False) -> dict:
    if task == "classification":
        return get_classification_models(quick=quick)
    elif task == "regression":
        return get_regression_models(quick=quick)
    else:
        raise ValueError(f"Unknown task: {task}")
