"""
Reporting — generates leaderboard table, comparison plots, README updates.
"""

from __future__ import annotations
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)


def _format_metric(val) -> str:
    if pd.isna(val) or val is None:
        return "—"
    if isinstance(val, (int, np.integer)):
        return str(val)
    return f"{val:.4f}"


def _medal(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}")


def generate_leaderboard_md(df: pd.DataFrame, metadata: dict) -> str:
    """Generate markdown leaderboard table from results."""
    task = metadata["task"]
    primary = "test_roc_auc" if task == "classification" else "test_r2"

    if task == "classification":
        cols = ["rank", "model", "test_accuracy", "test_f1", "test_roc_auc",
                "cv_roc_auc_mean", "train_time_sec"]
        headers = ["Rank", "Model", "Accuracy", "F1", "ROC-AUC", "CV ROC-AUC", "Time (s)"]
    else:
        cols = ["rank", "model", "test_r2", "test_mae", "test_rmse",
                "cv_r2_mean", "train_time_sec"]
        headers = ["Rank", "Model", "R²", "MAE", "RMSE", "CV R²", "Time (s)"]

    df = df.copy()
    available_cols = [c for c in cols if c in df.columns]
    available_headers = [headers[cols.index(c)] for c in available_cols]

    lines = []
    lines.append(f"## 🏆 Leaderboard — {metadata['dataset'].replace('_', ' ').title()}\n")
    lines.append(f"**Task**: {task} | **Train**: {metadata['n_train']} | **Test**: {metadata['n_test']} | **Features**: {metadata['n_features']}")
    lines.append(f"**Updated**: {metadata['timestamp']} (commit `{metadata['git_sha']}`)\n")

    # Build markdown table
    lines.append("| " + " | ".join(available_headers) + " |")
    lines.append("|" + "|".join(["---"] * len(available_headers)) + "|")

    for _, row in df.iterrows():
        row_vals = []
        for c in available_cols:
            v = row.get(c)
            if c == "rank":
                row_vals.append(_medal(int(v)))
            elif c == "model":
                rank = int(row["rank"])
                model_str = f"**{v}**" if rank == 1 else str(v)
                row_vals.append(model_str)
            else:
                row_vals.append(_format_metric(v))
        lines.append("| " + " | ".join(row_vals) + " |")

    lines.append(f"\n_Primary metric: **{primary}** · Best model wins 🥇_\n")
    return "\n".join(lines)


def generate_comparison_plot(df: pd.DataFrame, dataset_name: str, task: str):
    """Bar chart comparing all models on the primary metric."""
    primary = "test_roc_auc" if task == "classification" else "test_r2"
    if primary not in df.columns:
        primary = "test_accuracy" if task == "classification" else "test_r2"

    df = df.dropna(subset=[primary]).sort_values(primary, ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(df))))
    colors = ["#2ecc71" if i == len(df) - 1 else "#3498db" for i in range(len(df))]
    bars = ax.barh(df["model"], df[primary], color=colors, edgecolor="white", linewidth=1)

    for bar, val in zip(bars, df[primary]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    ax.set_xlabel(primary.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"Model Comparison — {dataset_name.replace('_', ' ').title()}",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(0, max(df[primary]) * 1.12)
    plt.tight_layout()
    out_path = ASSETS_DIR / f"comparison_{dataset_name}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  → Saved {out_path}")


def generate_confusion_matrix_grid(results: list, X_test, y_test, target_names, dataset_name: str):
    """Confusion matrix grid for top-4 classification models."""
    from sklearn.metrics import confusion_matrix
    from models import get_all_models

    # Get top 4 by test_roc_auc or test_accuracy
    df = pd.DataFrame(results)
    sort_key = "test_roc_auc" if "test_roc_auc" in df.columns else "test_accuracy"
    if sort_key not in df.columns:
        return
    top = df.dropna(subset=[sort_key]).nlargest(4, sort_key)

    models_dict = get_all_models("classification")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for i, (_, row) in enumerate(top.iterrows()):
        if i >= 4: break
        name = row["model"]
        try:
            model = models_dict[name]()
            from data_loader import load_dataset
            X_train, _, y_train, _, _, _, _ = load_dataset(dataset_name)
            model.fit(X_train, y_train)
            cm = confusion_matrix(y_test, model.predict(X_test))
            labels = target_names if target_names is not None else range(cm.shape[0])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels, yticklabels=labels,
                        ax=axes[i], cbar=False)
            axes[i].set_title(f"{name}\n({sort_key}={row[sort_key]:.4f})", fontsize=11)
            axes[i].set_xlabel("Predicted"); axes[i].set_ylabel("Actual")
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Failed: {e}", ha="center", va="center")
            axes[i].set_axis_off()

    for j in range(len(top), 4):
        axes[j].set_axis_off()

    plt.suptitle(f"Confusion Matrices — Top 4 Models ({dataset_name})",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    out_path = ASSETS_DIR / f"confusion_{dataset_name}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  → Saved {out_path}")


def generate_feature_importance_plot(results: list, feature_names: list, dataset_name: str):
    """Feature importance from the best tree-based model."""
    from data_loader import load_dataset
    from models import get_all_models

    df = pd.DataFrame(results)
    sort_key = "test_roc_auc" if "test_roc_auc" in df.columns else "test_r2"
    if sort_key not in df.columns:
        return

    tree_models = {"Random Forest", "Gradient Boosting", "Extra Trees", "XGBoost", "LightGBM"}
    eligible = df[df["model"].isin(tree_models)].dropna(subset=[sort_key])
    if eligible.empty:
        return

    best_name = eligible.nlargest(1, sort_key)["model"].iloc[0]
    models_dict = get_all_models("classification") | get_all_models("regression")
    if best_name not in models_dict:
        return

    X_train, _, y_train, _, _, _, _ = load_dataset(dataset_name)
    model = models_dict[best_name]()
    model.fit(X_train, y_train)

    if not hasattr(model, "feature_importances_"):
        return

    importances = pd.Series(model.feature_importances_, index=feature_names).nlargest(15)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(importances))))
    ax.barh(importances.index[::-1], importances.values[::-1],
            color="#9b59b6", edgecolor="white")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Top 15 Features — {best_name} on {dataset_name}",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    out_path = ASSETS_DIR / f"features_{dataset_name}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  → Saved {out_path}")


def save_results_json(results: list, metadata: dict, path: str):
    payload = {"metadata": metadata, "results": results}
    Path(path).write_text(json.dumps(payload, indent=2, default=str))


def update_readme_leaderboard(df: pd.DataFrame, metadata: dict):
    """Replace the leaderboard block in README.md between markers."""
    readme_path = Path("README.md")
    if not readme_path.exists():
        return

    leaderboard_md = generate_leaderboard_md(df, metadata)
    dataset = metadata["dataset"]
    plot_block = (
        f"\n![Comparison]({ASSETS_DIR}/comparison_{dataset}.png)\n"
    )
    if metadata["task"] == "classification":
        plot_block += f"![Confusion]({ASSETS_DIR}/confusion_{dataset}.png)\n"
    plot_block += f"![Features]({ASSETS_DIR}/features_{dataset}.png)\n"

    block = leaderboard_md + plot_block

    pattern = re.compile(r"<!-- LEADERBOARD_START -->.*?<!-- LEADERBOARD_END -->", re.DOTALL)
    content = readme_path.read_text()
    new_block = f"<!-- LEADERBOARD_START -->\n{block}\n<!-- LEADERBOARD_END -->"
    if pattern.search(content):
        updated = pattern.sub(new_block, content)
    else:
        updated = content + "\n\n" + new_block + "\n"
    readme_path.write_text(updated)
    print(f"  → Updated README.md leaderboard block")
