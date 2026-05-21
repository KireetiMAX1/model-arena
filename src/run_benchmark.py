"""
ML Benchmark Suite — runs all models, generates leaderboard, plots, and reports.

Run:
    python src/run_benchmark.py
    python src/run_benchmark.py --dataset wine --quick
"""

from __future__ import annotations
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_dataset, DATASETS
from models import get_all_models
from evaluation import evaluate_model, cross_validate_model
from reporting import (
    generate_leaderboard_md,
    generate_comparison_plot,
    generate_confusion_matrix_grid,
    generate_feature_importance_plot,
    update_readme_leaderboard,
    save_results_json,
)


def run_benchmark(dataset_name: str = "breast_cancer", quick: bool = False, seed: int = 42):
    """Run all models against a dataset, save results + plots."""
    print(f"\n{'=' * 60}")
    print(f"🏆 ML Benchmark Suite — Dataset: {dataset_name}")
    print(f"{'=' * 60}\n")

    # Load data
    X_train, X_test, y_train, y_test, feature_names, target_names, task = load_dataset(dataset_name)
    print(f"📊 Dataset: {dataset_name}")
    print(f"  Task           : {task}")
    print(f"  Train samples  : {len(X_train)}")
    print(f"  Test samples   : {len(X_test)}")
    print(f"  Features       : {len(feature_names)}")
    print(f"  Classes        : {len(target_names) if target_names is not None else 'N/A'}\n")

    # Get all models
    models = get_all_models(task=task, quick=quick)
    print(f"🤖 Models to benchmark: {len(models)}\n")

    # Run each model
    results = []
    for name, model_factory in models.items():
        print(f"▶ Running {name}...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            model = model_factory()
            cv_metrics = cross_validate_model(model, X_train, y_train, task=task, cv=5)
            model.fit(X_train, y_train)
            test_metrics = evaluate_model(model, X_test, y_test, task=task)
            elapsed = round(time.perf_counter() - t0, 2)

            row = {
                "model": name,
                **{f"cv_{k}": v for k, v in cv_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
                "train_time_sec": elapsed,
            }
            results.append(row)
            primary = "test_roc_auc" if task == "classification" else "test_r2"
            score = row.get(primary, row.get("test_accuracy", 0))
            print(f"✅ {primary}={score:.4f} ({elapsed}s)")
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({"model": name, "error": str(e), "train_time_sec": 0})

    # Save results
    results_df = pd.DataFrame(results).sort_values(
        by="test_roc_auc" if task == "classification" else "test_r2",
        ascending=False, na_position="last",
    ).reset_index(drop=True)
    results_df.insert(0, "rank", range(1, len(results_df) + 1))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    metadata = {
        "dataset": dataset_name,
        "task": task,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_names),
        "timestamp": timestamp,
        "git_sha": _get_git_sha(),
    }

    Path("results").mkdir(exist_ok=True)
    results_df.to_csv(f"results/{dataset_name}_results.csv", index=False)
    save_results_json(results, metadata, f"results/{dataset_name}_results.json")

    # Generate plots + leaderboard
    print(f"\n📈 Generating plots & leaderboard...")
    generate_comparison_plot(results_df, dataset_name, task)
    if task == "classification":
        generate_confusion_matrix_grid(results, X_test, y_test, target_names, dataset_name)
    generate_feature_importance_plot(results, feature_names, dataset_name)

    leaderboard_md = generate_leaderboard_md(results_df, metadata)
    Path(f"results/{dataset_name}_leaderboard.md").write_text(leaderboard_md)

    # Update main README with current best results
    update_readme_leaderboard(results_df, metadata)

    print(f"\n{'=' * 60}")
    print(f"✅ Benchmark complete! Results saved to results/")
    print(f"{'=' * 60}\n")
    print(results_df.to_string(index=False))
    return results_df


def _get_git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="breast_cancer", choices=list(DATASETS.keys()))
    parser.add_argument("--quick", action="store_true", help="Skip slow models for fast CI runs")
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    args = parser.parse_args()

    if args.all:
        for ds in DATASETS:
            run_benchmark(ds, quick=args.quick)
    else:
        run_benchmark(args.dataset, quick=args.quick)


if __name__ == "__main__":
    main()
