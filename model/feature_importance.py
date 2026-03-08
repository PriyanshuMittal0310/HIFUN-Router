"""Feature importance analysis using SHAP and built-in feature importances."""

import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import (
    CLASSIFIER_PATH, MODEL_DIR, LABELED_RUNS_CSV, RESULTS_DIR,
)
from features.feature_extractor import FEATURE_NAMES


def generate_shap_report(
    model=None,
    X_train: np.ndarray = None,
    feature_names: list = None,
    output_dir: str = None,
):
    """Generate SHAP feature importance report with visualizations.

    Produces:
      - shap_summary.pdf: SHAP beeswarm summary plot
      - shap_bar.pdf: Mean absolute SHAP bar chart
      - shap_values.json: Numeric SHAP importance per feature
    """
    if output_dir is None:
        output_dir = os.path.join(MODEL_DIR, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    if model is None:
        model = joblib.load(CLASSIFIER_PATH)

    if X_train is None or feature_names is None:
        df = pd.read_csv(LABELED_RUNS_CSV)
        feature_names = FEATURE_NAMES
        X_train = df[feature_names].values.astype(np.float32)

    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Extract class-1 (GRAPH) SHAP values for visualization
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv_plot = shap_values[:, :, 1]
    elif isinstance(shap_values, list) and len(shap_values) > 1:
        sv_plot = shap_values[1]
    else:
        sv_plot = shap_values

    # Summary beeswarm plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        sv_plot, X_train,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot of mean |SHAP|
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        sv_plot, X_train,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_bar.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Numeric importance values
    mean_abs_shap = np.abs(sv_plot).mean(axis=0)
    importance_dict = dict(zip(feature_names, mean_abs_shap.tolist()))
    sorted_importance = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )

    with open(os.path.join(output_dir, "shap_values.json"), "w") as f:
        json.dump(sorted_importance, f, indent=2)

    print(f"SHAP analysis saved to {output_dir}/")
    print("Top-5 features by mean |SHAP|:")
    for i, (feat, val) in enumerate(sorted_importance.items()):
        if i >= 5:
            break
        print(f"  {i+1}. {feat}: {val:.4f}")

    return sorted_importance, shap_values


def generate_builtin_importance(
    model=None,
    feature_names: list = None,
    output_dir: str = None,
):
    """Generate feature importance from model's built-in importance scores."""
    if output_dir is None:
        output_dir = os.path.join(MODEL_DIR, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    if model is None:
        model = joblib.load(CLASSIFIER_PATH)
    if feature_names is None:
        feature_names = FEATURE_NAMES

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[sorted_idx], color="steelblue")
    plt.xticks(
        range(len(importances)),
        [feature_names[i] for i in sorted_idx],
        rotation=45,
        ha="right",
    )
    plt.title("Built-in Feature Importance (XGBoost gain)")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "builtin_importance.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "builtin_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    importance_dict = dict(zip(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx].tolist(),
    ))
    with open(os.path.join(output_dir, "builtin_importance.json"), "w") as f:
        json.dump(importance_dict, f, indent=2)

    print(f"Built-in importance saved to {output_dir}/")
    return importance_dict


def run_ablation_study(
    model_class=None,
    X: np.ndarray = None,
    y: np.ndarray = None,
    feature_names: list = None,
    output_dir: str = None,
):
    """Ablation study: retrain model with each feature removed to measure impact."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    if output_dir is None:
        output_dir = os.path.join(MODEL_DIR, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    if X is None or y is None:
        df = pd.read_csv(LABELED_RUNS_CSV)
        feature_names = FEATURE_NAMES
        X = df[feature_names].values.astype(np.float32)
        y = (df["label"] == "GRAPH").astype(int).values

    if model_class is None:
        import xgboost as xgb
        model_class = lambda: xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            random_state=42, use_label_encoder=False, eval_metric="logloss",
        )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline: all features
    baseline_model = model_class()
    baseline_f1 = cross_val_score(baseline_model, X, y, cv=skf, scoring="f1").mean()

    ablation_results = {"baseline_f1": float(baseline_f1), "features": {}}

    for i, feat in enumerate(feature_names):
        X_ablated = np.delete(X, i, axis=1)
        m = model_class()
        f1 = cross_val_score(m, X_ablated, y, cv=skf, scoring="f1").mean()
        drop = baseline_f1 - f1
        ablation_results["features"][feat] = {
            "f1_without": float(f1),
            "f1_drop": float(drop),
        }

    # Save
    with open(os.path.join(output_dir, "ablation_results.json"), "w") as f:
        json.dump(ablation_results, f, indent=2)

    # Plot
    features = list(ablation_results["features"].keys())
    drops = [ablation_results["features"][f]["f1_drop"] for f in features]
    sorted_idx = np.argsort(drops)[::-1]

    plt.figure(figsize=(12, 6))
    colors = ["#d9534f" if drops[i] > 0 else "#5cb85c" for i in sorted_idx]
    plt.bar(range(len(features)), [drops[i] for i in sorted_idx], color=colors)
    plt.xticks(
        range(len(features)),
        [features[i] for i in sorted_idx],
        rotation=45,
        ha="right",
    )
    plt.axhline(y=0, color="black", linewidth=0.5)
    plt.title("Ablation Study: F1 Drop When Feature Removed")
    plt.ylabel("F1 Drop (positive = feature helps)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_study.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "ablation_study.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Ablation study saved to {output_dir}/")
    return ablation_results


if __name__ == "__main__":
    print("=== SHAP Analysis ===")
    generate_shap_report()

    print("\n=== Built-in Feature Importance ===")
    generate_builtin_importance()

    print("\n=== Ablation Study ===")
    run_ablation_study()
