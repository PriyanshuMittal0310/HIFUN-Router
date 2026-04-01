"""Model training pipeline: Decision Tree + XGBoost classifiers for SQL/GRAPH routing."""

import json
import os
import sys
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import LABELED_RUNS_CSV, MODEL_DIR, CLASSIFIER_PATH, FEATURE_SCHEMA_PATH
from features.feature_extractor import FEATURE_NAMES


def _default_labeled_path() -> str:
    """Prefer fixed balanced train split, then fallback datasets."""
    candidates = [
        os.path.join(PROJECT_ROOT, "training_data", "fixed_train_balanced.csv"),
        os.path.join(PROJECT_ROOT, "training_data", "fixed_train_base.csv"),
        os.path.join(PROJECT_ROOT, "training_data", "real_labeled_runs_balanced.csv"),
        os.path.join(PROJECT_ROOT, "training_data", "real_labeled_runs.csv"),
        LABELED_RUNS_CSV,
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return LABELED_RUNS_CSV


def load_data(labeled_data_path: str):
    """Load labeled training data and return (X, y, df)."""
    df = pd.read_csv(labeled_data_path)
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = (df["label"] == "GRAPH").astype(int).values  # 0=SQL, 1=GRAPH
    return X, y, df


def train_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    class_weight: str | None = "balanced",
):
    """Train a Decision Tree classifier with cross-validation."""
    dt = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        class_weight=class_weight,
    )
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    f1_scores = cross_val_score(dt, X, y, cv=skf, scoring="f1")
    acc_scores = cross_val_score(dt, X, y, cv=skf, scoring="accuracy")

    # Fit on full data for final model
    dt.fit(X, y)

    return dt, {
        "model": "DecisionTree",
        "cv_f1_mean": float(f1_scores.mean()),
        "cv_f1_std": float(f1_scores.std()),
        "cv_accuracy_mean": float(acc_scores.mean()),
        "cv_accuracy_std": float(acc_scores.std()),
    }


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    use_scale_pos_weight: bool = True,
):
    """Train an XGBoost classifier with cross-validation."""
    sql_count = int((y == 0).sum())
    graph_count = int((y == 1).sum())
    scale_pos_weight = float(sql_count / max(graph_count, 1)) if use_scale_pos_weight else 1.0

    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    f1_scores = cross_val_score(xgb_clf, X, y, cv=skf, scoring="f1")
    acc_scores = cross_val_score(xgb_clf, X, y, cv=skf, scoring="accuracy")

    # Fit on full data for final model
    xgb_clf.fit(X, y)

    return xgb_clf, {
        "model": "XGBoost",
        "cv_f1_mean": float(f1_scores.mean()),
        "cv_f1_std": float(f1_scores.std()),
        "cv_accuracy_mean": float(acc_scores.mean()),
        "cv_accuracy_std": float(acc_scores.std()),
        "scale_pos_weight": float(scale_pos_weight),
    }


def evaluate_model(model, X: np.ndarray, y: np.ndarray, name: str) -> Dict:
    """Evaluate a fitted model and return metrics dict."""
    y_pred = model.predict(X)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(
            y, y_pred, target_names=["SQL", "GRAPH"], output_dict=True
        ),
    }


def train(
    labeled_data_path: str | None = None,
    model_out: str = CLASSIFIER_PATH,
    cv_folds: int = 5,
    holdout_fraction: float = 0.2,
    class_weight: str | None = "balanced",
    use_scale_pos_weight: bool = True,
) -> Dict:
    """Full training pipeline: load data, train both models, save best one.

    Returns dict with training results and metrics.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    labeled_path = labeled_data_path or _default_labeled_path()
    X, y, df = load_data(labeled_path)
    print(f"Loaded {len(df)} samples ({y.sum()} GRAPH, {len(y) - y.sum()} SQL)")

    # Create a holdout split for honest reporting in addition to CV metrics.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=holdout_fraction,
        stratify=y,
        random_state=42,
    )

    # Train both models
    dt_model, dt_cv = train_decision_tree(
        X_train,
        y_train,
        cv_folds,
        class_weight=class_weight,
    )
    print(f"Decision Tree CV F1: {dt_cv['cv_f1_mean']:.3f} +/- {dt_cv['cv_f1_std']:.3f}")

    xgb_model, xgb_cv = train_xgboost(
        X_train,
        y_train,
        cv_folds,
        use_scale_pos_weight=use_scale_pos_weight,
    )
    print(f"XGBoost CV F1:       {xgb_cv['cv_f1_mean']:.3f} +/- {xgb_cv['cv_f1_std']:.3f}")

    # Select best model
    if xgb_cv["cv_f1_mean"] >= dt_cv["cv_f1_mean"]:
        best_model, best_name = xgb_model, "XGBoost"
    else:
        best_model, best_name = dt_model, "DecisionTree"
    print(f"Best model: {best_name}")

    # Save best model
    joblib.dump(best_model, model_out)
    print(f"Model saved to {model_out}")

    # Save Decision Tree separately for interpretability
    dt_path = model_out.replace(".pkl", "_dt.pkl")
    joblib.dump(dt_model, dt_path)

    # Save feature schema alongside model
    schema = {
        "feature_names": FEATURE_NAMES,
        "num_features": len(FEATURE_NAMES),
        "label_map": {0: "SQL", 1: "GRAPH"},
    }
    with open(FEATURE_SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2)

    # Full evaluation
    dt_eval = evaluate_model(dt_model, X_test, y_test, "DecisionTree")
    xgb_eval = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    results = {
        "dataset_size": len(df),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "label_distribution": {"SQL": int(len(y) - y.sum()), "GRAPH": int(y.sum())},
        "cv_folds": cv_folds,
        "labeled_data_path": labeled_path,
        "class_weight": class_weight,
        "use_scale_pos_weight": use_scale_pos_weight,
        "decision_tree": {**dt_cv, **dt_eval},
        "xgboost": {**xgb_cv, **xgb_eval},
        "best_model": best_name,
        "model_path": model_out,
    }

    # Save results
    results_path = os.path.join(MODEL_DIR, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")

    return results


if __name__ == "__main__":
    train()
