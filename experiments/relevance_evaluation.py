"""Robust relevance evaluation for SQL/GRAPH routing quality.

This script addresses common research-quality gaps by:
1) Using separate train and evaluation datasets.
2) Comparing against stronger baselines.
3) Reporting class-sensitive metrics on an untouched evaluation split.
4) Running a no-history ablation to detect historical-feature leakage.

Usage:
  python experiments/relevance_evaluation.py
  python experiments/relevance_evaluation.py \
      --train training_data/real_labeled_runs_balanced.csv \
      --eval training_data/real_labeled_runs.csv
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import RESULTS_DIR
from features.feature_extractor import FEATURE_NAMES
from router.baselines import ThresholdBaseline


HISTORY_FEATURES = ["hist_avg_runtime_ms", "hist_runtime_variance"]


@dataclass
class EvalMetrics:
    model: str
    accuracy: float
    f1: float
    precision: float
    recall: float
    tn: int
    fp: int
    fn: int
    tp: int


def _default_train_path() -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, "training_data", "fixed_train_balanced.csv"),
        os.path.join(PROJECT_ROOT, "training_data", "fixed_train_base.csv"),
        os.path.join(PROJECT_ROOT, "training_data", "real_labeled_runs_balanced.csv"),
        os.path.join(PROJECT_ROOT, "training_data", "real_labeled_runs.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No train dataset found under training_data/")


def _default_eval_path() -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, "training_data", "fixed_eval_set.csv"),
        os.path.join(PROJECT_ROOT, "training_data", "real_labeled_runs.csv"),
        os.path.join(PROJECT_ROOT, "training_data", "real_labeled_runs_balanced.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No eval dataset found under training_data/")


def _load_xy(path: str, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(path)
    X = df[feature_names].values.astype(np.float32)
    y = (df["label"] == "GRAPH").astype(int).values
    return X, y, df


def _metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> EvalMetrics:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return EvalMetrics(
        model=name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def _render_markdown(summary: dict) -> str:
    lines = []
    lines.append("# Routing Relevance Evaluation")
    lines.append("")
    lines.append(f"- Train dataset: {summary['train_path']}")
    lines.append(f"- Eval dataset: {summary['eval_path']}")
    lines.append(f"- Train rows: {summary['train_rows']}")
    lines.append(f"- Eval rows: {summary['eval_rows']}")
    lines.append(f"- Train label distribution: {summary['train_label_distribution']}")
    lines.append(f"- Eval label distribution: {summary['eval_label_distribution']}")
    lines.append("")
    lines.append("## Model Comparison")
    lines.append("")
    lines.append("| Model | Accuracy | F1 | Precision | Recall | TN | FP | FN | TP |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in summary["metrics"]:
        lines.append(
            f"| {m['model']} | {m['accuracy']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['tn']} | {m['fp']} | {m['fn']} | {m['tp']} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- NoHistory models drop historical runtime features to test leakage risk.")
    lines.append("- AlwaysSQL/AlwaysGRAPH quantify class-imbalance floor and ceiling behavior.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust relevance evaluation for routing models")
    parser.add_argument("--train", default=None, help="Training CSV path")
    parser.add_argument("--eval", dest="eval_path", default=None, help="Evaluation CSV path")
    parser.add_argument("--out_json", default=os.path.join(RESULTS_DIR, "relevance_eval.json"))
    parser.add_argument("--out_md", default=os.path.join(RESULTS_DIR, "relevance_eval.md"))
    args = parser.parse_args()

    train_path = args.train or _default_train_path()
    eval_path = args.eval_path or _default_eval_path()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    feature_names = [f for f in FEATURE_NAMES if f in pd.read_csv(train_path, nrows=1).columns]
    feature_no_hist = [f for f in feature_names if f not in HISTORY_FEATURES]

    X_train, y_train, train_df = _load_xy(train_path, feature_names)
    X_eval, y_eval, eval_df = _load_xy(eval_path, feature_names)

    X_train_no_hist = train_df[feature_no_hist].values.astype(np.float32)
    X_eval_no_hist = eval_df[feature_no_hist].values.astype(np.float32)

    metrics = []

    # Baseline floors
    y_pred = np.zeros_like(y_eval)
    metrics.append(asdict(_metrics("AlwaysSQL", y_eval, y_pred)))

    y_pred = np.ones_like(y_eval)
    metrics.append(asdict(_metrics("AlwaysGRAPH", y_eval, y_pred)))

    # Trivial traversal baseline
    has_traversal_idx = feature_names.index("has_traversal") if "has_traversal" in feature_names else None
    if has_traversal_idx is not None:
        y_pred = (X_eval[:, has_traversal_idx] >= 0.5).astype(int)
        metrics.append(asdict(_metrics("TraversalRule", y_eval, y_pred)))

    # Threshold baseline tuned on train set
    threshold = ThresholdBaseline.tune_thresholds(train_path, feature_names)
    y_pred = np.array([1 if threshold.route(row, feature_names) == "GRAPH" else 0 for row in X_eval])
    metrics.append(asdict(_metrics("ThresholdBaseline", y_eval, y_pred)))

    # Logistic regression baseline
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_eval)
    metrics.append(asdict(_metrics("LogRegBalanced", y_eval, y_pred)))

    # Decision tree
    dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_eval)
    metrics.append(asdict(_metrics("DecisionTreeBalanced", y_eval, y_pred)))

    # XGBoost full features
    sql_count = int((y_train == 0).sum())
    graph_count = int((y_train == 1).sum())
    spw = float(sql_count / max(graph_count, 1))
    xgb_full = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=spw,
    )
    xgb_full.fit(X_train, y_train)
    y_pred = xgb_full.predict(X_eval)
    metrics.append(asdict(_metrics("XGBoostBalanced", y_eval, y_pred)))

    # XGBoost no-history ablation
    xgb_no_hist = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=spw,
    )
    xgb_no_hist.fit(X_train_no_hist, y_train)
    y_pred = xgb_no_hist.predict(X_eval_no_hist)
    metrics.append(asdict(_metrics("XGBoostNoHistory", y_eval, y_pred)))

    summary = {
        "train_path": train_path,
        "eval_path": eval_path,
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "train_label_distribution": {
            "SQL": int((y_train == 0).sum()),
            "GRAPH": int((y_train == 1).sum()),
        },
        "eval_label_distribution": {
            "SQL": int((y_eval == 0).sum()),
            "GRAPH": int((y_eval == 1).sum()),
        },
        "feature_count": len(feature_names),
        "feature_count_no_history": len(feature_no_hist),
        "metrics": metrics,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(_render_markdown(summary))

    print(f"Saved: {args.out_json}")
    print(f"Saved: {args.out_md}")


if __name__ == "__main__":
    main()
