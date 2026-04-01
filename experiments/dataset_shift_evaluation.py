"""Cross-dataset generalization evaluation.

Trains on one dataset family and evaluates on each other dataset family to
measure transfer robustness of routing decisions.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import RESULTS_DIR
from features.feature_extractor import FEATURE_NAMES


def _default_source() -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, "training_data", "real_labeled_runs.csv"),
        os.path.join(PROJECT_ROOT, "training_data", "fixed_train_base.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No source dataset found for dataset-shift eval")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset-shift evaluation for routing model")
    parser.add_argument("--source", default=None, help="Source CSV with dataset column")
    parser.add_argument("--out_json", default=os.path.join(RESULTS_DIR, "dataset_shift_eval.json"))
    parser.add_argument("--out_md", default=os.path.join(RESULTS_DIR, "dataset_shift_eval.md"))
    args = parser.parse_args()

    source = args.source or _default_source()
    df = pd.read_csv(source)

    feature_names = [f for f in FEATURE_NAMES if f in df.columns]
    if "dataset" not in df.columns:
        raise ValueError("dataset column required")

    rows = []
    datasets = sorted(df["dataset"].dropna().unique().tolist())

    for train_ds in datasets:
        train_df = df[df["dataset"] == train_ds].copy()
        y_train = (train_df["label"] == "GRAPH").astype(int).values

        # Need both classes to fit a classifier.
        if len(np.unique(y_train)) < 2:
            for eval_ds in datasets:
                if eval_ds == train_ds:
                    continue
                rows.append({
                    "train_dataset": train_ds,
                    "eval_dataset": eval_ds,
                    "status": "skipped_single_class_train",
                })
            continue

        X_train = train_df[feature_names].values.astype(np.float32)
        sql_count = int((y_train == 0).sum())
        graph_count = int((y_train == 1).sum())
        spw = float(sql_count / max(graph_count, 1))

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=spw,
        )
        model.fit(X_train, y_train)

        for eval_ds in datasets:
            if eval_ds == train_ds:
                continue
            eval_df = df[df["dataset"] == eval_ds].copy()
            y_eval = (eval_df["label"] == "GRAPH").astype(int).values
            X_eval = eval_df[feature_names].values.astype(np.float32)
            y_pred = model.predict(X_eval)

            rows.append({
                "train_dataset": train_ds,
                "eval_dataset": eval_ds,
                "status": "ok",
                "rows": int(len(eval_df)),
                "accuracy": float(accuracy_score(y_eval, y_pred)),
                "f1": float(f1_score(y_eval, y_pred, zero_division=0)),
                "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
                "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
                "eval_graph_rows": int((y_eval == 1).sum()),
                "eval_sql_rows": int((y_eval == 0).sum()),
            })

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"source": source, "results": rows}, f, indent=2)

    md = ["# Dataset Shift Evaluation", "", f"- Source: {source}", "", "| Train | Eval | Status | Rows | F1 | Precision | Recall |", "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["status"] != "ok":
            md.append(f"| {r['train_dataset']} | {r['eval_dataset']} | {r['status']} | - | - | - | - |")
        else:
            md.append(
                f"| {r['train_dataset']} | {r['eval_dataset']} | ok | {r['rows']} | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |"
            )

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Saved: {args.out_json}")
    print(f"Saved: {args.out_md}")


if __name__ == "__main__":
    main()
