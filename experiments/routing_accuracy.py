"""Evaluate routing accuracy of each strategy against cost-model labels.

Computes per-strategy classification metrics (accuracy, F1, precision, recall)
and breaks results down by ambiguity level:
  - Unambiguous SQL: cost model strongly favors SQL (ratio > threshold)
  - Unambiguous GRAPH: cost model strongly favors GRAPH (ratio > threshold)
  - Ambiguous: cost model margin is narrow

Usage:
    python experiments/routing_accuracy.py
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import LABELED_RUNS_CSV, CLASSIFIER_PATH, RESULTS_DIR
from features.feature_extractor import FEATURE_NAMES
from router.baselines import ThresholdBaseline


def classify_ambiguity(row, ratio_thresh=2.0):
    """Classify each sample as unambiguous-SQL, unambiguous-GRAPH, or ambiguous."""
    sql_rt = row["sql_runtime_ms"]
    graph_rt = row["graph_runtime_ms"]
    if sql_rt <= 0 or graph_rt <= 0:
        return "ambiguous"
    ratio = max(sql_rt, graph_rt) / min(sql_rt, graph_rt)
    if ratio < ratio_thresh:
        return "ambiguous"
    elif sql_rt < graph_rt:
        return "unambiguous_sql"
    else:
        return "unambiguous_graph"


def main():
    df = pd.read_csv(LABELED_RUNS_CSV)
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y_true = (df["label"] == "GRAPH").astype(int).values  # 1=GRAPH, 0=SQL
    y_true_str = df["label"].values

    # Classify ambiguity
    df["ambiguity"] = df.apply(classify_ambiguity, axis=1)
    print(f"Dataset: {len(df)} samples")
    print(f"  Unambiguous SQL:   {(df['ambiguity'] == 'unambiguous_sql').sum()}")
    print(f"  Unambiguous GRAPH: {(df['ambiguity'] == 'unambiguous_graph').sum()}")
    print(f"  Ambiguous:         {(df['ambiguity'] == 'ambiguous').sum()}")
    print()

    # --- Strategy predictions ---
    strategies = {}

    # 1. Always SQL
    strategies["Always SQL"] = np.zeros(len(df), dtype=int)  # all SQL=0

    # 2. Always Graph
    strategies["Always Graph"] = np.ones(len(df), dtype=int)  # all GRAPH=1

    # 3. Trivial Rule: has_traversal >= 0.5 -> GRAPH, else SQL
    #    (use >= 0.5 threshold since augmentation adds noise to binary feature)
    has_trav_idx = FEATURE_NAMES.index("has_traversal")
    strategies["Trivial Rule"] = (X[:, has_trav_idx] >= 0.5).astype(int)

    # 4. Threshold Rule (tuned) - also use >= 0.5 for has_traversal
    thresh = ThresholdBaseline.tune_thresholds(LABELED_RUNS_CSV, FEATURE_NAMES)
    thresh_preds = np.array([
        int(thresh.route(x, FEATURE_NAMES) == "GRAPH") for x in X
    ])
    strategies["Threshold Rule"] = thresh_preds

    # 5. Logistic Regression (5-fold CV predictions for fair evaluation)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    strategies["Logistic Regression"] = cross_val_predict(logreg, X, y_true, cv=skf)

    # 6. Decision Tree (5-fold CV predictions for fair evaluation)
    dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)
    strategies["Learned ML (DT)"] = cross_val_predict(dt, X, y_true, cv=skf)

    # --- Compute metrics ---
    results = []
    for name, preds in strategies.items():
        row = {"Strategy": name}
        # Overall metrics
        row["Accuracy"] = accuracy_score(y_true, preds)
        row["F1"] = f1_score(y_true, preds, zero_division=0)
        row["Precision"] = precision_score(y_true, preds, zero_division=0)
        row["Recall"] = recall_score(y_true, preds, zero_division=0)

        # Per-ambiguity-category accuracy
        for cat in ["unambiguous_sql", "unambiguous_graph", "ambiguous"]:
            mask = df["ambiguity"] == cat
            if mask.sum() > 0:
                row[f"Acc_{cat}"] = accuracy_score(y_true[mask], preds[mask])
                row[f"F1_{cat}"] = f1_score(y_true[mask], preds[mask], zero_division=0)
                row[f"N_{cat}"] = int(mask.sum())
            else:
                row[f"Acc_{cat}"] = float("nan")
                row[f"F1_{cat}"] = float("nan")
                row[f"N_{cat}"] = 0

        results.append(row)

    results_df = pd.DataFrame(results)

    # Print results
    print("=" * 80)
    print("ROUTING ACCURACY (Overall)")
    print("=" * 80)
    for _, r in results_df.iterrows():
        print(f"  {r['Strategy']:25s}  Acc={r['Accuracy']:.3f}  F1={r['F1']:.3f}  "
              f"Prec={r['Precision']:.3f}  Rec={r['Recall']:.3f}")

    print()
    print("=" * 80)
    print("ROUTING ACCURACY BY AMBIGUITY CATEGORY")
    print("=" * 80)
    for cat in ["unambiguous_sql", "unambiguous_graph", "ambiguous"]:
        n = results_df[f"N_{cat}"].iloc[0]
        print(f"\n  {cat} (N={n}):")
        for _, r in results_df.iterrows():
            print(f"    {r['Strategy']:25s}  Acc={r[f'Acc_{cat}']:.3f}  F1={r[f'F1_{cat}']:.3f}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "routing_accuracy.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Also save as JSON for easy paper reference
    json_path = os.path.join(RESULTS_DIR, "routing_accuracy.json")
    with open(json_path, "w") as f:
        json.dump(results_df.to_dict(orient="records"), f, indent=2)
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
