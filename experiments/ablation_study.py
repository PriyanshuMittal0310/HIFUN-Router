"""Feature ablation study: retrain with each feature group removed.

For each feature group: retrain model with that group removed, compare
CV F1 vs the full 22-feature model. Report delta accuracy per removed
group to identify the most critical features.

Usage:
    python experiments/ablation_study.py \
        --data training_data/labeled_runs.csv \
        --output experiments/results/ablation.csv
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import LABELED_RUNS_CSV, RESULTS_DIR, CLASSIFIER_PATH
from features.feature_extractor import FEATURE_NAMES

logger = logging.getLogger(__name__)

# Feature groups for ablation (must match FEATURE_NAMES from config/feature_schema.json)
FEATURE_GROUPS = {
    "operation_counts": [
        "op_count_filter", "op_count_join", "op_count_traversal",
        "op_count_aggregate", "op_count_map",
    ],
    "structure": [
        "ast_depth", "num_projected_columns", "num_tables_joined",
    ],
    "graph_features": [
        "has_traversal", "max_hops", "avg_degree", "max_degree", "degree_skew",
    ],
    "cardinality": [
        "input_cardinality_log", "output_cardinality_log", "selectivity",
    ],
    "data_characteristics": [
        "has_index", "join_fanout", "estimated_shuffle_bytes_log",
        "estimated_traversal_ops",
    ],
    "historical": [
        "hist_avg_runtime_ms", "hist_runtime_variance",
    ],
}


def run_ablation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    cv_folds: int = 5,
) -> dict:
    """Run feature ablation study.

    Returns dict with baseline F1 and per-feature/group ablation results.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    def make_model():
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)

    # Baseline: all features
    baseline_model = make_model()
    baseline_f1_scores = cross_val_score(baseline_model, X, y, cv=skf, scoring="f1")
    baseline_f1 = float(baseline_f1_scores.mean())
    baseline_f1_std = float(baseline_f1_scores.std())

    results = {
        "baseline_f1": baseline_f1,
        "baseline_f1_std": baseline_f1_std,
        "individual_features": {},
        "feature_groups": {},
    }

    # Per-feature ablation
    print(f"\nBaseline F1 (all {len(feature_names)} features): {baseline_f1:.4f} ± {baseline_f1_std:.4f}")
    print("\nPer-feature ablation:")

    for i, feat in enumerate(feature_names):
        X_ablated = np.delete(X, i, axis=1)
        model = make_model()
        f1_scores = cross_val_score(model, X_ablated, y, cv=skf, scoring="f1")
        f1_mean = float(f1_scores.mean())
        f1_std = float(f1_scores.std())
        f1_drop = baseline_f1 - f1_mean

        results["individual_features"][feat] = {
            "f1_without": f1_mean,
            "f1_std": f1_std,
            "f1_drop": f1_drop,
        }
        marker = " ***" if abs(f1_drop) > 0.01 else ""
        print(f"  Drop '{feat}': F1={f1_mean:.4f} (Δ={f1_drop:+.4f}){marker}")

    # Feature group ablation
    print("\nFeature group ablation:")
    for group_name, group_features in FEATURE_GROUPS.items():
        # Find indices of features in this group
        drop_indices = []
        for feat in group_features:
            if feat in feature_names:
                drop_indices.append(feature_names.index(feat))

        if not drop_indices:
            continue

        X_ablated = np.delete(X, drop_indices, axis=1)
        model = make_model()
        f1_scores = cross_val_score(model, X_ablated, y, cv=skf, scoring="f1")
        f1_mean = float(f1_scores.mean())
        f1_std = float(f1_scores.std())
        f1_drop = baseline_f1 - f1_mean

        results["feature_groups"][group_name] = {
            "features_removed": group_features,
            "n_features_removed": len(drop_indices),
            "f1_without": f1_mean,
            "f1_std": f1_std,
            "f1_drop": f1_drop,
        }
        marker = " ***" if abs(f1_drop) > 0.01 else ""
        print(f"  Drop '{group_name}' ({len(drop_indices)} features): F1={f1_mean:.4f} (Δ={f1_drop:+.4f}){marker}")

    return results


def results_to_dataframes(results: dict) -> tuple:
    """Convert ablation results to DataFrames for CSV output."""
    # Individual features
    ind_rows = []
    for feat, metrics in results["individual_features"].items():
        ind_rows.append({
            "feature": feat,
            "f1_without": metrics["f1_without"],
            "f1_std": metrics["f1_std"],
            "f1_drop": metrics["f1_drop"],
            "baseline_f1": results["baseline_f1"],
        })
    individual_df = pd.DataFrame(ind_rows)
    individual_df = individual_df.sort_values("f1_drop", ascending=False)

    # Feature groups
    group_rows = []
    for group_name, metrics in results["feature_groups"].items():
        group_rows.append({
            "group": group_name,
            "features_removed": ", ".join(metrics["features_removed"]),
            "n_features_removed": metrics["n_features_removed"],
            "f1_without": metrics["f1_without"],
            "f1_std": metrics["f1_std"],
            "f1_drop": metrics["f1_drop"],
            "baseline_f1": results["baseline_f1"],
        })
    group_df = pd.DataFrame(group_rows)
    group_df = group_df.sort_values("f1_drop", ascending=False)

    return individual_df, group_df


def main():
    parser = argparse.ArgumentParser(description="Feature ablation study")
    parser.add_argument("--data", default=LABELED_RUNS_CSV,
                        help="Path to labeled training data CSV")
    parser.add_argument("--output", default=None,
                        help="Output CSV path for ablation results")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of cross-validation folds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load data
    df = pd.read_csv(args.data)
    feature_names = FEATURE_NAMES
    X = df[feature_names].values.astype(np.float32)
    y = (df["label"] == "GRAPH").astype(int).values

    print(f"Training data: {len(df)} samples ({y.sum()} GRAPH / {(1-y).sum()} SQL)")
    print(f"Features: {len(feature_names)}")

    # Run ablation
    results = run_ablation(X, y, feature_names, cv_folds=args.cv_folds)

    # Save results
    output_base = args.output or os.path.join(RESULTS_DIR, "ablation.csv")
    os.makedirs(os.path.dirname(output_base), exist_ok=True)

    individual_df, group_df = results_to_dataframes(results)

    # Save individual feature ablation
    individual_df.to_csv(output_base, index=False)
    print(f"\nIndividual feature ablation saved to {output_base}")

    # Save group ablation
    group_path = output_base.replace(".csv", "_groups.csv")
    group_df.to_csv(group_path, index=False)
    print(f"Feature group ablation saved to {group_path}")

    # Save full JSON results
    json_path = output_base.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {json_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("TOP-5 MOST IMPACTFUL INDIVIDUAL FEATURES (by F1 drop)")
    print("=" * 70)
    top5 = individual_df.head(5)
    for _, row in top5.iterrows():
        print(f"  {row['feature']:30s}  F1 drop: {row['f1_drop']:+.4f}")

    print("\nMOST IMPACTFUL FEATURE GROUPS:")
    print("=" * 70)
    for _, row in group_df.iterrows():
        print(f"  {row['group']:25s}  ({row['n_features_removed']} features)  F1 drop: {row['f1_drop']:+.4f}")


if __name__ == "__main__":
    main()
