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


def _default_data_path() -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, "training_data", "fixed_train_base_strict.csv"),
        os.path.join(PROJECT_ROOT, "training_data", "fixed_train_base.csv"),
        os.path.join(PROJECT_ROOT, "training_data", "real_labeled_runs_strict_curated.csv"),
        LABELED_RUNS_CSV,
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return LABELED_RUNS_CSV

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
    cv_repeats: int = 3,
    min_graph_rows: int = 100,
    allow_degenerate: bool = False,
) -> dict:
    """Run feature ablation study.

    Returns dict with baseline F1 and per-feature/group ablation results.
    """
    graph_rows = int((y == 1).sum())
    sql_rows = int((y == 0).sum())
    if graph_rows == 0 or sql_rows == 0:
        raise ValueError("Ablation requires both SQL and GRAPH labels")
    if graph_rows < min_graph_rows and not allow_degenerate:
        raise ValueError(
            f"Degenerate ablation dataset: GRAPH rows={graph_rows} < required {min_graph_rows}. "
            "Collect additional real graph-winning labels before interpreting feature ablation."
        )

    def make_model():
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)

    def repeated_cv_scores(X_input: np.ndarray) -> np.ndarray:
        all_scores = []
        for r in range(cv_repeats):
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42 + r)
            model = make_model()
            scores = cross_val_score(model, X_input, y, cv=skf, scoring="f1")
            all_scores.extend(scores.tolist())
        return np.asarray(all_scores, dtype=np.float32)

    # Baseline: all features
    baseline_f1_scores = repeated_cv_scores(X)
    baseline_f1 = float(baseline_f1_scores.mean())
    baseline_f1_std = float(baseline_f1_scores.std())
    baseline_ci_low = float(np.percentile(baseline_f1_scores, 2.5))
    baseline_ci_high = float(np.percentile(baseline_f1_scores, 97.5))

    results = {
        "baseline_f1": baseline_f1,
        "baseline_f1_std": baseline_f1_std,
        "baseline_f1_ci": {
            "p2_5": baseline_ci_low,
            "p97_5": baseline_ci_high,
        },
        "cv_folds": cv_folds,
        "cv_repeats": cv_repeats,
        "individual_features": {},
        "feature_groups": {},
    }

    # Per-feature ablation
    print(
        f"\nBaseline F1 (all {len(feature_names)} features): {baseline_f1:.4f} ± {baseline_f1_std:.4f} "
        f"[95%~ {baseline_ci_low:.4f}, {baseline_ci_high:.4f}]"
    )
    print("\nPer-feature ablation:")

    for i, feat in enumerate(feature_names):
        X_ablated = np.delete(X, i, axis=1)
        f1_scores = repeated_cv_scores(X_ablated)
        f1_mean = float(f1_scores.mean())
        f1_std = float(f1_scores.std())
        f1_drop = baseline_f1 - f1_mean
        f1_ci_low = float(np.percentile(f1_scores, 2.5))
        f1_ci_high = float(np.percentile(f1_scores, 97.5))
        drop_scores = baseline_f1_scores - f1_scores
        f1_drop_std = float(drop_scores.std())
        f1_drop_ci_low = float(np.percentile(drop_scores, 2.5))
        f1_drop_ci_high = float(np.percentile(drop_scores, 97.5))

        results["individual_features"][feat] = {
            "f1_without": f1_mean,
            "f1_std": f1_std,
            "f1_drop": f1_drop,
            "f1_ci": {"p2_5": f1_ci_low, "p97_5": f1_ci_high},
            "f1_drop_std": f1_drop_std,
            "f1_drop_ci": {"p2_5": f1_drop_ci_low, "p97_5": f1_drop_ci_high},
        }
        marker = " ***" if abs(f1_drop) > 0.01 else ""
        print(
            f"  Drop '{feat}': F1={f1_mean:.4f} (Δ={f1_drop:+.4f}, "
            f"Δ95%~[{f1_drop_ci_low:+.4f},{f1_drop_ci_high:+.4f}]){marker}"
        )

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
        f1_scores = repeated_cv_scores(X_ablated)
        f1_mean = float(f1_scores.mean())
        f1_std = float(f1_scores.std())
        f1_drop = baseline_f1 - f1_mean
        f1_ci_low = float(np.percentile(f1_scores, 2.5))
        f1_ci_high = float(np.percentile(f1_scores, 97.5))
        drop_scores = baseline_f1_scores - f1_scores
        f1_drop_std = float(drop_scores.std())
        f1_drop_ci_low = float(np.percentile(drop_scores, 2.5))
        f1_drop_ci_high = float(np.percentile(drop_scores, 97.5))

        results["feature_groups"][group_name] = {
            "features_removed": group_features,
            "n_features_removed": len(drop_indices),
            "f1_without": f1_mean,
            "f1_std": f1_std,
            "f1_drop": f1_drop,
            "f1_ci": {"p2_5": f1_ci_low, "p97_5": f1_ci_high},
            "f1_drop_std": f1_drop_std,
            "f1_drop_ci": {"p2_5": f1_drop_ci_low, "p97_5": f1_drop_ci_high},
        }
        marker = " ***" if abs(f1_drop) > 0.01 else ""
        print(
            f"  Drop '{group_name}' ({len(drop_indices)} features): F1={f1_mean:.4f} "
            f"(Δ={f1_drop:+.4f}, Δ95%~[{f1_drop_ci_low:+.4f},{f1_drop_ci_high:+.4f}]){marker}"
        )

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
            "f1_ci_low": metrics.get("f1_ci", {}).get("p2_5"),
            "f1_ci_high": metrics.get("f1_ci", {}).get("p97_5"),
            "f1_drop": metrics["f1_drop"],
            "f1_drop_std": metrics.get("f1_drop_std"),
            "f1_drop_ci_low": metrics.get("f1_drop_ci", {}).get("p2_5"),
            "f1_drop_ci_high": metrics.get("f1_drop_ci", {}).get("p97_5"),
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
            "f1_ci_low": metrics.get("f1_ci", {}).get("p2_5"),
            "f1_ci_high": metrics.get("f1_ci", {}).get("p97_5"),
            "f1_drop": metrics["f1_drop"],
            "f1_drop_std": metrics.get("f1_drop_std"),
            "f1_drop_ci_low": metrics.get("f1_drop_ci", {}).get("p2_5"),
            "f1_drop_ci_high": metrics.get("f1_drop_ci", {}).get("p97_5"),
            "baseline_f1": results["baseline_f1"],
        })
    group_df = pd.DataFrame(group_rows)
    group_df = group_df.sort_values("f1_drop", ascending=False)

    return individual_df, group_df


def write_markdown_summary(results: dict, individual_df: pd.DataFrame, group_df: pd.DataFrame, md_path: str) -> None:
    """Write a concise markdown report for ablation/stability evidence."""
    lines = []
    lines.append("# Strict Ablation Summary")
    lines.append("")
    lines.append(f"- Baseline F1: {results['baseline_f1']:.4f}")
    lines.append(f"- Baseline std: {results['baseline_f1_std']:.4f}")
    lines.append(
        f"- Baseline 95% interval: [{results.get('baseline_f1_ci', {}).get('p2_5', 0.0):.4f}, "
        f"{results.get('baseline_f1_ci', {}).get('p97_5', 0.0):.4f}]"
    )
    lines.append(f"- CV folds: {results.get('cv_folds', 'n/a')}")
    lines.append(f"- CV repeats: {results.get('cv_repeats', 'n/a')}")

    max_feature_drop = float(individual_df['f1_drop'].max()) if not individual_df.empty else 0.0
    max_group_drop = float(group_df['f1_drop'].max()) if not group_df.empty else 0.0
    lines.append(f"- Max individual feature F1 drop: {max_feature_drop:+.4f}")
    lines.append(f"- Max feature-group F1 drop: {max_group_drop:+.4f}")
    lines.append("")

    lines.append("## Top Individual Feature Drops")
    lines.append("")
    lines.append("| Feature | F1 drop | Drop std | Drop CI low | Drop CI high |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, row in individual_df.head(10).iterrows():
        lines.append(
            f"| {row['feature']} | {row['f1_drop']:+.4f} | {row.get('f1_drop_std', 0.0):.4f} | "
            f"{row.get('f1_drop_ci_low', 0.0):+.4f} | {row.get('f1_drop_ci_high', 0.0):+.4f} |"
        )
    lines.append("")

    lines.append("## Feature Group Drops")
    lines.append("")
    lines.append("| Group | Removed | F1 drop | Drop std | Drop CI low | Drop CI high |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in group_df.iterrows():
        lines.append(
            f"| {row['group']} | {int(row['n_features_removed'])} | {row['f1_drop']:+.4f} | "
            f"{row.get('f1_drop_std', 0.0):.4f} | {row.get('f1_drop_ci_low', 0.0):+.4f} | "
            f"{row.get('f1_drop_ci_high', 0.0):+.4f} |"
        )

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Feature ablation study")
    parser.add_argument("--data", default=None,
                        help="Path to labeled training data CSV")
    parser.add_argument("--output", default=None,
                        help="Output CSV path for ablation results")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--cv-repeats", type=int, default=3,
                        help="Number of repeated CV rounds with different seeds")
    parser.add_argument("--min-graph-rows", type=int, default=100,
                        help="Minimum GRAPH rows required for valid ablation")
    parser.add_argument("--allow-degenerate", action="store_true",
                        help="Allow running on degenerate datasets (debug only)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load data
    data_path = args.data or _default_data_path()
    df = pd.read_csv(data_path)
    feature_names = FEATURE_NAMES
    X = df[feature_names].values.astype(np.float32)
    y = (df["label"] == "GRAPH").astype(int).values

    print(f"Training data: {len(df)} samples ({y.sum()} GRAPH / {(1-y).sum()} SQL)")
    print(f"Data path: {data_path}")
    print(f"Features: {len(feature_names)}")

    # Run ablation
    results = run_ablation(
        X,
        y,
        feature_names,
        cv_folds=args.cv_folds,
        cv_repeats=args.cv_repeats,
        min_graph_rows=args.min_graph_rows,
        allow_degenerate=args.allow_degenerate,
    )

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

    # Save markdown summary
    md_path = output_base.replace(".csv", ".md")
    write_markdown_summary(results, individual_df, group_df, md_path)
    print(f"Markdown summary saved to {md_path}")

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
