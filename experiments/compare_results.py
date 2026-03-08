"""Compare all routing strategies and produce a comparison table.

Usage:
    python experiments/compare_results.py \
        --results experiments/results/ \
        --metrics latency_ms shuffle_bytes routing_accuracy \
        --output experiments/results/comparison_table.csv
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import RESULTS_DIR, SAMPLE_QUERIES_DIR, LABELED_RUNS_CSV

logger = logging.getLogger(__name__)


def load_result_csvs(results_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all strategy result CSVs from the results directory."""
    strategy_dfs = {}
    required_cols = {"strategy", "median_latency_ms", "success"}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".csv") or fname == "comparison_table.csv" or fname == "ablation.csv":
            continue
        strategy_name = fname.replace(".csv", "")
        fpath = os.path.join(results_dir, fname)
        df = pd.read_csv(fpath)
        if "strategy" in df.columns and len(df) > 0 and required_cols.issubset(df.columns):
            strategy_dfs[strategy_name] = df
    return strategy_dfs


def compute_routing_accuracy(strategy_df: pd.DataFrame, reference_df: pd.DataFrame = None) -> float:
    """Compute routing accuracy by comparing against the learned model's decisions.

    If no reference is available, estimate from labeled training data.
    For 'learned' strategy, use average confidence as a proxy.
    For baselines, compute fraction of queries where the engine choice
    matched the optimal (learned) choice.
    """
    if "avg_confidence" in strategy_df.columns:
        # Learned strategy: use confidence as accuracy proxy
        valid = strategy_df[strategy_df["success"]]
        return float(valid["avg_confidence"].mean()) if len(valid) > 0 else 0.0

    if reference_df is not None:
        # Compare engine choices against learned model's decisions
        merged = strategy_df.merge(
            reference_df[["query_id", "engines_used"]],
            on="query_id", suffixes=("_base", "_ref"),
        )
        if len(merged) > 0:
            matches = (merged["engines_used_base"] == merged["engines_used_ref"]).sum()
            return float(matches / len(merged))
    return float("nan")


def compute_strategy_metrics(df: pd.DataFrame) -> Dict:
    """Compute aggregate metrics for a single strategy."""
    successful = df[df["success"]] if "success" in df.columns else df

    if len(successful) == 0:
        return {
            "median_latency_ms": float("nan"),
            "p95_latency_ms": float("nan"),
            "mean_latency_ms": float("nan"),
            "total_queries": len(df),
            "successful_queries": 0,
            "success_rate": 0.0,
        }

    return {
        "median_latency_ms": float(successful["median_latency_ms"].median()),
        "p95_latency_ms": float(successful["median_latency_ms"].quantile(0.95)),
        "mean_latency_ms": float(successful["mean_latency_ms"].mean()) if "mean_latency_ms" in successful.columns else float(successful["median_latency_ms"].mean()),
        "total_queries": len(df),
        "successful_queries": int(len(successful)),
        "success_rate": float(len(successful) / len(df)),
    }


def build_comparison_table(
    strategy_dfs: Dict[str, pd.DataFrame],
    reference_strategy: str = "learned_routing",
) -> pd.DataFrame:
    """Build the final comparison table across all strategies.

    Returns a DataFrame matching the expected format from the project spec:
    Strategy | Median Latency (ms) | p95 Latency (ms) | Routing Accuracy
    """
    ref_df = strategy_dfs.get(reference_strategy)
    rows = []

    # Desired display order
    display_order = ["always_sql", "always_graph", "rule_based", "learned_routing"]
    display_names = {
        "always_sql": "Always SQL",
        "always_graph": "Always Graph",
        "rule_based": "Rule-Based",
        "learned_routing": "Learned (ML)",
    }

    for strategy_key in display_order:
        if strategy_key not in strategy_dfs:
            continue
        df = strategy_dfs[strategy_key]
        metrics = compute_strategy_metrics(df)
        accuracy = compute_routing_accuracy(df, ref_df if strategy_key != reference_strategy else None)

        rows.append({
            "Strategy": display_names.get(strategy_key, strategy_key),
            "Median Latency (ms)": round(metrics["median_latency_ms"], 1),
            "p95 Latency (ms)": round(metrics["p95_latency_ms"], 1),
            "Mean Latency (ms)": round(metrics["mean_latency_ms"], 1),
            "Successful Queries": metrics["successful_queries"],
            "Total Queries": metrics["total_queries"],
            "Routing Accuracy": round(accuracy, 3) if not np.isnan(accuracy) else "N/A",
        })

    # Also include any other strategies not in the standard list
    for strategy_key, df in strategy_dfs.items():
        if strategy_key in display_order:
            continue
        metrics = compute_strategy_metrics(df)
        accuracy = compute_routing_accuracy(df, ref_df)
        rows.append({
            "Strategy": strategy_key,
            "Median Latency (ms)": round(metrics["median_latency_ms"], 1),
            "p95 Latency (ms)": round(metrics["p95_latency_ms"], 1),
            "Mean Latency (ms)": round(metrics["mean_latency_ms"], 1),
            "Successful Queries": metrics["successful_queries"],
            "Total Queries": metrics["total_queries"],
            "Routing Accuracy": round(accuracy, 3) if not np.isnan(accuracy) else "N/A",
        })

    return pd.DataFrame(rows)


def compute_relative_improvements(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Add columns showing improvement relative to Always-SQL baseline."""
    df = comparison_df.copy()
    baseline_row = df[df["Strategy"] == "Always SQL"]
    if len(baseline_row) == 0:
        return df

    baseline_median = baseline_row["Median Latency (ms)"].iloc[0]
    baseline_p95 = baseline_row["p95 Latency (ms)"].iloc[0]

    if baseline_median > 0:
        df["Median vs SQL (%)"] = df["Median Latency (ms)"].apply(
            lambda x: round((x - baseline_median) / baseline_median * 100, 1)
            if not np.isnan(x) else "N/A"
        )
    if baseline_p95 > 0:
        df["p95 vs SQL (%)"] = df["p95 Latency (ms)"].apply(
            lambda x: round((x - baseline_p95) / baseline_p95 * 100, 1)
            if not np.isnan(x) else "N/A"
        )
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare routing strategy results")
    parser.add_argument("--results", default=RESULTS_DIR,
                        help="Directory containing per-strategy CSV files")
    parser.add_argument("--output", default=None,
                        help="Output CSV path for comparison table")
    parser.add_argument("--metrics", nargs="*",
                        default=["latency_ms", "routing_accuracy"],
                        help="Metrics to include (for display)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load all result CSVs
    strategy_dfs = load_result_csvs(args.results)
    if not strategy_dfs:
        print(f"No result CSVs found in {args.results}")
        print("Run baseline and learned experiments first.")
        sys.exit(1)

    print(f"Loaded strategies: {list(strategy_dfs.keys())}")
    for name, df in strategy_dfs.items():
        print(f"  {name}: {len(df)} queries")

    # Build comparison table
    comparison_df = build_comparison_table(strategy_dfs)
    comparison_df = compute_relative_improvements(comparison_df)

    # Save
    output_path = args.output or os.path.join(args.results, "comparison_table.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    print(f"\nComparison table saved to {output_path}")

    # Print table
    print("\n" + "=" * 100)
    print("ROUTING STRATEGY COMPARISON")
    print("=" * 100)
    print(comparison_df.to_string(index=False))
    print("=" * 100)

    # Also save as JSON for programmatic use
    json_path = output_path.replace(".csv", ".json")
    comparison_df.to_json(json_path, orient="records", indent=2)
    print(f"JSON version saved to {json_path}")


if __name__ == "__main__":
    main()
