"""Run baseline routing strategies: Always-SQL, Always-Graph, Rule-Based.

Usage:
    python experiments/run_baselines.py --strategy always_sql --queries dsl/sample_queries/ --output experiments/results/always_sql.csv
    python experiments/run_baselines.py --strategy always_graph --queries dsl/sample_queries/ --output experiments/results/always_graph.csv
    python experiments/run_baselines.py --strategy rule_based --queries dsl/sample_queries/ --output experiments/results/rule_based.csv
"""

import argparse
import json
import logging
import os
import sys
import time

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from router.hybrid_router import HybridRouter
from config.paths import SAMPLE_QUERIES_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

STRATEGIES = ["always_sql", "always_graph", "rule_based"]


def load_all_queries(queries_dir: str) -> list:
    """Load all query JSON files from the given directory."""
    all_queries = []
    for fname in sorted(os.listdir(queries_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(queries_dir, fname)
        with open(fpath) as f:
            queries = json.load(f)
        if isinstance(queries, list):
            all_queries.extend(queries)
        else:
            all_queries.append(queries)
    return all_queries


def run_baseline(strategy: str, queries: list, num_runs: int = 3) -> pd.DataFrame:
    """Run a baseline strategy on all queries.

    Args:
        strategy: one of 'always_sql', 'always_graph', 'rule_based'
        queries: list of query JSON dicts
        num_runs: number of repetitions for timing stability

    Returns:
        DataFrame with per-query results
    """
    if strategy == "always_sql":
        router = HybridRouter(force_engine="SQL")
    elif strategy == "always_graph":
        router = HybridRouter(force_engine="GRAPH")
    elif strategy == "rule_based":
        router = HybridRouter(force_engine="RULE_BASED")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    rows = []
    for query in queries:
        query_id = query.get("query_id", "unknown")
        description = query.get("description", "")
        latencies = []
        last_result = None

        for run_idx in range(num_runs):
            try:
                # For rule_based, we override force_engine handling
                if strategy == "rule_based":
                    router.force_engine = None  # Let heuristic route
                    # Temporarily disable ML predictor to force heuristic fallback
                    router._model_path = "/nonexistent/model.pkl"
                    router._predictor = None

                result = router.execute_query(query)
                latencies.append(result["total_time_ms"])
                last_result = result
            except Exception as e:
                logger.warning(f"Query {query_id} run {run_idx} failed: {e}")
                latencies.append(float("nan"))

        if last_result and "error" not in last_result:
            routing = last_result.get("routing_decisions", [])
            engines = [d["engine"] for d in routing]
            result_rows = len(last_result.get("result", pd.DataFrame()))
        else:
            engines = []
            result_rows = 0

        import numpy as np
        valid_latencies = [l for l in latencies if not (isinstance(l, float) and l != l)]

        rows.append({
            "query_id": query_id,
            "description": description,
            "strategy": strategy,
            "median_latency_ms": float(np.median(valid_latencies)) if valid_latencies else float("nan"),
            "mean_latency_ms": float(np.mean(valid_latencies)) if valid_latencies else float("nan"),
            "min_latency_ms": float(np.min(valid_latencies)) if valid_latencies else float("nan"),
            "max_latency_ms": float(np.max(valid_latencies)) if valid_latencies else float("nan"),
            "num_subexpressions": len(engines),
            "engines_used": ",".join(engines),
            "result_rows": result_rows,
            "num_runs": len(valid_latencies),
            "success": len(valid_latencies) > 0,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Run baseline routing strategies")
    parser.add_argument("--strategy", choices=STRATEGIES, required=True,
                        help="Baseline strategy to run")
    parser.add_argument("--queries", default=SAMPLE_QUERIES_DIR,
                        help="Path to query directory or single JSON file")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: experiments/results/<strategy>.csv)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of repetitions per query for timing stability")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load queries
    if os.path.isdir(args.queries):
        queries = load_all_queries(args.queries)
    else:
        with open(args.queries) as f:
            queries = json.load(f)
        if not isinstance(queries, list):
            queries = [queries]

    print(f"Strategy: {args.strategy}")
    print(f"Queries loaded: {len(queries)}")

    # Run
    results_df = run_baseline(args.strategy, queries, num_runs=args.runs)

    # Save
    output_path = args.output or os.path.join(RESULTS_DIR, f"{args.strategy}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Summary
    successful = results_df[results_df["success"]]
    if len(successful) > 0:
        print(f"\nSummary ({args.strategy}):")
        print(f"  Queries succeeded: {len(successful)}/{len(results_df)}")
        print(f"  Median latency: {successful['median_latency_ms'].median():.1f} ms")
        print(f"  Mean latency:   {successful['mean_latency_ms'].mean():.1f} ms")
        print(f"  p95 latency:    {successful['median_latency_ms'].quantile(0.95):.1f} ms")
    else:
        print("  No queries succeeded.")


if __name__ == "__main__":
    main()