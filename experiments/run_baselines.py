"""Run baseline routing strategies: Always-SQL, Always-Graph, Rule-Based,
Threshold, Logistic Regression, and ML (Decision Tree / XGBoost).

Usage:
    python experiments/run_baselines.py --strategy always_sql
    python experiments/run_baselines.py --strategy threshold
    python experiments/run_baselines.py --strategy logreg
    python experiments/run_baselines.py --strategy all
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from router.hybrid_router import HybridRouter
from router.baselines import (
    trivial_rule_route,
    ThresholdBaseline,
    LogisticRegressionBaseline,
)
from config.paths import (
    SAMPLE_QUERIES_DIR, RESULTS_DIR, LABELED_RUNS_CSV, STATS_DIR,
    TPCH_PARQUET_DIR, SNB_PARQUET_DIR, GRAPHS_DIR,
)
from features.feature_extractor import FEATURE_NAMES

logger = logging.getLogger(__name__)

STRATEGIES = [
    "always_sql",
    "always_graph",
    "trivial_rule",
    "threshold",
    "logreg",
    "learned_ml",
]


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


def _make_router(strategy, query, force_engine=None, custom_router=None,
                 threshold_baseline=None, logreg_baseline=None):
    """Create a HybridRouter with correct data paths for the query."""
    query_id = query.get("query_id", "")
    is_snb = query_id.startswith("q_snb")
    parquet_dir = SNB_PARQUET_DIR if is_snb else TPCH_PARQUET_DIR

    kwargs = dict(parquet_dir=parquet_dir, graph_dir=GRAPHS_DIR)
    if force_engine:
        kwargs["force_engine"] = force_engine
    if custom_router:
        kwargs["custom_router"] = custom_router
    return HybridRouter(**kwargs)


def run_baseline(strategy: str, queries: list, num_runs: int = 3) -> pd.DataFrame:
    """Run a baseline strategy on all queries.

    Args:
        strategy: one of the STRATEGIES
        queries: list of query JSON dicts
        num_runs: number of repetitions for timing stability

    Returns:
        DataFrame with per-query results
    """
    # Prepare threshold and logreg baselines if needed
    threshold_baseline = None
    logreg_baseline = None

    if strategy == "threshold":
        if os.path.exists(LABELED_RUNS_CSV):
            threshold_baseline = ThresholdBaseline.tune_thresholds(
                LABELED_RUNS_CSV, FEATURE_NAMES
            )
        else:
            logger.warning("No labeled data for tuning; using default thresholds")
            threshold_baseline = ThresholdBaseline()

    if strategy == "logreg":
        if os.path.exists(LABELED_RUNS_CSV):
            df_train = pd.read_csv(LABELED_RUNS_CSV)
            available_cols = [c for c in FEATURE_NAMES if c in df_train.columns]
            if available_cols:
                X_train = df_train[available_cols].values
                y_train = (df_train["label"] == "GRAPH").astype(int).values
                logreg_baseline = LogisticRegressionBaseline()
                logreg_baseline.fit(X_train, y_train)
            else:
                logger.warning("Feature columns not found in labeled data")
                return pd.DataFrame()
        else:
            logger.warning("No labeled data for logistic regression training")
            return pd.DataFrame()

    if strategy in ("always_sql",):
        make_kwargs = lambda q: dict(force_engine="SQL")
    elif strategy == "always_graph":
        make_kwargs = lambda q: dict(force_engine="GRAPH")
    elif strategy in ("trivial_rule", "rule_based"):
        make_kwargs = lambda q: dict(
            custom_router=lambda sub, fv, fn: (
                "GRAPH" if sub.primary_op_type == "TRAVERSAL" else "SQL"
            )
        )
    elif strategy == "threshold":
        make_kwargs = lambda q: dict(
            custom_router=lambda sub, fv, fn: threshold_baseline.route(fv, fn)
        )
    elif strategy == "logreg":
        make_kwargs = lambda q: dict(
            custom_router=lambda sub, fv, fn: logreg_baseline.route(fv)
        )
    elif strategy == "learned_ml":
        make_kwargs = lambda q: {}
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    rows = []
    for query in queries:
        query_id = query.get("query_id", "unknown")
        description = query.get("description", "")
        latencies = []
        last_result = None

        # Create per-query router with correct data paths
        is_snb = query_id.startswith("q_snb")
        parquet_dir = SNB_PARQUET_DIR if is_snb else TPCH_PARQUET_DIR
        graph_dir = os.path.join(GRAPHS_DIR, "snb") if is_snb else os.path.join(GRAPHS_DIR, "synthetic")
        router = HybridRouter(parquet_dir=parquet_dir, graph_dir=graph_dir,
                              **make_kwargs(query))

        for run_idx in range(num_runs):
            try:
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
    parser.add_argument("--strategy", choices=STRATEGIES + ["all"], required=True,
                        help="Baseline strategy to run (or 'all' to run all)")
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

    strategies_to_run = STRATEGIES if args.strategy == "all" else [args.strategy]

    for strategy in strategies_to_run:
        print(f"\n{'=' * 60}")
        print(f"Strategy: {strategy}")
        print(f"Queries loaded: {len(queries)}")

        results_df = run_baseline(strategy, queries, num_runs=args.runs)

        if results_df.empty:
            print(f"  Skipped (no data available)")
            continue

        output_path = args.output or os.path.join(RESULTS_DIR, f"{strategy}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

        successful = results_df[results_df["success"]]
        if len(successful) > 0:
            print(f"  Queries succeeded: {len(successful)}/{len(results_df)}")
            print(f"  Median latency: {successful['median_latency_ms'].median():.1f} ms")
            print(f"  Mean latency:   {successful['mean_latency_ms'].mean():.1f} ms")
            print(f"  p95 latency:    {successful['median_latency_ms'].quantile(0.95):.1f} ms")
        else:
            print("  No queries succeeded.")


if __name__ == "__main__":
    main()