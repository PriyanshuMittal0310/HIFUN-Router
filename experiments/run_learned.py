"""Run ML-routed (learned) strategy using the trained classifier.

Usage:
    python experiments/run_learned.py \
        --model model/artifacts/classifier_v1.pkl \
        --queries dsl/sample_queries/ \
        --output experiments/results/learned_routing.csv
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
from config.paths import SAMPLE_QUERIES_DIR, RESULTS_DIR, CLASSIFIER_PATH

logger = logging.getLogger(__name__)


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


def run_learned(model_path: str, queries: list, num_runs: int = 3) -> pd.DataFrame:
    """Run ML-routed strategy and collect per-query metrics.

    Args:
        model_path: path to trained classifier pickle
        queries: list of query JSON dicts
        num_runs: number of repetitions for timing stability

    Returns:
        DataFrame with per-query results including routing decisions and confidence
    """
    router = HybridRouter(model_path=model_path)

    rows = []
    for query in queries:
        query_id = query.get("query_id", "unknown")
        description = query.get("description", "")
        latencies = []
        last_result = None

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
            confidences = [d["confidence"] for d in routing]
            inference_times = [d["inference_ms"] for d in routing]
            result_rows = len(last_result.get("result", pd.DataFrame()))
            parse_ms = last_result.get("parse_time_ms", 0)
            decompose_ms = last_result.get("decompose_time_ms", 0)
            exec_times = last_result.get("execution_times", {})
        else:
            engines = []
            confidences = []
            inference_times = []
            result_rows = 0
            parse_ms = 0
            decompose_ms = 0
            exec_times = {}

        valid_latencies = [l for l in latencies if not (isinstance(l, float) and l != l)]

        rows.append({
            "query_id": query_id,
            "description": description,
            "strategy": "learned",
            "median_latency_ms": float(np.median(valid_latencies)) if valid_latencies else float("nan"),
            "mean_latency_ms": float(np.mean(valid_latencies)) if valid_latencies else float("nan"),
            "min_latency_ms": float(np.min(valid_latencies)) if valid_latencies else float("nan"),
            "max_latency_ms": float(np.max(valid_latencies)) if valid_latencies else float("nan"),
            "num_subexpressions": len(engines),
            "engines_used": ",".join(engines),
            "avg_confidence": float(np.mean(confidences)) if confidences else float("nan"),
            "avg_inference_ms": float(np.mean(inference_times)) if inference_times else float("nan"),
            "total_inference_ms": float(np.sum(inference_times)) if inference_times else 0.0,
            "parse_ms": parse_ms,
            "decompose_ms": decompose_ms,
            "result_rows": result_rows,
            "num_runs": len(valid_latencies),
            "success": len(valid_latencies) > 0,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Run ML-routed (learned) strategy")
    parser.add_argument("--model", default=CLASSIFIER_PATH,
                        help="Path to trained classifier")
    parser.add_argument("--queries", default=SAMPLE_QUERIES_DIR,
                        help="Path to query directory or single JSON file")
    parser.add_argument("--output", default=None,
                        help="Output CSV path")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of repetitions per query")
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

    print(f"Strategy: learned (ML-routed)")
    print(f"Model: {args.model}")
    print(f"Queries loaded: {len(queries)}")

    # Run
    results_df = run_learned(args.model, queries, num_runs=args.runs)

    # Save
    output_path = args.output or os.path.join(RESULTS_DIR, "learned_routing.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Summary
    successful = results_df[results_df["success"]]
    if len(successful) > 0:
        print(f"\nSummary (learned routing):")
        print(f"  Queries succeeded: {len(successful)}/{len(results_df)}")
        print(f"  Median latency: {successful['median_latency_ms'].median():.1f} ms")
        print(f"  Mean latency:   {successful['mean_latency_ms'].mean():.1f} ms")
        print(f"  p95 latency:    {successful['median_latency_ms'].quantile(0.95):.1f} ms")
        print(f"  Avg confidence: {successful['avg_confidence'].mean():.3f}")
        print(f"  Total inference overhead: {successful['total_inference_ms'].sum():.1f} ms")
    else:
        print("  No queries succeeded.")


if __name__ == "__main__":
    main()