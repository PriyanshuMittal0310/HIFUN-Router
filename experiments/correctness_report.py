"""Generate a per-query correctness report (CSV) comparing
HybridRouter output against the ReferenceExecutor using SHA256 checksums.

Usage:
    python experiments/correctness_report.py
    python experiments/correctness_report.py --queries dsl/sample_queries/ --output experiments/results/correctness_report.csv
"""

import argparse
import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tests.reference_executor import ReferenceExecutor, compare_results
from router.hybrid_router import HybridRouter
from config.paths import (
    TPCH_PARQUET_DIR, SNB_PARQUET_DIR, GRAPHS_DIR, STATS_DIR,
    SAMPLE_QUERIES_DIR, RESULTS_DIR,
)


def _get_paths_for_query(qid: str):
    """Return (parquet_dir, graph_dir) based on query dataset."""
    import os
    if qid.startswith("q_snb"):
        return SNB_PARQUET_DIR, os.path.join(GRAPHS_DIR, "snb")
    return TPCH_PARQUET_DIR, os.path.join(GRAPHS_DIR, "synthetic")


def generate_correctness_table(query_dir: str, output_csv: str):
    """Run all queries through both executors and generate a CSV report."""
    results = []

    for fname in sorted(os.listdir(query_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(query_dir, fname)) as f:
            data = json.load(f)
        queries = data if isinstance(data, list) else [data]

        for q in queries:
            qid = q.get("query_id", "?")
            parquet_dir, graph_dir = _get_paths_for_query(qid)
            router = HybridRouter(
                parquet_dir=parquet_dir,
                graph_dir=graph_dir,
                stats_dir=STATS_DIR,
                force_engine="SQL",
            )
            ref_exec = ReferenceExecutor(
                parquet_dir=parquet_dir,
                graph_dir=graph_dir,
            )
            try:
                ref_df = ref_exec.execute(q)
                result_dict = router.execute_query(q)
                test_df = result_dict["result"]
                report = compare_results(ref_df, test_df, qid)
                report["source_file"] = fname
                results.append(report)
            except Exception as e:
                results.append({
                    "query_id": qid,
                    "pass": False,
                    "error": str(e),
                    "source_file": fname,
                })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    total = len(df)
    passed = df["pass"].sum()
    print(f"\nCorrectness Summary: {passed}/{total} queries pass "
          f"({100 * passed / total:.1f}%)")
    cols = ["query_id", "row_count_match", "sha256_match", "col_mismatches", "pass"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate correctness report")
    parser.add_argument("--queries", default=SAMPLE_QUERIES_DIR,
                        help="Directory containing query JSON files")
    parser.add_argument("--output", default=os.path.join(RESULTS_DIR,
                        "correctness_report.csv"),
                        help="Output CSV path")
    args = parser.parse_args()

    generate_correctness_table(args.queries, args.output)


if __name__ == "__main__":
    main()
