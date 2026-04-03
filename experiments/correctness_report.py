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


def _is_missing_data_error(exc: Exception) -> bool:
    """Return True when an exception indicates unavailable local tables/data."""
    if isinstance(exc, FileNotFoundError):
        return True
    msg = str(exc).lower()
    return "not found" in msg and "table" in msg


def _is_unsupported_schema_error(exc: Exception) -> bool:
    """Return True when an exception indicates a query/schema mismatch."""
    msg = str(exc).lower()
    return "missing filter column" in msg


def _is_tpch_query(qid: str) -> bool:
    return qid.startswith("q_tpch") or qid.startswith("q_synth")


def generate_correctness_table(query_dir: str, output_csv: str, require_native_tpch: bool = False):
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

            if require_native_tpch and _is_tpch_query(qid):
                native_customer = os.path.exists(os.path.join(parquet_dir, "customer"))
                native_orders = os.path.exists(os.path.join(parquet_dir, "orders"))
                if not (native_customer and native_orders):
                    results.append({
                        "query_id": qid,
                        "pass": False,
                        "status": "skipped_missing_data",
                        "error": "native_tpch_required_but_missing_customer_or_orders",
                        "source_file": fname,
                    })
                    continue

            router = HybridRouter(
                parquet_dir=parquet_dir,
                graph_dir=graph_dir,
                stats_dir=STATS_DIR,
                force_engine="SQL",
                strict_schema=True,
            )
            ref_exec = ReferenceExecutor(
                parquet_dir=parquet_dir,
                graph_dir=graph_dir,
                strict_schema=True,
            )
            try:
                ref_df = ref_exec.execute(q)
                result_dict = router.execute_query(q)
                test_df = result_dict["result"]
                report = compare_results(ref_df, test_df, qid)
                report["source_file"] = fname
                report["status"] = "pass" if report.get("pass", False) else "fail"
                results.append(report)
            except Exception as e:
                if _is_missing_data_error(e):
                    status = "skipped_missing_data"
                elif _is_unsupported_schema_error(e):
                    status = "skipped_unsupported_schema"
                else:
                    status = "fail"
                results.append({
                    "query_id": qid,
                    "pass": False,
                    "status": status,
                    "error": str(e),
                    "source_file": fname,
                })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    total = len(df)
    passed = int(df["pass"].sum())
    skipped_missing_data = int((df.get("status") == "skipped_missing_data").sum()) if "status" in df.columns else 0
    skipped_unsupported_schema = int((df.get("status") == "skipped_unsupported_schema").sum()) if "status" in df.columns else 0
    skipped = skipped_missing_data + skipped_unsupported_schema
    executable = total - skipped
    exec_pass_pct = (100 * passed / executable) if executable else 0.0
    total_pass_pct = (100 * passed / total) if total else 0.0

    print(
        f"\nCorrectness Summary: {passed}/{total} queries pass "
        f"({total_pass_pct:.1f}%). Executable-only: {passed}/{executable} "
        f"({exec_pass_pct:.1f}%), skipped_missing_data={skipped_missing_data}, "
        f"skipped_unsupported_schema={skipped_unsupported_schema}."
    )
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
    parser.add_argument("--require_native_tpch", action="store_true",
                        help="Require native TPCH customer/orders parquet for TPCH/synthetic queries")
    args = parser.parse_args()

    generate_correctness_table(args.queries, args.output, require_native_tpch=args.require_native_tpch)


if __name__ == "__main__":
    main()
