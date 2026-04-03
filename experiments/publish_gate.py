"""Publication gate for strict evaluation artifacts.

Fails fast when required strict artifacts are missing or key metrics do not meet
minimum thresholds.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import TPCH_PARQUET_DIR


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _require_file(path: str, label: str, errors: list[str]) -> None:
    if not os.path.exists(path):
        errors.append(f"Missing {label}: {path}")


def _check_manifest(manifest: Dict[str, Any], max_query_overlap: int, errors: list[str]) -> None:
    overlap = manifest.get("overlap", {})
    query_overlap = int(overlap.get("query_id", 0))
    if query_overlap > max_query_overlap:
        errors.append(
            f"Split leakage detected: query_id overlap={query_overlap} > allowed {max_query_overlap}"
        )


def _find_metric(metrics: list[Dict[str, Any]], model: str) -> Dict[str, Any] | None:
    for m in metrics:
        if m.get("model") == model:
            return m
    return None


def _check_relevance(relevance: Dict[str, Any], min_xgb_f1: float, errors: list[str]) -> None:
    metrics = relevance.get("metrics", [])
    xgb = _find_metric(metrics, "XGBoostBalanced")
    if xgb is None:
        errors.append("relevance_eval: missing model metric 'XGBoostBalanced'")
        return

    f1 = float(xgb.get("f1", 0.0))
    if f1 < min_xgb_f1:
        errors.append(
            f"Relevance gate failed: XGBoostBalanced f1={f1:.4f} < min {min_xgb_f1:.4f}"
        )


def _check_robustness(robustness: Dict[str, Any], min_robust_f1: float, errors: list[str]) -> None:
    f1 = float(robustness.get("xgb_eval_f1", 0.0))
    if f1 < min_robust_f1:
        errors.append(
            f"Robustness gate failed: xgb_eval_f1={f1:.4f} < min {min_robust_f1:.4f}"
        )


def _check_correctness(
    correctness_csv: str,
    min_executable_pass_rate: float,
    max_skipped_missing_data: int,
    max_skipped_unsupported_schema: int,
    errors: list[str],
) -> None:
    df = pd.read_csv(correctness_csv)
    if df.empty:
        errors.append("Correctness gate failed: report is empty")
        return

    if "status" in df.columns:
        status = df["status"].fillna("fail")
    else:
        status = pd.Series(["pass" if bool(v) else "fail" for v in df.get("pass", [])])

    skipped_missing = int((status == "skipped_missing_data").sum())
    skipped_schema = int((status == "skipped_unsupported_schema").sum())
    executable = df[~status.isin(["skipped_missing_data", "skipped_unsupported_schema"])]

    executable_pass_rate = float(executable["pass"].mean()) if len(executable) else 0.0

    if executable_pass_rate < min_executable_pass_rate:
        errors.append(
            "Correctness gate failed: executable_pass_rate="
            f"{executable_pass_rate:.4f} < min {min_executable_pass_rate:.4f}"
        )

    if skipped_missing > max_skipped_missing_data:
        errors.append(
            f"Correctness gate failed: skipped_missing_data={skipped_missing} "
            f"> allowed {max_skipped_missing_data}"
        )

    if skipped_schema > max_skipped_unsupported_schema:
        errors.append(
            f"Correctness gate failed: skipped_unsupported_schema={skipped_schema} "
            f"> allowed {max_skipped_unsupported_schema}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate strict publication artifacts")
    parser.add_argument("--manifest", default="training_data/fixed_split_manifest_strict.json")
    parser.add_argument("--relevance_json", default="experiments/results/relevance_eval_strict_runtime.json")
    parser.add_argument("--robustness_json", default="experiments/results/strict_robustness_eval_runtime.json")
    parser.add_argument("--correctness_csv", default="experiments/results/correctness_report_runtime.csv")
    parser.add_argument("--max_query_overlap", type=int, default=0)
    parser.add_argument("--min_xgb_f1", type=float, default=0.95)
    parser.add_argument("--min_robust_f1", type=float, default=0.95)
    parser.add_argument("--min_executable_pass_rate", type=float, default=1.0)
    parser.add_argument("--max_skipped_missing_data", type=int, default=0)
    parser.add_argument("--max_skipped_unsupported_schema", type=int, default=0)
    parser.add_argument("--require_native_tpch", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []

    _require_file(args.manifest, "split manifest", errors)
    _require_file(args.relevance_json, "relevance report", errors)
    _require_file(args.robustness_json, "robustness report", errors)
    _require_file(args.correctness_csv, "correctness report", errors)

    if args.require_native_tpch and not os.path.exists(TPCH_PARQUET_DIR):
        errors.append(
            f"Native TPCH requirement failed: directory not found at {TPCH_PARQUET_DIR}"
        )

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        return 1

    manifest = _load_json(args.manifest)
    relevance = _load_json(args.relevance_json)
    robustness = _load_json(args.robustness_json)

    _check_manifest(manifest, args.max_query_overlap, errors)
    _check_relevance(relevance, args.min_xgb_f1, errors)
    _check_robustness(robustness, args.min_robust_f1, errors)
    _check_correctness(
        args.correctness_csv,
        args.min_executable_pass_rate,
        args.max_skipped_missing_data,
        args.max_skipped_unsupported_schema,
        errors,
    )

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        return 1

    print("[PASS] Strict publication gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
