"""Generate a one-page project status checklist from strict runtime artifacts."""

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_bool(v: bool) -> str:
    return "PASS" if v else "FAIL"


def _pct(v: float) -> str:
    return f"{100.0 * v:.2f}%"


def _model(metrics: list[dict], name: str) -> dict | None:
    for m in metrics:
        if m.get("model") == name:
            return m
    return None


def build_snapshot(results_dir: str, training_dir: str) -> str:
    quality = _load_json(os.path.join(training_dir, "dataset_quality_report_strict_runtime.json"))
    relevance = _load_json(os.path.join(results_dir, "relevance_eval_strict_runtime.json"))
    robustness = _load_json(os.path.join(results_dir, "strict_robustness_eval_runtime.json"))
    ablation = _load_json(os.path.join(results_dir, "ablation_strict_runtime.json"))
    correctness = pd.read_csv(os.path.join(results_dir, "correctness_report_runtime.csv"))

    xgb = _model(relevance.get("metrics", []), "XGBoostBalanced") or {}
    always_sql = _model(relevance.get("metrics", []), "AlwaysSQL") or {}

    status = correctness["status"].fillna("fail") if "status" in correctness.columns else correctness["pass"].map(
        lambda x: "pass" if bool(x) else "fail"
    )
    executable = correctness[~status.isin(["skipped_missing_data", "skipped_unsupported_schema"])]
    exec_pass_rate = float(executable["pass"].mean()) if len(executable) else 0.0

    feature_drops = [float(v.get("f1_drop", 0.0)) for v in ablation.get("individual_features", {}).values()]
    group_drops = [float(v.get("f1_drop", 0.0)) for v in ablation.get("feature_groups", {}).values()]
    perm = robustness.get("permutation_importance", [])
    perm_drops = [float(v.get("mean_f1_drop", 0.0)) for v in perm]

    max_feature_drop = max(feature_drops) if feature_drops else 0.0
    max_group_drop = max(group_drops) if group_drops else 0.0
    max_perm_drop = max(perm_drops) if perm_drops else 0.0

    native_tpch_dir = os.path.join("data", "parquet", "tpch")
    native_customer = os.path.join(native_tpch_dir, "customer")
    native_orders = os.path.join(native_tpch_dir, "orders")
    native_tpch_ready = os.path.isdir(native_customer) and os.path.isdir(native_orders)

    lines = []
    lines.append("# Project Status Snapshot")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Checklist")
    lines.append("")

    lines.append(
        f"- [{ 'x' if quality.get('passed', False) else ' ' }] Dataset quality gate: {_fmt_bool(quality.get('passed', False))}"
    )
    lines.append(
        f"- [x] Eval class coverage: SQL={relevance.get('eval_label_distribution', {}).get('SQL', 0)}, "
        f"GRAPH={relevance.get('eval_label_distribution', {}).get('GRAPH', 0)}"
    )

    xgb_f1 = float(xgb.get("f1", 0.0))
    lines.append(f"- [{'x' if xgb_f1 >= 0.95 else ' '}] Relevance (XGBoostBalanced F1>=0.95): {_pct(xgb_f1)}")

    robust_f1 = float(robustness.get("xgb_eval_f1", 0.0))
    lines.append(f"- [{'x' if robust_f1 >= 0.95 else ' '}] Robustness (XGB eval F1>=0.95): {_pct(robust_f1)}")

    lines.append(
        f"- [{'x' if exec_pass_rate >= 1.0 else ' '}] Correctness executable pass rate (==100%): {_pct(exec_pass_rate)}"
    )

    non_flat = (max_feature_drop >= 0.005 and max_group_drop >= 0.005) or (max_perm_drop >= 0.05)
    lines.append(
        f"- [{'x' if non_flat else ' '}] Non-flat feature evidence (ablation/group or permutation): "
        f"feature={max_feature_drop:.4f}, group={max_group_drop:.4f}, permutation={max_perm_drop:.4f}"
    )

    lines.append(
        f"- [{'x' if native_tpch_ready else ' '}] Native TPCH parquet available: {_fmt_bool(native_tpch_ready)}"
    )

    lines.append("")
    lines.append("## Key Metrics")
    lines.append("")
    lines.append(
        f"- AlwaysSQL baseline: accuracy={_pct(float(always_sql.get('accuracy', 0.0)))}, "
        f"F1(GRAPH)={float(always_sql.get('f1', 0.0)):.4f}"
    )
    lines.append(
        f"- XGBoostBalanced: accuracy={_pct(float(xgb.get('accuracy', 0.0)))}, "
        f"F1(GRAPH)={float(xgb.get('f1', 0.0)):.4f}, precision={float(xgb.get('precision', 0.0)):.4f}, "
        f"recall={float(xgb.get('recall', 0.0)):.4f}"
    )

    lines.append("")
    lines.append("## Open Blockers")
    lines.append("")
    if not native_tpch_ready:
        lines.append("- Native TPCH gate is blocked: missing data/parquet/tpch/customer and/or data/parquet/tpch/orders")
        lines.append("- To fix: provide TPCH raw .tbl files then run make data-tpch")
    if not non_flat:
        lines.append("- Non-flatness evidence gate is blocked: both ablation and permutation drops are below thresholds")
    if native_tpch_ready and non_flat:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one-page strict status snapshot")
    parser.add_argument("--results_dir", default="experiments/results")
    parser.add_argument("--training_dir", default="training_data")
    parser.add_argument("--output", default="experiments/results/project_status_snapshot.md")
    args = parser.parse_args()

    report = build_snapshot(args.results_dir, args.training_dir)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote status snapshot: {args.output}")


if __name__ == "__main__":
    main()
