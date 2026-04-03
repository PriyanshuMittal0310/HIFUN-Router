"""Dataset quality gate for defensible SQL/GRAPH routing evaluation.

This script enforces minimum quality constraints before model training/reporting.
It is intentionally strict: failing checks should block headline experiments.
"""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, asdict

import pandas as pd


@dataclass
class GateResult:
    passed: bool
    checks: list[dict]
    summary: dict


def _row_signature(row: pd.Series, cols: list[str]) -> str:
    payload = "|".join(str(row[c]) for c in cols)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_load(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _graph_duplication_factor(df: pd.DataFrame) -> tuple[float, int, int]:
    g = df[df["label"] == "GRAPH"].copy()
    total = int(len(g))
    if total == 0:
        return float("inf"), 0, 0

    if "source_row_id" in g.columns:
        unique = int(g["source_row_id"].nunique())
    else:
        sig_cols = [
            c
            for c in [
                "dataset",
                "query_id",
                "sub_id",
                "label",
                "sql_median_ms",
                "graph_median_ms",
                "speedup",
            ]
            if c in g.columns
        ]
        if not sig_cols:
            sig_cols = list(g.columns)
        g["_sig"] = [_row_signature(r, sig_cols) for _, r in g.iterrows()]
        unique = int(g["_sig"].nunique())

    if unique == 0:
        return float("inf"), total, unique
    return float(total / unique), total, unique


def _add_check(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def evaluate_gate(
    source_df: pd.DataFrame,
    train_df: pd.DataFrame | None,
    eval_df: pd.DataFrame | None,
    min_graph_total: int,
    min_graph_train: int,
    min_graph_eval: int,
    min_graph_ratio: float,
    max_graph_duplication_factor: float,
    min_graph_datasets: int,
    min_real_measurement_share: float,
) -> GateResult:
    checks: list[dict] = []

    if "label" not in source_df.columns:
        raise ValueError("Source dataset must contain 'label' column")

    n_total = int(len(source_df))
    n_sql = int((source_df["label"] == "SQL").sum())
    n_graph = int((source_df["label"] == "GRAPH").sum())
    graph_ratio = float(n_graph / max(n_total, 1))

    _add_check(
        checks,
        "both_labels_present",
        n_sql > 0 and n_graph > 0,
        f"SQL={n_sql}, GRAPH={n_graph}",
    )

    _add_check(
        checks,
        "min_graph_total",
        n_graph >= min_graph_total,
        f"GRAPH={n_graph}, required>={min_graph_total}",
    )

    _add_check(
        checks,
        "min_graph_ratio",
        graph_ratio >= min_graph_ratio,
        f"ratio={graph_ratio:.4f}, required>={min_graph_ratio:.4f}",
    )

    if "dataset" in source_df.columns:
        graph_dataset_count = int(source_df[source_df["label"] == "GRAPH"]["dataset"].nunique())
    else:
        graph_dataset_count = 0
    _add_check(
        checks,
        "graph_across_datasets",
        graph_dataset_count >= min_graph_datasets,
        f"GRAPH present in {graph_dataset_count} dataset(s), required>={min_graph_datasets}",
    )

    dup_factor, graph_total, graph_unique = _graph_duplication_factor(source_df)
    _add_check(
        checks,
        "graph_duplication_factor",
        dup_factor <= max_graph_duplication_factor,
        f"dup_factor={dup_factor:.2f}, total={graph_total}, unique={graph_unique}, allowed<={max_graph_duplication_factor:.2f}",
    )

    if "label_source" in source_df.columns:
        real_share = float((source_df["label_source"] == "real_measurement").mean())
        _add_check(
            checks,
            "real_measurement_share",
            real_share >= min_real_measurement_share,
            f"share={real_share:.4f}, required>={min_real_measurement_share:.4f}",
        )

    if train_df is not None:
        if "label" not in train_df.columns:
            raise ValueError("Train dataset must contain 'label' column")
        n_train_graph = int((train_df["label"] == "GRAPH").sum())
        _add_check(
            checks,
            "min_graph_train",
            n_train_graph >= min_graph_train,
            f"train GRAPH={n_train_graph}, required>={min_graph_train}",
        )

    if eval_df is not None:
        if "label" not in eval_df.columns:
            raise ValueError("Eval dataset must contain 'label' column")
        n_eval_graph = int((eval_df["label"] == "GRAPH").sum())
        n_eval_sql = int((eval_df["label"] == "SQL").sum())
        _add_check(
            checks,
            "eval_both_labels_present",
            n_eval_graph > 0 and n_eval_sql > 0,
            f"eval SQL={n_eval_sql}, GRAPH={n_eval_graph}",
        )
        _add_check(
            checks,
            "min_graph_eval",
            n_eval_graph >= min_graph_eval,
            f"eval GRAPH={n_eval_graph}, required>={min_graph_eval}",
        )

    passed = all(c["passed"] for c in checks)
    summary = {
        "rows_total": n_total,
        "rows_sql": n_sql,
        "rows_graph": n_graph,
        "graph_ratio": graph_ratio,
        "graph_duplication_factor": dup_factor,
        "graph_unique_rows": graph_unique,
        "graph_total_rows": graph_total,
    }

    return GateResult(passed=passed, checks=checks, summary=summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset quality gate for defensible evaluation")
    parser.add_argument("--source", default="training_data/real_labeled_runs.csv")
    parser.add_argument("--train", default="training_data/fixed_train_base.csv")
    parser.add_argument("--eval", default="training_data/fixed_eval_set.csv")
    parser.add_argument("--min_graph_total", type=int, default=100)
    parser.add_argument("--min_graph_train", type=int, default=100)
    parser.add_argument("--min_graph_eval", type=int, default=25)
    parser.add_argument("--min_graph_ratio", type=float, default=0.10)
    parser.add_argument("--max_graph_duplication_factor", type=float, default=3.0)
    parser.add_argument("--min_graph_datasets", type=int, default=2)
    parser.add_argument("--min_real_measurement_share", type=float, default=0.90)
    parser.add_argument("--out_json", default="training_data/dataset_quality_report.json")
    args = parser.parse_args()

    source_df = pd.read_csv(args.source)
    train_df = _safe_load(args.train)
    eval_df = _safe_load(args.eval)

    result = evaluate_gate(
        source_df=source_df,
        train_df=train_df,
        eval_df=eval_df,
        min_graph_total=args.min_graph_total,
        min_graph_train=args.min_graph_train,
        min_graph_eval=args.min_graph_eval,
        min_graph_ratio=args.min_graph_ratio,
        max_graph_duplication_factor=args.max_graph_duplication_factor,
        min_graph_datasets=args.min_graph_datasets,
        min_real_measurement_share=args.min_real_measurement_share,
    )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)

    print(json.dumps(asdict(result), indent=2))
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
