"""Build a validated all-data source for training/evaluation.

Combines strict curated real-measurement labels with additional real-measurement
rows from other datasets (e.g., SNB-BI/JOB/TPCDS), while deduplicating exact
row signatures for leakage safety.
"""

import argparse
import hashlib
import json
import os
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _row_signature(row: pd.Series, cols: List[str]) -> str:
    payload = "|".join(str(row[c]) for c in cols)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pick_signature_cols(df: pd.DataFrame) -> List[str]:
    cols = [
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
        if c in df.columns
    ]
    return cols if cols else list(df.columns)


def _to_set(csv_list: str) -> set[str]:
    return {x.strip() for x in csv_list.split(",") if x.strip()}


def build_dataset(
    strict_source: str,
    full_source: str,
    output: str,
    include_datasets: set[str],
    require_real_measurement: bool,
    summary_out: str | None,
) -> Dict[str, object]:
    strict_df = pd.read_csv(strict_source)
    full_df = pd.read_csv(full_source)

    if "label" not in strict_df.columns or "label" not in full_df.columns:
        raise ValueError("Both inputs must have a 'label' column")

    base = strict_df.copy()
    extra = full_df.copy()

    if include_datasets and "dataset" in extra.columns:
        extra = extra[extra["dataset"].isin(include_datasets)].copy()

    if require_real_measurement and "label_source" in extra.columns:
        extra = extra[extra["label_source"] == "real_measurement"].copy()

    extra = extra[extra["label"].isin(["SQL", "GRAPH"])].copy()

    merged = pd.concat([base, extra], axis=0, ignore_index=True)

    sig_cols = _pick_signature_cols(merged)
    merged["source_row_id"] = [_row_signature(r, sig_cols) for _, r in merged.iterrows()]
    before_dedup = int(len(merged))
    merged = merged.drop_duplicates(subset=["source_row_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    merged.to_csv(output, index=False)

    summary: Dict[str, object] = {
        "strict_source": strict_source,
        "full_source": full_source,
        "output": output,
        "include_datasets": sorted(list(include_datasets)),
        "require_real_measurement": bool(require_real_measurement),
        "rows": {
            "strict": int(len(base)),
            "extra_filtered": int(len(extra)),
            "merged_before_dedup": before_dedup,
            "merged_after_dedup": int(len(merged)),
        },
        "labels": merged["label"].value_counts().to_dict(),
        "datasets": merged["dataset"].value_counts().to_dict() if "dataset" in merged.columns else {},
    }

    if summary_out:
        os.makedirs(os.path.dirname(summary_out), exist_ok=True)
        with open(summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validated all-data training source")
    parser.add_argument(
        "--strict_source",
        default=os.path.join(PROJECT_ROOT, "training_data", "real_labeled_runs_strict_curated.csv"),
    )
    parser.add_argument(
        "--full_source",
        default=os.path.join(PROJECT_ROOT, "training_data", "real_labeled_runs.csv"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "training_data", "real_labeled_runs_strict_all.csv"),
    )
    parser.add_argument(
        "--include_datasets",
        default="snb_real_queries,ogb_real_queries,snb_bi_real_queries,job_real_queries,tpcds_real_queries",
    )
    parser.add_argument("--allow_non_real_measurement", action="store_true")
    parser.add_argument(
        "--summary_out",
        default=os.path.join(PROJECT_ROOT, "training_data", "real_labeled_runs_strict_all_summary.json"),
    )
    args = parser.parse_args()

    summary = build_dataset(
        strict_source=args.strict_source,
        full_source=args.full_source,
        output=args.output,
        include_datasets=_to_set(args.include_datasets),
        require_real_measurement=not args.allow_non_real_measurement,
        summary_out=args.summary_out,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
