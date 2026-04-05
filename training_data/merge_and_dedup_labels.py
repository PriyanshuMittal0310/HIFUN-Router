"""Merge labeled CSV files and deduplicate stable row signatures."""

import argparse
import hashlib
import json
import os
from typing import List

import pandas as pd


def _sig(row: pd.Series, cols: List[str]) -> str:
    payload = "|".join(str(row[c]) for c in cols)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Merge and dedup labeled run CSVs")
    p.add_argument("--base", required=True)
    p.add_argument("--extra", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary_out", default="")
    args = p.parse_args()

    b = pd.read_csv(args.base)
    e = pd.read_csv(args.extra)

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
        if c in b.columns and c in e.columns
    ]
    if not cols:
        cols = list(set(b.columns).intersection(set(e.columns)))

    merged = pd.concat([b, e], ignore_index=True)
    merged["merge_row_id"] = [_sig(r, cols) for _, r in merged.iterrows()]
    merged = merged.drop_duplicates(subset=["merge_row_id"]).drop(columns=["merge_row_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    merged.to_csv(args.output, index=False)

    summary = {
        "base_rows": int(len(b)),
        "extra_rows": int(len(e)),
        "merged_rows": int(len(merged)),
        "labels": merged["label"].value_counts().to_dict() if "label" in merged.columns else {},
        "by_dataset": merged.groupby(["dataset", "label"]).size().unstack(fill_value=0).to_dict() if {"dataset", "label"}.issubset(set(merged.columns)) else {},
    }
    if args.summary_out:
        os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
        with open(args.summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
