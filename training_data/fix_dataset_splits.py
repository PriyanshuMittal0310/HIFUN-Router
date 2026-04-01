"""Create fixed train/eval splits with leakage-safe balancing.

Why this exists:
- Training and evaluation must be deterministic across runs.
- Balanced training should not duplicate rows that also appear in evaluation.

Outputs (default):
- training_data/fixed_train_base.csv
- training_data/fixed_eval_set.csv
- training_data/fixed_train_balanced.csv
- training_data/fixed_split_manifest.json
"""

import argparse
import hashlib
import json
import os
from typing import Dict

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_source() -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, "training_data", "real_labeled_runs.csv"),
        os.path.join(PROJECT_ROOT, "training_data", "labeled_runs.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No source labeled dataset found.")


def _row_signature(row: pd.Series, cols: list[str]) -> str:
    payload = "|".join(str(row[c]) for c in cols)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _deterministic_eval_split(
    df: pd.DataFrame,
    eval_fraction: float,
    min_eval_per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "label" not in df.columns:
        raise ValueError("Input dataset must contain 'label' column")

    eval_parts = []
    for label, group in df.groupby("label", sort=True):
        n = len(group)
        if n < 2:
            continue
        n_eval = max(min_eval_per_class, int(round(n * eval_fraction)))
        n_eval = min(max(1, n_eval), n - 1)
        part = group.sample(n=n_eval, random_state=seed)
        eval_parts.append(part)

    if not eval_parts:
        raise ValueError("Could not create eval split: every class has <2 samples")

    eval_df = pd.concat(eval_parts, axis=0).sort_values("source_row_id")
    train_df = df.drop(index=eval_df.index).sort_values("source_row_id")
    return train_df, eval_df


def _balance_train(train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = train_df["label"].value_counts()
    if len(counts) < 2:
        return train_df.copy()

    major_label = counts.idxmax()
    minor_label = counts.idxmin()

    major = train_df[train_df["label"] == major_label].copy()
    minor = train_df[train_df["label"] == minor_label].copy()

    if len(minor) == 0:
        return train_df.copy()

    minor_up = minor.sample(n=len(major), replace=True, random_state=seed).copy()
    major["resampled"] = 0
    minor_up["resampled"] = 1

    out = pd.concat([major, minor_up], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed train/eval splits")
    parser.add_argument("--source", default=None, help="Source labeled CSV")
    parser.add_argument("--eval_fraction", type=float, default=0.2)
    parser.add_argument("--min_eval_per_class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_base_out", default=os.path.join(PROJECT_ROOT, "training_data", "fixed_train_base.csv"))
    parser.add_argument("--eval_out", default=os.path.join(PROJECT_ROOT, "training_data", "fixed_eval_set.csv"))
    parser.add_argument("--train_balanced_out", default=os.path.join(PROJECT_ROOT, "training_data", "fixed_train_balanced.csv"))
    parser.add_argument("--manifest_out", default=os.path.join(PROJECT_ROOT, "training_data", "fixed_split_manifest.json"))
    args = parser.parse_args()

    source_path = args.source or _default_source()
    df = pd.read_csv(source_path)

    sig_cols = [c for c in ["dataset", "query_id", "sub_id", "label", "sql_median_ms", "graph_median_ms", "speedup"] if c in df.columns]
    if not sig_cols:
        sig_cols = list(df.columns)

    df = df.copy()
    df["source_row_id"] = [
        _row_signature(row, sig_cols)
        for _, row in df.iterrows()
    ]

    # Deduplicate exact repeated signatures to avoid artificial leakage before split.
    df = df.drop_duplicates(subset=["source_row_id"]).reset_index(drop=True)

    train_base, eval_df = _deterministic_eval_split(
        df,
        eval_fraction=args.eval_fraction,
        min_eval_per_class=args.min_eval_per_class,
        seed=args.seed,
    )
    train_balanced = _balance_train(train_base, seed=args.seed)

    os.makedirs(os.path.dirname(args.train_base_out), exist_ok=True)
    train_base.to_csv(args.train_base_out, index=False)
    eval_df.to_csv(args.eval_out, index=False)
    train_balanced.to_csv(args.train_balanced_out, index=False)

    manifest: Dict[str, object] = {
        "source": source_path,
        "seed": args.seed,
        "eval_fraction": args.eval_fraction,
        "min_eval_per_class": args.min_eval_per_class,
        "rows": {
            "source_deduped": int(len(df)),
            "train_base": int(len(train_base)),
            "eval": int(len(eval_df)),
            "train_balanced": int(len(train_balanced)),
        },
        "labels": {
            "source_deduped": df["label"].value_counts().to_dict(),
            "train_base": train_base["label"].value_counts().to_dict(),
            "eval": eval_df["label"].value_counts().to_dict(),
            "train_balanced": train_balanced["label"].value_counts().to_dict(),
        },
        "outputs": {
            "train_base": args.train_base_out,
            "eval": args.eval_out,
            "train_balanced": args.train_balanced_out,
        },
    }

    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
