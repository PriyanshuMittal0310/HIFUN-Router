"""Create fixed train/eval splits with leakage-safe balancing.

Why this exists:
- Training and evaluation must be deterministic across runs.
- Balanced training should not duplicate rows that also appear in evaluation.

Outputs (default):
- training_data/fixed_train_base.csv
- training_data/fixed_eval_set.csv
- training_data/fixed_eval_graph_only.csv
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


def _deterministic_group_eval_split(
    df: pd.DataFrame,
    eval_fraction: float,
    min_eval_per_class: int,
    seed: int,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "label" not in df.columns:
        raise ValueError("Input dataset must contain 'label' column")
    if group_col not in df.columns:
        raise ValueError(f"Group-disjoint split requested but '{group_col}' column is missing")

    rng = pd.Series(df[group_col].dropna().unique()).sample(frac=1.0, random_state=seed).tolist()
    if not rng:
        raise ValueError("No groups found for group-disjoint split")

    label_counts = df["label"].value_counts().to_dict()
    target_eval_rows = max(1, int(round(len(df) * eval_fraction)))

    eval_groups: set[str] = set()

    # First satisfy per-class minimum rows in eval.
    for label in sorted(df["label"].unique().tolist()):
        need = min_eval_per_class
        current = 0
        candidate_groups = (
            df[df["label"] == label][group_col]
            .dropna()
            .drop_duplicates()
            .sample(frac=1.0, random_state=seed)
            .tolist()
        )
        for g in candidate_groups:
            if g in eval_groups:
                current = int((df[(df[group_col].isin(eval_groups)) & (df["label"] == label)]).shape[0])
                if current >= need:
                    break
                continue
            eval_groups.add(g)
            current = int((df[(df[group_col].isin(eval_groups)) & (df["label"] == label)]).shape[0])
            if current >= need:
                break

    # Then fill up to target eval rows.
    for g in rng:
        if g in eval_groups:
            continue
        current_rows = int(df[df[group_col].isin(eval_groups)].shape[0])
        if current_rows >= target_eval_rows:
            break
        eval_groups.add(g)

    eval_df = df[df[group_col].isin(eval_groups)].copy()
    train_df = df[~df[group_col].isin(eval_groups)].copy()

    # Safety checks: both classes must remain in train and eval.
    for label, total in label_counts.items():
        tr = int((train_df["label"] == label).sum())
        ev = int((eval_df["label"] == label).sum())
        if total > 1 and (tr == 0 or ev == 0):
            raise ValueError(
                f"Group split invalid for label={label}: train={tr}, eval={ev}, total={total}."
            )

    return train_df.sort_values("source_row_id"), eval_df.sort_values("source_row_id")


def _balance_train(
    train_df: pd.DataFrame,
    seed: int,
    min_unique_minority_for_upsample: int,
) -> tuple[pd.DataFrame, bool, str]:
    counts = train_df["label"].value_counts()
    if len(counts) < 2:
        out = train_df.copy()
        out["resampled"] = 0
        return out, False, "single_class"

    major_label = counts.idxmax()
    minor_label = counts.idxmin()

    major = train_df[train_df["label"] == major_label].copy()
    minor = train_df[train_df["label"] == minor_label].copy()

    if len(minor) == 0:
        out = train_df.copy()
        out["resampled"] = 0
        return out, False, "no_minority_class"

    unique_minority = int(minor["source_row_id"].nunique()) if "source_row_id" in minor.columns else int(len(minor))
    if unique_minority < min_unique_minority_for_upsample:
        # Refuse to upsample when the minority class has too few unique examples;
        # this prevents memorizing a handful of rows and reporting inflated metrics.
        out = train_df.copy()
        out["resampled"] = 0
        return out, False, f"minority_unique_too_small:{unique_minority}"

    minor_up = minor.sample(n=len(major), replace=True, random_state=seed).copy()
    major["resampled"] = 0
    minor_up["resampled"] = 1

    out = pd.concat([major, minor_up], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out, True, "ok"


def _assert_min_graph_support(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    min_graph_train: int,
    min_graph_eval: int,
    allow_degenerate: bool,
) -> None:
    train_graph = int((train_df["label"] == "GRAPH").sum())
    eval_graph = int((eval_df["label"] == "GRAPH").sum())

    problems = []
    if train_graph < min_graph_train:
        problems.append(
            f"train GRAPH rows={train_graph} < required {min_graph_train}"
        )
    if eval_graph < min_graph_eval:
        problems.append(
            f"eval GRAPH rows={eval_graph} < required {min_graph_eval}"
        )

    if problems and not allow_degenerate:
        joined = "; ".join(problems)
        raise ValueError(
            "Degenerate label distribution detected: "
            + joined
            + ". Generate additional real graph-winning workloads before training/evaluation, "
            + "or rerun with --allow_degenerate for debugging only."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed train/eval splits")
    parser.add_argument("--source", default=None, help="Source labeled CSV")
    parser.add_argument("--eval_fraction", type=float, default=0.2)
    parser.add_argument("--min_eval_per_class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_mode", choices=["group", "row"], default="group")
    parser.add_argument("--group_col", default="query_id")
    parser.add_argument("--min_graph_train", type=int, default=100)
    parser.add_argument("--min_graph_eval", type=int, default=25)
    parser.add_argument("--allow_degenerate", action="store_true")
    parser.add_argument("--min_unique_graph_for_upsample", type=int, default=40)
    parser.add_argument("--train_base_out", default=os.path.join(PROJECT_ROOT, "training_data", "fixed_train_base.csv"))
    parser.add_argument("--eval_out", default=os.path.join(PROJECT_ROOT, "training_data", "fixed_eval_set.csv"))
    parser.add_argument("--graph_eval_out", default=os.path.join(PROJECT_ROOT, "training_data", "fixed_eval_graph_only.csv"))
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

    split_mode_used = args.split_mode
    if args.split_mode == "group" and args.group_col in df.columns and df[args.group_col].nunique() > 1:
        train_base, eval_df = _deterministic_group_eval_split(
            df,
            eval_fraction=args.eval_fraction,
            min_eval_per_class=args.min_eval_per_class,
            seed=args.seed,
            group_col=args.group_col,
        )
    else:
        split_mode_used = "row"
        train_base, eval_df = _deterministic_eval_split(
            df,
            eval_fraction=args.eval_fraction,
            min_eval_per_class=args.min_eval_per_class,
            seed=args.seed,
        )

    _assert_min_graph_support(
        train_base,
        eval_df,
        min_graph_train=args.min_graph_train,
        min_graph_eval=args.min_graph_eval,
        allow_degenerate=args.allow_degenerate,
    )

    train_balanced, balanced_applied, balanced_reason = _balance_train(
        train_base,
        seed=args.seed,
        min_unique_minority_for_upsample=args.min_unique_graph_for_upsample,
    )

    graph_eval_df = eval_df[eval_df["label"] == "GRAPH"].copy().reset_index(drop=True)

    os.makedirs(os.path.dirname(args.train_base_out), exist_ok=True)
    train_base.to_csv(args.train_base_out, index=False)
    eval_df.to_csv(args.eval_out, index=False)
    graph_eval_df.to_csv(args.graph_eval_out, index=False)
    train_balanced.to_csv(args.train_balanced_out, index=False)

    manifest: Dict[str, object] = {
        "source": source_path,
        "seed": args.seed,
        "split_mode": split_mode_used,
        "group_col": args.group_col,
        "eval_fraction": args.eval_fraction,
        "min_eval_per_class": args.min_eval_per_class,
        "min_graph_train": args.min_graph_train,
        "min_graph_eval": args.min_graph_eval,
        "allow_degenerate": bool(args.allow_degenerate),
        "min_unique_graph_for_upsample": args.min_unique_graph_for_upsample,
        "rows": {
            "source_deduped": int(len(df)),
            "train_base": int(len(train_base)),
            "eval": int(len(eval_df)),
            "eval_graph_only": int(len(graph_eval_df)),
            "train_balanced": int(len(train_balanced)),
        },
        "labels": {
            "source_deduped": df["label"].value_counts().to_dict(),
            "train_base": train_base["label"].value_counts().to_dict(),
            "eval": eval_df["label"].value_counts().to_dict(),
            "eval_graph_only": graph_eval_df["label"].value_counts().to_dict(),
            "train_balanced": train_balanced["label"].value_counts().to_dict(),
        },
        "balancing": {
            "applied": bool(balanced_applied),
            "reason": balanced_reason,
        },
        "overlap": {
            "source_row_id": int(len(set(train_base["source_row_id"]).intersection(set(eval_df["source_row_id"])))),
            "query_id": int(len(set(train_base["query_id"]).intersection(set(eval_df["query_id"])))) if "query_id" in train_base.columns and "query_id" in eval_df.columns else None,
            "sub_id": int(len(set(train_base["sub_id"]).intersection(set(eval_df["sub_id"])))) if "sub_id" in train_base.columns and "sub_id" in eval_df.columns else None,
        },
        "outputs": {
            "train_base": args.train_base_out,
            "eval": args.eval_out,
            "eval_graph_only": args.graph_eval_out,
            "train_balanced": args.train_balanced_out,
        },
    }

    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
