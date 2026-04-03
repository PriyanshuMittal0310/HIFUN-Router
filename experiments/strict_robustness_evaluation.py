"""Strict robustness evaluation for the SQL/GRAPH router.

This script complements relevance and ablation reports with:
1) Bootstrap confidence intervals on untouched eval predictions.
2) Label-permutation sanity checks (detects leakage or trivial memorization).
3) Permutation feature importance on strict eval.
4) Cross-dataset transfer matrix on strict curated data.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import RESULTS_DIR
from features.feature_extractor import FEATURE_NAMES


def _default_train() -> str:
    return os.path.join(PROJECT_ROOT, "training_data", "fixed_train_base_strict.csv")


def _default_eval() -> str:
    return os.path.join(PROJECT_ROOT, "training_data", "fixed_eval_set_strict.csv")


def _default_transfer_source() -> str:
    return os.path.join(PROJECT_ROOT, "training_data", "real_labeled_runs_strict_curated.csv")


def _load_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _xy(df: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_names].values.astype(np.float32)
    y = (df["label"] == "GRAPH").astype(int).values
    return X, y


def _xgb(scale_pos_weight: float, seed: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )


def _bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1s = []
    precs = []
    recs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        f1s.append(float(f1_score(yt, yp, zero_division=0)))
        precs.append(float(precision_score(yt, yp, zero_division=0)))
        recs.append(float(recall_score(yt, yp, zero_division=0)))

    def q(arr: list[float], p: float) -> float:
        return float(np.quantile(np.array(arr, dtype=float), p))

    return {
        "n_bootstrap": n_bootstrap,
        "f1": {"p2_5": q(f1s, 0.025), "p50": q(f1s, 0.5), "p97_5": q(f1s, 0.975)},
        "precision": {"p2_5": q(precs, 0.025), "p50": q(precs, 0.5), "p97_5": q(precs, 0.975)},
        "recall": {"p2_5": q(recs, 0.025), "p50": q(recs, 0.5), "p97_5": q(recs, 0.975)},
    }


def _label_permutation_sanity(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    scale_pos_weight: float,
    n_perm: int,
    seed: int,
    train_groups: np.ndarray | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    f1_scores = []
    for i in range(n_perm):
        y_perm = np.array(y_train, copy=True)
        if train_groups is None:
            rng.shuffle(y_perm)
        else:
            # Preserve per-group label counts while destroying per-row correspondence.
            for g in np.unique(train_groups):
                idx = np.where(train_groups == g)[0]
                if len(idx) > 1:
                    rng.shuffle(y_perm[idx])
        model = _xgb(scale_pos_weight=scale_pos_weight, seed=seed + i + 1)
        model.fit(X_train, y_perm)
        y_pred = model.predict(X_eval)
        f1_scores.append(float(f1_score(y_eval, y_pred, zero_division=0)))

    return {
        "n_permutations": n_perm,
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "f1_min": float(np.min(f1_scores)),
        "f1_max": float(np.max(f1_scores)),
        "scores": f1_scores,
    }


def _permutation_importance(
    model: xgb.XGBClassifier,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_names: list[str],
    repeats: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    base_pred = model.predict(X_eval)
    base_f1 = float(f1_score(y_eval, base_pred, zero_division=0))

    rows = []
    for col, name in enumerate(feature_names):
        drops = []
        for _ in range(repeats):
            Xp = np.array(X_eval, copy=True)
            perm_idx = rng.permutation(len(Xp))
            Xp[:, col] = Xp[perm_idx, col]
            yp = model.predict(Xp)
            f1p = float(f1_score(y_eval, yp, zero_division=0))
            drops.append(base_f1 - f1p)
        rows.append({
            "feature": name,
            "mean_f1_drop": float(np.mean(drops)),
            "std_f1_drop": float(np.std(drops)),
            "max_f1_drop": float(np.max(drops)),
        })

    rows.sort(key=lambda r: r["mean_f1_drop"], reverse=True)
    return rows


def _cross_dataset_transfer(df: pd.DataFrame, feature_names: list[str]) -> list[dict]:
    if "dataset" not in df.columns:
        return []
    results = []
    datasets = sorted(df["dataset"].dropna().unique().tolist())
    for train_ds in datasets:
        train_df = df[df["dataset"] == train_ds]
        X_train, y_train = _xy(train_df, feature_names)
        if len(np.unique(y_train)) < 2:
            continue
        sql_count = int((y_train == 0).sum())
        graph_count = int((y_train == 1).sum())
        model = _xgb(scale_pos_weight=float(sql_count / max(graph_count, 1)), seed=42)
        model.fit(X_train, y_train)
        for eval_ds in datasets:
            if eval_ds == train_ds:
                continue
            eval_df = df[df["dataset"] == eval_ds]
            X_eval, y_eval = _xy(eval_df, feature_names)
            y_pred = model.predict(X_eval)
            results.append({
                "train_dataset": train_ds,
                "eval_dataset": eval_ds,
                "rows": int(len(eval_df)),
                "f1": float(f1_score(y_eval, y_pred, zero_division=0)),
                "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
                "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
            })
    return results


def _render_md(report: dict) -> str:
    lines = [
        "# Strict Robustness Evaluation",
        "",
        f"- Train: {report['train_path']}",
        f"- Eval: {report['eval_path']}",
        f"- Transfer source: {report['transfer_source']}",
        "",
        "## Core Strict Metric",
        "",
        f"- XGBoost eval F1: {report['xgb_eval_f1']:.4f}",
        "",
        "## Bootstrap 95% CI (Eval)",
        "",
    ]
    ci = report["bootstrap_ci"]
    lines.append(
        f"- F1: [{ci['f1']['p2_5']:.4f}, {ci['f1']['p97_5']:.4f}] (median {ci['f1']['p50']:.4f})"
    )
    lines.append(
        f"- Precision: [{ci['precision']['p2_5']:.4f}, {ci['precision']['p97_5']:.4f}] (median {ci['precision']['p50']:.4f})"
    )
    lines.append(
        f"- Recall: [{ci['recall']['p2_5']:.4f}, {ci['recall']['p97_5']:.4f}] (median {ci['recall']['p50']:.4f})"
    )

    lp = report["label_permutation_sanity"]
    lines.extend([
        "",
        "## Label-Permutation Sanity",
        "",
        f"- Mode: {lp.get('mode', 'global_shuffle')}",
        f"- Permuted-label mean F1: {lp['f1_mean']:.4f} ± {lp['f1_std']:.4f}",
        f"- Permuted-label range: [{lp['f1_min']:.4f}, {lp['f1_max']:.4f}]",
    ])

    overlap = report.get("overlap_audit", {})
    if overlap:
        lines.extend([
            "",
            "## Overlap Audit",
            "",
            f"- Overlap on source_row_id: {overlap.get('source_row_id_overlap', 'n/a')}",
            f"- Overlap on query_id: {overlap.get('query_id_overlap', 'n/a')}",
            f"- Overlap on sub_id: {overlap.get('sub_id_overlap', 'n/a')}",
        ])

    lines.extend([
        "",
        "## Top Permutation Importance (Eval F1 Drop)",
        "",
        "| Feature | Mean Drop | Std | Max Drop |",
        "|---|---:|---:|---:|",
    ])
    for row in report["permutation_importance"][:10]:
        lines.append(
            f"| {row['feature']} | {row['mean_f1_drop']:.4f} | {row['std_f1_drop']:.4f} | {row['max_f1_drop']:.4f} |"
        )

    if report["cross_dataset_transfer"]:
        lines.extend([
            "",
            "## Cross-Dataset Transfer",
            "",
            "| Train | Eval | Rows | F1 | Precision | Recall |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for row in report["cross_dataset_transfer"]:
            lines.append(
                f"| {row['train_dataset']} | {row['eval_dataset']} | {row['rows']} | {row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict robustness evaluation")
    parser.add_argument("--train", default=_default_train())
    parser.add_argument("--eval", dest="eval_path", default=_default_eval())
    parser.add_argument("--transfer_source", default=_default_transfer_source())
    parser.add_argument("--out_json", default=os.path.join(RESULTS_DIR, "strict_robustness_eval.json"))
    parser.add_argument("--out_md", default=os.path.join(RESULTS_DIR, "strict_robustness_eval.md"))
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--n_perm_labels", type=int, default=100)
    parser.add_argument("--n_perm_features", type=int, default=30)
    parser.add_argument("--perm_group_col", default="query_id", help="Group column for within-group label permutation sanity")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_df = _load_df(args.train)
    eval_df = _load_df(args.eval_path)
    transfer_df = _load_df(args.transfer_source)

    feature_names = [f for f in FEATURE_NAMES if f in train_df.columns and f in eval_df.columns]
    if not feature_names:
        raise ValueError("No shared features found in train/eval")

    X_train, y_train = _xy(train_df, feature_names)
    X_eval, y_eval = _xy(eval_df, feature_names)

    overlap_audit = {}
    if "source_row_id" in train_df.columns and "source_row_id" in eval_df.columns:
        overlap_audit["source_row_id_overlap"] = int(
            len(set(train_df["source_row_id"].astype(str)).intersection(set(eval_df["source_row_id"].astype(str))))
        )
    if "query_id" in train_df.columns and "query_id" in eval_df.columns:
        overlap_audit["query_id_overlap"] = int(
            len(set(train_df["query_id"].astype(str)).intersection(set(eval_df["query_id"].astype(str))))
        )
    if "sub_id" in train_df.columns and "sub_id" in eval_df.columns:
        overlap_audit["sub_id_overlap"] = int(
            len(set(train_df["sub_id"].astype(str)).intersection(set(eval_df["sub_id"].astype(str))))
        )

    sql_count = int((y_train == 0).sum())
    graph_count = int((y_train == 1).sum())
    spw = float(sql_count / max(graph_count, 1))

    xgb_model = _xgb(scale_pos_weight=spw, seed=args.seed)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_eval)
    xgb_eval_f1 = float(f1_score(y_eval, y_pred, zero_division=0))

    # Secondary baseline for context.
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_eval)
    logreg_eval_f1 = float(f1_score(y_eval, y_pred_lr, zero_division=0))

    bootstrap_ci = _bootstrap_ci(y_eval, y_pred, n_bootstrap=args.n_bootstrap, seed=args.seed)
    perm_groups = None
    perm_mode = "global_shuffle"
    if args.perm_group_col and args.perm_group_col in train_df.columns:
        perm_groups = train_df[args.perm_group_col].astype(str).values
        perm_mode = f"within_group:{args.perm_group_col}"

    perm_sanity = _label_permutation_sanity(
        X_train,
        y_train,
        X_eval,
        y_eval,
        scale_pos_weight=spw,
        n_perm=args.n_perm_labels,
        seed=args.seed,
        train_groups=perm_groups,
    )
    perm_sanity["mode"] = perm_mode
    perm_importance = _permutation_importance(
        xgb_model,
        X_eval,
        y_eval,
        feature_names=feature_names,
        repeats=args.n_perm_features,
        seed=args.seed,
    )

    transfer_features = [f for f in FEATURE_NAMES if f in transfer_df.columns]
    transfer_rows = _cross_dataset_transfer(transfer_df, transfer_features)

    report = {
        "train_path": args.train,
        "eval_path": args.eval_path,
        "transfer_source": args.transfer_source,
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "xgb_eval_f1": xgb_eval_f1,
        "logreg_eval_f1": logreg_eval_f1,
        "bootstrap_ci": bootstrap_ci,
        "label_permutation_sanity": perm_sanity,
        "overlap_audit": overlap_audit,
        "permutation_importance": perm_importance,
        "cross_dataset_transfer": transfer_rows,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(_render_md(report))

    print(f"Saved: {args.out_json}")
    print(f"Saved: {args.out_md}")


if __name__ == "__main__":
    main()