"""cross_dataset_generalization.py — Task 2.3.

Tests whether the XGBoost routing classifier generalises across datasets:
  - Train on dataset A, evaluate on dataset B (cross-dataset F1)
  - Fine-tune with 20% of target domain, re-evaluate

Produces:
  experiments/results/cross_dataset_results.csv
  experiments/results/cross_dataset_heatmap.pdf   (base + fine-tuned F1)
  experiments/results/cross_dataset_report.txt    (summary for paper)

Usage:
    python experiments/cross_dataset_generalization.py
    python experiments/cross_dataset_generalization.py \
        --labeled_csv training_data/real_labeled_runs.csv \
        --output_dir  experiments/results/
"""

import argparse
import logging
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ─── Default paths ────────────────────────────────────────────────────────────
DEFAULT_LABELED_CSV = "training_data/labeled_runs.csv"
DEFAULT_OUTPUT_DIR  = "experiments/results/"

# ─── Feature columns (22-dim, from feature_schema.json) ──────────────────────
FEATURE_COLS = [
    "op_count_filter", "op_count_join", "op_count_traversal",
    "op_count_aggregate", "op_count_map", "ast_depth",
    "has_traversal", "max_hops", "input_cardinality_log",
    "output_cardinality_log", "selectivity", "avg_degree",
    "max_degree", "degree_skew", "num_projected_columns",
    "has_index", "join_fanout", "estimated_shuffle_bytes_log",
    "estimated_traversal_ops", "hist_avg_runtime_ms",
    "hist_runtime_variance", "num_tables_joined",
]
LABEL_COL = "label"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_splits(labeled_csv: str) -> Dict[str, pd.DataFrame]:
    """Load labeled CSV and split by 'dataset' column."""
    df = pd.read_csv(labeled_csv)

    # Normalise infinite / NaN values
    df[FEATURE_COLS] = (
        df[FEATURE_COLS]
        .replace([float("inf"), float("-inf")], np.nan)
        .fillna(0.0)
    )

    splits: Dict[str, pd.DataFrame] = {}
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds].copy()
        if len(sub) >= 5:            # need at least 5 samples to be useful
            splits[ds] = sub
    return splits


def _train_xgb(X: np.ndarray, y: np.ndarray):
    """Train an XGBoost binary classifier."""
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def _f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(y_true, y_pred))


def _prepare_Xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    avail = [c for c in FEATURE_COLS if c in df.columns]
    X = df[avail].values.astype(np.float32)
    y = (df[LABEL_COL] == "GRAPH").astype(int).values
    return X, y


def _finetune_split(
    X_test: np.ndarray, y_test: np.ndarray, ft_fraction: float = 0.20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_ft, y_ft, X_holdout, y_holdout).

    ft_fraction of test samples used for fine-tuning; remainder for evaluation.
    Stratified when possible.
    """
    from sklearn.model_selection import train_test_split

    if len(np.unique(y_test)) < 2 or len(X_test) < 5:
        n = max(1, int(len(X_test) * ft_fraction))
        return X_test[:n], y_test[:n], X_test[n:], y_test[n:]

    try:
        X_ft, X_hold, y_ft, y_hold = train_test_split(
            X_test, y_test,
            train_size=ft_fraction,
            stratify=y_test,
            random_state=42,
        )
    except ValueError:
        n = max(1, int(len(X_test) * ft_fraction))
        X_ft, X_hold = X_test[:n], X_test[n:]
        y_ft, y_hold = y_test[:n], y_test[n:]

    return X_ft, y_ft, X_hold, y_hold


# ─── Main experiment ─────────────────────────────────────────────────────────

def cross_dataset_experiment(
    labeled_csv: str = DEFAULT_LABELED_CSV,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    ft_fraction: float = 0.20,
) -> pd.DataFrame:
    """Run all train→test cross-dataset pairs and produce results + plots.

    Returns a DataFrame with one row per (train_dataset, test_dataset) pair.
    """
    splits = _load_splits(labeled_csv)
    datasets = sorted(splits.keys())
    logger.info("Datasets found: %s", datasets)

    if len(datasets) < 2:
        logger.warning(
            "Need ≥ 2 datasets for cross-dataset experiment; found %d. "
            "Generate more queries with training_data/query_generator.py.",
            len(datasets),
        )

    rows = []

    for train_ds in datasets:
        X_train, y_train = _prepare_Xy(splits[train_ds])
        if len(X_train) < 10:
            logger.warning("Skipping train dataset '%s': only %d samples", train_ds, len(X_train))
            continue

        base_model = _train_xgb(X_train, y_train)

        for test_ds in datasets:
            X_test_full, y_test_full = _prepare_Xy(splits[test_ds])
            if len(X_test_full) < 5:
                continue

            same = train_ds == test_ds

            # ── Evaluate base model (no fine-tuning) ─────────────────────
            if same:
                # In-distribution: report leave-one-out cross-val instead
                from sklearn.model_selection import StratifiedKFold, cross_val_predict
                cv = StratifiedKFold(n_splits=min(5, len(X_test_full)), shuffle=True, random_state=42)
                import xgboost as xgb
                clone_model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=42, verbosity=0,
                )
                if len(np.unique(y_test_full)) > 1:
                    y_pred_base = cross_val_predict(clone_model, X_test_full, y_test_full, cv=cv)
                else:
                    y_pred_base = np.zeros_like(y_test_full)
            else:
                y_pred_base = base_model.predict(X_test_full)

            f1_base  = _f1_weighted(y_test_full, y_pred_base)
            acc_base = _accuracy(y_test_full, y_pred_base)

            # ── Fine-tune with ft_fraction of test domain ─────────────────
            X_ft, y_ft, X_hold, y_hold = _finetune_split(
                X_test_full, y_test_full, ft_fraction
            )

            # Combine train + fine-tune split and retrain
            X_combined = np.vstack([X_train, X_ft])
            y_combined = np.hstack([y_train, y_ft])
            ft_model = _train_xgb(X_combined, y_combined)

            if len(X_hold) >= 2:
                y_pred_ft = ft_model.predict(X_hold)
                y_true_ft = y_hold
            else:
                # Fallback: evaluate on full test set (slight optimistic bias)
                y_pred_ft = ft_model.predict(X_test_full)
                y_true_ft = y_test_full

            f1_ft  = _f1_weighted(y_true_ft, y_pred_ft)
            acc_ft = _accuracy(y_true_ft, y_pred_ft)

            logger.info(
                "  Train=%-20s → Test=%-20s | "
                "F1_base=%.3f  F1_ft=%.3f  (Δ%+.3f)  acc_base=%.3f",
                train_ds, test_ds, f1_base, f1_ft, f1_ft - f1_base, acc_base,
            )

            rows.append({
                "train_dataset":  train_ds,
                "test_dataset":   test_ds,
                "train_samples":  len(X_train),
                "test_samples":   len(X_test_full),
                "ft_samples":     len(X_ft),
                "f1_base":        round(f1_base,          3),
                "acc_base":       round(acc_base,         3),
                "f1_finetuned":   round(f1_ft,            3),
                "acc_finetuned":  round(acc_ft,           3),
                "f1_improvement": round(f1_ft - f1_base,  3),
                "same_dataset":   same,
            })

    results_df = pd.DataFrame(rows)

    if results_df.empty:
        logger.warning("No cross-dataset pairs produced results; check labeled_csv.")
        return results_df

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "cross_dataset_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved cross-dataset results → %s", csv_path)

    # ── Heat-maps ───────────────────────────────────────────────────────────
    _plot_heatmaps(results_df, output_dir)

    # ── Text report for paper ────────────────────────────────────────────────
    _write_report(results_df, output_dir)

    return results_df


def _plot_heatmaps(df: pd.DataFrame, output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed; skipping heatmap plots")
        return

    for metric, title_suffix in [
        ("f1_base",      "Base (No Fine-Tuning)"),
        ("f1_finetuned", f"After 20% Fine-Tuning"),
    ]:
        try:
            pivot = df.pivot(
                index="train_dataset", columns="test_dataset", values=metric
            )
        except Exception as exc:
            logger.warning("Cannot pivot for '%s': %s", metric, exc)
            continue

        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5),
                                        max(4, len(pivot.index) * 1.2)))
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="YlGn",
            vmin=0.0, vmax=1.0, linewidths=0.5, ax=ax,
        )
        ax.set_title(f"Cross-Dataset Routing F1 — {title_suffix}", fontsize=13, pad=12)
        ax.set_xlabel("Test Dataset",  fontsize=11)
        ax.set_ylabel("Train Dataset", fontsize=11)
        plt.tight_layout()

        fname = f"cross_dataset_heatmap_{metric}.pdf"
        fig.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved heatmap → %s", os.path.join(output_dir, fname))


def _write_report(df: pd.DataFrame, output_dir: str) -> None:
    """Write a plain-text summary suitable for inclusion in the paper."""
    lines = [
        "HIFUN Router — Cross-Dataset Generalisation Report",
        "=" * 60,
        "",
    ]

    same    = df[df["same_dataset"]]
    cross   = df[~df["same_dataset"]]

    if not same.empty:
        lines.append(
            f"In-distribution F1 (diagonal):  "
            f"mean={same['f1_base'].mean():.3f}  "
            f"min={same['f1_base'].min():.3f}  "
            f"max={same['f1_base'].max():.3f}"
        )
    if not cross.empty:
        lines.append(
            f"Cross-dataset F1  (off-diag):   "
            f"mean={cross['f1_base'].mean():.3f}  "
            f"min={cross['f1_base'].min():.3f}  "
            f"max={cross['f1_base'].max():.3f}"
        )
        lines.append(
            f"After 20%% fine-tuning:          "
            f"mean={cross['f1_finetuned'].mean():.3f}  "
            f"mean improvement={cross['f1_improvement'].mean():+.3f}"
        )

    lines.append("")
    lines.append(df[[
        "train_dataset", "test_dataset",
        "train_samples", "test_samples",
        "f1_base", "f1_finetuned", "f1_improvement",
    ]].to_string(index=False))

    report_path = os.path.join(output_dir, "cross_dataset_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved text report → %s", report_path)
    print("\n".join(lines))


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-dataset generalisation experiment for HIFUN Router"
    )
    p.add_argument("--labeled_csv",  default=DEFAULT_LABELED_CSV)
    p.add_argument("--output_dir",   default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--ft_fraction",  type=float, default=0.20,
                   help="Fraction of test domain used for fine-tuning (default 0.20)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cross_dataset_experiment(
        labeled_csv=args.labeled_csv,
        output_dir=args.output_dir,
        ft_fraction=args.ft_fraction,
    )
