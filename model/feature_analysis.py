"""feature_analysis.py — Task 2.5: Variance Inflation Factor (VIF) analysis.

Detects multi-collinearity among the 22 routing features. Features with
VIF > 10 are severely collinear; > 5 indicates moderate collinearity.

For tree ensemble models (XGBoost) collinearity does NOT degrade predictive
performance, but it does make feature-importance plots misleading.  This
analysis lets us:
  1. Report collinearity explicitly in the paper (Section III.B).
  2. Optionally remove redundant features and verify F1 is unchanged (ablation).
  3. Address reviewer concerns about feature independence.

Produces:
  experiments/results/vif_analysis.csv
  experiments/results/vif_correlation_heatmap.pdf
  experiments/results/vif_report.txt

Usage:
    python model/feature_analysis.py
    python model/feature_analysis.py \
        --labeled_csv training_data/real_labeled_runs.csv \
        --output_dir  experiments/results/
"""

import argparse
import logging
import os
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ─── Default paths ────────────────────────────────────────────────────────────
DEFAULT_LABELED_CSV = "training_data/labeled_runs.csv"
DEFAULT_OUTPUT_DIR  = "experiments/results/"

# ─── Feature columns (22-dim, from feature_schema.json) ──────────────────────
FEATURE_COLS: List[str] = [
    "op_count_filter", "op_count_join", "op_count_traversal",
    "op_count_aggregate", "op_count_map", "ast_depth",
    "has_traversal", "max_hops", "input_cardinality_log",
    "output_cardinality_log", "selectivity", "avg_degree",
    "max_degree", "degree_skew", "num_projected_columns",
    "has_index", "join_fanout", "estimated_shuffle_bytes_log",
    "estimated_traversal_ops", "hist_avg_runtime_ms",
    "hist_runtime_variance", "num_tables_joined",
]

# Features that are derived from others — safe to remove if VIF is very high
DERIVED_FEATURES = [
    "estimated_shuffle_bytes_log",  # derived from output_cardinality_log
    "output_cardinality_log",       # derived from input_cardinality_log × selectivity
    "estimated_traversal_ops",      # derived from avg_degree, max_hops
    "degree_skew",                  # derived from avg_degree, max_degree
]

# VIF thresholds
VIF_SEVERE   = 10.0
VIF_MODERATE = 5.0


# ─── VIF computation ──────────────────────────────────────────────────────────

def compute_vif(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each feature column.

    Args:
        df:          DataFrame containing feature columns.
        feature_cols: Ordered list of column names to include.

    Returns:
        DataFrame with columns [Feature, VIF, Collinearity_Level].
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        logger.warning(
            "statsmodels not installed.  Run: pip install statsmodels\n"
            "Falling back to correlation-based collinearity estimate."
        )
        return _correlation_based_vif(df, feature_cols)

    avail = [c for c in feature_cols if c in df.columns]
    X = (
        df[avail]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values
        .astype(float)
    )

    vif_rows = []
    for i, col in enumerate(avail):
        try:
            vif = variance_inflation_factor(X, i)
        except Exception:
            vif = float("inf")
        level = (
            "severe"   if vif > VIF_SEVERE   else
            "moderate" if vif > VIF_MODERATE else
            "ok"
        )
        vif_rows.append({"Feature": col, "VIF": round(vif, 2), "Collinearity_Level": level})

    return pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)


def _correlation_based_vif(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Approximate VIF via R² from OLS for each feature (fallback)."""
    from sklearn.linear_model import LinearRegression

    avail = [c for c in feature_cols if c in df.columns]
    X = (
        df[avail]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values
        .astype(float)
    )

    vif_rows = []
    for i, col in enumerate(avail):
        y    = X[:, i]
        X_ot = np.delete(X, i, axis=1)
        r2   = LinearRegression().fit(X_ot, y).score(X_ot, y)
        vif  = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        level = (
            "severe"   if vif > VIF_SEVERE   else
            "moderate" if vif > VIF_MODERATE else
            "ok"
        )
        vif_rows.append({"Feature": col, "VIF": round(vif, 2), "Collinearity_Level": level})

    return pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)


# ─── Feature removal recommendation ──────────────────────────────────────────

def recommend_removals(
    vif_df: pd.DataFrame,
    threshold: float = VIF_SEVERE,
    prefer_derived: bool = True,
) -> List[str]:
    """Return an ordered list of features to remove to reduce VIF below threshold.

    Args:
        vif_df:         Output of compute_vif().
        threshold:      VIF level above which features are candidates for removal.
        prefer_derived: If True, prefer removing features in DERIVED_FEATURES
                        before primary features.
    Returns:
        Ordered list of feature names to remove.
    """
    high_vif = set(vif_df[vif_df["VIF"] > threshold]["Feature"].tolist())
    if not high_vif:
        return []

    if prefer_derived:
        removal_order = [f for f in DERIVED_FEATURES if f in high_vif]
        removal_order += [f for f in high_vif if f not in DERIVED_FEATURES]
    else:
        removal_order = vif_df[vif_df["VIF"] > threshold]["Feature"].tolist()

    return removal_order


def ablation_without_features(
    labeled_csv: str,
    features_to_remove: List[str],
    n_cv_folds: int = 5,
) -> dict:
    """Cross-validate XGBoost with the flagged features removed.

    Returns a dict with F1 (full set) vs F1 (reduced set).
    """
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    df = pd.read_csv(labeled_csv)
    df[FEATURE_COLS] = (
        df[FEATURE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    y = (df["label"] == "GRAPH").astype(int).values

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)

    # Full feature set
    avail_full = [c for c in FEATURE_COLS if c in df.columns]
    X_full = df[avail_full].values.astype(np.float32)
    scores_full = cross_val_score(model, X_full, y, cv=cv, scoring="f1_weighted")

    # Reduced feature set
    reduced = [c for c in avail_full if c not in features_to_remove]
    X_red   = df[reduced].values.astype(np.float32)
    scores_red = cross_val_score(model, X_red, y, cv=cv, scoring="f1_weighted")

    return {
        "f1_full":        round(scores_full.mean(), 3),
        "f1_full_std":    round(scores_full.std(),  3),
        "f1_reduced":     round(scores_red.mean(),  3),
        "f1_reduced_std": round(scores_red.std(),   3),
        "features_removed": features_to_remove,
        "n_features_full":  len(avail_full),
        "n_features_reduced": len(reduced),
    }


# ─── Correlation heat-map ─────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """Save a Pearson correlation heat-map of all feature pairs."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available; skipping correlation heatmap")
        return

    avail = [c for c in FEATURE_COLS if c in df.columns]
    X = (
        df[avail]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    corr = X.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # upper triangle
    sns.heatmap(
        corr, mask=mask, annot=False, cmap="coolwarm",
        vmin=-1, vmax=1, linewidths=0.3, ax=ax,
    )
    ax.set_title("HIFUN Router — Feature Pairwise Correlation", fontsize=13, pad=12)
    plt.tight_layout()
    out = os.path.join(output_dir, "vif_correlation_heatmap.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correlation heatmap → %s", out)


# ─── Main analysis ────────────────────────────────────────────────────────────

def run_vif_analysis(
    labeled_csv: str = DEFAULT_LABELED_CSV,
    output_dir: str  = DEFAULT_OUTPUT_DIR,
    vif_threshold: float = VIF_SEVERE,
    run_ablation: bool   = True,
) -> pd.DataFrame:
    """Full VIF analysis pipeline.

    1. Load labeled CSV.
    2. Compute per-feature VIF.
    3. Recommend features to remove.
    4. (Optional) ablation: re-run CV without removed features.
    5. Save CSV, text report, and correlation heatmap.

    Returns the VIF DataFrame.
    """
    if not os.path.exists(labeled_csv):
        logger.error(
            "Labeled CSV not found: %s\nRun training_data/real_collection_script.py first.",
            labeled_csv,
        )
        # Return empty but valid DataFrame so downstream code doesn't crash
        return pd.DataFrame(columns=["Feature", "VIF", "Collinearity_Level"])

    df = pd.read_csv(labeled_csv)
    logger.info("Loaded %d rows from %s", len(df), labeled_csv)

    os.makedirs(output_dir, exist_ok=True)

    # ── VIF table ────────────────────────────────────────────────────────────
    vif_df = compute_vif(df, FEATURE_COLS)
    csv_path = os.path.join(output_dir, "vif_analysis.csv")
    vif_df.to_csv(csv_path, index=False)
    logger.info("VIF Analysis:\n%s", vif_df.to_string(index=False))

    # ── Recommendations ──────────────────────────────────────────────────────
    to_remove = recommend_removals(vif_df, threshold=vif_threshold)

    # ── Correlation heat-map ─────────────────────────────────────────────────
    plot_correlation_heatmap(df, output_dir)

    # ── Ablation ─────────────────────────────────────────────────────────────
    ablation_result: Optional[dict] = None
    if run_ablation and to_remove:
        try:
            ablation_result = ablation_without_features(labeled_csv, to_remove)
            logger.info(
                "Ablation: F1_full=%.3f ± %.3f  F1_reduced=%.3f ± %.3f  (removed %s)",
                ablation_result["f1_full"],
                ablation_result["f1_full_std"],
                ablation_result["f1_reduced"],
                ablation_result["f1_reduced_std"],
                to_remove,
            )
        except Exception as exc:
            logger.warning("Ablation failed: %s", exc)

    # ── Text report ──────────────────────────────────────────────────────────
    _write_vif_report(vif_df, to_remove, ablation_result, vif_threshold, output_dir)

    return vif_df


def _write_vif_report(
    vif_df: pd.DataFrame,
    to_remove: List[str],
    ablation: Optional[dict],
    threshold: float,
    output_dir: str,
) -> None:
    lines = [
        "HIFUN Router — Feature Collinearity (VIF) Report",
        "=" * 60,
        f"VIF threshold for concern: > {VIF_MODERATE:.0f} (moderate), > {VIF_SEVERE:.0f} (severe)",
        "",
        vif_df.to_string(index=False),
        "",
    ]

    severe   = vif_df[vif_df["VIF"] > VIF_SEVERE]
    moderate = vif_df[(vif_df["VIF"] > VIF_MODERATE) & (vif_df["VIF"] <= VIF_SEVERE)]

    lines.append(f"Severe collinearity  (VIF > {VIF_SEVERE:.0f}): {len(severe)} features")
    if not severe.empty:
        lines.append(f"  → {severe['Feature'].tolist()}")
    lines.append(f"Moderate collinearity (VIF > {VIF_MODERATE:.0f}): {len(moderate)} features")
    if not moderate.empty:
        lines.append(f"  → {moderate['Feature'].tolist()}")
    lines.append("")

    if to_remove:
        lines.append(f"Recommended removals: {to_remove}")
        lines.append(
            "Note: XGBoost / tree ensembles are inherently robust to collinear features;\n"
            "      removing these features may not improve accuracy but does simplify\n"
            "      the feature-importance plot and reduces reviewer concern."
        )
    else:
        lines.append(f"No features have VIF > {threshold:.0f}.  Feature set is well-conditioned.")

    if ablation:
        lines += [
            "",
            "Ablation study (CV F1 without high-VIF features):",
            f"  Full feature set ({ablation['n_features_full']} features):    "
            f"F1 = {ablation['f1_full']:.3f} ± {ablation['f1_full_std']:.3f}",
            f"  Reduced set     ({ablation['n_features_reduced']} features):    "
            f"F1 = {ablation['f1_reduced']:.3f} ± {ablation['f1_reduced_std']:.3f}",
            f"  Difference: {ablation['f1_reduced'] - ablation['f1_full']:+.3f}",
        ]

    report_path = os.path.join(output_dir, "vif_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved VIF report → %s", report_path)
    print("\n".join(lines))


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VIF feature collinearity analysis for HIFUN Router"
    )
    p.add_argument("--labeled_csv",   default=DEFAULT_LABELED_CSV)
    p.add_argument("--output_dir",    default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--vif_threshold", type=float, default=VIF_SEVERE,
                   help="VIF level above which features are flagged (default 10)")
    p.add_argument("--no_ablation",   action="store_true",
                   help="Skip the ablation study (faster)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_vif_analysis(
        labeled_csv=args.labeled_csv,
        output_dir=args.output_dir,
        vif_threshold=args.vif_threshold,
        run_ablation=not args.no_ablation,
    )
