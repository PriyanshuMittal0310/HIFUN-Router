"""scale_factor_experiment.py — Task 2.4: latency vs data scale.

Runs all HIFUN routing strategies at TPC-H Scale Factor 1 and SF=5,
reporting median and p95 latency per strategy and query.

Produces:
  experiments/results/scale_factor_results.csv
  experiments/results/scale_factor_summary.csv
  experiments/results/scale_factor_plot.pdf

Usage:
    python experiments/scale_factor_experiment.py
    python experiments/scale_factor_experiment.py \
        --spark_master spark://spark-master:7077 \
        --output_dir   experiments/results/
"""

import argparse
import json
import logging
import os
import time
import traceback
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ─── Default paths ────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR   = "experiments/results/"
DEFAULT_QUERIES_FILE = "dsl/sample_queries/tpch_queries.json"

# Scale factor → Parquet directory mapping
SCALE_FACTORS: Dict[str, str] = {
    "sf1": "data/parquet/tpch/",
    "sf5": "data/parquet/tpch_sf5/",
}

STRATEGIES = ["always_sql", "always_graph", "threshold", "logreg", "learned_xgb"]

# ─── Routing strategy implementations ────────────────────────────────────────

def _build_router_for_strategy(
    strategy: str,
    parquet_dir: str,
    spark=None,
    use_real_engines: bool = True,
):
    """Construct a HybridRouter configured for the given strategy."""
    from config.paths    import GRAPHS_DIR, STATS_DIR, CLASSIFIER_PATH
    from router.hybrid_router import HybridRouter
    from features.feature_extractor import FEATURE_NAMES

    common = dict(
        parquet_dir=parquet_dir,
        graph_dir=os.path.join("data", "graphs", "synthetic"),
        stats_dir=STATS_DIR,
        model_path=CLASSIFIER_PATH,
        use_real_engines=use_real_engines,
        spark=spark,
    )

    if strategy == "always_sql":
        return HybridRouter(**common, force_engine="SQL")

    if strategy == "always_graph":
        return HybridRouter(**common, force_engine="GRAPH")

    if strategy == "threshold":
        from router.baselines import ThresholdBaseline
        baseline = ThresholdBaseline()
        router = HybridRouter(
            **common,
            custom_router=lambda sub, fv, fn: baseline.route(fv, fn),
        )
        return router

    if strategy == "logreg":
        from router.baselines import LogisticRegressionBaseline
        from config.paths import LABELED_RUNS_CSV
        lb = LogisticRegressionBaseline()
        try:
            df = pd.read_csv(LABELED_RUNS_CSV)
            X  = df[FEATURE_NAMES].values
            y  = (df["label"] == "GRAPH").astype(int).values
            lb.fit(X, y)
        except Exception as exc:
            logger.warning("LogReg fit failed (%s), using random init", exc)
        return HybridRouter(
            **common,
            custom_router=lambda sub, fv, fn: lb.route(fv),
        )

    if strategy == "learned_xgb":
        # Default HybridRouter uses XGBoost via ModelPredictor
        return HybridRouter(**common)

    raise ValueError(f"Unknown strategy: '{strategy}'")


# ─── Timing helper ────────────────────────────────────────────────────────────

def _time_query(router, query: dict, n_runs: int = 1) -> Dict[str, Any]:
    """Execute query n_runs times and return timing + metadata."""
    times_ms = []
    row_count = -1
    status   = "ok"
    engine_decisions = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        try:
            out = router.execute_query(query)
            result = out.get("result")
            if result is not None:
                if hasattr(result, "count"):       # Spark DF
                    row_count = result.count()
                elif isinstance(result, pd.DataFrame):
                    row_count = len(result)
            engine_decisions = [
                d.get("engine", "?")
                for d in out.get("routing_decisions", [])
            ]
        except Exception as exc:
            status = str(exc)[:120]
            logger.warning("Query %s failed: %s", query.get("query_id"), exc)
            times_ms.append(float("inf"))
            continue
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    finite = [t for t in times_ms if t != float("inf")]
    return {
        "latency_median_ms": round(float(np.median(finite)), 2) if finite else float("inf"),
        "latency_p95_ms":    round(float(np.percentile(finite, 95)), 2) if finite else float("inf"),
        "latency_min_ms":    round(min(finite), 2) if finite else float("inf"),
        "row_count":         row_count,
        "status":            status,
        "engine_decisions":  ",".join(engine_decisions),
    }


# ─── Main experiment ─────────────────────────────────────────────────────────

def run_scale_experiment(
    queries_file: str = DEFAULT_QUERIES_FILE,
    output_dir: str   = DEFAULT_OUTPUT_DIR,
    spark_master: Optional[str] = None,
    use_real_engines: bool = True,
    n_runs: int = 1,
) -> pd.DataFrame:
    """Run the scale factor experiment.

    For each (scale_factor, strategy, query) triple:
      - Instantiate HybridRouter with the appropriate Parquet directory
      - Execute the query and measure latency
      - Record results

    Args:
        queries_file:      Path to TPC-H DSL query JSON.
        output_dir:        Directory for result CSVs and plots.
        spark_master:      Override Spark master URL (e.g. for cluster mode).
        use_real_engines:  True → PySpark + GraphFrames; False → pandas.
        n_runs:            Number of timed executions per query (median taken).
    """
    with open(queries_file) as f:
        queries = json.load(f)
    if not isinstance(queries, list):
        queries = [queries]

    rows: List[dict] = []
    os.makedirs(output_dir, exist_ok=True)

    for sf_label, parquet_dir in SCALE_FACTORS.items():
        if not os.path.isdir(parquet_dir):
            logger.warning(
                "Parquet directory not found for %s: %s  (skipping, run data/scripts/generate_tpch_sf5.sh first)",
                sf_label, parquet_dir,
            )
            continue

        logger.info("\n=== Scale Factor: %s  (%s) ===", sf_label, parquet_dir)

        # Build a single SparkSession per scale factor (expensive to start)
        spark = None
        if use_real_engines:
            from config.spark_config import get_spark_session
            extra = {}
            if spark_master:
                extra["master"] = spark_master
            spark = get_spark_session(
                app_name=f"HIFUN_ScaleFactor_{sf_label}",
                **extra,   # type: ignore[arg-type]
            )

        for strategy in STRATEGIES:
            logger.info("  Strategy: %s", strategy)
            try:
                router = _build_router_for_strategy(
                    strategy, parquet_dir,
                    spark=spark,
                    use_real_engines=use_real_engines,
                )
            except Exception as exc:
                logger.warning(
                    "   Could not build router for '%s': %s", strategy, exc
                )
                continue

            for query in queries:
                qid = query.get("query_id", "unknown")
                timing = _time_query(router, query, n_runs=n_runs)

                row = {
                    "scale_factor":     sf_label,
                    "strategy":         strategy,
                    "query_id":         qid,
                    **timing,
                }
                rows.append(row)
                logger.info(
                    "    [%s][%s][%s]: %.1fms  rows=%d  engines=%s",
                    sf_label, strategy, qid,
                    timing["latency_median_ms"],
                    timing["row_count"],
                    timing["engine_decisions"],
                )

        if spark is not None and use_real_engines:
            spark.stop()

    df = pd.DataFrame(rows)
    results_csv = os.path.join(output_dir, "scale_factor_results.csv")
    df.to_csv(results_csv, index=False)
    logger.info("\n✓  Saved %d result rows → %s", len(df), results_csv)

    # ── Summary table (median + p95 per scale/strategy) ──────────────────────
    _write_summary(df, output_dir)
    _plot_results(df, output_dir)

    return df


def _write_summary(df: pd.DataFrame, output_dir: str) -> None:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        logger.warning("No successful queries to summarise")
        return

    summary = (
        ok.groupby(["scale_factor", "strategy"])["latency_median_ms"]
        .agg(
            queries="count",
            median_ms="median",
            p95_ms=lambda x: x.quantile(0.95),
            max_ms="max",
        )
        .round(2)
        .reset_index()
    )
    summary_csv = os.path.join(output_dir, "scale_factor_summary.csv")
    summary.to_csv(summary_csv, index=False)
    logger.info("Scale Factor Summary:\n%s", summary.to_string(index=False))


def _plot_results(df: pd.DataFrame, output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available; skipping plots")
        return

    ok = df[(df["status"] == "ok") & (df["latency_median_ms"] != float("inf"))].copy()
    if ok.empty:
        return

    fig, axes = plt.subplots(1, len(SCALE_FACTORS), figsize=(6 * len(SCALE_FACTORS), 5),
                              sharey=False)
    if len(SCALE_FACTORS) == 1:
        axes = [axes]

    for ax, (sf_label, _) in zip(axes, SCALE_FACTORS.items()):
        sub = ok[ok["scale_factor"] == sf_label]
        if sub.empty:
            ax.set_title(f"SF={sf_label} (no data)")
            continue

        strat_latency = (
            sub.groupby("strategy")["latency_median_ms"]
            .median()
            .sort_values()
            .reset_index()
        )
        sns.barplot(
            data=strat_latency,
            x="strategy", y="latency_median_ms",
            palette="muted", ax=ax,
        )
        ax.set_title(f"TPC-H Scale Factor = {sf_label}")
        ax.set_xlabel("Routing Strategy")
        ax.set_ylabel("Median Query Latency (ms)")
        ax.tick_params(axis="x", rotation=25)

    plt.suptitle("HIFUN Router — Latency by Scale Factor & Strategy", fontsize=13, y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "scale_factor_plot.pdf")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scale factor plot → %s", plot_path)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scale factor latency experiment for HIFUN Router"
    )
    p.add_argument("--queries_file",      default=DEFAULT_QUERIES_FILE)
    p.add_argument("--output_dir",        default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--spark_master",      default=None,
                   help="Spark master URL (default: local[*])")
    p.add_argument("--use_real_engines",  action="store_true", default=False,
                   help="Use PySpark + GraphFrames instead of pandas")
    p.add_argument("--n_runs",            type=int, default=1,
                   help="Number of timed runs per query (median taken)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_scale_experiment(
        queries_file=args.queries_file,
        output_dir=args.output_dir,
        spark_master=args.spark_master,
        use_real_engines=args.use_real_engines,
        n_runs=args.n_runs,
    )
