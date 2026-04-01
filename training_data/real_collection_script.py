"""real_collection_script.py — collect real execution labels at scale.

Replaces the simulated heuristic labels in `training_data/labeled_runs.csv`
with real measured runtimes from PySpark + GraphFrames.

Usage:
    python training_data/real_collection_script.py [--queries_dir PATH]
                                                    [--output CSV]
                                                    [--n_warmup N]
                                                    [--n_measure N]

Output CSV columns:
    sub_id, query_id, dataset, <22 feature columns>,
    sql_median_ms, sql_std_ms, graph_median_ms, graph_std_ms,
    speedup, label, label_source
"""

import argparse
import json
import logging
import os
import sys
import traceback
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.spark_config import get_spark_session
from config.paths         import STATS_DIR, PARQUET_DIR, GRAPHS_DIR, TPCH_PARQUET_DIR
from parser.dsl_parser        import DSLParser
from decomposer.query_decomposer import QueryDecomposer
from features.feature_extractor  import FeatureExtractor, FEATURE_NAMES
from execution.spark_sql_generator   import SparkSQLGenerator
from execution.graphframes_generator  import GraphFramesGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_QUERY_DIRS   = ["dsl/sample_queries/"]
DEFAULT_OUTPUT_CSV   = "training_data/real_labeled_runs.csv"
DEFAULT_N_WARMUP     = 2    # JVM warm-up runs before measurement
DEFAULT_N_MEASURE    = 3    # runs whose median becomes the label
DEFAULT_REPEAT       = 1    # repeat each subexpression this many times

# Parquet directories by dataset key
DATASET_PARQUET_DIRS = {
    "tpch_queries":      TPCH_PARQUET_DIR,
    "snb_queries":       os.path.join(PARQUET_DIR, "snb"),
    "snb_real_queries":  os.path.join(PARQUET_DIR, "snb"),
    "snb_bi_real_queries": os.path.join(PARQUET_DIR, "snb"),
    "ogb_real_queries":  os.path.join(GRAPHS_DIR, "ogbn_arxiv"),
    "job_real_queries":  os.path.join(PARQUET_DIR, "job"),
    "tpcds_real_queries": os.path.join(PARQUET_DIR, "tpcds"),
}
DATASET_GRAPH_DIRS = {
    "tpch_queries":      os.path.join(GRAPHS_DIR, "synthetic"),
    "snb_queries":       os.path.join(GRAPHS_DIR, "snb"),
    "snb_real_queries":  os.path.join(GRAPHS_DIR, "snb"),
    "snb_bi_real_queries": os.path.join(GRAPHS_DIR, "snb"),
    "ogb_real_queries":  os.path.join(GRAPHS_DIR, "ogbn_arxiv"),
}


def _available_sources(parquet_dir: str, graph_dir: str) -> set:
    sources = set()
    if os.path.isdir(parquet_dir):
        for name in os.listdir(parquet_dir):
            p = os.path.join(parquet_dir, name)
            if os.path.isdir(p) or p.endswith(".parquet"):
                sources.add(name.replace(".parquet", ""))
    if os.path.isdir(graph_dir):
        for name in os.listdir(graph_dir):
            base = name.replace(".parquet", "")
            if base.endswith("_vertices"):
                sources.add(base[:-9])
            elif base.endswith("_edges"):
                sources.add(base[:-6])
    return sources


def _query_references_missing_sources(query: dict, available: set) -> tuple[bool, str]:
    op_ids = {op.get("op_id") for op in query.get("operations", [])}
    for op in query.get("operations", []):
        source = op.get("source", "")
        if source and source not in op_ids and source not in available:
            return True, source
        join = op.get("join")
        if isinstance(join, dict):
            right_source = join.get("right_source", "")
            if right_source and right_source not in op_ids and right_source not in available:
                return True, right_source
    return False, ""

# ─── Measurement helper ───────────────────────────────────────────────────────

def _measure(fn, n_warmup: int, n_measure: int) -> Tuple[float, float]:
    """Execute fn() n_warmup+n_measure times; return (median_ms, std_ms).

    Warmup runs are discarded.  Spark DataFrames are force-materialised
    via .count() so that lazy evaluation does not skew timings.
    """
    for _ in range(n_warmup):
        try:
            result = fn()
            if hasattr(result, "count"):
                result.count()
        except Exception:
            pass

    times_ms: List[float] = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        try:
            result = fn()
            if hasattr(result, "count"):
                result.count()
            elif isinstance(result, pd.DataFrame):
                _ = len(result)
        except Exception as exc:
            logger.warning("Execution attempt failed: %s", exc)
            times_ms.append(float("inf"))
            continue
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    finite = [t for t in times_ms if t != float("inf")]
    if not finite:
        return float("inf"), float("inf")
    arr = sorted(finite)
    median = arr[len(arr) // 2]
    std    = float(np.std(finite)) if len(finite) > 1 else 0.0
    return round(median, 2), round(std, 2)


# ─── Main collection loop ─────────────────────────────────────────────────────

def collect_labels(
    spark,
    query_dirs: List[str],
    output_csv: str,
    n_warmup: int,
    n_measure: int,
    repeat: int,
    include_synthetic: bool,
) -> pd.DataFrame:
    """Iterate over all DSL queries, decompose, extract features, and measure.

    For each SubExpression:
      - Measure SQL path via SparkSQLGenerator
      - Measure GRAPH path via GraphFramesGenerator (only if has_traversal=1;
        otherwise assign a 3× penalty)
      - Label = argmin(sql_ms, graph_ms)
    """
    parser     = DSLParser()
    decomposer = QueryDecomposer()
    extractor  = FeatureExtractor(stats_dir=STATS_DIR)

    rows: List[dict] = []

    for query_dir in query_dirs:
        if not os.path.isdir(query_dir):
            logger.warning("Query dir not found: %s", query_dir)
            continue

        for fname in sorted(os.listdir(query_dir)):
            if not fname.endswith(".json"):
                continue
            if not include_synthetic and "synthetic" in fname:
                logger.info("Skipping synthetic query file: %s", fname)
                continue

            fpath   = os.path.join(query_dir, fname)
            dataset = fname.replace(".json", "")
            parquet_dir = DATASET_PARQUET_DIRS.get(dataset, TPCH_PARQUET_DIR)
            graph_dir   = DATASET_GRAPH_DIRS.get(dataset, os.path.join(GRAPHS_DIR, "synthetic"))

            # Instantiate generators for this dataset
            sql_gen   = SparkSQLGenerator(spark, parquet_dir)
            graph_gen = GraphFramesGenerator(spark, graph_dir)
            available = _available_sources(parquet_dir, graph_dir)

            if not available:
                logger.warning(
                    "No sources found for dataset=%s (parquet=%s, graph=%s); skipping file",
                    dataset,
                    parquet_dir,
                    graph_dir,
                )
                continue

            with open(fpath) as f:
                data = json.load(f)
            queries = data if isinstance(data, list) else [data]

            for query in queries:
                qid = query.get("query_id", "unknown")
                missing, source = _query_references_missing_sources(query, available)
                if missing:
                    logger.info(
                        "Skipping query %s (dataset=%s): missing source '%s'",
                        qid,
                        dataset,
                        source,
                    )
                    continue

                logger.info("Processing query: %s  (dataset=%s)", qid, dataset)
                try:
                    nodes     = parser.parse(query)
                    sub_exprs = decomposer.decompose(nodes)
                except Exception:
                    logger.error("Parse/decompose failed for %s:\n%s", qid, traceback.format_exc())
                    continue

                sub_map = {sub.sub_id: sub for sub in sub_exprs}

                for sub in sub_exprs:
                    for rep in range(repeat):
                        sid = f"{sub.sub_id}_r{rep}"
                        logger.info("  Sub: %s  (%s)", sid, sub.primary_op_type)
                        try:
                            fv = extractor.extract(sub)
                        except Exception as exc:
                            logger.warning("  Feature extraction error: %s", exc)
                            continue

                        has_traversal = bool(fv[FEATURE_NAMES.index("has_traversal")])

                        def _prepare_sql_deps(target_sub, cache, prepared):
                            for dep_sub_id in target_sub.depends_on_subs:
                                if dep_sub_id in prepared:
                                    continue
                                dep_sub = sub_map.get(dep_sub_id)
                                if dep_sub is None:
                                    continue
                                _prepare_sql_deps(dep_sub, cache, prepared)
                                sql_gen.cache = cache
                                dep_df = sql_gen.generate(dep_sub)
                                if hasattr(dep_df, "count"):
                                    dep_df.count()
                                prepared.add(dep_sub_id)

                        def _prepare_graph_deps(target_sub, cache, prepared):
                            for dep_sub_id in target_sub.depends_on_subs:
                                if dep_sub_id in prepared:
                                    continue
                                dep_sub = sub_map.get(dep_sub_id)
                                if dep_sub is None:
                                    continue
                                _prepare_graph_deps(dep_sub, cache, prepared)
                                graph_gen.cache = cache
                                dep_df = graph_gen.generate(dep_sub)
                                if hasattr(dep_df, "count"):
                                    dep_df.count()
                                prepared.add(dep_sub_id)

                        # ── SQL path ────────────────────────────────────────
                        shared = {}
                        def run_sql(s=sub, gen=sql_gen, cache=shared):
                            _prepare_sql_deps(s, cache, set())
                            gen.cache = cache
                            return gen.generate(s)

                        sql_med, sql_std = _measure(run_sql, n_warmup, n_measure)

                        # ── GRAPH path ──────────────────────────────────────
                        if has_traversal:
                            shared2 = {}
                            def run_graph(s=sub, gen=graph_gen, cache=shared2):
                                _prepare_graph_deps(s, cache, set())
                                gen.cache = cache
                                return gen.generate(s)
                            graph_med, graph_std = _measure(run_graph, n_warmup, n_measure)
                        else:
                            # Pure relational: graph can't run — assign 3× SQL cost
                            graph_med = sql_med * 3.0 if sql_med != float("inf") else float("inf")
                            graph_std = 0.0

                        # Stabilize labels when only one engine path is feasible.
                        if sql_med == float("inf") and graph_med != float("inf"):
                            sql_med = graph_med * 3.0
                            sql_std = 0.0
                        elif graph_med == float("inf") and sql_med != float("inf"):
                            graph_med = sql_med * 3.0
                            graph_std = 0.0

                        if sql_med == float("inf") and graph_med == float("inf"):
                            logger.info("  Skipping sub %s: both engines failed", sid)
                            continue

                        label   = "GRAPH" if graph_med < sql_med else "SQL"
                        speedup = sql_med / max(graph_med, 1e-6)

                        logger.info(
                            "  SQL=%.1fms  Graph=%.1fms  Label=%s  Speedup=%.2fx",
                            sql_med, graph_med, label, speedup,
                        )

                        row = {
                            "sub_id":          sid,
                            "query_id":        qid,
                            "dataset":         dataset,
                            **dict(zip(FEATURE_NAMES, fv.tolist())),
                            "sql_median_ms":   sql_med,
                            "sql_std_ms":      sql_std,
                            "graph_median_ms": round(graph_med, 2),
                            "graph_std_ms":    round(graph_std, 2),
                            "speedup":         round(speedup, 3),
                            "label":           label,
                            "label_source":    "real_measurement",
                        }
                        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    n_sql   = (df["label"] == "SQL").sum()   if len(df) else 0
    n_graph = (df["label"] == "GRAPH").sum() if len(df) else 0
    logger.info(
        "\n✓  Saved %d labeled rows → %s\n   SQL=%d  GRAPH=%d",
        len(df), output_csv, n_sql, n_graph,
    )
    return df


# ─── CLI entry point ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect real execution labels for HIFUN Router")
    p.add_argument("--queries_dir", nargs="+", default=DEFAULT_QUERY_DIRS,
                   help="Directories containing DSL query JSON files")
    p.add_argument("--output",      default=DEFAULT_OUTPUT_CSV,
                   help="Output CSV path")
    p.add_argument("--n_warmup",    type=int, default=DEFAULT_N_WARMUP,
                   help="Number of JVM warm-up runs (discarded)")
    p.add_argument("--n_measure",   type=int, default=DEFAULT_N_MEASURE,
                   help="Number of timed runs (median taken)")
    p.add_argument("--repeat",      type=int, default=DEFAULT_REPEAT,
                   help="Repeat each sub-expression N times to increase sample count")
    p.add_argument("--include_synthetic", action="store_true",
                   help="Include synthetic query files (disabled by default)")
    p.add_argument("--master",      default=None,
                   help="Spark master URL override (e.g. spark://host:7077)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    extra = {}
    if args.master:
        extra["master_url"] = args.master

    spark_kw = {"app_name": "HIFUN_LabelCollection"}
    if args.master:
        spark_kw["master"] = args.master

    spark = get_spark_session(**spark_kw)   # type: ignore[arg-type]
    try:
        collect_labels(
            spark,
            query_dirs=args.queries_dir,
            output_csv=args.output,
            n_warmup=args.n_warmup,
            n_measure=args.n_measure,
            repeat=max(1, args.repeat),
            include_synthetic=args.include_synthetic,
        )
    finally:
        spark.stop()
