"""Training data collection: parse DSL queries, decompose, extract features, label.

Since both execution engines (Spark SQL, GraphFrames) are not running simultaneously,
we use a heuristic cost model to simulate runtime measurements and generate labels.
The heuristic is based on subexpression characteristics:
  - Pure TRAVERSAL → GRAPH engine faster
  - Pure relational (FILTER/JOIN/AGG) → SQL engine faster
  - Mixed → depends on traversal cost vs relational cost ratio
"""

import csv
import json
import math
import os
import random
import sys

import numpy as np

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import SAMPLE_QUERIES_DIR, TRAINING_DATA_DIR, STATS_DIR
from parser.dsl_parser import DSLParser
from decomposer.query_decomposer import QueryDecomposer
from features.feature_extractor import FeatureExtractor, FEATURE_NAMES


def _simulate_runtimes(feature_dict: dict, primary_op_type: str):
    """Heuristic cost model to simulate SQL and GRAPH runtimes (ms).

    Based on the feature vector characteristics, estimates which engine
    would be faster and generates realistic-looking runtime numbers.

    Design goal: produce roughly balanced labels (40-60% SQL vs GRAPH)
    so the ML classifier learns meaningful decision boundaries.
    """
    rng = random.Random()

    input_card = 10 ** max(feature_dict["input_cardinality_log"], 0)
    has_trav = feature_dict["has_traversal"]
    max_hops = max(feature_dict["max_hops"], 0)
    avg_degree = feature_dict["avg_degree"]
    n_joins = feature_dict["op_count_join"]
    n_filters = feature_dict["op_count_filter"]
    n_aggs = feature_dict["op_count_aggregate"]
    n_trav = feature_dict["op_count_traversal"]
    selectivity = feature_dict["selectivity"]
    trav_ops = feature_dict["estimated_traversal_ops"]
    degree_skew = feature_dict["degree_skew"]

    # --- SQL cost model (milliseconds) ---
    sql_base = 10.0 + 0.001 * input_card  # scan cost
    sql_base += n_filters * 3.0
    sql_base += n_joins * (20.0 + 0.003 * input_card * selectivity)
    sql_base += n_aggs * 10.0
    if has_trav:
        # SQL must simulate traversals via recursive self-joins — very expensive
        sql_base += max_hops * 80.0 + n_trav * 60.0
        if avg_degree > 0:
            sql_base += (avg_degree ** min(max_hops, 3)) * 5.0
        sql_base += degree_skew * 20.0  # skewed graphs are harder for SQL
    sql_ms = max(1.0, sql_base * rng.uniform(0.85, 1.15))

    # --- GRAPH cost model (milliseconds) ---
    if has_trav:
        # Graph engine is native for traversals — low overhead
        graph_base = 8.0  # small fixed overhead for BFS/DFS
        start_v = max(1, int(input_card * selectivity))
        graph_base += start_v * max_hops * 0.3
        if avg_degree > 0:
            graph_base += start_v * (avg_degree ** min(max_hops, 3)) * 0.005
        graph_base += n_aggs * 5.0  # post-traversal aggregation
        graph_base += n_filters * 2.0  # vertex/edge filtering
        graph_base += n_joins * 12.0  # cross-engine join overhead
    else:
        # Graph engine overhead for pure relational workloads
        graph_base = 35.0 + 0.004 * input_card
        graph_base += n_joins * 30.0
        graph_base += n_aggs * 18.0
        graph_base += n_filters * 5.0
    graph_ms = max(1.0, graph_base * rng.uniform(0.85, 1.15))

    label = "GRAPH" if graph_ms < sql_ms else "SQL"
    return sql_ms, graph_ms, label


def collect_training_data(
    queries_dir: str = SAMPLE_QUERIES_DIR,
    stats_dir: str = STATS_DIR,
    output_path: str = None,
    augment_factor: int = 10,
):
    """Parse all DSL queries, decompose, extract features, simulate labels.

    Args:
        queries_dir: Directory containing DSL JSON query files.
        stats_dir: Directory containing precomputed statistics.
        output_path: Path for labeled_runs.csv output.
        augment_factor: Number of augmented variants per real sub-expression.
    """
    if output_path is None:
        output_path = os.path.join(TRAINING_DATA_DIR, "labeled_runs.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    parser = DSLParser()
    decomposer = QueryDecomposer()
    extractor = FeatureExtractor(stats_dir=stats_dir)

    columns = (
        ["sub_id", "query_id", "dataset"]
        + FEATURE_NAMES
        + ["sql_runtime_ms", "graph_runtime_ms", "label"]
    )

    rows = []

    # Process all JSON query files
    for fname in sorted(os.listdir(queries_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(queries_dir, fname)
        dataset = fname.replace("_queries.json", "")

        parsed = parser.parse_file(fpath)
        for query_id, query_nodes in parsed.items():
            sub_exprs = decomposer.decompose(query_nodes)
            for sub in sub_exprs:
                feat_dict = extractor.extract_dict(sub)
                sql_ms, graph_ms, label = _simulate_runtimes(
                    feat_dict, sub.primary_op_type
                )

                row = {
                    "sub_id": sub.sub_id,
                    "query_id": query_id,
                    "dataset": dataset,
                }
                row.update(feat_dict)
                row["sql_runtime_ms"] = round(sql_ms, 3)
                row["graph_runtime_ms"] = round(graph_ms, 3)
                row["label"] = label
                rows.append(row)

    # Augment: create perturbed variants to increase training set size
    augmented = []
    rng = random.Random(42)
    for _ in range(augment_factor):
        for row in rows:
            new_row = dict(row)
            new_row["sub_id"] = row["sub_id"] + f"_aug{len(augmented)}"
            # Perturb numeric features slightly
            for feat in FEATURE_NAMES:
                val = row[feat]
                if isinstance(val, (int, float)) and val != 0:
                    noise = rng.gauss(0, abs(val) * 0.05)
                    new_row[feat] = round(val + noise, 6)
            # Re-simulate runtimes with perturbed features
            sql_ms, graph_ms, label = _simulate_runtimes(
                new_row, "TRAVERSAL" if new_row["has_traversal"] else "RELATIONAL"
            )
            new_row["sql_runtime_ms"] = round(sql_ms, 3)
            new_row["graph_runtime_ms"] = round(graph_ms, 3)
            new_row["label"] = label
            augmented.append(new_row)

    all_rows = rows + augmented

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)

    extractor.close()
    print(f"Collected {len(rows)} base + {len(augmented)} augmented = "
          f"{len(all_rows)} total rows -> {output_path}")
    return output_path, len(all_rows)


if __name__ == "__main__":
    collect_training_data()
