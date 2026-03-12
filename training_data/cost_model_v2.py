"""Cost model v2: Heuristic runtime simulator that produces ambiguous labels.

Key improvement over v1: introduces three regime-switching penalties that make
the SQL/GRAPH decision non-trivially separable by a single binary feature.

Ambiguous regimes (SQL can win even when has_traversal=1):
  A. Ultra-selective start filter (selectivity < 0.005): so few start vertices
     that a single indexed SQL lookup + BFS-equivalent self-join is cheaper
     than BFS graph-engine initialisation overhead.
  B. Sparse graph (avg_degree < 2.0): almost a tree; SQL recursive-CTE joins
     are competitive and avoid graph-engine startup cost.
  C. Heavy mixed workload (traversal + multiple joins + aggregations):
     cross-engine data transfer overhead erodes the graph engine's BFS advantage.

These three regimes push 15-25% of has_traversal=1 samples to SQL labels,
dropping expected ML F1 from the degenerate 1.000 into the 0.82-0.92 range
on the current benchmark set.
"""

import random


def simulate_runtimes_v2(feature_dict: dict, rng: random.Random = None):
    """Return (sql_ms, graph_ms, label) using the ambiguity-aware cost model.

    Args:
        feature_dict: dictionary of feature name -> value as returned by
                      FeatureExtractor.extract_dict().
        rng: optional seeded Random instance for reproducibility.

    Returns:
        Tuple (sql_ms: float, graph_ms: float, label: str).
    """
    if rng is None:
        rng = random.Random()

    input_card = 10 ** max(feature_dict.get("input_cardinality_log", 0), 0)
    has_trav    = feature_dict.get("has_traversal", 0)
    max_hops    = max(feature_dict.get("max_hops", 0), 0)
    avg_degree  = feature_dict.get("avg_degree", 1.0)
    n_joins     = feature_dict.get("op_count_join", 0)
    n_filters   = feature_dict.get("op_count_filter", 0)
    n_aggs      = feature_dict.get("op_count_aggregate", 0)
    n_trav      = feature_dict.get("op_count_traversal", 0)
    selectivity = feature_dict.get("selectivity", 1.0)
    degree_skew = feature_dict.get("degree_skew", 0.0)

    # ── SQL cost (milliseconds) ───────────────────────────────────────────────
    sql_base = 10.0 + 0.001 * input_card
    sql_base += n_filters * 3.0
    sql_base += n_joins * (20.0 + 0.003 * input_card * selectivity)
    sql_base += n_aggs * 10.0
    if has_trav:
        # Normal traversal: SQL is expensive (recursive self-joins)
        sql_trav_penalty = max_hops * 55.0 + n_trav * 40.0
        if avg_degree > 0:
            sql_trav_penalty += (avg_degree ** min(max_hops, 3)) * 4.0
        sql_trav_penalty += degree_skew * 15.0

        # Regime A: ultra-selective 1-hop → SQL treats as indexed point-lookup
        # + single foreign-key join; multi-hop still favours BFS.
        if selectivity < 0.005 and max_hops <= 1.2:
            sql_trav_penalty = max_hops * 4.0 + n_trav * 4.0

        # Regime B: sparse graph (avg_degree < 2) → recursive-CTE joins are
        # competitive; reduce penalty proportionally.
        elif avg_degree < 2.0:
            sql_trav_penalty *= 0.30

        sql_base += sql_trav_penalty

    sql_ms = max(1.0, sql_base * rng.uniform(0.85, 1.15))

    # ── GRAPH cost (milliseconds) ─────────────────────────────────────────────
    if has_trav:
        graph_base = 8.0                        # BFS engine fixed overhead
        start_v = max(1, int(input_card * selectivity))
        graph_base += start_v * max_hops * 0.3
        if avg_degree > 0:
            graph_base += start_v * (avg_degree ** min(max_hops, 3)) * 0.005
        graph_base += n_aggs * 5.0
        graph_base += n_filters * 2.0
        graph_base += n_joins * 12.0

        # Regime A: tiny traversal → engine init cost becomes significant
        if selectivity < 0.005 and max_hops <= 1.2:
            graph_base += 32.0

        # Regime B: sparse graph → BFS setup overhead visible
        if avg_degree < 2.0:
            graph_base += 18.0

        # Regime C: mixed workload (trav + joins + aggs) → cross-engine overhead
        if n_joins >= 2 and n_aggs >= 1:
            graph_base += n_joins * 8.0 + n_aggs * 6.0

    else:
        # Pure relational: graph engine is at a disadvantage
        graph_base = 35.0 + 0.004 * input_card
        graph_base += n_joins * 30.0
        graph_base += n_aggs * 18.0
        graph_base += n_filters * 5.0

    graph_ms = max(1.0, graph_base * rng.uniform(0.85, 1.15))

    label = "GRAPH" if graph_ms < sql_ms else "SQL"
    return sql_ms, graph_ms, label
