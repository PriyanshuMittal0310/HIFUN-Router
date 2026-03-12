"""query_generator.py — Task 2.2 supplementary: generate DSL query variants.

Produces additional labeled query variants by sweeping:
  - Predicate selectivity (FILTER queries)
  - Max traversal hops  (TRAVERSAL queries)
  - Join combinations   (JOIN queries)

These variants increase the training set from ~186 to 500+ samples,
providing the coverage needed for robust cross-dataset generalisation.

Usage:
    python training_data/query_generator.py
    # Writes to dsl/sample_queries/synthetic_variants.json
    #            dsl/sample_queries/snb_traversal_variants.json
    #            dsl/sample_queries/tpch_join_variants.json
"""

import copy
import json
import os
import logging

logger = logging.getLogger(__name__)

SAMPLE_QUERIES_DIR = "dsl/sample_queries/"

# ─── Variant generators ───────────────────────────────────────────────────────

def generate_filter_variants(base_query: dict, selectivity_values: list) -> list:
    """Sweep predicate value / selectivity for FILTER queries.

    Args:
        base_query:         A DSL query dict with at least one FILTER operation.
        selectivity_values: List of approximate selectivity targets [0, 1].
                            The predicate value is annotated with a
                            ``_selectivity_hint`` field that the FeatureExtractor
                            uses when computing the ``selectivity`` feature.
    Returns:
        List of variant query dicts.
    """
    variants = []
    for i, sv in enumerate(selectivity_values):
        q = copy.deepcopy(base_query)
        q["query_id"] = f"{base_query['query_id']}_sel{int(sv * 100):03d}"
        for op in q.get("operations", []):
            if op.get("type") == "FILTER" and "predicate" in op:
                op["predicate"]["_selectivity_hint"] = sv
        variants.append(q)
    return variants


def generate_hop_variants(base_query: dict, hop_values: list) -> list:
    """Sweep max_hops for TRAVERSAL queries.

    Args:
        base_query:  A DSL query dict with at least one TRAVERSAL operation.
        hop_values:  List of integer hop depths to generate variants for.
    Returns:
        List of variant query dicts.
    """
    variants = []
    for h in hop_values:
        q = copy.deepcopy(base_query)
        q["query_id"] = f"{base_query['query_id']}_hops{h}"
        for op in q.get("operations", []):
            if op.get("type") == "TRAVERSAL" and "traversal" in op:
                op["traversal"]["max_hops"] = h
        variants.append(q)
    return variants


def generate_join_variants(base_query: dict, join_types: list) -> list:
    """Sweep join strategy (inner / left / semi) for JOIN queries.

    Args:
        base_query:  A DSL query dict with at least one JOIN operation.
        join_types:  List of join type strings, e.g. ["inner", "left", "semi"].
    Returns:
        List of variant query dicts (one per join_type).
    """
    variants = []
    for jt in join_types:
        q = copy.deepcopy(base_query)
        q["query_id"] = f"{base_query['query_id']}_join{jt}"
        for op in q.get("operations", []):
            if op.get("type") == "JOIN" and "join" in op:
                op["join"]["join_type"] = jt
        variants.append(q)
    return variants


def generate_combined_variants(base_query: dict) -> list:
    """Generate a grid of (hops, selectivity) variants for mixed queries."""
    variants = []
    hops_grid         = [1, 2, 3, 4]
    selectivity_grid  = [0.01, 0.05, 0.10, 0.25, 0.50]

    has_traversal = any(
        op.get("type") == "TRAVERSAL"
        for op in base_query.get("operations", [])
    )
    has_filter = any(
        op.get("type") == "FILTER"
        for op in base_query.get("operations", [])
    )

    if has_traversal:
        variants.extend(generate_hop_variants(base_query, hops_grid))
    if has_filter:
        variants.extend(generate_filter_variants(base_query, selectivity_grid))
    if has_traversal and has_filter:
        # Cross-product: vary both hops and selectivity
        for h in hops_grid:
            for sv in selectivity_grid:
                q = copy.deepcopy(base_query)
                q["query_id"] = (
                    f"{base_query['query_id']}_h{h}_sv{int(sv * 100):03d}"
                )
                for op in q.get("operations", []):
                    if op.get("type") == "TRAVERSAL" and "traversal" in op:
                        op["traversal"]["max_hops"] = h
                    if op.get("type") == "FILTER" and "predicate" in op:
                        op["predicate"]["_selectivity_hint"] = sv
                variants.append(q)
    return variants


# ─── Orchestration ────────────────────────────────────────────────────────────

def _load_queries(filepath: str) -> list:
    with open(filepath) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def _save_variants(variants: list, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(variants, f, indent=2)
    logger.info("Saved %d variants → %s", len(variants), output_path)


def generate_all_variants(queries_dir: str = SAMPLE_QUERIES_DIR) -> dict:
    """Generate all variant sets and write them to the queries directory.

    Returns a dict mapping output filename → list of variant queries.
    """
    results = {}

    # ── SNB traversal variants (hop sweep) ───────────────────────────────────
    snb_path = os.path.join(queries_dir, "snb_queries.json")
    if os.path.exists(snb_path):
        snb_queries = _load_queries(snb_path)
        traversal_variants = []
        for q in snb_queries:
            if any(op.get("type") == "TRAVERSAL" for op in q.get("operations", [])):
                traversal_variants.extend(generate_combined_variants(q))
        out = os.path.join(queries_dir, "snb_traversal_variants.json")
        _save_variants(traversal_variants, out)
        results["snb_traversal_variants"] = traversal_variants
        logger.info("SNB traversal variants: %d", len(traversal_variants))

    # ── Synthetic graph variants ──────────────────────────────────────────────
    syn_path = os.path.join(queries_dir, "synthetic_queries.json")
    if os.path.exists(syn_path):
        syn_queries = _load_queries(syn_path)
        syn_variants = []
        for q in syn_queries:
            syn_variants.extend(generate_combined_variants(q))
        out = os.path.join(queries_dir, "synthetic_variants.json")
        _save_variants(syn_variants, out)
        results["synthetic_variants"] = syn_variants
        logger.info("Synthetic variants: %d", len(syn_variants))

    # ── TPC-H join variants ───────────────────────────────────────────────────
    tpch_path = os.path.join(queries_dir, "tpch_queries.json")
    if os.path.exists(tpch_path):
        tpch_queries = _load_queries(tpch_path)
        tpch_variants = []
        for q in tpch_queries:
            # Selectivity sweep for filter-heavy TPC-H queries
            tpch_variants.extend(
                generate_filter_variants(q, [0.005, 0.01, 0.05, 0.10, 0.25])
            )
            # Join strategy sweep
            if any(op.get("type") == "JOIN" for op in q.get("operations", [])):
                tpch_variants.extend(
                    generate_join_variants(q, ["inner", "left"])
                )
        out = os.path.join(queries_dir, "tpch_join_variants.json")
        _save_variants(tpch_variants, out)
        results["tpch_join_variants"] = tpch_variants
        logger.info("TPC-H join variants: %d", len(tpch_variants))

    total = sum(len(v) for v in results.values())
    logger.info("\nTotal variants generated: %d", total)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_all_variants(SAMPLE_QUERIES_DIR)
