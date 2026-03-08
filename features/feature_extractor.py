"""FeatureExtractor: builds 22-dimensional feature vectors for ML routing."""

import json
import math
import os
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from parser.ast_nodes import QueryNode, SubExpression
from features.stats_collector import StatsCollector
from features.historical_store import HistoricalStore

# Load canonical feature names from schema
_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "feature_schema.json",
)

def _load_feature_names() -> List[str]:
    with open(_SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    return [feat["name"] for feat in schema["features"]]

FEATURE_NAMES = _load_feature_names()
NUM_FEATURES = len(FEATURE_NAMES)  # 22


class FeatureExtractor:
    """Builds a fixed-length (22) numeric feature vector for a SubExpression.

    Uses precomputed stats (cardinality, degree distributions) and optional
    historical runtime data from SQLite to construct features for the ML classifier.
    """

    def __init__(self, stats_dir: str = "data/stats",
                 history_db_path: str = "data/runtime_history.db"):
        self.stats = StatsCollector(stats_dir)
        self.history = HistoricalStore(history_db_path)

    def extract(self, sub_expr: SubExpression) -> np.ndarray:
        """Build the canonical 22-feature vector for a SubExpression."""
        nodes = sub_expr.nodes

        # --- STEP 1: Query Shape Features ---
        op_counts = self._count_op_types(nodes)
        ast_depth = self._compute_ast_depth(nodes)
        has_traversal = 1 if op_counts["TRAVERSAL"] > 0 else 0
        max_hops = self._get_max_hops(nodes)

        # --- STEP 2: Data Statistics Features ---
        source = self._get_primary_source(nodes)
        is_graph = self.stats.is_graph_source(source)

        input_cardinality = max(self.stats.get_row_count(source), 1)
        input_cardinality_log = math.log10(input_cardinality + 1)

        selectivity = self._compute_selectivity(nodes, source)
        output_cardinality = max(int(input_cardinality * selectivity), 1)
        output_cardinality_log = math.log10(output_cardinality + 1)

        if is_graph:
            graph_stats = self.stats.get_graph_stats(source) or {}
            avg_degree = graph_stats.get("avg_degree", 0.0)
            max_degree = graph_stats.get("max_degree", 0.0)
            stddev_degree = graph_stats.get("stddev_degree", 0.0)
            degree_skew = stddev_degree / avg_degree if avg_degree > 0 else 0.0
        else:
            avg_degree = 0.0
            max_degree = 0.0
            degree_skew = 0.0

        # Number of output columns from last node in the chain
        num_projected_columns = len(nodes[-1].fields) if nodes[-1].fields else 1

        has_index = self._check_has_index(nodes, source)
        join_fanout = self._compute_join_fanout(nodes, input_cardinality)

        # --- STEP 3: Engine Cost Proxies ---
        avg_row_bytes = num_projected_columns * 8  # rough estimate
        estimated_shuffle_bytes_log = math.log10(
            output_cardinality * avg_row_bytes + 1
        )

        if has_traversal:
            start_vertices = max(int(input_cardinality * selectivity), 1)
            estimated_traversal_ops = start_vertices * (
                avg_degree ** max_hops if avg_degree > 0 and max_hops > 0 else 1.0
            )
        else:
            estimated_traversal_ops = 0.0

        # Count distinct source table/graph names
        num_tables_joined = self._count_distinct_sources(nodes)

        # --- STEP 4: Historical Features ---
        op_types = [n.op_type for n in nodes]
        fingerprint = HistoricalStore.compute_fingerprint(op_types, source)
        hist_avg_runtime_ms, hist_runtime_variance = self.history.lookup(fingerprint)

        # --- STEP 5: Assemble vector in schema order ---
        vector = np.array([
            op_counts["FILTER"],
            op_counts["JOIN"],
            op_counts["TRAVERSAL"],
            op_counts["AGGREGATE"],
            op_counts["MAP"],
            ast_depth,
            has_traversal,
            max_hops,
            input_cardinality_log,
            output_cardinality_log,
            selectivity,
            avg_degree,
            max_degree,
            degree_skew,
            num_projected_columns,
            has_index,
            join_fanout,
            estimated_shuffle_bytes_log,
            estimated_traversal_ops,
            hist_avg_runtime_ms,
            hist_runtime_variance,
            num_tables_joined,
        ], dtype=np.float32)

        # Update estimated output rows on the sub-expression
        sub_expr.estimated_output_rows = output_cardinality

        return vector

    def extract_dict(self, sub_expr: SubExpression) -> Dict[str, float]:
        """Extract features as a named dictionary (useful for debugging/CSV export)."""
        vector = self.extract(sub_expr)
        return dict(zip(FEATURE_NAMES, vector.tolist()))

    # --- Helper methods ---

    @staticmethod
    def _count_op_types(nodes: List[QueryNode]) -> Dict[str, int]:
        counts = {"FILTER": 0, "JOIN": 0, "TRAVERSAL": 0, "AGGREGATE": 0, "MAP": 0}
        for node in nodes:
            if node.op_type in counts:
                counts[node.op_type] += 1
        return counts

    @staticmethod
    def _compute_ast_depth(nodes: List[QueryNode]) -> int:
        """Compute max depth of the AST subtree via BFS from root nodes."""
        if not nodes:
            return 0

        node_ids = {n.op_id for n in nodes}
        children: Dict[str, List[str]] = {n.op_id: [] for n in nodes}
        for node in nodes:
            for dep in node.depends_on:
                if dep in node_ids:
                    children[dep].append(node.op_id)

        roots = [n.op_id for n in nodes
                 if not any(d in node_ids for d in n.depends_on)]
        if not roots:
            roots = [nodes[0].op_id]

        max_depth = 0
        queue = deque((r, 1) for r in roots)
        visited = set()
        while queue:
            oid, depth = queue.popleft()
            if oid in visited:
                continue
            visited.add(oid)
            max_depth = max(max_depth, depth)
            for child_id in children.get(oid, []):
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

        return max_depth

    @staticmethod
    def _get_max_hops(nodes: List[QueryNode]) -> int:
        max_hops = 0
        for node in nodes:
            if node.traversal and "max_hops" in node.traversal:
                max_hops = max(max_hops, node.traversal["max_hops"])
        return max_hops

    def _get_primary_source(self, nodes: List[QueryNode]) -> str:
        """Get the primary data source for the sub-expression.

        Returns the first node's source that isn't a reference to another op.
        """
        op_ids = {n.op_id for n in nodes}
        for node in nodes:
            if node.source not in op_ids:
                return node.source
        return nodes[0].source

    def _compute_selectivity(self, nodes: List[QueryNode], source: str) -> float:
        """Estimate selectivity of filter predicates.

        Uses column stats (distinct count) to approximate selectivity.
        Falls back to heuristic defaults.
        Also accounts for TRAVERSAL start_vertex_filter as an implicit filter.
        """
        selectivity = 1.0

        for node in nodes:
            # Handle TRAVERSAL start_vertex_filter as an equality predicate
            if node.op_type == "TRAVERSAL" and node.traversal:
                svf = node.traversal.get("start_vertex_filter")
                if svf:
                    row_count = self.stats.get_row_count(node.source)
                    if row_count > 0:
                        selectivity *= 1.0 / row_count
                    else:
                        selectivity *= 0.001  # single vertex from unknown graph
                continue

            if node.op_type != "FILTER" or node.predicate is None:
                continue

            pred = node.predicate
            col = pred.get("column", "")
            operator = pred.get("operator", "=")

            col_stats = self.stats.get_column_stats(source, col)

            if operator == "=":
                if col_stats and col_stats.get("distinct_count", 0) > 0:
                    selectivity *= 1.0 / col_stats["distinct_count"]
                else:
                    selectivity *= 0.1  # default for equality
            elif operator in (">", "<", ">=", "<="):
                if col_stats and col_stats.get("min") is not None and col_stats.get("max") is not None:
                    value = pred.get("value", 0)
                    try:
                        val = float(value)
                        col_min = float(col_stats["min"])
                        col_max = float(col_stats["max"])
                        col_range = col_max - col_min
                        if col_range > 0:
                            if operator in (">", ">="):
                                selectivity *= max(0.01, (col_max - val) / col_range)
                            else:
                                selectivity *= max(0.01, (val - col_min) / col_range)
                        else:
                            selectivity *= 0.5
                    except (ValueError, TypeError):
                        selectivity *= 0.33
                else:
                    selectivity *= 0.33
            elif operator == "IN":
                values = pred.get("value", [])
                if isinstance(values, list) and col_stats and col_stats.get("distinct_count", 0) > 0:
                    selectivity *= min(1.0, len(values) / col_stats["distinct_count"])
                else:
                    selectivity *= 0.2
            elif operator == "LIKE":
                selectivity *= 0.1
            else:
                selectivity *= 0.5

        return max(0.001, min(1.0, selectivity))

    def _check_has_index(self, nodes: List[QueryNode], source: str) -> int:
        """Check if predicate column likely has an index (heuristic: primary keys)."""
        for node in nodes:
            if node.predicate:
                col = node.predicate.get("column", "")
                # Heuristic: columns ending in 'key' or 'id' are typically indexed
                if col.endswith("key") or col.endswith("id") or col == "id":
                    return 1
        return 0

    def _compute_join_fanout(self, nodes: List[QueryNode],
                             input_cardinality: int) -> float:
        """Estimate join fanout ratio."""
        for node in nodes:
            if node.op_type == "JOIN" and node.join:
                right_source = node.join.get("right_source", "")
                right_rows = self.stats.get_row_count(right_source)
                if right_rows > 0 and input_cardinality > 0:
                    # For inner joins, fanout ~ min(left, right) / left
                    estimated_out = min(input_cardinality, right_rows)
                    return estimated_out / input_cardinality
                return 1.0
        return 1.0

    @staticmethod
    def _count_distinct_sources(nodes: List[QueryNode]) -> int:
        """Count distinct table/graph names referenced (excluding op_id references)."""
        op_ids = {n.op_id for n in nodes}
        sources = set()
        for node in nodes:
            if node.source not in op_ids:
                sources.add(node.source)
            if node.join:
                rs = node.join.get("right_source", "")
                if rs not in op_ids:
                    sources.add(rs)
        return max(len(sources), 1)

    def close(self):
        self.history.close()
