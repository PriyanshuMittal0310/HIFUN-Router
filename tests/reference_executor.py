"""ReferenceExecutor: naive Python/pandas executor for correctness testing.

Executes ALL operations through a single pandas-based engine (no ML routing),
providing a ground-truth result to compare against the HybridRouter's output.
"""

import os
import logging
from typing import Dict, Optional

import pandas as pd

from parser.dsl_parser import DSLParser
from parser.ast_nodes import QueryNode

logger = logging.getLogger(__name__)


class ReferenceExecutor:
    """Executes HIFUN DSL queries naively using pure pandas.

    All operations go through the same pandas code path, regardless of whether
    they are relational or graph operations. This provides a correctness oracle.
    """

    def __init__(self, parquet_dir: str, graph_dir: str):
        """
        Args:
            parquet_dir: directory with table parquet folders (e.g. tpch/)
            graph_dir: directory with graph parquet files
        """
        self.parquet_dir = parquet_dir
        self.graph_dir = graph_dir
        self._table_cache: Dict[str, pd.DataFrame] = {}
        self.parser = DSLParser()
        # Intermediate results keyed by op_id
        self._op_results: Dict[str, pd.DataFrame] = {}

    def load_table(self, source_name: str) -> pd.DataFrame:
        """Load a table from parquet."""
        if source_name in self._table_cache:
            return self._table_cache[source_name]

        candidates = [
            os.path.join(self.parquet_dir, source_name),
            os.path.join(self.parquet_dir, f"{source_name}.parquet"),
            os.path.join(self.graph_dir, f"{source_name}.parquet"),
            os.path.join(self.graph_dir, source_name),
        ]
        for path in candidates:
            if os.path.exists(path):
                df = pd.read_parquet(path)
                self._table_cache[source_name] = df
                return df

        # Handle graph-specific patterns
        if "vertices" in source_name or "vertex" in source_name:
            path = os.path.join(self.graph_dir, "synthetic_vertices.parquet")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                self._table_cache[source_name] = df
                return df

        if "edges" in source_name or "edge" in source_name:
            path = os.path.join(self.graph_dir, "synthetic_edges.parquet")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                self._table_cache[source_name] = df
                return df

        raise FileNotFoundError(f"Table '{source_name}' not found")

    def execute(self, query_json: dict) -> pd.DataFrame:
        """Execute a query and return the final result."""
        self._op_results.clear()
        nodes = self.parser.parse(query_json)

        result = None
        last_idx = len(nodes) - 1
        for i, node in enumerate(nodes):
            is_final = (i == last_idx)
            result = self._execute_node(node, apply_projection=is_final)
            self._op_results[node.op_id] = result

        return result if result is not None else pd.DataFrame()

    def _resolve_source(self, source_name: str) -> pd.DataFrame:
        """Resolve a source name: check op results first, then load table."""
        if source_name in self._op_results:
            return self._op_results[source_name]
        return self.load_table(source_name)

    def _execute_node(self, node: QueryNode, apply_projection: bool = True) -> pd.DataFrame:
        """Execute a single node."""
        if node.op_type == "FILTER":
            return self._exec_filter(node, apply_projection)
        elif node.op_type == "JOIN":
            return self._exec_join(node, apply_projection)
        elif node.op_type == "AGGREGATE":
            return self._exec_aggregate(node)
        elif node.op_type == "MAP":
            return self._exec_map(node, apply_projection)
        elif node.op_type == "TRAVERSAL":
            return self._exec_traversal(node)
        else:
            raise ValueError(f"Unknown op_type: {node.op_type}")

    def _exec_filter(self, node: QueryNode, apply_projection: bool = True) -> pd.DataFrame:
        """Execute FILTER operation."""
        df = self._resolve_source(node.source)
        p = node.predicate
        if p is None:
            if apply_projection and node.fields:
                available = [c for c in node.fields if c in df.columns]
                return df[available].reset_index(drop=True) if available else df
            return df

        col = p["column"]
        op = p["operator"]
        val = p["value"]

        if col not in df.columns:
            if apply_projection and node.fields:
                available = [c for c in node.fields if c in df.columns]
                return df[available].reset_index(drop=True) if available else df
            return df

        if op == "=":
            mask = df[col] == val
        elif op == ">":
            mask = df[col] > val
        elif op == "<":
            mask = df[col] < val
        elif op == ">=":
            mask = df[col] >= val
        elif op == "<=":
            mask = df[col] <= val
        elif op == "IN":
            mask = df[col].isin(val if isinstance(val, list) else [val])
        elif op == "LIKE":
            pattern = str(val).replace("%", ".*").replace("_", ".")
            mask = df[col].astype(str).str.match(pattern, na=False)
        else:
            raise ValueError(f"Unsupported operator: {op}")

        result = df[mask]
        if apply_projection and node.fields:
            available = [c for c in node.fields if c in result.columns]
            if available:
                result = result[available]
        return result.reset_index(drop=True)

    def _exec_join(self, node: QueryNode, apply_projection: bool = True) -> pd.DataFrame:
        """Execute JOIN operation."""
        j = node.join
        if j is None:
            raise ValueError(f"JOIN node {node.op_id} missing join spec")

        left_df = self._resolve_source(node.source)
        right_df = self._resolve_source(j["right_source"])

        left_key = j["left_key"]
        right_key = j["right_key"]
        how = j.get("join_type", "INNER").lower()

        result = pd.merge(
            left_df, right_df,
            left_on=left_key, right_on=right_key,
            how=how, suffixes=("", "_right")
        )

        if apply_projection and node.fields:
            available = [c for c in node.fields if c in result.columns]
            if available:
                result = result[available]
        return result.reset_index(drop=True)

    def _exec_aggregate(self, node: QueryNode) -> pd.DataFrame:
        """Execute AGGREGATE operation."""
        df = self._resolve_source(node.source)
        agg = node.aggregate
        if agg is None:
            raise ValueError(f"AGGREGATE node {node.op_id} missing aggregate spec")

        group_by = agg["group_by"]
        functions = agg["functions"]

        agg_exprs = {}
        for fn in functions:
            func_name = fn["func"].upper()
            col = fn["column"]
            pandas_func = {
                "SUM": "sum", "COUNT": "count", "AVG": "mean",
                "MAX": "max", "MIN": "min"
            }.get(func_name, "sum")
            agg_name = f"{func_name}_{col}"
            agg_exprs[agg_name] = pd.NamedAgg(column=col, aggfunc=pandas_func)

        if group_by:
            result = df.groupby(group_by, as_index=False).agg(**agg_exprs)
        else:
            # No group_by: aggregate entire DataFrame
            row = {}
            for agg_name, named_agg in agg_exprs.items():
                col = named_agg.column
                func = named_agg.aggfunc
                row[agg_name] = df[col].agg(func)
            result = pd.DataFrame([row])

        return result.reset_index(drop=True)

    def _exec_map(self, node: QueryNode, apply_projection: bool = True) -> pd.DataFrame:
        """Execute MAP (project) operation."""
        df = self._resolve_source(node.source)
        if apply_projection and node.fields:
            available = [c for c in node.fields if c in df.columns]
            if available:
                return df[available].reset_index(drop=True)
        return df

    def _exec_traversal(self, node: QueryNode) -> pd.DataFrame:
        """Execute TRAVERSAL via BFS over pandas DataFrames."""
        t = node.traversal
        if t is None:
            raise ValueError(f"TRAVERSAL node {node.op_id} missing traversal spec")

        source = node.source

        # Load vertices and edges
        try:
            vertices = self.load_table(f"{source}_vertices")
        except Exception:
            vertices = self.load_table("synthetic_vertices")

        try:
            edges = self.load_table(f"{source}_edges")
        except Exception:
            edges = self.load_table("synthetic_edges")

        start_filter = t["start_vertex_filter"]
        max_hops = t.get("max_hops", 1)
        direction = t.get("direction", "BOTH")
        return_fields = t.get("return_fields", ["id"])

        # Find start vertices
        col = start_filter["column"]
        val = start_filter["value"]
        if col in vertices.columns:
            start_ids = set(vertices[vertices[col] == val]["id"].tolist())
        else:
            start_ids = set()

        # Build adjacency
        out_adj = {}
        in_adj = {}
        for _, row in edges.iterrows():
            src, dst = row["src"], row["dst"]
            out_adj.setdefault(src, []).append(dst)
            in_adj.setdefault(dst, []).append(src)

        # BFS
        visited = set(start_ids)
        frontier = set(start_ids)

        for _ in range(max_hops):
            next_frontier = set()
            for vid in frontier:
                if direction in ("OUT", "BOTH"):
                    next_frontier.update(out_adj.get(vid, []))
                if direction in ("IN", "BOTH"):
                    next_frontier.update(in_adj.get(vid, []))
            next_frontier -= visited
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier

        result = vertices[vertices["id"].isin(visited)]
        available = [c for c in return_fields if c in result.columns]
        if available:
            result = result[available]
        return result.reset_index(drop=True)
