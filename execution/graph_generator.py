"""GraphGenerator: translates GRAPH-routed SubExpressions into graph operations.

Uses pandas-based BFS for local/testing mode. For production, can use
GraphFrames on PySpark.
"""

import logging
from typing import Dict, Optional

import pandas as pd

from parser.ast_nodes import QueryNode, SubExpression

logger = logging.getLogger(__name__)


class GraphGenerator:
    """Translates a SubExpression labelled 'GRAPH' into graph traversal operations.

    Uses pandas DataFrames for BFS simulation (local mode).
    """

    def __init__(self, graph_dir: str, cache: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Args:
            graph_dir: directory containing vertices.parquet and edges.parquet
            cache: shared result cache {sub_id or op_id -> DataFrame}
        """
        self.graph_dir = graph_dir
        self.cache = cache if cache is not None else {}
        self._vertices = None
        self._edges = None

    @property
    def vertices(self) -> pd.DataFrame:
        if self._vertices is None:
            self._vertices = self._load_vertices()
        return self._vertices

    @property
    def edges(self) -> pd.DataFrame:
        if self._edges is None:
            self._edges = self._load_edges()
        return self._edges

    def _load_vertices(self) -> pd.DataFrame:
        """Load vertex data from parquet."""
        import os
        for name in ["vertices.parquet", "synthetic_vertices.parquet",
                      "snb_vertices.parquet", "vertex.parquet"]:
            path = os.path.join(self.graph_dir, name)
            if os.path.exists(path):
                return pd.read_parquet(path)
        raise FileNotFoundError(f"No vertex parquet found in {self.graph_dir}")

    def _load_edges(self) -> pd.DataFrame:
        """Load edge data from parquet."""
        import os
        for name in ["edges.parquet", "synthetic_edges.parquet",
                      "snb_edges.parquet", "edge.parquet"]:
            path = os.path.join(self.graph_dir, name)
            if os.path.exists(path):
                return pd.read_parquet(path)
        raise FileNotFoundError(f"No edge parquet found in {self.graph_dir}")

    def generate(self, sub_expr: SubExpression) -> pd.DataFrame:
        """Execute graph-routed subexpression and return result DataFrame."""
        df = None
        for node in sub_expr.nodes:
            df = self._execute_node(node, df)
            # Cache intermediate result by op_id for downstream references
            self.cache[node.op_id] = df
        return df

    def _execute_node(self, node: QueryNode, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Dispatch a single QueryNode."""
        # If node's source is a cached op result, use that instead of pipeline df
        if node.source in self.cache:
            df = self.cache[node.source]
        
        if node.op_type == "TRAVERSAL":
            return self._apply_traversal(node, df)
        elif node.op_type == "FILTER":
            return self._apply_filter(node, df)
        elif node.op_type == "AGGREGATE":
            return self._apply_aggregate(node, df)
        elif node.op_type == "MAP":
            return self._apply_map(node, df)
        elif node.op_type == "JOIN":
            return self._apply_join(node, df)
        else:
            raise ValueError(f"Unsupported op_type in graph engine: {node.op_type}")

    def _apply_traversal(self, node: QueryNode, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Execute BFS traversal on graph data."""
        t = node.traversal
        if t is None:
            raise ValueError(f"TRAVERSAL node {node.op_id} missing traversal spec")

        vertices = self.vertices
        edges = self.edges

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

        # Build adjacency lists for efficiency
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

        # Build result from visited vertices
        result = vertices[vertices["id"].isin(visited)]
        available = [c for c in return_fields if c in result.columns]
        if available:
            result = result[available]
        return result.reset_index(drop=True)

    def _apply_filter(self, node: QueryNode, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Apply post-traversal filter."""
        if df is None:
            df = self.vertices
        p = node.predicate
        if p is None:
            if node.fields:
                return df[[c for c in node.fields if c in df.columns]]
            return df

        col = p["column"]
        op = p["operator"]
        val = p["value"]

        if col not in df.columns:
            if node.fields:
                return df[[c for c in node.fields if c in df.columns]]
            return df

        ops = {"=": "eq", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}
        if op in ops:
            mask = getattr(df[col], ops[op])(val)
        elif op == "IN":
            mask = df[col].isin(val if isinstance(val, list) else [val])
        elif op == "LIKE":
            pattern = str(val).replace("%", ".*").replace("_", ".")
            mask = df[col].astype(str).str.match(pattern, na=False)
        else:
            raise ValueError(f"Unsupported operator: {op}")

        result = df[mask]
        if node.fields:
            available = [c for c in node.fields if c in result.columns]
            if available:
                result = result[available]
        return result.reset_index(drop=True)

    def _apply_aggregate(self, node: QueryNode, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Apply aggregation on graph results."""
        if df is None:
            df = self.vertices
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
            row = {}
            for agg_name, named_agg in agg_exprs.items():
                col = named_agg.column
                func = named_agg.aggfunc
                row[agg_name] = df[col].agg(func)
            result = pd.DataFrame([row])
        return result.reset_index(drop=True)

    def _apply_map(self, node: QueryNode, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Project columns."""
        if df is None:
            df = self.vertices
        if node.fields:
            available = [c for c in node.fields if c in df.columns]
            if available:
                return df[available].reset_index(drop=True)
        return df

    def _apply_join(self, node: QueryNode, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Handle join within graph context."""
        j = node.join
        if j is None:
            raise ValueError(f"JOIN node {node.op_id} missing join spec")

        left_df = df if df is not None else self.vertices

        right_source = j["right_source"]
        right_df = self.cache.get(right_source)
        if right_df is None:
            right_df = self.vertices

        left_key = j["left_key"]
        right_key = j["right_key"]
        join_type = j.get("join_type", "INNER").lower()

        result = pd.merge(
            left_df, right_df,
            left_on=left_key, right_on=right_key,
            how=join_type, suffixes=("", "_right")
        )

        if node.fields:
            available = [c for c in node.fields if c in result.columns]
            if available:
                result = result[available]
        return result.reset_index(drop=True)
