"""SQLGenerator: translates SQL-routed SubExpressions into executable operations.

Supports two backends:
  - PySpark DataFrame API (production, requires SparkSession)
  - Pandas DataFrame API (lightweight, for testing without Spark)
"""

import logging
from typing import Dict, Optional

import pandas as pd

from parser.ast_nodes import QueryNode, SubExpression

logger = logging.getLogger(__name__)


class SQLGenerator:
    """Translates a SubExpression labelled 'SQL' into DataFrame operations.

    Uses pandas DataFrames for local execution. For Spark execution, use
    SparkSQLGenerator which wraps PySpark DataFrame API.
    """

    def __init__(self, data_loader, cache: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Args:
            data_loader: callable(source_name) -> pd.DataFrame
                         Loads a table/source by name.
            cache: shared result cache {sub_id or op_id -> DataFrame}
                   for resolving depends_on references.
        """
        self.data_loader = data_loader
        self.cache = cache if cache is not None else {}

    def generate(self, sub_expr: SubExpression) -> pd.DataFrame:
        """Execute SQL-routed subexpression and return result DataFrame."""
        df = None
        last_idx = len(sub_expr.nodes) - 1
        for i, node in enumerate(sub_expr.nodes):
            df = self._execute_node(node, df, is_final=(i == last_idx))
            # Cache the FULL result (before field projection) by op_id
            # so downstream references get all columns
            self.cache[node.op_id] = df
        return df

    def _execute_node(self, node: QueryNode, df: Optional[pd.DataFrame],
                      is_final: bool = True) -> pd.DataFrame:
        """Dispatch a single QueryNode to the appropriate handler.

        Resolution logic: If node.source refers to a cached op_id, use that
        result (not the pipeline df). This ensures correct data flow when
        nodes reference specific upstream operations.
        
        Field projection is only applied on the final node of a sub-expression
        to avoid dropping columns needed by downstream operations.
        """
        # If node's source is an earlier op result, resolve from cache
        if node.source in self.cache:
            df = self.cache[node.source]
        
        # For intermediate nodes, suppress field projection
        apply_fields = node.fields if is_final else None
        
        if node.op_type == "FILTER":
            return self._apply_filter(node, df, fields_override=apply_fields)
        elif node.op_type == "JOIN":
            return self._apply_join(node, df, fields_override=apply_fields)
        elif node.op_type == "AGGREGATE":
            return self._apply_aggregate(node, df)
        elif node.op_type == "MAP":
            return self._apply_map(node, df, fields_override=apply_fields)
        elif node.op_type == "TRAVERSAL":
            return self._apply_traversal_as_sql(node, df)
        else:
            raise ValueError(f"Unknown op_type: {node.op_type}")

    def _resolve_source(self, source_name: str, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Resolve a source name to a DataFrame: cache first, then loader."""
        if df is not None:
            return df
        if source_name in self.cache:
            return self.cache[source_name]
        return self.data_loader(source_name)

    def _apply_filter(self, node: QueryNode, df: Optional[pd.DataFrame],
                      fields_override=None) -> pd.DataFrame:
        """Apply FILTER operation."""
        df = self._resolve_source(node.source, df)
        fields = fields_override
        p = node.predicate
        if p is None:
            if fields:
                return df[[c for c in fields if c in df.columns]]
            return df

        col = p["column"]
        op = p["operator"]
        val = p["value"]

        if col not in df.columns:
            logger.warning(f"Column {col} not found, returning unfiltered")
            if fields:
                return df[[c for c in fields if c in df.columns]]
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
            raise ValueError(f"Unsupported filter operator: {op}")

        result = df[mask]
        if fields:
            available = [c for c in fields if c in result.columns]
            if available:
                result = result[available]
        return result.reset_index(drop=True)

    def _apply_join(self, node: QueryNode, df: Optional[pd.DataFrame],
                    fields_override=None) -> pd.DataFrame:
        """Apply JOIN operation."""
        j = node.join
        if j is None:
            raise ValueError(f"JOIN node {node.op_id} missing join spec")
        fields = fields_override

        # Left side: always resolve from source (table name or cached op_id)
        left_df = self._resolve_source(node.source, None)

        # Right side: from cache (upstream op_id) or load from table
        right_source = j["right_source"]
        right_df = self.cache.get(right_source)
        if right_df is None:
            right_df = self.data_loader(right_source)

        left_key = j["left_key"]
        right_key = j["right_key"]
        join_type = j.get("join_type", "INNER").lower()

        # Map DSL join types to pandas merge how
        how_map = {"inner": "inner", "left": "left", "right": "right"}
        how = how_map.get(join_type, "inner")

        result = pd.merge(
            left_df, right_df,
            left_on=left_key, right_on=right_key,
            how=how, suffixes=("", "_right")
        )

        if fields:
            available = [c for c in fields if c in result.columns]
            if available:
                result = result[available]

        return result.reset_index(drop=True)

    def _apply_aggregate(self, node: QueryNode, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Apply AGGREGATE operation."""
        df = self._resolve_source(node.source, df)
        agg = node.aggregate
        if agg is None:
            raise ValueError(f"AGGREGATE node {node.op_id} missing aggregate spec")

        group_by = agg["group_by"]
        functions = agg["functions"]

        # Build pandas aggregation dict
        agg_dict = {}
        rename_map = {}
        for fn in functions:
            func_name = fn["func"].upper()
            col = fn["column"]
            pandas_func = {
                "SUM": "sum", "COUNT": "count", "AVG": "mean",
                "MAX": "max", "MIN": "min"
            }.get(func_name, "sum")

            agg_key = f"{func_name}_{col}"
            agg_dict[col] = pandas_func
            rename_map[col] = agg_key

        # Need to handle multiple aggs on same column
        # Use groupby + agg with named aggregations
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

    def _apply_map(self, node: QueryNode, df: Optional[pd.DataFrame],
                   fields_override=None) -> pd.DataFrame:
        """Apply MAP (project) operation — select specified columns."""
        df = self._resolve_source(node.source, df)
        fields = fields_override if fields_override is not None else node.fields
        if fields:
            available = [c for c in fields if c in df.columns]
            if available:
                return df[available].reset_index(drop=True)
        return df

    def _apply_traversal_as_sql(self, node: QueryNode, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """SQL fallback for TRAVERSAL: simulate BFS via iterative self-joins.

        This is intentionally slow (to demonstrate SQL is suboptimal for graph ops).
        """
        t = node.traversal
        if t is None:
            raise ValueError(f"TRAVERSAL node {node.op_id} missing traversal spec")

        # Load vertex and edge data
        source = node.source
        try:
            vertices = self.data_loader(f"{source}_vertices")
        except Exception:
            vertices = self.data_loader(source)
        try:
            edges = self.data_loader(f"{source}_edges")
        except Exception:
            # Return empty DataFrame if no edge data
            return pd.DataFrame(columns=t.get("return_fields", ["id"]))

        start_filter = t["start_vertex_filter"]
        max_hops = t.get("max_hops", 1)
        return_fields = t.get("return_fields", ["id"])

        # Find start vertices
        col = start_filter["column"]
        val = start_filter["value"]
        if col in vertices.columns:
            current_ids = set(vertices[vertices[col] == val]["id"].tolist())
        else:
            current_ids = set()

        # BFS via repeated joins
        visited = set(current_ids)
        for _ in range(max_hops):
            neighbors = set()
            for vid in current_ids:
                out_neighbors = edges[edges["src"] == vid]["dst"].tolist()
                in_neighbors = edges[edges["dst"] == vid]["src"].tolist()
                direction = t.get("direction", "BOTH")
                if direction == "OUT":
                    neighbors.update(out_neighbors)
                elif direction == "IN":
                    neighbors.update(in_neighbors)
                else:  # BOTH
                    neighbors.update(out_neighbors + in_neighbors)
            new_ids = neighbors - visited
            if not new_ids:
                break
            visited.update(new_ids)
            current_ids = new_ids

        # Filter vertices to visited set
        result = vertices[vertices["id"].isin(visited)]
        available = [c for c in return_fields if c in result.columns]
        if available:
            result = result[available]
        return result.reset_index(drop=True)
