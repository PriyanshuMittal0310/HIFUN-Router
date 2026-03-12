"""GraphFramesGenerator: TRAVERSAL SubExpressions via actual GraphFrames BFS.

Requires the GraphFrames JAR on the Spark classpath:
  graphframes:graphframes:0.8.3-spark3.4-s_2.12

Graph data is loaded from Parquet (local or HDFS):
  <graph_dir>/<graph_name>_vertices.parquet
  <graph_dir>/<graph_name>_edges.parquet

Edge schema requirements (MANDATORY):
  edges:    src (LongType or StringType), dst, relationship (StringType)
  vertices: id  (must match src/dst type)

BFS produces columns: from, e0, v1, [e1, v2, ...], to
We extract destination vertex fields from the "to" struct.
"""

import logging
import os
from typing import Dict, Optional

from graphframes import GraphFrame
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

from parser.ast_nodes import QueryNode, SubExpression

logger = logging.getLogger(__name__)

# If a graph has more vertices than this, skip the cache() call (memory safety)
_CACHE_VERTEX_THRESHOLD = 2_000_000
# Default edge label when traversal spec does not specify one
_DEFAULT_EDGE_LABEL = "KNOWS"


class GraphFramesGenerator:
    """Executes TRAVERSAL SubExpressions using real GraphFrames BFS.

    Caches loaded GraphFrame objects across SubExpressions to avoid
    repeated Parquet reads when multiple traversals use the same graph.

    Results are returned as Spark DataFrames (lazy — not yet materialized).
    """

    def __init__(
        self,
        spark: SparkSession,
        graph_dir: str,
        cache: Optional[Dict[str, DataFrame]] = None,
        hdfs_root: Optional[str] = None,
    ):
        """
        Args:
            spark:     Active SparkSession (must include GraphFrames JAR).
            graph_dir: Local path containing <name>_vertices.parquet / <name>_edges.parquet.
            cache:     Shared result cache {op_id -> Spark DataFrame}.
            hdfs_root: Optional HDFS base URI for reading graph data from HDFS.
        """
        self.spark    = spark
        self.graph_dir = graph_dir
        self.cache: Dict[str, DataFrame] = cache if cache is not None else {}
        self.hdfs_root = hdfs_root
        # GraphFrame objects cached by graph_name
        self._gf_cache: Dict[str, GraphFrame] = {}

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def generate(self, sub_expr: SubExpression) -> DataFrame:
        """Execute a graph-routed SubExpression via GraphFrames.

        Returns a lazy Spark DataFrame.  Callers should call .count() or
        .toPandas() to materialise results.
        """
        df: Optional[DataFrame] = None
        for node in sub_expr.nodes:
            df = self._execute_node(node, df)
            self.cache[node.op_id] = df  # type: ignore[assignment]
        return df  # type: ignore[return-value]

    # ─────────────────────────────────────────────
    # Node dispatch
    # ─────────────────────────────────────────────

    def _execute_node(
        self, node: QueryNode, df: Optional[DataFrame]
    ) -> DataFrame:
        if node.source in self.cache:
            df = self.cache[node.source]

        dispatch = {
            "TRAVERSAL": self._apply_traversal,
            "FILTER":    self._apply_filter,
            "AGGREGATE": self._apply_aggregate,
            "MAP":       self._apply_map,
            "JOIN":      self._apply_join,
        }
        handler = dispatch.get(node.op_type)
        if handler is None:
            raise ValueError(f"GraphFramesGenerator: unknown op_type '{node.op_type}'")
        return handler(node, df)

    # ─────────────────────────────────────────────
    # Graph loading
    # ─────────────────────────────────────────────

    def _resolve_path(self, filename: str) -> str:
        """Try HDFS, then local graph_dir."""
        if self.hdfs_root:
            hdfs_path = f"{self.hdfs_root.rstrip('/')}/{filename}"
            try:
                jvm  = self.spark._jvm  # type: ignore[attr-defined]
                conf = self.spark._jsc.hadoopConfiguration()  # type: ignore
                path_obj = jvm.org.apache.hadoop.fs.Path(hdfs_path)
                fs = path_obj.getFileSystem(conf)
                if fs.exists(path_obj):
                    return hdfs_path
            except Exception as exc:
                logger.warning("HDFS check failed for '%s': %s", hdfs_path, exc)

        local = os.path.join(self.graph_dir, filename)
        if os.path.exists(local):
            return local
        raise FileNotFoundError(
            f"GraphFramesGenerator: cannot find '{filename}' "
            f"in HDFS or local graph_dir '{self.graph_dir}'"
        )

    def _load_graph(self, graph_name: str) -> GraphFrame:
        """Load (or return cached) GraphFrame for the named graph."""
        if graph_name in self._gf_cache:
            return self._gf_cache[graph_name]

        # Try name-specific files first, then generic fallbacks
        v_candidates = [
            f"{graph_name}_vertices.parquet",
            f"{graph_name}/vertices.parquet",
            "vertices.parquet",
        ]
        e_candidates = [
            f"{graph_name}_edges.parquet",
            f"{graph_name}/edges.parquet",
            "edges.parquet",
        ]

        v_path = self._find_candidate(v_candidates)
        e_path = self._find_candidate(e_candidates)

        vertices = self.spark.read.parquet(v_path)
        edges    = self.spark.read.parquet(e_path)

        # GraphFrames requires columns named exactly "id", "src", "dst"
        if "id" not in vertices.columns:
            first_col = vertices.columns[0]
            vertices  = vertices.withColumnRenamed(first_col, "id")
            logger.warning("Renamed vertex id column from '%s' to 'id'", first_col)

        if "src" not in edges.columns or "dst" not in edges.columns:
            raise ValueError(
                f"Edge parquet at '{e_path}' must have 'src' and 'dst' columns. "
                f"Found: {edges.columns}"
            )

        # Ensure 'relationship' column exists (used in BFS edgeFilter)
        if "relationship" not in edges.columns:
            edges = edges.withColumn("relationship", F.lit(_DEFAULT_EDGE_LABEL))

        gf = GraphFrame(vertices, edges)

        # Cache vertex/edge DFs in Spark memory for repeated traversals
        v_count = vertices.count()
        if v_count < _CACHE_VERTEX_THRESHOLD:
            gf.vertices.cache()
            gf.edges.cache()
            logger.info(
                "Cached GraphFrame '%s': %d vertices", graph_name, v_count
            )

        self._gf_cache[graph_name] = gf
        return gf

    def _find_candidate(self, candidates: list) -> str:
        for name in candidates:
            try:
                return self._resolve_path(name)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(
            f"None of the candidate files found: {candidates}"
        )

    # ─────────────────────────────────────────────
    # Operation handlers
    # ─────────────────────────────────────────────

    def _apply_traversal(
        self, node: QueryNode, df: Optional[DataFrame]
    ) -> DataFrame:
        """Execute BFS traversal using GraphFrames .bfs()."""
        t = node.traversal
        if t is None:
            raise ValueError(f"TRAVERSAL node '{node.op_id}' missing traversal spec")

        graph_name   = node.source
        gf           = self._load_graph(graph_name)
        max_hops     = int(t.get("max_hops", 1))
        direction    = t.get("direction", "BOTH").upper()
        edge_label   = t.get("edge_label", _DEFAULT_EDGE_LABEL)
        return_fields = t.get("return_fields", ["id"])

        sf = t.get("start_vertex_filter", {})
        from_expr = self._build_vertex_expr(sf, gf.vertices.columns)

        # For IN direction, reverse the edge direction before BFS
        if direction == "IN":
            reversed_edges = gf.edges.select(
                F.col("dst").alias("src"),
                F.col("src").alias("dst"),
                F.col("relationship"),
            )
            gf = GraphFrame(gf.vertices, reversed_edges)

        edge_filter = f"relationship = '{edge_label}'"

        logger.info(
            "GraphFrames BFS: graph='%s', fromExpr='%s', "
            "edgeFilter='%s', maxPathLength=%d",
            graph_name, from_expr, edge_filter, max_hops,
        )

        bfs_result = gf.bfs(
            fromExpr=from_expr,
            toExpr="id IS NOT NULL",
            edgeFilter=edge_filter,
            maxPathLength=max_hops,
        )

        # BFS result schema: from (struct), e0 (struct), [v1...], to (struct)
        result = self._extract_destination(bfs_result, return_fields)

        # Apply optional filter on destination vertex attributes
        dest_filter = t.get("destination_filter")
        if dest_filter:
            cond = self._col_condition(dest_filter)
            if cond is not None:
                result = result.filter(cond)

        return result.distinct()

    def _apply_filter(
        self, node: QueryNode, df: Optional[DataFrame]
    ) -> DataFrame:
        """Post-traversal filter on pipeline DataFrame."""
        if df is None:
            gf = self._load_graph(node.source)
            df = gf.vertices
        p = node.predicate
        if p is not None:
            cond = self._col_condition(p)
            if cond is not None:
                df = df.filter(cond)
        if node.fields:
            available = set(df.columns)
            proj = [c for c in node.fields if c in available]
            if proj:
                df = df.select(proj)
        return df

    def _apply_aggregate(
        self, node: QueryNode, df: Optional[DataFrame]
    ) -> DataFrame:
        if df is None:
            gf = self._load_graph(node.source)
            df = gf.vertices

        agg = node.aggregate
        if agg is None:
            return df

        group_by = agg.get("group_by", [])
        funcs    = agg.get("functions", [])
        agg_exprs = []
        for fn_spec in funcs:
            func_name = fn_spec.get("func", "count").lower()
            col_name  = fn_spec.get("column", "*")
            alias     = fn_spec.get("alias", f"{func_name}_{col_name}")
            spark_fn  = getattr(F, func_name, None)
            if spark_fn is None:
                continue
            expr = spark_fn(F.col(col_name)) if col_name != "*" else spark_fn("*")
            agg_exprs.append(expr.alias(alias))

        if not agg_exprs:
            return df
        if group_by:
            return df.groupBy(*group_by).agg(*agg_exprs)
        return df.agg(*agg_exprs)

    def _apply_map(
        self, node: QueryNode, df: Optional[DataFrame]
    ) -> DataFrame:
        if df is None:
            gf = self._load_graph(node.source)
            df = gf.vertices
        if node.fields:
            available = set(df.columns)
            proj = [c for c in node.fields if c in available]
            if proj:
                return df.select(proj)
        return df

    def _apply_join(
        self, node: QueryNode, df: Optional[DataFrame]
    ) -> DataFrame:
        """Join graph traversal result with a relational table via Spark SQL."""
        if df is None:
            gf = self._load_graph(node.source)
            df = gf.vertices

        j = node.join
        if j is None:
            return df

        right_source = j.get("right_source", "")
        if right_source in self.cache:
            right_df = self.cache[right_source]
        else:
            right_path = os.path.join(self.graph_dir, f"{right_source}.parquet")
            if not os.path.exists(right_path):
                logger.warning("Join right-side '%s' not found; returning left side", right_source)
                return df
            right_df = self.spark.read.parquet(right_path)

        left_key  = j.get("left_key",  "id")
        right_key = j.get("right_key", "id")
        join_type = j.get("join_type", "inner").lower()

        joined = df.join(right_df, df[left_key] == right_df[right_key], how=join_type)
        if node.fields:
            available = set(joined.columns)
            proj = [c for c in node.fields if c in available]
            if proj:
                return joined.select(proj)
        return joined

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _build_vertex_expr(sf: dict, columns: list) -> str:
        """Build a GraphFrames BFS fromExpr string from a start-vertex filter."""
        if not sf:
            return "id IS NOT NULL"
        col   = sf.get("column", "id")
        op    = sf.get("operator", "=")
        value = sf.get("value")

        if col not in columns:
            logger.warning("Vertex filter column '%s' not in schema; using 'id IS NOT NULL'", col)
            return "id IS NOT NULL"

        if op == "=" and isinstance(value, str):
            return f"{col} = '{value}'"
        if op == "=" and value is not None:
            return f"{col} = {value}"
        if op == "IN" and isinstance(value, list):
            vals = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in value)
            return f"{col} IN ({vals})"
        return "id IS NOT NULL"

    @staticmethod
    def _col_condition(predicate: dict):
        """Build a PySpark Column condition from a predicate dict."""
        col   = predicate.get("column")
        op    = predicate.get("operator", "=")
        value = predicate.get("value")

        if col is None:
            return None

        c = F.col(col)
        op_map = {
            "=":    lambda: c == value,
            "!=":   lambda: c != value,
            ">":    lambda: c > value,
            "<":    lambda: c < value,
            ">=":   lambda: c >= value,
            "<=":   lambda: c <= value,
            "IN":   lambda: c.isin(value if isinstance(value, list) else [value]),
            "LIKE": lambda: c.like(str(value)),
        }
        fn = op_map.get(op)
        return fn() if fn else None

    @staticmethod
    def _extract_destination(bfs_result: DataFrame, return_fields: list) -> DataFrame:
        """Extract fields from the 'to' struct column of a BFS result."""
        if "to" not in bfs_result.columns:
            # Degenerate case: fromExpr matched the start vertices only
            return bfs_result

        to_schema: StructType = bfs_result.schema["to"].dataType  # type: ignore
        available_fields = {f.name for f in to_schema.fields}

        proj = [
            F.col(f"to.{field}").alias(field)
            for field in return_fields
            if field in available_fields
        ]

        if not proj:
            # Fallback: return "to.id"
            if "id" in available_fields:
                proj = [F.col("to.id").alias("id")]
            else:
                return bfs_result.select("to")

        return bfs_result.select(proj)
