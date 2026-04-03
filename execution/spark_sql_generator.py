"""SparkSQLGenerator: translates SQL-routed SubExpressions into real PySpark operations.

Uses the PySpark DataFrame API with Catalyst optimizer enabled.
Predicate pushdown, AQE (Adaptive Query Execution), and broadcast join
hints are all applied transparently via SparkSession configuration.

Supports both local Parquet data and HDFS-backed Parquet tables.
"""

import logging
import os
from typing import Dict, Optional, Tuple, Union

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType

from parser.ast_nodes import QueryNode, SubExpression

logger = logging.getLogger(__name__)

# Threshold (rows) below which a table is broadcast-joined (avoids shuffle)
_BROADCAST_ROW_THRESHOLD = 100_000
# Average row size in bytes (conservative estimate for broadcast size calc)
_AVG_ROW_BYTES = 256


class SparkSQLGenerator:
    """Executes SQL-routed SubExpressions using the PySpark DataFrame API.

    Catalyst optimizer handles:
    - Predicate pushdown into Parquet row-group statistics
    - Columnar reads (only projected columns are loaded)
    - AQE: dynamic partition coalescing and skew join mitigation
    - Broadcast join for tables below threshold

    Supports upstream results from previous SubExpressions stored as
    Spark DataFrames in the shared cache.
    """

    def __init__(
        self,
        spark: SparkSession,
        parquet_dir: str,
        cache: Optional[Dict[str, DataFrame]] = None,
        hdfs_root: Optional[str] = None,
    ):
        """
        Args:
            spark:       Active SparkSession (configured with AQE, GraphFrames JAR).
            parquet_dir: Local path to Parquet table directory.  If hdfs_root is
                         set, paths are resolved under HDFS first.
            cache:       Shared result cache {sub_id or op_id -> Spark DataFrame}.
            hdfs_root:   Optional HDFS base URI, e.g. 'hdfs://namenode:9000/data'.
                         When set, table lookup order is HDFS → local Parquet.
        """
        self.spark = spark
        self.parquet_dir = parquet_dir
        self.cache: Dict[str, DataFrame] = cache if cache is not None else {}
        self.hdfs_root = hdfs_root
        # internal Parquet/temp-view cache keyed by source name
        self._source_cache: Dict[str, DataFrame] = {}
        self._traversal_graph_cache: Dict[str, Tuple[DataFrame, DataFrame]] = {}

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def generate(self, sub_expr: SubExpression) -> DataFrame:
        """Translate and execute a SQL-routed SubExpression.

        Returns a *lazy* Spark DataFrame (not yet materialized).
        Callers that need a row count should call .count() explicitly.
        """
        df: Optional[DataFrame] = None
        last_idx = len(sub_expr.nodes) - 1
        for i, node in enumerate(sub_expr.nodes):
            df = self._execute_node(node, df, is_final=(i == last_idx))
            self.cache[node.op_id] = df
        return df  # type: ignore[return-value]

    # ─────────────────────────────────────────────
    # Node dispatch
    # ─────────────────────────────────────────────

    def _execute_node(
        self, node: QueryNode, df: Optional[DataFrame], is_final: bool = True
    ) -> DataFrame:
        # Resolve cached upstream result
        if node.source in self.cache:
            df = self.cache[node.source]

        apply_fields = node.fields if is_final else None

        dispatch = {
            "FILTER":    self._apply_filter,
            "JOIN":      self._apply_join,
            "AGGREGATE": self._apply_aggregate,
            "MAP":       self._apply_map,
            "TRAVERSAL": self._apply_traversal_via_joins,
        }
        handler = dispatch.get(node.op_type)
        if handler is None:
            raise ValueError(f"SparkSQLGenerator: unknown op_type '{node.op_type}'")
        return handler(node, df, fields_override=apply_fields)

    # ─────────────────────────────────────────────
    # Source loading  (HDFS → local Parquet → cache)
    # ─────────────────────────────────────────────

    def _load_source(self, source_name: str) -> DataFrame:
        """Load a Parquet source into a Spark DataFrame with temp-view registration."""
        if source_name in self._source_cache:
            return self._source_cache[source_name]

        path = self._resolve_path(source_name)
        df = (
            self.spark.read
            .option("mergeSchema", "true")
            .parquet(path)
        )
        # Register as a temp SQL view for debugging and Spark SQL fallback
        safe_name = source_name.replace("/", "_").replace(".", "_")
        df.createOrReplaceTempView(safe_name)
        self._source_cache[source_name] = df
        logger.debug("Loaded source '%s' from '%s'", source_name, path)
        return df

    def _resolve_path(self, source_name: str) -> str:
        """Try HDFS first, then local Parquet directory."""
        if self.hdfs_root:
            hdfs_path = f"{self.hdfs_root.rstrip('/')}/{source_name}"
            try:
                # Validate HDFS path exists by touching the Hadoop FS
                jvm = self.spark._jvm  # type: ignore[attr-defined]
                conf = self.spark._jsc.hadoopConfiguration()  # type: ignore
                path_obj = jvm.org.apache.hadoop.fs.Path(hdfs_path)
                fs = path_obj.getFileSystem(conf)
                if fs.exists(path_obj):
                    return hdfs_path
            except Exception as exc:
                logger.warning("HDFS path check failed for '%s': %s", hdfs_path, exc)

        # Local candidates
        candidates = [
            os.path.join(self.parquet_dir, source_name),
            os.path.join(self.parquet_dir, f"{source_name}.parquet"),
            os.path.join(self.parquet_dir, f"{source_name}_vertices"),
            os.path.join(self.parquet_dir, f"{source_name}_vertices.parquet"),
            os.path.join(self.parquet_dir, f"{source_name}_edges"),
            os.path.join(self.parquet_dir, f"{source_name}_edges.parquet"),
            source_name,  # absolute path
        ]

        # Alias graph-style source names to common relational vertex tables.
        alias_sources = {
            "snb": ["person", "persons"],
        }
        for alias in alias_sources.get(source_name, []):
            candidates.extend([
                os.path.join(self.parquet_dir, alias),
                os.path.join(self.parquet_dir, f"{alias}.parquet"),
            ])
        for c in candidates:
            if os.path.exists(c):
                return c
        raise FileNotFoundError(
            f"SparkSQLGenerator: cannot find source '{source_name}' "
            f"in HDFS or local path '{self.parquet_dir}'"
        )

    def _resolve_df(
        self, source_name: str, pipeline_df: Optional[DataFrame]
    ) -> DataFrame:
        """Return pipeline DF if available, else load from source."""
        if pipeline_df is not None:
            return pipeline_df
        if source_name in self.cache:
            return self.cache[source_name]
        return self._load_source(source_name)

    # ─────────────────────────────────────────────
    # Operation handlers
    # ─────────────────────────────────────────────

    def _apply_filter(
        self,
        node: QueryNode,
        df: Optional[DataFrame],
        fields_override=None,
    ) -> DataFrame:
        df = self._resolve_df(node.source, df)
        p = node.predicate
        if p is not None:
            condition = self._build_condition(p)
            if condition is not None:
                # Catalyst pushes this filter into the Parquet scan automatically
                df = df.filter(condition)

        fields = fields_override or node.fields
        if fields:
            available = set(df.columns)
            proj = [c for c in fields if c in available]
            if proj:
                df = df.select(proj)
        return df

    def _apply_join(
        self,
        node: QueryNode,
        df: Optional[DataFrame],
        fields_override=None,
    ) -> DataFrame:
        if df is None:
            df = self._resolve_df(node.source, None)

        j = node.join
        if j is None:
            return df

        right_source = j.get("right_source", "")
        right_df = (
            self.cache[right_source]
            if right_source in self.cache
            else self._load_source(right_source)
        )

        # Use broadcast hint for small right-side tables to skip a shuffle stage
        right_count = self._estimate_count(right_df)
        if right_count < _BROADCAST_ROW_THRESHOLD:
            right_df = F.broadcast(right_df)
            logger.debug("Broadcast join applied for '%s' (%d rows)", right_source, right_count)

        left_key  = j.get("left_key",  "id")
        right_key = j.get("right_key", "id")
        join_type = j.get("join_type", "inner").lower()

        # Disambiguate columns with the same name pre-join
        right_cols = set(right_df.columns) - {right_key}
        for col in right_cols:
            if col in df.columns and col != left_key:
                right_df = right_df.withColumnRenamed(col, f"{right_source}_{col}")

        joined = df.join(right_df, df[left_key] == right_df[right_key], how=join_type)

        fields = fields_override or node.fields
        if fields:
            available = set(joined.columns)
            proj = [c for c in fields if c in available]
            if proj:
                return joined.select(proj)
        return joined

    def _apply_aggregate(
        self,
        node: QueryNode,
        df: Optional[DataFrame],
        fields_override=None,
    ) -> DataFrame:
        if df is None:
            df = self._resolve_df(node.source, None)

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
                logger.warning("Unknown aggregate function '%s', skipping", func_name)
                continue
            expr = spark_fn(F.col(col_name)) if col_name != "*" else spark_fn("*")
            agg_exprs.append(expr.alias(alias))

        if not agg_exprs:
            return df

        if group_by:
            return df.groupBy(*group_by).agg(*agg_exprs)
        return df.agg(*agg_exprs)

    def _apply_map(
        self,
        node: QueryNode,
        df: Optional[DataFrame],
        fields_override=None,
    ) -> DataFrame:
        if df is None:
            df = self._resolve_df(node.source, None)

        fields = fields_override or node.fields
        if fields:
            available = set(df.columns)
            proj = [c for c in fields if c in available]
            if proj:
                return df.select(proj)
        return df

    def _apply_traversal_via_joins(
        self,
        node: QueryNode,
        df: Optional[DataFrame],
        fields_override=None,
    ) -> DataFrame:
        """Execute TRAVERSAL in SQL path via iterative edge joins (Spark DataFrame BFS)."""
        t = node.traversal
        if t is None:
            raise ValueError(f"TRAVERSAL node '{node.op_id}' missing traversal spec")

        vertices, edges = self._load_traversal_graph(node.source)

        start_filter = t.get("start_vertex_filter", {})
        sf_condition = self._build_condition(start_filter) if start_filter else None
        if sf_condition is not None:
            frontier = vertices.filter(sf_condition).select(F.col("id")).distinct()
        else:
            # If no start filter exists, default to all vertices to keep behavior deterministic.
            frontier = vertices.select(F.col("id")).distinct()

        visited = frontier

        max_hops = int(t.get("max_hops", 1))
        direction = str(t.get("direction", "BOTH")).upper()
        edge_label = t.get("edge_label")
        if edge_label and "relationship" in edges.columns:
            edges = edges.filter(F.col("relationship") == edge_label)

        for _ in range(max_hops):
            next_parts = []

            if direction in ("OUT", "BOTH"):
                next_out = (
                    frontier.alias("f")
                    .join(edges.alias("e"), F.col("f.id") == F.col("e.src"), "inner")
                    .select(F.col("e.dst").alias("id"))
                )
                next_parts.append(next_out)

            if direction in ("IN", "BOTH"):
                next_in = (
                    frontier.alias("f")
                    .join(edges.alias("e"), F.col("f.id") == F.col("e.dst"), "inner")
                    .select(F.col("e.src").alias("id"))
                )
                next_parts.append(next_in)

            if not next_parts:
                break

            next_frontier = next_parts[0]
            for p in next_parts[1:]:
                next_frontier = next_frontier.unionByName(p)
            next_frontier = next_frontier.distinct()
            next_frontier = next_frontier.join(visited, on="id", how="left_anti")

            if next_frontier.limit(1).count() == 0:
                break

            visited = visited.unionByName(next_frontier).distinct()
            frontier = next_frontier

        result = visited.join(vertices, on="id", how="inner")

        return_fields = t.get("return_fields") or node.fields
        fields = fields_override or return_fields
        if fields:
            available = set(result.columns)
            proj = [c for c in fields if c in available]
            if proj:
                result = result.select(*proj)

        return result.distinct()

    def _load_traversal_graph(self, graph_name: str) -> Tuple[DataFrame, DataFrame]:
        """Load traversal vertices/edges for SQL-path BFS; cache by graph name."""
        if graph_name in self._traversal_graph_cache:
            return self._traversal_graph_cache[graph_name]

        # Start with the configured parquet directory, then attempt common graph siblings.
        data_root = os.path.dirname(os.path.dirname(self.parquet_dir))
        candidate_dirs = [
            self.parquet_dir,
            os.path.join(data_root, "graphs", graph_name),
            os.path.join(data_root, "graphs"),
        ]

        def _resolve_from_dirs(names: list[str]) -> str:
            for d in candidate_dirs:
                for n in names:
                    p = os.path.join(d, n)
                    if os.path.exists(p):
                        return p
            raise FileNotFoundError(f"Traversal source file not found for {names} in {candidate_dirs}")

        v_path = _resolve_from_dirs([
            f"{graph_name}_vertices.parquet",
            f"{graph_name}_vertices",
            "vertices.parquet",
            "vertices",
        ])
        e_path = _resolve_from_dirs([
            f"{graph_name}_edges.parquet",
            f"{graph_name}_edges",
            "edges.parquet",
            "edges",
            "snb_edges.parquet",
        ])

        vertices = self.spark.read.parquet(v_path)
        edges = self.spark.read.parquet(e_path)

        if "id" not in vertices.columns:
            vertices = vertices.withColumnRenamed(vertices.columns[0], "id")

        if "src" not in edges.columns or "dst" not in edges.columns:
            # Light schema normalization fallback for alternate edge exports.
            if "source" in edges.columns and "target" in edges.columns:
                edges = edges.withColumnRenamed("source", "src").withColumnRenamed("target", "dst")
            else:
                raise ValueError(
                    f"Traversal edges must contain src/dst columns; found {edges.columns}"
                )

        self._traversal_graph_cache[graph_name] = (vertices, edges)
        return vertices, edges

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _build_condition(predicate: dict):
        """Translate a DSL predicate dict into a PySpark Column expression."""
        col   = predicate.get("column")
        op    = predicate.get("operator", "=")
        value = predicate.get("value")

        if col is None:
            return None

        col_expr = F.col(col)
        op_map = {
            "=":    lambda c, v: c == v,
            "!=":   lambda c, v: c != v,
            ">":    lambda c, v: c > v,
            "<":    lambda c, v: c < v,
            ">=":   lambda c, v: c >= v,
            "<=":   lambda c, v: c <= v,
            "IN":   lambda c, v: c.isin(v if isinstance(v, list) else [v]),
            "LIKE": lambda c, v: c.like(str(v)),
            "NOT IN": lambda c, v: ~c.isin(v if isinstance(v, list) else [v]),
        }
        fn = op_map.get(op)
        if fn is None:
            logger.warning("SparkSQLGenerator: unknown operator '%s', skipping predicate", op)
            return None
        return fn(col_expr, value)

    @staticmethod
    def _estimate_count(df: DataFrame) -> int:
        """Estimate row count from Spark stats without a full scan when possible."""
        try:
            stats = df._jdf.queryExecution().optimizedPlan().stats()  # type: ignore
            row_count = stats.rowCount()
            # rowCount() returns a scala Option; check isDefined
            if row_count.isDefined():
                return int(str(row_count.get()))
        except Exception:
            pass
        # Fall back to an explicit count (triggers a Spark action)
        return df.count()
