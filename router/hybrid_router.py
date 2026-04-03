"""HybridRouter: end-to-end query orchestrator.

Orchestrates: parse -> decompose -> extract features -> predict engine
-> execute via SQL/GRAPH generators -> compose final result.

Engine modes
────────────
  use_real_engines=False (default)
      Uses pandas-based SQLGenerator + GraphGenerator.
      No SparkSession required.  Suitable for unit tests and CI.

  use_real_engines=True
      Uses SparkSQLGenerator (PySpark + Catalyst) and
      GraphFramesGenerator (GraphFrames BFS).
      Requires a SparkSession and the GraphFrames JAR.
      Results are Spark DataFrames until the final .toPandas() call.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from parser.dsl_parser import DSLParser
from parser.ast_nodes import QueryNode, SubExpression
from decomposer.query_decomposer import QueryDecomposer
from features.feature_extractor import FeatureExtractor
from model.predictor import ModelPredictor
from execution.sql_generator import SQLGenerator
from execution.graph_generator import GraphGenerator
from execution.result_composer import ResultComposer
from config.paths import (
    TPCH_PARQUET_DIR, GRAPHS_DIR, CLASSIFIER_PATH,
    STATS_DIR, SAMPLE_QUERIES_DIR,
)

logger = logging.getLogger(__name__)


class HybridRouter:
    """Orchestrates full HIFUN query execution with ML-guided engine routing.

    Pipeline:
        1. Parse DSL JSON → QueryNodes
        2. Decompose → SubExpressions
        3. For each SubExpression: extract features → predict engine
        4. Execute each SubExpression via assigned engine
        5. Compose partial results into final DataFrame

    When ``use_real_engines=True`` (Phase 2 mode):
        - SQL path  → PySpark DataFrame API  (SparkSQLGenerator)
        - GRAPH path → GraphFrames BFS       (GraphFramesGenerator)
        Final result is converted to pandas via .toPandas().

    When ``use_real_engines=False`` (default / test mode):
        - SQL path  → pandas  (SQLGenerator)
        - GRAPH path → pandas BFS (GraphGenerator)
    """

    def __init__(
        self,
        parquet_dir: str = TPCH_PARQUET_DIR,
        graph_dir: str = GRAPHS_DIR,
        stats_dir: str = STATS_DIR,
        model_path: str = CLASSIFIER_PATH,
        force_engine: Optional[str] = None,
        custom_router: Optional[callable] = None,
        use_real_engines: bool = False,
        strict_schema: bool = False,
        spark=None,
        hdfs_root: Optional[str] = None,
    ):
        """
        Args:
            parquet_dir:      path to Parquet tables (SQL engine)
            graph_dir:        path to graph Parquet data (GRAPH engine)
            stats_dir:        path to precomputed statistics
            model_path:       path to trained XGBoost classifier
            force_engine:     if set ("SQL" or "GRAPH"), bypass ML routing
            custom_router:    optional callable(sub_expr, fv, feature_names) → str
            use_real_engines: True  → PySpark + GraphFrames (Phase 2)
                              False → pandas (unit tests / Phase 1)
            strict_schema:    when True, missing predicate columns in pandas SQL
                              execution raise errors instead of being ignored.
            spark:            existing SparkSession (created automatically when
                              use_real_engines=True and spark is None)
            hdfs_root:        HDFS base URI, e.g. 'hdfs://namenode:9000/data'.
                              Passed to Spark generators for HDFS-backed data.
        """
        self.parquet_dir      = parquet_dir
        self.graph_dir        = graph_dir
        self.stats_dir        = stats_dir
        self.force_engine     = force_engine
        self.custom_router    = custom_router
        self.use_real_engines = use_real_engines
        self.strict_schema    = strict_schema
        self.hdfs_root        = hdfs_root

        # Core components (always pandas-based for parsing / feature extraction)
        self.parser    = DSLParser()
        self.decomposer = QueryDecomposer()
        self.feature_extractor = FeatureExtractor(stats_dir=stats_dir)

        # Predictor (lazy-loaded)
        self._predictor  = None
        self._model_path = model_path

        # Pandas table loader cache (used in non-Spark mode)
        self._table_cache: Dict[str, pd.DataFrame] = {}

        # Spark / real-engine initialisation
        self._spark = spark
        if use_real_engines and spark is None:
            self._spark = self._init_spark()

        # Pre-build Spark generators when using real engines
        if use_real_engines:
            from execution.spark_sql_generator   import SparkSQLGenerator
            from execution.graphframes_generator  import GraphFramesGenerator
            self._spark_sql_gen_cls   = SparkSQLGenerator
            self._spark_graph_gen_cls = GraphFramesGenerator
        else:
            self._spark_sql_gen_cls   = None
            self._spark_graph_gen_cls = None

    @property
    def predictor(self) -> ModelPredictor:
        """Lazy-load predictor to avoid failures when model doesn't exist."""
        if self._predictor is None:
            self._predictor = ModelPredictor(self._model_path)
        return self._predictor

    @property
    def spark(self):
        """Return the active SparkSession (only valid when use_real_engines=True)."""
        return self._spark

    def _init_spark(self):
        """Create a SparkSession using the project spark_config."""
        from config.spark_config import get_spark_session, get_hdfs_root
        extra = {}
        if self.hdfs_root:
            namenode = self.hdfs_root.split("/")[2] if "://" in self.hdfs_root else ""
            if namenode:
                extra["spark.hadoop.fs.defaultFS"] = f"hdfs://{namenode}"
        return get_spark_session(
            app_name="HIFUN_Router_RealEngines",
            extra_configs=extra if extra else None,
        )

    def _normalize_loaded_table(self, source_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Apply small schema compatibility fixes for known datasets."""
        if source_name == "works_at" and "company_name" not in df.columns and "organisation_id" in df.columns:
            fixed = df.copy()
            # Backward-compatible alias used by existing SNB sample queries.
            fixed["company_name"] = fixed["organisation_id"].astype(str)
            return fixed
        if source_name == "person" and "country" not in df.columns:
            vertices_path = os.path.join(self.graph_dir, "vertices.parquet")
            if os.path.exists(vertices_path):
                vdf = pd.read_parquet(vertices_path)
                if "country" in vdf.columns:
                    key_col = "person_id" if "person_id" in vdf.columns else "id"
                    if key_col in vdf.columns:
                        fixed = df.copy()
                        person_key = "id" if "id" in fixed.columns else "person_id"
                        if person_key in fixed.columns:
                            country_map = (
                                vdf[[key_col, "country"]]
                                .dropna(subset=[key_col])
                                .drop_duplicates(subset=[key_col])
                                .set_index(key_col)["country"]
                            )
                            fixed["country"] = fixed[person_key].map(country_map)
                            return fixed
        return df

    def _build_tpch_compat_table(self, source_name: str) -> Optional[pd.DataFrame]:
        """Build minimal TPCH-like tables from TPC-DS parquet when TPCH is absent."""
        if source_name not in {"customer", "orders"}:
            return None

        tpcds_dir = os.path.join(os.path.dirname(self.parquet_dir), "tpcds")
        if not os.path.exists(tpcds_dir):
            return None

        customer_path = os.path.join(tpcds_dir, "customer")
        if not os.path.exists(customer_path):
            return None

        cust_raw = pd.read_parquet(customer_path)
        if "c0" not in cust_raw.columns:
            return None

        c_custkey = pd.to_numeric(cust_raw["c0"], errors="coerce").fillna(0).astype(int)
        customer = pd.DataFrame({
            "c_custkey": c_custkey,
            "c_name": "CUST_" + c_custkey.astype(str),
            "c_nationkey": (c_custkey % 25).astype(int),
            "c_acctbal": (c_custkey % 10000).astype(float),
            "c_mktsegment": ["AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"][0],
        })
        customer["c_mktsegment"] = c_custkey.map(
            lambda x: ["AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"][x % 5]
        )

        if source_name == "customer":
            return customer

        store_sales_path = os.path.join(tpcds_dir, "store_sales")
        if not os.path.exists(store_sales_path):
            return None
        sales_raw = pd.read_parquet(store_sales_path)
        if sales_raw.empty:
            return None

        sales_raw = sales_raw.iloc[:200000].copy()
        if "c3" in sales_raw.columns:
            o_custkey = pd.to_numeric(sales_raw["c3"], errors="coerce")
        else:
            o_custkey = pd.Series(range(1, len(sales_raw) + 1), index=sales_raw.index)

        max_key = int(customer["c_custkey"].max()) if not customer.empty else 1
        o_custkey = o_custkey.fillna(0).astype(int)
        o_custkey = (o_custkey % max_key) + 1

        if "c20" in sales_raw.columns:
            o_totalprice = pd.to_numeric(sales_raw["c20"], errors="coerce")
        else:
            o_totalprice = pd.Series(range(1, len(sales_raw) + 1), index=sales_raw.index, dtype=float)
        o_totalprice = o_totalprice.fillna(0.0).abs() + 1.0

        order_idx = pd.Series(range(1, len(sales_raw) + 1), index=sales_raw.index)
        priorities = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
        statuses = ["F", "O", "P"]

        orders = pd.DataFrame({
            "o_orderkey": order_idx.astype(int),
            "o_custkey": o_custkey.astype(int),
            "o_totalprice": o_totalprice.astype(float),
            "o_orderpriority": order_idx.map(lambda x: priorities[x % len(priorities)]),
            "o_orderstatus": order_idx.map(lambda x: statuses[x % len(statuses)]),
        })
        return orders

    def _load_table(self, source_name: str) -> pd.DataFrame:
        """Load a table by name from parquet directory or graph directory."""
        if source_name in self._table_cache:
            return self._table_cache[source_name]

        # Try parquet directory first
        candidates = [
            os.path.join(self.parquet_dir, source_name),
            os.path.join(self.parquet_dir, f"{source_name}.parquet"),
            os.path.join(self.graph_dir, f"{source_name}.parquet"),
            os.path.join(self.graph_dir, source_name),
        ]
        for path in candidates:
            if os.path.exists(path):
                if os.path.isdir(path):
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_parquet(path)
                df = self._normalize_loaded_table(source_name, df)
                self._table_cache[source_name] = df
                return df

        # Handle graph-specific sources (vertices/edges)
        if "vertices" in source_name or "vertex" in source_name:
            for name in ["vertices.parquet", "snb_vertices.parquet",
                         "synthetic_vertices.parquet"]:
                path = os.path.join(self.graph_dir, name)
                if os.path.exists(path):
                    df = pd.read_parquet(path)
                    df = self._normalize_loaded_table(source_name, df)
                    self._table_cache[source_name] = df
                    return df

        if "edges" in source_name or "edge" in source_name:
            for name in ["edges.parquet", "snb_edges.parquet",
                         "synthetic_edges.parquet"]:
                path = os.path.join(self.graph_dir, name)
                if os.path.exists(path):
                    df = pd.read_parquet(path)
                    df = self._normalize_loaded_table(source_name, df)
                    self._table_cache[source_name] = df
                    return df

        compat_df = self._build_tpch_compat_table(source_name)
        if compat_df is not None:
            compat_df = self._normalize_loaded_table(source_name, compat_df)
            self._table_cache[source_name] = compat_df
            return compat_df

        raise FileNotFoundError(
            f"Could not find table '{source_name}' in {self.parquet_dir} or {self.graph_dir}"
        )

    def execute_query(self, query_json: dict) -> Dict:
        """Execute a full HIFUN query end-to-end.

        Args:
            query_json: parsed JSON dict of the DSL query

        Returns:
            dict with keys:
                result: final pd.DataFrame
                routing_decisions: list of {sub_id, engine, confidence, inference_ms}
                execution_times: dict of {sub_id: elapsed_ms}
                total_time_ms: float
                sub_expressions: list of SubExpression objects
        """
        t_start = time.perf_counter()

        # Step 1: Parse
        nodes = self.parser.parse(query_json)
        t_parse = time.perf_counter()

        # Step 2: Decompose
        sub_expressions = self.decomposer.decompose(nodes)
        t_decompose = time.perf_counter()

        # Step 3 & 4: Route + Execute
        shared_cache: Dict[str, pd.DataFrame] = {}
        routing_decisions = []
        execution_times = {}
        composer = ResultComposer()

        # Build dependency levels for topological execution
        levels = self._build_execution_levels(sub_expressions)

        for level in levels:
            for sub_expr in level:
                # Extract features & predict engine
                engine, confidence, inference_ms = self._route_subexpression(sub_expr)
                routing_decisions.append({
                    "sub_id": sub_expr.sub_id,
                    "engine": engine,
                    "primary_op_type": sub_expr.primary_op_type,
                    "n_nodes": len(sub_expr.nodes),
                    "confidence": confidence,
                    "inference_ms": inference_ms,
                })

                # Execute
                t_exec_start = time.perf_counter()
                result_df = self._execute_subexpression(sub_expr, engine, shared_cache)
                t_exec_end = time.perf_counter()
                execution_times[sub_expr.sub_id] = (t_exec_end - t_exec_start) * 1000

                # Store in caches
                shared_cache[sub_expr.sub_id] = result_df
                composer.register_result(sub_expr.sub_id, result_df)

        # Step 5: Compose
        final_result = composer.compose(sub_expressions)
        # If Spark mode returned a Spark DataFrame, materialise it now
        if hasattr(final_result, "toPandas"):
            final_result = final_result.toPandas()
        t_end = time.perf_counter()

        return {
            "result": final_result,
            "routing_decisions": routing_decisions,
            "execution_times": execution_times,
            "total_time_ms": (t_end - t_start) * 1000,
            "parse_time_ms": (t_parse - t_start) * 1000,
            "decompose_time_ms": (t_decompose - t_parse) * 1000,
            "sub_expressions": sub_expressions,
        }

    def execute_file(self, query_file: str) -> Dict:
        """Execute a query from a JSON file."""
        with open(query_file, "r") as f:
            query_json = json.load(f)
        return self.execute_query(query_json)

    def execute_batch(self, queries: List[dict]) -> List[Dict]:
        """Execute multiple queries and return all results."""
        results = []
        for q in queries:
            try:
                results.append(self.execute_query(q))
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                results.append({"error": str(e)})
        return results

    def _route_subexpression(self, sub_expr: SubExpression) -> Tuple[str, float, float]:
        """Predict engine for a SubExpression.

        Returns: (engine_label, confidence, inference_ms)
        """
        if self.force_engine:
            return self.force_engine, 1.0, 0.0

        # Custom router callback (used by baselines)
        if self.custom_router is not None:
            try:
                from features.feature_extractor import FEATURE_NAMES
                fv = self.feature_extractor.extract(sub_expr)
                t0 = time.perf_counter()
                engine = self.custom_router(sub_expr, fv, FEATURE_NAMES)
                inference_ms = (time.perf_counter() - t0) * 1000
                return engine, 0.8, inference_ms
            except Exception as e:
                logger.warning(f"Custom router failed, using heuristic: {e}")
                return self._heuristic_route(sub_expr), 0.5, 0.0

        try:
            fv = self.feature_extractor.extract(sub_expr)
            result = self.predictor.predict_proba(fv)
            engine = result["label"]
            confidence = max(result["sql_prob"], result["graph_prob"])
            return engine, confidence, result["inference_ms"]
        except Exception as e:
            # Fallback: heuristic routing
            logger.warning(f"ML prediction failed, using heuristic: {e}")
            engine = self._heuristic_route(sub_expr)
            return engine, 0.5, 0.0

    def _heuristic_route(self, sub_expr: SubExpression) -> str:
        """Fallback heuristic: TRAVERSAL → GRAPH, else → SQL."""
        if sub_expr.primary_op_type == "TRAVERSAL":
            return "GRAPH"
        return "SQL"

    def _execute_subexpression(
        self, sub_expr: SubExpression, engine: str,
        shared_cache: Dict[str, Any]
    ) -> Any:
        """Execute a sub-expression using the specified engine.

        Returns a pandas DataFrame in non-Spark mode, or a Spark DataFrame
        (lazy) in real-engine mode.  The caller is responsible for calling
        .toPandas() when a pandas result is ultimately required.
        """
        if self.use_real_engines and self._spark is not None:
            if engine == "GRAPH":
                generator = self._spark_graph_gen_cls(
                    self._spark, self.graph_dir,
                    cache=shared_cache,
                    hdfs_root=self.hdfs_root,
                )
            else:
                generator = self._spark_sql_gen_cls(
                    self._spark, self.parquet_dir,
                    cache=shared_cache,
                    hdfs_root=self.hdfs_root,
                )
            return generator.generate(sub_expr)

        # Pandas (test / Phase-1) mode
        if engine == "GRAPH":
            generator = GraphGenerator(self.graph_dir, cache=shared_cache)
            return generator.generate(sub_expr)
        else:
            generator = SQLGenerator(
                self._load_table,
                cache=shared_cache,
                strict_schema=self.strict_schema,
            )
            return generator.generate(sub_expr)

    @staticmethod
    def _build_execution_levels(
        sub_expressions: List[SubExpression]
    ) -> List[List[SubExpression]]:
        """Group sub-expressions into dependency levels for execution ordering.

        Level 0 = no dependencies, Level 1 = depends only on level-0, etc.
        """
        if not sub_expressions:
            return []

        sub_map = {se.sub_id: se for se in sub_expressions}
        levels: List[List[SubExpression]] = []
        assigned: Dict[str, int] = {}

        # Kahn's-like level assignment
        remaining = list(sub_expressions)
        while remaining:
            current_level = []
            still_remaining = []
            for se in remaining:
                deps_satisfied = all(
                    dep_id in assigned for dep_id in se.depends_on_subs
                )
                if deps_satisfied:
                    current_level.append(se)
                else:
                    still_remaining.append(se)

            if not current_level:
                # Circular dependency or unresolvable — force assign remaining
                current_level = still_remaining
                still_remaining = []

            for se in current_level:
                assigned[se.sub_id] = len(levels)

            levels.append(current_level)
            remaining = still_remaining

        return levels
