"""HybridRouter: end-to-end query orchestrator.

Orchestrates: parse -> decompose -> extract features -> predict engine
-> execute via SQL/GRAPH generators -> compose final result.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

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
    """

    def __init__(
        self,
        parquet_dir: str = TPCH_PARQUET_DIR,
        graph_dir: str = GRAPHS_DIR,
        stats_dir: str = STATS_DIR,
        model_path: str = CLASSIFIER_PATH,
        force_engine: Optional[str] = None,
    ):
        """
        Args:
            parquet_dir: path to parquet tables (for SQL engine)
            graph_dir: path to graph data (for GRAPH engine)
            stats_dir: path to precomputed statistics
            model_path: path to trained classifier
            force_engine: if set ("SQL" or "GRAPH"), bypass ML and force all
                         sub-expressions to use this engine
        """
        self.parquet_dir = parquet_dir
        self.graph_dir = graph_dir
        self.stats_dir = stats_dir
        self.force_engine = force_engine

        # Core components
        self.parser = DSLParser()
        self.decomposer = QueryDecomposer()
        self.feature_extractor = FeatureExtractor(stats_dir=stats_dir)

        # Predictor is optional (may not have trained model yet)
        self._predictor = None
        self._model_path = model_path

        # Table loader cache
        self._table_cache: Dict[str, pd.DataFrame] = {}

    @property
    def predictor(self) -> ModelPredictor:
        """Lazy-load predictor to avoid failures when model doesn't exist."""
        if self._predictor is None:
            self._predictor = ModelPredictor(self._model_path)
        return self._predictor

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
                self._table_cache[source_name] = df
                return df

        # Handle graph-specific sources
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
        shared_cache: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Execute a sub-expression using the specified engine."""
        if engine == "GRAPH":
            generator = GraphGenerator(self.graph_dir, cache=shared_cache)
            return generator.generate(sub_expr)
        else:
            generator = SQLGenerator(self._load_table, cache=shared_cache)
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
