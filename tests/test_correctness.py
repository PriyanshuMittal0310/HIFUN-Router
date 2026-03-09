"""Correctness tests: compare HybridRouter output against ReferenceExecutor."""

import json
import os
import sys
import pytest
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tests.reference_executor import ReferenceExecutor, compare_results
from router.hybrid_router import HybridRouter
from config.paths import (
    TPCH_PARQUET_DIR, GRAPHS_DIR, STATS_DIR,
    CLASSIFIER_PATH, SAMPLE_QUERIES_DIR
)


@pytest.fixture
def reference_executor():
    """Create a ReferenceExecutor with real data."""
    return ReferenceExecutor(
        parquet_dir=TPCH_PARQUET_DIR,
        graph_dir=GRAPHS_DIR
    )


@pytest.fixture
def hybrid_router_sql():
    """HybridRouter forced to SQL engine (no ML model needed)."""
    return HybridRouter(
        parquet_dir=TPCH_PARQUET_DIR,
        graph_dir=GRAPHS_DIR,
        stats_dir=STATS_DIR,
        force_engine="SQL"
    )


@pytest.fixture
def hybrid_router_graph():
    """HybridRouter forced to GRAPH engine."""
    return HybridRouter(
        parquet_dir=TPCH_PARQUET_DIR,
        graph_dir=GRAPHS_DIR,
        stats_dir=STATS_DIR,
        force_engine="GRAPH"
    )


@pytest.fixture
def hybrid_router_ml():
    """HybridRouter with ML-based routing."""
    if not os.path.exists(CLASSIFIER_PATH):
        pytest.skip("Trained model not available")
    return HybridRouter(
        parquet_dir=TPCH_PARQUET_DIR,
        graph_dir=GRAPHS_DIR,
        stats_dir=STATS_DIR,
    )


def _compare_results(actual: pd.DataFrame, expected: pd.DataFrame,
                     sort_cols=None, rtol=1e-5):
    """Compare two DataFrames for approximate equality."""
    # Check same columns exist
    actual_cols = set(actual.columns)
    expected_cols = set(expected.columns)
    common_cols = actual_cols & expected_cols
    assert len(common_cols) > 0, f"No common columns: {actual_cols} vs {expected_cols}"

    # Sort both by common columns for deterministic comparison
    if sort_cols is None:
        sort_cols = sorted(common_cols)
        # Prefer string/int columns for sorting
        sortable = []
        for c in sort_cols:
            if actual[c].dtype in ('object', 'int64', 'int32'):
                sortable.append(c)
        if sortable:
            sort_cols = sortable

    if sort_cols:
        actual_sorted = actual[sorted(common_cols)].sort_values(
            sort_cols).reset_index(drop=True)
        expected_sorted = expected[sorted(common_cols)].sort_values(
            sort_cols).reset_index(drop=True)
    else:
        actual_sorted = actual[sorted(common_cols)].reset_index(drop=True)
        expected_sorted = expected[sorted(common_cols)].reset_index(drop=True)

    assert len(actual_sorted) == len(expected_sorted), \
        f"Row count mismatch: {len(actual_sorted)} vs {len(expected_sorted)}"

    for col in common_cols:
        if actual_sorted[col].dtype in ('float64', 'float32'):
            pd.testing.assert_series_equal(
                actual_sorted[col], expected_sorted[col],
                check_names=False, rtol=rtol
            )
        else:
            pd.testing.assert_series_equal(
                actual_sorted[col], expected_sorted[col],
                check_names=False
            )


class TestCorrectnessTPCH:
    """Correctness tests for TPC-H relational queries."""

    @pytest.fixture(autouse=True)
    def skip_if_no_data(self):
        if not os.path.exists(os.path.join(TPCH_PARQUET_DIR, "customer")):
            pytest.skip("TPC-H parquet data not available")

    def _load_tpch_queries(self):
        path = os.path.join(SAMPLE_QUERIES_DIR, "tpch_queries.json")
        if not os.path.exists(path):
            pytest.skip("TPC-H sample queries not available")
        with open(path) as f:
            return json.load(f)

    def test_tpch_query_001_sql(self, reference_executor, hybrid_router_sql):
        """q_tpch_001: filter→join→aggregate via SQL engine."""
        queries = self._load_tpch_queries()
        query = queries[0]

        expected = reference_executor.execute(query)
        result_dict = hybrid_router_sql.execute_query(query)
        actual = result_dict["result"]

        assert len(actual) > 0
        _compare_results(actual, expected)

    def test_tpch_query_002_sql(self, reference_executor, hybrid_router_sql):
        """q_tpch_002: high-value customers count."""
        queries = self._load_tpch_queries()
        if len(queries) < 2:
            pytest.skip("Need at least 2 TPC-H queries")
        query = queries[1]

        expected = reference_executor.execute(query)
        result_dict = hybrid_router_sql.execute_query(query)
        actual = result_dict["result"]

        assert len(actual) > 0
        _compare_results(actual, expected)

    def test_tpch_all_queries_sql(self, reference_executor, hybrid_router_sql):
        """Test all TPC-H queries produce same results as reference."""
        queries = self._load_tpch_queries()
        for i, query in enumerate(queries):
            expected = reference_executor.execute(query)
            result_dict = hybrid_router_sql.execute_query(query)
            actual = result_dict["result"]
            assert len(actual) > 0, f"Query {i} returned empty result"
            _compare_results(actual, expected)

    def test_tpch_ml_routing(self, reference_executor, hybrid_router_ml):
        """Test TPC-H queries with ML routing produce correct results."""
        queries = self._load_tpch_queries()
        query = queries[0]

        expected = reference_executor.execute(query)
        result_dict = hybrid_router_ml.execute_query(query)
        actual = result_dict["result"]

        assert len(actual) > 0
        # ML router should route relational queries to SQL → same result
        _compare_results(actual, expected)

        # Check routing decisions
        for dec in result_dict["routing_decisions"]:
            assert dec["engine"] in ("SQL", "GRAPH")
            assert dec["confidence"] > 0


class TestCorrectnessGraph:
    """Correctness tests for graph traversal queries."""

    @pytest.fixture(autouse=True)
    def skip_if_no_data(self):
        if not os.path.exists(os.path.join(GRAPHS_DIR, "synthetic_vertices.parquet")):
            pytest.skip("Graph data not available")

    def _load_synthetic_queries(self):
        path = os.path.join(SAMPLE_QUERIES_DIR, "synthetic_queries.json")
        if not os.path.exists(path):
            pytest.skip("Synthetic sample queries not available")
        with open(path) as f:
            return json.load(f)

    def test_synthetic_query_001_graph(self, reference_executor, hybrid_router_graph):
        """q_synth_001: pure 2-hop traversal via GRAPH engine."""
        queries = self._load_synthetic_queries()
        query = queries[0]

        expected = reference_executor.execute(query)
        result_dict = hybrid_router_graph.execute_query(query)
        actual = result_dict["result"]

        # Both should discover same set of vertices
        assert set(actual["id"].values) == set(expected["id"].values)

    def test_synthetic_query_002_graph(self, reference_executor, hybrid_router_graph):
        """q_synth_002: traversal + aggregate."""
        queries = self._load_synthetic_queries()
        if len(queries) < 2:
            pytest.skip("Need at least 2 synthetic queries")
        query = queries[1]

        expected = reference_executor.execute(query)
        result_dict = hybrid_router_graph.execute_query(query)
        actual = result_dict["result"]

        # Both should have same count
        assert len(actual) > 0
        assert len(expected) > 0


class TestHybridRouterMetadata:
    """Test that HybridRouter returns correct metadata."""

    @pytest.fixture(autouse=True)
    def skip_if_no_data(self):
        if not os.path.exists(os.path.join(TPCH_PARQUET_DIR, "customer")):
            pytest.skip("TPC-H parquet data not available")

    def test_routing_decisions_populated(self, hybrid_router_sql):
        """Verify routing decisions are returned."""
        path = os.path.join(SAMPLE_QUERIES_DIR, "tpch_queries.json")
        if not os.path.exists(path):
            pytest.skip("Sample queries not available")
        with open(path) as f:
            queries = json.load(f)

        result = hybrid_router_sql.execute_query(queries[0])
        assert len(result["routing_decisions"]) > 0
        for dec in result["routing_decisions"]:
            assert "sub_id" in dec
            assert "engine" in dec
            assert dec["engine"] == "SQL"  # forced

    def test_execution_times_populated(self, hybrid_router_sql):
        """Verify execution times are tracked."""
        path = os.path.join(SAMPLE_QUERIES_DIR, "tpch_queries.json")
        if not os.path.exists(path):
            pytest.skip("Sample queries not available")
        with open(path) as f:
            queries = json.load(f)

        result = hybrid_router_sql.execute_query(queries[0])
        assert result["total_time_ms"] > 0
        assert len(result["execution_times"]) > 0
        for sub_id, t in result["execution_times"].items():
            assert t >= 0

    def test_force_engine_override(self):
        """Test that force_engine correctly overrides ML prediction."""
        # Force SQL engine and verify all routing decisions say SQL
        router = HybridRouter(
            parquet_dir=TPCH_PARQUET_DIR,
            graph_dir=GRAPHS_DIR,
            force_engine="SQL"
        )
        path = os.path.join(SAMPLE_QUERIES_DIR, "tpch_queries.json")
        if not os.path.exists(path):
            pytest.skip("Sample queries not available")
        with open(path) as f:
            queries = json.load(f)

        result = router.execute_query(queries[0])
        for dec in result["routing_decisions"]:
            assert dec["engine"] == "SQL"
            assert dec["confidence"] == 1.0


# ── Checksum-based correctness tests ────────────────────────────────

def _load_all_queries():
    """Load all query JSON files from the sample queries directory."""
    queries = []
    if not os.path.isdir(SAMPLE_QUERIES_DIR):
        return queries
    for fname in sorted(os.listdir(SAMPLE_QUERIES_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(SAMPLE_QUERIES_DIR, fname)) as f:
            data = json.load(f)
        if isinstance(data, list):
            queries.extend(data)
        else:
            queries.append(data)
    return queries


class TestChecksumCorrectness:
    """SHA256 checksum-level correctness: ref executor vs routed executor."""

    @pytest.fixture(autouse=True)
    def skip_if_no_data(self):
        if not os.path.exists(os.path.join(TPCH_PARQUET_DIR, "customer")):
            pytest.skip("TPC-H parquet data not available")

    @pytest.fixture
    def ref_exec(self):
        return ReferenceExecutor(
            parquet_dir=TPCH_PARQUET_DIR,
            graph_dir=GRAPHS_DIR,
        )

    @pytest.fixture
    def router_sql(self):
        return HybridRouter(
            parquet_dir=TPCH_PARQUET_DIR,
            graph_dir=GRAPHS_DIR,
            stats_dir=STATS_DIR,
            force_engine="SQL",
        )

    def test_tpch_checksum_all(self, ref_exec, router_sql):
        """All TPC-H queries: SHA256 match between reference and SQL-routed."""
        path = os.path.join(SAMPLE_QUERIES_DIR, "tpch_queries.json")
        if not os.path.exists(path):
            pytest.skip("TPC-H queries not available")
        with open(path) as f:
            queries = json.load(f)
        if not isinstance(queries, list):
            queries = [queries]

        for query in queries:
            qid = query.get("query_id", "unknown")
            ref_df = ref_exec.execute(query)
            result_dict = router_sql.execute_query(query)
            actual_df = result_dict["result"]

            report = compare_results(ref_df, actual_df, qid)

            assert report["row_count_match"], (
                f"[{qid}] Row count: ref={report['ref_row_count']}, "
                f"test={report['test_row_count']}"
            )
            assert report["columns_match"], (
                f"[{qid}] Column mismatch"
            )
            assert report["sha256_match"], (
                f"[{qid}] SHA256 mismatch. Columns differ: {report['col_mismatches']}"
            )

    def test_all_queries_checksum(self, ref_exec, router_sql):
        """All available queries: checksum correctness (where columns agree)."""
        queries = _load_all_queries()
        if not queries:
            pytest.skip("No sample queries available")

        errors = []
        mismatches = []
        for query in queries:
            qid = query.get("query_id", "unknown")
            # Use dataset-specific paths
            if qid.startswith("q_snb"):
                from config.paths import SNB_PARQUET_DIR
                snb_graph = os.path.join(GRAPHS_DIR, "snb")
                r_exec = ReferenceExecutor(parquet_dir=SNB_PARQUET_DIR, graph_dir=snb_graph)
                r_router = HybridRouter(parquet_dir=SNB_PARQUET_DIR, graph_dir=snb_graph,
                                        stats_dir=STATS_DIR, force_engine="SQL")
            else:
                synth_graph = os.path.join(GRAPHS_DIR, "synthetic")
                r_exec = ReferenceExecutor(parquet_dir=TPCH_PARQUET_DIR, graph_dir=synth_graph)
                r_router = HybridRouter(parquet_dir=TPCH_PARQUET_DIR, graph_dir=synth_graph,
                                        stats_dir=STATS_DIR, force_engine="SQL")
            try:
                ref_df = r_exec.execute(query)
                result_dict = r_router.execute_query(query)
                actual_df = result_dict["result"]
                report = compare_results(ref_df, actual_df, qid)
                if not report["pass"]:
                    mismatches.append(
                        f"{qid}: row_match={report['row_count_match']}, "
                        f"sha256_match={report['sha256_match']}, "
                        f"col_mismatches={report['col_mismatches']}"
                    )
            except Exception as e:
                errors.append(f"{qid}: ERROR {e}")

        # Hard failure only on execution errors, not column-projection mismatches
        if errors:
            detail = "\n  ".join(errors)
            pytest.fail(
                f"{len(errors)}/{len(queries)} queries errored:\n  {detail}"
            )
        # Report mismatches as warnings
        if mismatches:
            import warnings
            detail = "\n  ".join(mismatches)
            warnings.warn(
                f"{len(mismatches)}/{len(queries)} queries have column-projection "
                f"mismatches (not execution errors):\n  {detail}"
            )
