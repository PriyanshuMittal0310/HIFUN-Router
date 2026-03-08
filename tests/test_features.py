"""Tests for the feature extraction module."""

import os
import sys
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.ast_nodes import QueryNode, SubExpression
from features.feature_extractor import FeatureExtractor, FEATURE_NAMES, NUM_FEATURES
from features.stats_collector import StatsCollector
from features.historical_store import HistoricalStore


@pytest.fixture
def extractor():
    ex = FeatureExtractor(stats_dir='data/stats', history_db_path=':memory:')
    yield ex
    ex.close()


@pytest.fixture
def stats_collector():
    return StatsCollector('data/stats')


def _make_sub(nodes, sub_id="sub_0", primary_type="RELATIONAL"):
    return SubExpression(
        sub_id=sub_id,
        nodes=nodes,
        primary_op_type=primary_type,
        depends_on_subs=[],
        parallelizable=True,
    )


class TestFeatureNames:
    def test_feature_count(self):
        assert NUM_FEATURES == 22

    def test_feature_names_loaded(self):
        assert len(FEATURE_NAMES) == 22
        assert "op_count_filter" in FEATURE_NAMES
        assert "has_traversal" in FEATURE_NAMES
        assert "num_tables_joined" in FEATURE_NAMES


class TestStatsCollector:
    def test_load_customer_stats(self, stats_collector):
        stats = stats_collector.get_table_stats("customer")
        assert stats is not None
        assert stats["row_count"] == 150000

    def test_graph_stats(self, stats_collector):
        stats = stats_collector.get_graph_stats("synthetic_graph")
        assert stats is not None
        assert stats["avg_degree"] > 0

    def test_is_graph_source(self, stats_collector):
        assert stats_collector.is_graph_source("social_graph")
        assert stats_collector.is_graph_source("synthetic_graph")
        assert not stats_collector.is_graph_source("customer")

    def test_get_row_count(self, stats_collector):
        assert stats_collector.get_row_count("customer") == 150000

    def test_get_column_stats(self, stats_collector):
        col = stats_collector.get_column_stats("customer", "c_custkey")
        assert col is not None
        assert col["distinct_count"] == 150000


class TestHistoricalStore:
    def test_empty_lookup(self):
        store = HistoricalStore(":memory:")
        avg, var = store.lookup("nonexistent")
        assert avg == -1.0
        assert var == -1.0
        store.close()

    def test_record_and_lookup(self):
        store = HistoricalStore(":memory:")
        fp = "test_fp_123"
        store.record(fp, "SQL", 100.0)
        store.record(fp, "SQL", 200.0)
        avg, var = store.lookup(fp)
        assert abs(avg - 150.0) < 0.1
        assert var > 0
        store.close()

    def test_compute_fingerprint(self):
        fp1 = HistoricalStore.compute_fingerprint(["FILTER", "JOIN"], "customer")
        fp2 = HistoricalStore.compute_fingerprint(["JOIN", "FILTER"], "customer")
        # Same ops sorted -> same fingerprint
        assert fp1 == fp2

        fp3 = HistoricalStore.compute_fingerprint(["FILTER"], "customer")
        assert fp1 != fp3


class TestFeatureExtractor:
    def test_vector_shape_and_dtype(self, extractor):
        nodes = [
            QueryNode("s1", "FILTER", "customer", ["c_custkey", "c_name"],
                      predicate={"column": "c_nationkey", "operator": "=", "value": 7})
        ]
        sub = _make_sub(nodes)
        vec = extractor.extract(sub)
        assert vec.shape == (22,)
        assert vec.dtype == np.float32

    def test_relational_features(self, extractor):
        nodes = [
            QueryNode("s1", "FILTER", "customer", ["c_custkey", "c_name"],
                      predicate={"column": "c_nationkey", "operator": "=", "value": 7}),
            QueryNode("s2", "JOIN", "orders", ["o_orderkey", "o_totalprice"],
                      join={"right_source": "s1", "left_key": "o_custkey",
                            "right_key": "c_custkey", "join_type": "INNER"},
                      depends_on=["s1"]),
        ]
        sub = _make_sub(nodes)
        feat = extractor.extract_dict(sub)

        assert feat["op_count_filter"] == 1
        assert feat["op_count_join"] == 1
        assert feat["has_traversal"] == 0
        assert feat["avg_degree"] == 0.0
        assert feat["num_tables_joined"] >= 2

    def test_traversal_features(self, extractor):
        nodes = [
            QueryNode("s1", "TRAVERSAL", "social_graph", ["person_id"],
                      traversal={"start_vertex_filter": {"column": "id", "value": 1},
                                 "edge_label": "KNOWS", "direction": "OUT",
                                 "max_hops": 2, "return_fields": ["person_id"]})
        ]
        sub = _make_sub(nodes, primary_type="TRAVERSAL")
        feat = extractor.extract_dict(sub)

        assert feat["has_traversal"] == 1
        assert feat["max_hops"] == 2
        assert feat["op_count_traversal"] == 1
        assert feat["avg_degree"] > 0
        assert feat["estimated_traversal_ops"] > 0

    def test_selectivity_equality(self, extractor):
        nodes = [
            QueryNode("s1", "FILTER", "customer", ["c_custkey"],
                      predicate={"column": "c_nationkey", "operator": "=", "value": 5})
        ]
        sub = _make_sub(nodes)
        feat = extractor.extract_dict(sub)
        # c_nationkey has 25 distinct values -> selectivity ≈ 1/25 = 0.04
        assert 0.03 < feat["selectivity"] < 0.05

    def test_selectivity_range(self, extractor):
        nodes = [
            QueryNode("s1", "FILTER", "customer", ["c_custkey"],
                      predicate={"column": "c_acctbal", "operator": ">", "value": 5000})
        ]
        sub = _make_sub(nodes)
        feat = extractor.extract_dict(sub)
        # c_acctbal ranges from -999.99 to 9999.99 → ~45% above 5000
        assert 0.3 < feat["selectivity"] < 0.6

    def test_extract_dict_keys(self, extractor):
        nodes = [QueryNode("s1", "FILTER", "customer", ["c_custkey"])]
        sub = _make_sub(nodes)
        feat = extractor.extract_dict(sub)
        assert set(feat.keys()) == set(FEATURE_NAMES)

    def test_estimated_output_rows_set(self, extractor):
        nodes = [QueryNode("s1", "FILTER", "customer", ["c_custkey"])]
        sub = _make_sub(nodes)
        extractor.extract(sub)
        assert sub.estimated_output_rows > 0

    def test_full_pipeline_all_queries(self, extractor):
        """Process all sample queries through the full pipeline."""
        from parser.dsl_parser import DSLParser
        from decomposer.query_decomposer import QueryDecomposer

        parser = DSLParser()
        decomposer = QueryDecomposer()

        total = 0
        for qfile in ['dsl/sample_queries/tpch_queries.json',
                       'dsl/sample_queries/snb_queries.json',
                       'dsl/sample_queries/synthetic_queries.json']:
            with open(qfile) as f:
                queries = json.load(f)
            for q in queries:
                nodes = parser.parse(q)
                subs = decomposer.decompose(nodes)
                for sub in subs:
                    vec = extractor.extract(sub)
                    assert vec.shape == (22,)
                    assert not np.any(np.isnan(vec))
                    total += 1

        assert total > 0
