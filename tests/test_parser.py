"""Tests for the DSL parser module."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.dsl_parser import DSLParser
from parser.ast_nodes import QueryNode


@pytest.fixture
def parser():
    return DSLParser()


@pytest.fixture
def tpch_query():
    return {
        "query_id": "test_tpch",
        "operations": [
            {
                "op_id": "s1", "type": "FILTER", "source": "customer",
                "fields": ["c_custkey", "c_name"],
                "predicate": {"column": "c_nationkey", "operator": "=", "value": 7},
                "depends_on": []
            },
            {
                "op_id": "s2", "type": "JOIN", "source": "orders",
                "fields": ["o_orderkey", "o_custkey", "o_totalprice"],
                "join": {"right_source": "s1", "left_key": "o_custkey",
                         "right_key": "c_custkey", "join_type": "INNER"},
                "depends_on": ["s1"]
            },
            {
                "op_id": "s3", "type": "AGGREGATE", "source": "s2",
                "fields": ["c_name", "total"],
                "aggregate": {"group_by": ["c_name"],
                              "functions": [{"func": "SUM", "column": "o_totalprice"}]},
                "depends_on": ["s2"]
            }
        ]
    }


@pytest.fixture
def mixed_query():
    return {
        "query_id": "test_mixed",
        "operations": [
            {
                "op_id": "s1", "type": "TRAVERSAL", "source": "social_graph",
                "fields": ["person_id"],
                "traversal": {"start_vertex_filter": {"column": "person_id", "value": 1},
                              "edge_label": "KNOWS", "direction": "BOTH", "max_hops": 2,
                              "return_fields": ["person_id"]},
                "depends_on": []
            },
            {
                "op_id": "s2", "type": "AGGREGATE", "source": "posts",
                "fields": ["creator_id", "cnt"],
                "aggregate": {"group_by": ["creator_id"],
                              "functions": [{"func": "COUNT", "column": "post_id"}]},
                "depends_on": []
            },
            {
                "op_id": "s3", "type": "JOIN", "source": "s1",
                "fields": ["person_id", "cnt"],
                "join": {"right_source": "s2", "left_key": "person_id",
                         "right_key": "creator_id", "join_type": "LEFT"},
                "depends_on": ["s1", "s2"]
            }
        ]
    }


class TestDSLParser:
    def test_parse_returns_query_nodes(self, parser, tpch_query):
        nodes = parser.parse(tpch_query)
        assert len(nodes) == 3
        assert all(isinstance(n, QueryNode) for n in nodes)

    def test_topological_order(self, parser, tpch_query):
        nodes = parser.parse(tpch_query)
        ids = [n.op_id for n in nodes]
        # s1 must come before s2, s2 before s3
        assert ids.index("s1") < ids.index("s2")
        assert ids.index("s2") < ids.index("s3")

    def test_node_attributes(self, parser, tpch_query):
        nodes = parser.parse(tpch_query)
        s1 = nodes[0]
        assert s1.op_type == "FILTER"
        assert s1.source == "customer"
        assert s1.predicate is not None
        assert s1.predicate["column"] == "c_nationkey"

    def test_mixed_query_parse(self, parser, mixed_query):
        nodes = parser.parse(mixed_query)
        assert len(nodes) == 3
        types = [n.op_type for n in nodes]
        assert "TRAVERSAL" in types
        assert "JOIN" in types

    def test_mixed_query_dependency_order(self, parser, mixed_query):
        nodes = parser.parse(mixed_query)
        ids = [n.op_id for n in nodes]
        # s3 depends on both s1 and s2 — must come last
        assert ids.index("s3") > ids.index("s1")
        assert ids.index("s3") > ids.index("s2")

    def test_invalid_query_raises(self, parser):
        bad = {"query_id": "bad", "operations": [{"op_id": "x", "source": "foo"}]}
        with pytest.raises(ValueError, match="Invalid DSL query"):
            parser.parse(bad)

    def test_circular_dependency_raises(self, parser):
        circular = {
            "query_id": "circ",
            "operations": [
                {"op_id": "a", "type": "FILTER", "source": "t",
                 "depends_on": ["b"]},
                {"op_id": "b", "type": "FILTER", "source": "t",
                 "depends_on": ["a"]},
            ]
        }
        with pytest.raises(ValueError):
            parser.parse(circular)

    def test_parse_file(self, parser, tmp_path):
        query = {
            "query_id": "file_test",
            "operations": [
                {"op_id": "s1", "type": "FILTER", "source": "t",
                 "fields": ["a"], "depends_on": []}
            ]
        }
        fpath = tmp_path / "test.json"
        fpath.write_text(json.dumps(query))
        result = parser.parse_file(str(fpath))
        assert "file_test" in result
        assert len(result["file_test"]) == 1

    def test_parse_file_list(self, parser, tmp_path):
        queries = [
            {"query_id": "q1", "operations": [
                {"op_id": "s1", "type": "FILTER", "source": "t", "depends_on": []}
            ]},
            {"query_id": "q2", "operations": [
                {"op_id": "s1", "type": "FILTER", "source": "t", "depends_on": []}
            ]},
        ]
        fpath = tmp_path / "multi.json"
        fpath.write_text(json.dumps(queries))
        result = parser.parse_file(str(fpath))
        assert "q1" in result
        assert "q2" in result
