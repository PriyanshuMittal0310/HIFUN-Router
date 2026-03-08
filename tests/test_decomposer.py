"""Tests for the query decomposer module."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.ast_nodes import QueryNode, SubExpression
from decomposer.query_decomposer import QueryDecomposer


@pytest.fixture
def decomposer():
    return QueryDecomposer()


def _make_nodes(ops):
    """Helper to create QueryNode list from simplified dicts."""
    return [
        QueryNode(
            op_id=op["id"],
            op_type=op["type"],
            source=op.get("source", "table"),
            fields=op.get("fields", ["col1"]),
            predicate=op.get("predicate"),
            join=op.get("join"),
            traversal=op.get("traversal"),
            aggregate=op.get("aggregate"),
            depends_on=op.get("deps", []),
        )
        for op in ops
    ]


class TestQueryDecomposer:
    def test_pure_relational_single_group(self, decomposer):
        """FILTER -> JOIN -> AGGREGATE should form one RELATIONAL group."""
        nodes = _make_nodes([
            {"id": "s1", "type": "FILTER", "deps": []},
            {"id": "s2", "type": "JOIN", "deps": ["s1"],
             "join": {"right_source": "s1", "left_key": "k", "right_key": "k", "join_type": "INNER"}},
            {"id": "s3", "type": "AGGREGATE", "deps": ["s2"],
             "aggregate": {"group_by": ["col1"], "functions": [{"func": "SUM", "column": "col1"}]}},
        ])
        subs = decomposer.decompose(nodes)
        assert len(subs) == 1
        assert subs[0].primary_op_type == "RELATIONAL"
        assert len(subs[0].nodes) == 3

    def test_traversal_isolated(self, decomposer):
        """A TRAVERSAL node should be its own SubExpression."""
        nodes = _make_nodes([
            {"id": "s1", "type": "TRAVERSAL", "source": "graph", "deps": [],
             "traversal": {"start_vertex_filter": {"column": "id", "value": 1},
                           "edge_label": "E", "direction": "OUT", "max_hops": 2}},
        ])
        subs = decomposer.decompose(nodes)
        assert len(subs) == 1
        assert subs[0].primary_op_type == "TRAVERSAL"

    def test_mixed_query_decomposition(self, decomposer):
        """TRAVERSAL (s1) + AGGREGATE (s2) + JOIN (s3 depends on both)."""
        nodes = _make_nodes([
            {"id": "s1", "type": "TRAVERSAL", "source": "graph", "deps": [],
             "traversal": {"start_vertex_filter": {"column": "id", "value": 1},
                           "edge_label": "E", "direction": "BOTH", "max_hops": 2}},
            {"id": "s2", "type": "AGGREGATE", "source": "posts", "deps": [],
             "aggregate": {"group_by": ["c"], "functions": [{"func": "COUNT", "column": "x"}]}},
            {"id": "s3", "type": "JOIN", "source": "s1", "deps": ["s1", "s2"],
             "join": {"right_source": "s2", "left_key": "id", "right_key": "c", "join_type": "LEFT"}},
        ])
        subs = decomposer.decompose(nodes)
        # Should produce at least 2 groups: TRAVERSAL + RELATIONAL parts
        types = {s.primary_op_type for s in subs}
        assert "TRAVERSAL" in types
        assert "RELATIONAL" in types

    def test_parallel_independence(self, decomposer):
        """Two independent root nodes should both be marked parallelizable."""
        nodes = _make_nodes([
            {"id": "s1", "type": "TRAVERSAL", "source": "graph", "deps": [],
             "traversal": {"start_vertex_filter": {"column": "id", "value": 1},
                           "edge_label": "E", "direction": "OUT", "max_hops": 1}},
            {"id": "s2", "type": "AGGREGATE", "source": "table", "deps": [],
             "aggregate": {"group_by": ["c"], "functions": [{"func": "COUNT", "column": "x"}]}},
        ])
        subs = decomposer.decompose(nodes)
        parallel_subs = [s for s in subs if s.parallelizable]
        assert len(parallel_subs) >= 2

    def test_sub_expression_dependencies(self, decomposer):
        """JOIN that depends on both TRAVERSAL and AGGREGATE should have dep links."""
        nodes = _make_nodes([
            {"id": "s1", "type": "TRAVERSAL", "source": "graph", "deps": [],
             "traversal": {"start_vertex_filter": {"column": "id", "value": 1},
                           "edge_label": "E", "direction": "OUT", "max_hops": 1}},
            {"id": "s2", "type": "AGGREGATE", "source": "table", "deps": [],
             "aggregate": {"group_by": ["c"], "functions": [{"func": "COUNT", "column": "x"}]}},
            {"id": "s3", "type": "JOIN", "source": "s1", "deps": ["s1", "s2"],
             "join": {"right_source": "s2", "left_key": "id", "right_key": "c", "join_type": "INNER"}},
        ])
        subs = decomposer.decompose(nodes)
        # Find the sub containing s3 (JOIN)
        join_sub = [s for s in subs if any(n.op_id == "s3" for n in s.nodes)][0]
        assert len(join_sub.depends_on_subs) >= 1
        assert not join_sub.parallelizable

    def test_empty_input(self, decomposer):
        subs = decomposer.decompose([])
        assert subs == []

    def test_single_node(self, decomposer):
        nodes = _make_nodes([
            {"id": "s1", "type": "FILTER", "deps": []}
        ])
        subs = decomposer.decompose(nodes)
        assert len(subs) == 1
        assert subs[0].primary_op_type == "RELATIONAL"

    def test_map_merged_with_parent(self, decomposer):
        """MAP node should be merged with parent relational chain."""
        nodes = _make_nodes([
            {"id": "s1", "type": "MAP", "deps": []},
            {"id": "s2", "type": "FILTER", "deps": ["s1"]},
        ])
        subs = decomposer.decompose(nodes)
        assert len(subs) == 1
        assert len(subs[0].nodes) == 2
