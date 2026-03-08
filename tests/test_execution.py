"""Tests for Phase 5: Execution engines, result composition, and hybrid router."""

import json
import os
import sys
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from parser.ast_nodes import QueryNode, SubExpression
from parser.dsl_parser import DSLParser
from decomposer.query_decomposer import QueryDecomposer
from execution.sql_generator import SQLGenerator
from execution.graph_generator import GraphGenerator
from execution.result_composer import ResultComposer
from config.paths import TPCH_PARQUET_DIR, GRAPHS_DIR, SAMPLE_QUERIES_DIR


# --- Fixtures ---

@pytest.fixture
def sample_tables():
    """Create small in-memory tables for testing."""
    customers = pd.DataFrame({
        "c_custkey": [1, 2, 3, 4, 5],
        "c_name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "c_acctbal": [1000.0, 6000.0, 3000.0, 8000.0, 500.0],
        "c_nationkey": [7, 7, 3, 7, 5],
    })
    orders = pd.DataFrame({
        "o_orderkey": [101, 102, 103, 104, 105, 106],
        "o_custkey": [1, 2, 2, 3, 4, 4],
        "o_totalprice": [150.0, 200.0, 300.0, 50.0, 400.0, 100.0],
    })
    return {"customer": customers, "orders": orders}


@pytest.fixture
def table_loader(sample_tables):
    """Create a data loader function from sample tables."""
    def loader(name):
        if name in sample_tables:
            return sample_tables[name]
        raise FileNotFoundError(f"Table '{name}' not found in test data")
    return loader


@pytest.fixture
def sample_graph_data(tmp_path):
    """Create small graph data for testing."""
    vertices = pd.DataFrame({
        "id": [0, 1, 2, 3, 4],
        "attr1": [10, 20, 30, 40, 50],
        "label": ["A", "B", "C", "A", "B"],
    })
    edges = pd.DataFrame({
        "src": [0, 0, 1, 2, 3],
        "dst": [1, 2, 3, 4, 4],
        "relationship": ["KNOWS", "KNOWS", "KNOWS", "KNOWS", "KNOWS"],
    })

    vertices.to_parquet(os.path.join(tmp_path, "synthetic_vertices.parquet"))
    edges.to_parquet(os.path.join(tmp_path, "synthetic_edges.parquet"))
    return str(tmp_path), vertices, edges


# --- SQLGenerator Tests ---

class TestSQLGenerator:
    """Tests for SQLGenerator."""

    def test_filter_basic(self, table_loader):
        """Test basic filter operation."""
        node = QueryNode(
            op_id="s1", op_type="FILTER", source="customer",
            fields=["c_custkey", "c_name", "c_nationkey"],
            predicate={"column": "c_nationkey", "operator": "=", "value": 7}
        )
        sub = SubExpression(
            sub_id="sub_0", nodes=[node],
            primary_op_type="RELATIONAL", depends_on_subs=[],
            parallelizable=True
        )
        gen = SQLGenerator(table_loader)
        result = gen.generate(sub)
        assert len(result) == 3  # Alice, Bob, Diana have nationkey=7
        assert set(result.columns) == {"c_custkey", "c_name", "c_nationkey"}

    def test_filter_greater_than(self, table_loader):
        """Test filter with > operator."""
        node = QueryNode(
            op_id="s1", op_type="FILTER", source="customer",
            fields=["c_custkey", "c_name", "c_acctbal"],
            predicate={"column": "c_acctbal", "operator": ">", "value": 5000}
        )
        sub = SubExpression(
            sub_id="sub_0", nodes=[node],
            primary_op_type="RELATIONAL", depends_on_subs=[],
            parallelizable=True
        )
        gen = SQLGenerator(table_loader)
        result = gen.generate(sub)
        assert len(result) == 2  # Bob=6000, Diana=8000

    def test_filter_in_operator(self, table_loader):
        """Test IN operator."""
        node = QueryNode(
            op_id="s1", op_type="FILTER", source="customer",
            fields=["c_custkey", "c_name"],
            predicate={"column": "c_custkey", "operator": "IN", "value": [1, 3, 5]}
        )
        sub = SubExpression(
            sub_id="sub_0", nodes=[node],
            primary_op_type="RELATIONAL", depends_on_subs=[],
            parallelizable=True
        )
        gen = SQLGenerator(table_loader)
        result = gen.generate(sub)
        assert len(result) == 3

    def test_join_operation(self, table_loader):
        """Test JOIN between two tables."""
        filter_node = QueryNode(
            op_id="s1", op_type="FILTER", source="customer",
            fields=["c_custkey", "c_name", "c_nationkey"],
            predicate={"column": "c_nationkey", "operator": "=", "value": 7}
        )
        join_node = QueryNode(
            op_id="s2", op_type="JOIN", source="orders",
            fields=["o_orderkey", "o_custkey", "o_totalprice", "c_name"],
            join={"right_source": "s1", "left_key": "o_custkey",
                  "right_key": "c_custkey", "join_type": "INNER"},
            depends_on=["s1"]
        )

        # First execute filter and cache result
        gen = SQLGenerator(table_loader)
        filter_sub = SubExpression(
            sub_id="sub_0", nodes=[filter_node],
            primary_op_type="RELATIONAL", depends_on_subs=[],
            parallelizable=True
        )
        filter_result = gen.generate(filter_sub)
        gen.cache["s1"] = filter_result

        # Now execute join
        join_sub = SubExpression(
            sub_id="sub_1", nodes=[join_node],
            primary_op_type="RELATIONAL", depends_on_subs=["sub_0"],
            parallelizable=False
        )
        result = gen.generate(join_sub)
        # Only orders for nation=7 customers (Alice=1, Bob=2, Diana=4)
        assert len(result) > 0
        assert "c_name" in result.columns

    def test_aggregate_operation(self, table_loader):
        """Test AGGREGATE with group_by."""
        agg_node = QueryNode(
            op_id="s1", op_type="AGGREGATE", source="orders",
            fields=["o_custkey", "SUM_o_totalprice"],
            aggregate={
                "group_by": ["o_custkey"],
                "functions": [{"func": "SUM", "column": "o_totalprice"}]
            }
        )
        sub = SubExpression(
            sub_id="sub_0", nodes=[agg_node],
            primary_op_type="RELATIONAL", depends_on_subs=[],
            parallelizable=True
        )
        gen = SQLGenerator(table_loader)
        result = gen.generate(sub)
        assert "SUM_o_totalprice" in result.columns
        # customer 4 has orders 400+100=500
        cust4_total = result[result["o_custkey"] == 4]["SUM_o_totalprice"].values[0]
        assert cust4_total == 500.0

    def test_aggregate_no_group_by(self, table_loader):
        """Test AGGREGATE without group_by (whole-table aggregation)."""
        agg_node = QueryNode(
            op_id="s1", op_type="AGGREGATE", source="orders",
            fields=["COUNT_o_orderkey"],
            aggregate={
                "group_by": [],
                "functions": [{"func": "COUNT", "column": "o_orderkey"}]
            }
        )
        sub = SubExpression(
            sub_id="sub_0", nodes=[agg_node],
            primary_op_type="RELATIONAL", depends_on_subs=[],
            parallelizable=True
        )
        gen = SQLGenerator(table_loader)
        result = gen.generate(sub)
        assert result["COUNT_o_orderkey"].values[0] == 6

    def test_map_operation(self, table_loader):
        """Test MAP (project) operation."""
        node = QueryNode(
            op_id="s1", op_type="MAP", source="customer",
            fields=["c_custkey", "c_name"]
        )
        sub = SubExpression(
            sub_id="sub_0", nodes=[node],
            primary_op_type="RELATIONAL", depends_on_subs=[],
            parallelizable=True
        )
        gen = SQLGenerator(table_loader)
        result = gen.generate(sub)
        assert list(result.columns) == ["c_custkey", "c_name"]
        assert len(result) == 5

    def test_multi_node_pipeline(self, table_loader):
        """Test a pipeline of FILTER → JOIN → AGGREGATE in a single SubExpression."""
        filter_node = QueryNode(
            op_id="s1", op_type="FILTER", source="customer",
            fields=["c_custkey", "c_name", "c_nationkey"],
            predicate={"column": "c_nationkey", "operator": "=", "value": 7}
        )
        join_node = QueryNode(
            op_id="s2", op_type="JOIN", source="orders",
            fields=["o_orderkey", "o_custkey", "o_totalprice", "c_name"],
            join={"right_source": "s1", "left_key": "o_custkey",
                  "right_key": "c_custkey", "join_type": "INNER"},
            depends_on=["s1"]
        )
        agg_node = QueryNode(
            op_id="s3", op_type="AGGREGATE", source="s2",
            fields=["c_name", "SUM_o_totalprice"],
            aggregate={
                "group_by": ["c_name"],
                "functions": [{"func": "SUM", "column": "o_totalprice"}]
            },
            depends_on=["s2"]
        )

        gen = SQLGenerator(table_loader)
        # Execute filter first, cache it
        filter_sub = SubExpression(
            sub_id="sub_0", nodes=[filter_node],
            primary_op_type="RELATIONAL", depends_on_subs=[],
            parallelizable=True
        )
        filter_result = gen.generate(filter_sub)
        gen.cache["s1"] = filter_result

        # Execute join + aggregate pipeline
        pipeline_sub = SubExpression(
            sub_id="sub_1", nodes=[join_node, agg_node],
            primary_op_type="RELATIONAL", depends_on_subs=["sub_0"],
            parallelizable=False
        )
        result = gen.generate(pipeline_sub)
        assert "SUM_o_totalprice" in result.columns
        assert "c_name" in result.columns


# --- GraphGenerator Tests ---

class TestGraphGenerator:
    """Tests for GraphGenerator."""

    def test_basic_traversal(self, sample_graph_data):
        """Test basic graph traversal."""
        graph_dir, vertices, edges = sample_graph_data
        node = QueryNode(
            op_id="t1", op_type="TRAVERSAL", source="synthetic_graph",
            fields=["id", "attr1"],
            traversal={
                "start_vertex_filter": {"column": "id", "value": 0},
                "edge_label": "KNOWS", "direction": "OUT",
                "max_hops": 1, "return_fields": ["id", "attr1"]
            }
        )
        sub = SubExpression(
            sub_id="sub_0", nodes=[node],
            primary_op_type="TRAVERSAL", depends_on_subs=[],
            parallelizable=True
        )
        gen = GraphGenerator(graph_dir)
        result = gen.generate(sub)
        # From vertex 0 with 1-hop OUT: reaches 1 and 2
        assert 0 in result["id"].values  # start vertex included
        assert 1 in result["id"].values
        assert 2 in result["id"].values

    def test_multi_hop_traversal(self, sample_graph_data):
        """Test 2-hop traversal."""
        graph_dir, vertices, edges = sample_graph_data
        node = QueryNode(
            op_id="t1", op_type="TRAVERSAL", source="synthetic_graph",
            fields=["id"],
            traversal={
                "start_vertex_filter": {"column": "id", "value": 0},
                "edge_label": "KNOWS", "direction": "OUT",
                "max_hops": 2, "return_fields": ["id"]
            }
        )
        sub = SubExpression(
            sub_id="sub_0", nodes=[node],
            primary_op_type="TRAVERSAL", depends_on_subs=[],
            parallelizable=True
        )
        gen = GraphGenerator(graph_dir)
        result = gen.generate(sub)
        # From 0: hop1→{1,2}, hop2→{3,4}
        assert set(result["id"].values) == {0, 1, 2, 3, 4}

    def test_traversal_with_aggregate(self, sample_graph_data):
        """Test TRAVERSAL followed by AGGREGATE."""
        graph_dir, vertices, edges = sample_graph_data
        trav_node = QueryNode(
            op_id="t1", op_type="TRAVERSAL", source="synthetic_graph",
            fields=["id"],
            traversal={
                "start_vertex_filter": {"column": "id", "value": 0},
                "edge_label": "KNOWS", "direction": "OUT",
                "max_hops": 1, "return_fields": ["id"]
            }
        )
        agg_node = QueryNode(
            op_id="a1", op_type="AGGREGATE", source="t1",
            fields=["COUNT_id"],
            aggregate={
                "group_by": [],
                "functions": [{"func": "COUNT", "column": "id"}]
            },
            depends_on=["t1"]
        )
        sub = SubExpression(
            sub_id="sub_0", nodes=[trav_node, agg_node],
            primary_op_type="TRAVERSAL", depends_on_subs=[],
            parallelizable=True
        )
        gen = GraphGenerator(graph_dir)
        result = gen.generate(sub)
        assert result["COUNT_id"].values[0] == 3  # vertices 0, 1, 2

    def test_bidirectional_traversal(self, sample_graph_data):
        """Test BOTH direction traversal."""
        graph_dir, vertices, edges = sample_graph_data
        node = QueryNode(
            op_id="t1", op_type="TRAVERSAL", source="synthetic_graph",
            fields=["id"],
            traversal={
                "start_vertex_filter": {"column": "id", "value": 2},
                "edge_label": "KNOWS", "direction": "BOTH",
                "max_hops": 1, "return_fields": ["id"]
            }
        )
        sub = SubExpression(
            sub_id="sub_0", nodes=[node],
            primary_op_type="TRAVERSAL", depends_on_subs=[],
            parallelizable=True
        )
        gen = GraphGenerator(graph_dir)
        result = gen.generate(sub)
        # Vertex 2: in-edges from 0, out-edges to 4
        assert 0 in result["id"].values  # in-neighbor
        assert 4 in result["id"].values  # out-neighbor
        assert 2 in result["id"].values  # self


# --- ResultComposer Tests ---

class TestResultComposer:
    """Tests for ResultComposer."""

    def test_single_result(self):
        """Test composing a single result."""
        composer = ResultComposer()
        df = pd.DataFrame({"a": [1, 2, 3]})
        sub = SubExpression(
            sub_id="sub_0", nodes=[], primary_op_type="RELATIONAL",
            depends_on_subs=[], parallelizable=True
        )
        composer.register_result("sub_0", df)
        result = composer.compose([sub])
        assert len(result) == 3

    def test_dependent_results(self):
        """Test composing results with dependencies."""
        composer = ResultComposer()
        df1 = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        df2 = pd.DataFrame({"id": [1, 2], "score": [100, 200]})

        sub0 = SubExpression(
            sub_id="sub_0", nodes=[], primary_op_type="TRAVERSAL",
            depends_on_subs=[], parallelizable=True
        )
        sub1 = SubExpression(
            sub_id="sub_1", nodes=[], primary_op_type="RELATIONAL",
            depends_on_subs=["sub_0"], parallelizable=False
        )

        composer.register_result("sub_0", df1)
        composer.register_result("sub_1", df2)
        result = composer.compose([sub0, sub1], merge_strategy="join")
        assert "id" in result.columns

    def test_clear(self):
        """Test clearing results."""
        composer = ResultComposer()
        composer.register_result("sub_0", pd.DataFrame({"a": [1]}))
        composer.clear()
        assert composer.get_result("sub_0") is None

    def test_empty_compose(self):
        """Test composing with no sub-expressions."""
        composer = ResultComposer()
        result = composer.compose([])
        assert len(result) == 0


# --- Integration test with real parquet data (if available) ---

class TestIntegrationWithData:
    """Integration tests using actual parquet data files."""

    @pytest.fixture(autouse=True)
    def skip_if_no_data(self):
        """Skip if parquet data not available."""
        if not os.path.exists(os.path.join(TPCH_PARQUET_DIR, "customer")):
            pytest.skip("TPC-H parquet data not available")

    def test_tpch_query_001(self):
        """Test TPC-H query 1: filter→join→aggregate pipeline."""
        query_path = os.path.join(SAMPLE_QUERIES_DIR, "tpch_queries.json")
        if not os.path.exists(query_path):
            pytest.skip("Sample queries not available")

        with open(query_path) as f:
            queries = json.load(f)

        query = queries[0]  # q_tpch_001
        parser = DSLParser()
        decomposer = QueryDecomposer()
        nodes = parser.parse(query)
        subs = decomposer.decompose(nodes)

        # Build loader for real data
        def real_loader(name):
            path = os.path.join(TPCH_PARQUET_DIR, name)
            if os.path.exists(path):
                return pd.read_parquet(path)
            raise FileNotFoundError(name)

        # Execute all subs with SQL generator
        gen = SQLGenerator(real_loader)
        result = None
        for sub in subs:
            result = gen.generate(sub)
            # Cache each node's result for downstream ops
            for node in sub.nodes:
                gen.cache[node.op_id] = result

        assert result is not None
        assert len(result) > 0
        assert "c_name" in result.columns

    def test_tpch_aggregate_correctness(self):
        """Verify aggregate values are correct."""
        customers = pd.read_parquet(os.path.join(TPCH_PARQUET_DIR, "customer"))
        orders = pd.read_parquet(os.path.join(TPCH_PARQUET_DIR, "orders"))

        # Direct pandas computation
        nation7 = customers[customers["c_nationkey"] == 7]
        merged = orders.merge(nation7, left_on="o_custkey", right_on="c_custkey")
        expected = merged.groupby("c_name")["o_totalprice"].sum().reset_index()
        expected.columns = ["c_name", "SUM_o_totalprice"]

        # Via SQLGenerator
        tables = {"customer": customers, "orders": orders}
        gen = SQLGenerator(lambda n: tables[n])

        filter_node = QueryNode(
            op_id="s1", op_type="FILTER", source="customer",
            fields=["c_custkey", "c_name", "c_nationkey"],
            predicate={"column": "c_nationkey", "operator": "=", "value": 7}
        )
        filter_sub = SubExpression(
            sub_id="sub_0", nodes=[filter_node],
            primary_op_type="RELATIONAL", depends_on_subs=[],
            parallelizable=True
        )
        gen.cache["s1"] = gen.generate(filter_sub)

        join_node = QueryNode(
            op_id="s2", op_type="JOIN", source="orders",
            fields=["o_orderkey", "o_custkey", "o_totalprice", "c_name"],
            join={"right_source": "s1", "left_key": "o_custkey",
                  "right_key": "c_custkey", "join_type": "INNER"},
            depends_on=["s1"]
        )
        agg_node = QueryNode(
            op_id="s3", op_type="AGGREGATE", source="s2",
            fields=["c_name", "SUM_o_totalprice"],
            aggregate={"group_by": ["c_name"],
                       "functions": [{"func": "SUM", "column": "o_totalprice"}]},
            depends_on=["s2"]
        )
        pipeline_sub = SubExpression(
            sub_id="sub_1", nodes=[join_node, agg_node],
            primary_op_type="RELATIONAL", depends_on_subs=["sub_0"],
            parallelizable=False
        )
        result = gen.generate(pipeline_sub)

        # Compare sorting by c_name
        expected_sorted = expected.sort_values("c_name").reset_index(drop=True)
        result_sorted = result.sort_values("c_name").reset_index(drop=True)
        assert len(result_sorted) == len(expected_sorted)
        pd.testing.assert_series_equal(
            result_sorted["SUM_o_totalprice"],
            expected_sorted["SUM_o_totalprice"],
            check_names=False
        )
