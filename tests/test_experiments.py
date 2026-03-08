"""Tests for Phase 6: Evaluation & Experiments.

Tests the baseline runners, learned model runner, comparison table builder,
and ablation study.
"""

import json
import os
import sys
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import (
    SAMPLE_QUERIES_DIR, RESULTS_DIR, TPCH_PARQUET_DIR,
    GRAPHS_DIR, STATS_DIR, CLASSIFIER_PATH, LABELED_RUNS_CSV,
)
from experiments.run_baselines import load_all_queries, run_baseline
from experiments.run_learned import run_learned
from experiments.compare_results import (
    load_result_csvs, compute_strategy_metrics,
    build_comparison_table, compute_relative_improvements,
)
from experiments.ablation_study import (
    run_ablation, results_to_dataframes, FEATURE_GROUPS,
)


# ─── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_queries():
    """Load a small subset of queries for testing."""
    queries = load_all_queries(SAMPLE_QUERIES_DIR)
    # Use only TPC-H queries (which are most reliable) for speed
    return [q for q in queries if q.get("query_id", "").startswith("q_tpch")][:2]


@pytest.fixture(scope="module")
def tpch_queries():
    """Load TPC-H queries."""
    fpath = os.path.join(SAMPLE_QUERIES_DIR, "tpch_queries.json")
    with open(fpath) as f:
        return json.load(f)[:3]


@pytest.fixture(scope="module")
def synthetic_queries():
    """Load synthetic queries that don't require graph data beyond what exists."""
    fpath = os.path.join(SAMPLE_QUERIES_DIR, "synthetic_queries.json")
    with open(fpath) as f:
        queries = json.load(f)
    # Return only purely relational ones for reliable testing
    return [q for q in queries if q.get("query_id") in ("q_synth_004", "q_synth_006")]


@pytest.fixture(scope="module")
def results_dir():
    """Create a temporary results directory."""
    tmpdir = tempfile.mkdtemp(prefix="hifun_test_results_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ─── Query Loading Tests ────────────────────────────────────────────

class TestQueryLoading:
    def test_load_all_queries(self):
        queries = load_all_queries(SAMPLE_QUERIES_DIR)
        assert len(queries) > 0
        assert all("query_id" in q for q in queries)
        assert all("operations" in q for q in queries)

    def test_load_tpch_queries(self):
        fpath = os.path.join(SAMPLE_QUERIES_DIR, "tpch_queries.json")
        with open(fpath) as f:
            queries = json.load(f)
        assert len(queries) == 5
        assert queries[0]["query_id"] == "q_tpch_001"

    def test_load_synthetic_queries(self):
        fpath = os.path.join(SAMPLE_QUERIES_DIR, "synthetic_queries.json")
        with open(fpath) as f:
            queries = json.load(f)
        assert len(queries) >= 5

    def test_load_snb_queries(self):
        fpath = os.path.join(SAMPLE_QUERIES_DIR, "snb_queries.json")
        with open(fpath) as f:
            queries = json.load(f)
        assert len(queries) >= 3


# ─── Baseline Runner Tests ──────────────────────────────────────────

class TestBaselines:
    def test_always_sql_runs(self, tpch_queries):
        """Always-SQL strategy should execute all TPC-H queries."""
        df = run_baseline("always_sql", tpch_queries, num_runs=1)
        assert len(df) == len(tpch_queries)
        assert "median_latency_ms" in df.columns
        assert "strategy" in df.columns
        assert (df["strategy"] == "always_sql").all()
        # TPC-H queries are pure SQL, should all succeed
        assert df["success"].all()

    def test_always_sql_engines(self, tpch_queries):
        """Always-SQL should use only SQL engine."""
        df = run_baseline("always_sql", tpch_queries[:1], num_runs=1)
        engines = df.iloc[0]["engines_used"].split(",")
        assert all(e == "SQL" for e in engines)

    def test_always_graph_runs(self, tpch_queries):
        """Always-Graph may fail on pure SQL queries but should not crash."""
        df = run_baseline("always_graph", tpch_queries[:1], num_runs=1)
        assert len(df) == 1
        assert "strategy" in df.columns

    def test_rule_based_runs(self, tpch_queries):
        """Rule-based strategy should execute TPC-H queries via SQL."""
        df = run_baseline("rule_based", tpch_queries[:2], num_runs=1)
        assert len(df) == 2
        assert (df["strategy"] == "rule_based").all()
        # Pure relational queries should be routed to SQL by heuristic
        for _, row in df.iterrows():
            if row["success"]:
                engines = row["engines_used"].split(",")
                assert all(e == "SQL" for e in engines)

    def test_baseline_timing(self, tpch_queries):
        """Baseline latencies should be positive numbers."""
        df = run_baseline("always_sql", tpch_queries[:1], num_runs=2)
        row = df.iloc[0]
        assert row["median_latency_ms"] > 0
        assert row["mean_latency_ms"] > 0
        assert row["min_latency_ms"] > 0
        assert row["num_runs"] == 2

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_baseline("invalid", [], num_runs=1)

    def test_baseline_multiple_runs(self, tpch_queries):
        """Multiple runs should produce stable timing results."""
        df = run_baseline("always_sql", tpch_queries[:1], num_runs=3)
        row = df.iloc[0]
        assert row["num_runs"] == 3
        assert row["min_latency_ms"] <= row["median_latency_ms"] <= row["max_latency_ms"]

    def test_baseline_csv_output(self, tpch_queries, results_dir):
        """Baseline results should be saveable as CSV."""
        df = run_baseline("always_sql", tpch_queries[:1], num_runs=1)
        path = os.path.join(results_dir, "always_sql.csv")
        df.to_csv(path, index=False)
        loaded = pd.read_csv(path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["strategy"] == "always_sql"


# ─── Learned Model Runner Tests ─────────────────────────────────────

class TestLearnedRunner:
    @pytest.fixture(autouse=True)
    def _check_model(self):
        if not os.path.exists(CLASSIFIER_PATH):
            pytest.skip("Trained model not found at " + CLASSIFIER_PATH)

    def test_learned_runs(self, tpch_queries):
        """Learned strategy should execute queries using ML predictor."""
        df = run_learned(CLASSIFIER_PATH, tpch_queries[:2], num_runs=1)
        assert len(df) == 2
        assert (df["strategy"] == "learned").all()
        assert "avg_confidence" in df.columns
        assert "avg_inference_ms" in df.columns

    def test_learned_confidence(self, tpch_queries):
        """Learned model should have confidence scores."""
        df = run_learned(CLASSIFIER_PATH, tpch_queries[:1], num_runs=1)
        row = df.iloc[0]
        if row["success"]:
            assert 0.0 <= row["avg_confidence"] <= 1.0
            assert row["avg_inference_ms"] >= 0

    def test_learned_has_timing_breakdown(self, tpch_queries):
        """Learned strategy should include parse + decompose timing."""
        df = run_learned(CLASSIFIER_PATH, tpch_queries[:1], num_runs=1)
        row = df.iloc[0]
        if row["success"]:
            assert "parse_ms" in df.columns
            assert "decompose_ms" in df.columns


# ─── Compare Results Tests ──────────────────────────────────────────

class TestCompareResults:
    @pytest.fixture
    def mock_strategy_dfs(self):
        """Create mock strategy DataFrames for comparison testing."""
        base_data = {
            "query_id": ["q1", "q2", "q3"],
            "description": ["desc1", "desc2", "desc3"],
            "median_latency_ms": [100.0, 200.0, 150.0],
            "mean_latency_ms": [110.0, 210.0, 160.0],
            "min_latency_ms": [90.0, 180.0, 140.0],
            "max_latency_ms": [120.0, 220.0, 170.0],
            "num_subexpressions": [2, 3, 2],
            "engines_used": ["SQL,SQL", "SQL,SQL,SQL", "SQL,SQL"],
            "result_rows": [10, 20, 15],
            "num_runs": [3, 3, 3],
            "success": [True, True, True],
        }

        sql_df = pd.DataFrame({**base_data, "strategy": ["always_sql"] * 3})
        graph_df = pd.DataFrame({
            **base_data,
            "strategy": ["always_graph"] * 3,
            "median_latency_ms": [120.0, 250.0, 180.0],
            "mean_latency_ms": [130.0, 260.0, 190.0],
            "engines_used": ["GRAPH,GRAPH", "GRAPH,GRAPH,GRAPH", "GRAPH,GRAPH"],
        })
        learned_df = pd.DataFrame({
            **base_data,
            "strategy": ["learned"] * 3,
            "median_latency_ms": [80.0, 180.0, 140.0],
            "mean_latency_ms": [85.0, 185.0, 145.0],
            "engines_used": ["SQL,SQL", "SQL,GRAPH,SQL", "SQL,SQL"],
            "avg_confidence": [0.95, 0.88, 0.92],
            "avg_inference_ms": [0.5, 0.6, 0.4],
            "total_inference_ms": [1.0, 1.8, 0.8],
        })

        return {
            "always_sql": sql_df,
            "always_graph": graph_df,
            "learned_routing": learned_df,
        }

    def test_compute_metrics(self, mock_strategy_dfs):
        metrics = compute_strategy_metrics(mock_strategy_dfs["always_sql"])
        assert "median_latency_ms" in metrics
        assert "p95_latency_ms" in metrics
        assert metrics["successful_queries"] == 3
        assert metrics["total_queries"] == 3

    def test_build_comparison_table(self, mock_strategy_dfs):
        table = build_comparison_table(mock_strategy_dfs)
        assert len(table) == 3  # 3 strategies
        assert "Strategy" in table.columns
        assert "Median Latency (ms)" in table.columns
        assert "p95 Latency (ms)" in table.columns

    def test_comparison_table_order(self, mock_strategy_dfs):
        table = build_comparison_table(mock_strategy_dfs)
        strategies = table["Strategy"].tolist()
        assert strategies[0] == "Always SQL"
        assert strategies[1] == "Always Graph"
        assert strategies[2] == "Learned (ML)"

    def test_relative_improvements(self, mock_strategy_dfs):
        table = build_comparison_table(mock_strategy_dfs)
        table = compute_relative_improvements(table)
        assert "Median vs SQL (%)" in table.columns
        # SQL baseline should be 0%
        sql_row = table[table["Strategy"] == "Always SQL"]
        assert sql_row["Median vs SQL (%)"].iloc[0] == 0.0

    def test_empty_results(self):
        empty = {"always_sql": pd.DataFrame({
            "query_id": [], "strategy": [], "median_latency_ms": [],
            "mean_latency_ms": [], "success": [],
        })}
        table = build_comparison_table(empty)
        assert len(table) == 1

    def test_load_result_csvs(self, results_dir):
        """Test loading CSV files from results directory."""
        # Create a dummy CSV with required columns
        df = pd.DataFrame({
            "query_id": ["q1"], "strategy": ["test"],
            "median_latency_ms": [100.0], "mean_latency_ms": [100.0],
            "success": [True],
        })
        df.to_csv(os.path.join(results_dir, "test_strategy.csv"), index=False)
        loaded = load_result_csvs(results_dir)
        assert "test_strategy" in loaded
        assert len(loaded["test_strategy"]) == 1


# ─── Ablation Study Tests ──────────────────────────────────────────

class TestAblationStudy:
    @pytest.fixture(autouse=True)
    def _check_training_data(self):
        if not os.path.exists(LABELED_RUNS_CSV):
            pytest.skip("Training data not found at " + LABELED_RUNS_CSV)

    def test_ablation_runs(self):
        """Ablation study should complete without errors."""
        from features.feature_extractor import FEATURE_NAMES
        df = pd.read_csv(LABELED_RUNS_CSV)
        X = df[FEATURE_NAMES].values.astype(np.float32)
        y = (df["label"] == "GRAPH").astype(int).values

        results = run_ablation(X, y, FEATURE_NAMES, cv_folds=3)

        assert "baseline_f1" in results
        assert results["baseline_f1"] > 0
        assert "individual_features" in results
        assert len(results["individual_features"]) == len(FEATURE_NAMES)
        assert "feature_groups" in results
        assert len(results["feature_groups"]) > 0

    def test_ablation_individual_features(self):
        """Each individual feature ablation should have correct fields."""
        from features.feature_extractor import FEATURE_NAMES
        df = pd.read_csv(LABELED_RUNS_CSV)
        X = df[FEATURE_NAMES].values.astype(np.float32)
        y = (df["label"] == "GRAPH").astype(int).values

        results = run_ablation(X, y, FEATURE_NAMES, cv_folds=3)

        for feat, metrics in results["individual_features"].items():
            assert "f1_without" in metrics
            assert "f1_drop" in metrics
            assert 0.0 <= metrics["f1_without"] <= 1.0

    def test_ablation_group_features(self):
        """Feature group ablation should have all defined groups."""
        from features.feature_extractor import FEATURE_NAMES
        df = pd.read_csv(LABELED_RUNS_CSV)
        X = df[FEATURE_NAMES].values.astype(np.float32)
        y = (df["label"] == "GRAPH").astype(int).values

        results = run_ablation(X, y, FEATURE_NAMES, cv_folds=3)

        for group_name in FEATURE_GROUPS:
            assert group_name in results["feature_groups"], f"Missing group: {group_name}"

    def test_results_to_dataframes(self):
        """Ablation results should convert to clean DataFrames."""
        from features.feature_extractor import FEATURE_NAMES
        df = pd.read_csv(LABELED_RUNS_CSV)
        X = df[FEATURE_NAMES].values.astype(np.float32)
        y = (df["label"] == "GRAPH").astype(int).values

        results = run_ablation(X, y, FEATURE_NAMES, cv_folds=3)
        individual_df, group_df = results_to_dataframes(results)

        assert len(individual_df) == len(FEATURE_NAMES)
        assert "feature" in individual_df.columns
        assert "f1_drop" in individual_df.columns

        assert len(group_df) == len(FEATURE_GROUPS)
        assert "group" in group_df.columns

    def test_feature_groups_cover_all_features(self):
        """All feature groups should reference existing features."""
        from features.feature_extractor import FEATURE_NAMES
        all_group_features = set()
        for features in FEATURE_GROUPS.values():
            for f in features:
                all_group_features.add(f)
        # Every group feature should be in FEATURE_NAMES
        for f in all_group_features:
            assert f in FEATURE_NAMES, f"Group feature '{f}' not in FEATURE_NAMES"


# ─── Integration Tests ──────────────────────────────────────────────

class TestIntegration:
    """End-to-end integration tests using real data."""

    def test_full_evaluation_pipeline(self, tpch_queries, results_dir):
        """Run all strategies, compare, and produce table."""
        # Run baselines with 1 run for speed
        sql_df = run_baseline("always_sql", tpch_queries[:2], num_runs=1)
        sql_df.to_csv(os.path.join(results_dir, "always_sql.csv"), index=False)

        rule_df = run_baseline("rule_based", tpch_queries[:2], num_runs=1)
        rule_df.to_csv(os.path.join(results_dir, "rule_based.csv"), index=False)

        # Run learned if model exists
        if os.path.exists(CLASSIFIER_PATH):
            learned_df = run_learned(CLASSIFIER_PATH, tpch_queries[:2], num_runs=1)
            learned_df.to_csv(os.path.join(results_dir, "learned_routing.csv"), index=False)

        # Compare
        strategy_dfs = load_result_csvs(results_dir)
        assert len(strategy_dfs) >= 2  # at least SQL + rule_based

        comparison = build_comparison_table(strategy_dfs)
        comparison = compute_relative_improvements(comparison)

        assert len(comparison) >= 2
        assert "Strategy" in comparison.columns

        # Save
        comparison.to_csv(os.path.join(results_dir, "comparison_table.csv"), index=False)
        assert os.path.exists(os.path.join(results_dir, "comparison_table.csv"))

    def test_baseline_result_format_consistent(self, tpch_queries):
        """All baseline strategies should produce same column structure."""
        sql_df = run_baseline("always_sql", tpch_queries[:1], num_runs=1)
        rule_df = run_baseline("rule_based", tpch_queries[:1], num_runs=1)

        shared_cols = {"query_id", "strategy", "median_latency_ms", "mean_latency_ms",
                       "engines_used", "result_rows", "success", "num_runs"}
        assert shared_cols.issubset(set(sql_df.columns))
        assert shared_cols.issubset(set(rule_df.columns))
