"""Centralized path configuration for the HIFUN Router project."""

import os

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw data
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TPCH_RAW_DIR = os.path.join(RAW_DIR, "tpch-kit", "dbgen")
SNB_RAW_DIR = os.path.join(RAW_DIR, "snb")

# Parquet tables
PARQUET_DIR = os.path.join(PROJECT_ROOT, "data", "parquet")
TPCH_PARQUET_DIR = os.path.join(PARQUET_DIR, "tpch")
SNB_PARQUET_DIR = os.path.join(PARQUET_DIR, "snb")

# Graph data
GRAPHS_DIR = os.path.join(PROJECT_ROOT, "data", "graphs")
SYNTHETIC_VERTICES = os.path.join(GRAPHS_DIR, "synthetic_vertices.parquet")
SYNTHETIC_EDGES = os.path.join(GRAPHS_DIR, "synthetic_edges.parquet")
SNB_EDGES = os.path.join(GRAPHS_DIR, "snb_edges.parquet")

# Precomputed statistics
STATS_DIR = os.path.join(PROJECT_ROOT, "data", "stats")
CUSTOMER_STATS = os.path.join(STATS_DIR, "customer_stats.json")
ORDERS_STATS = os.path.join(STATS_DIR, "orders_stats.json")
SYNTHETIC_GRAPH_STATS = os.path.join(STATS_DIR, "synthetic_graph_stats.json")

# DSL sample queries
SAMPLE_QUERIES_DIR = os.path.join(PROJECT_ROOT, "dsl", "sample_queries")
DSL_SCHEMA_PATH = os.path.join(PROJECT_ROOT, "dsl", "schema.json")

# Model artifacts
MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "artifacts")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier_v1.pkl")
FEATURE_SCHEMA_PATH = os.path.join(MODEL_DIR, "feature_schema_v1.json")

# Training data
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
LABELED_RUNS_CSV = os.path.join(TRAINING_DATA_DIR, "labeled_runs.csv")

# Experiment results
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
