# Project Implementation Blueprint: AI-Assisted HIFUN Query Decomposition

> **Version:** 1.0 | **Role:** Senior Big Data Architect & Research Lead
> **Target:** Developer Team Implementation Guide | **Audience:** Workshop / SIGMOD Paper Prototype

---

## Table of Contents

1. [Project Overview & Goals](#1-project-overview--goals)
2. [System Architecture Summary](#2-system-architecture-summary)
3. [HIFUN DSL Definition (JSON-Based Subset)](#3-hifun-dsl-definition-json-based-subset)
4. [Proposed Directory Structure](#4-proposed-directory-structure)
5. [Phase 1 — Environment Setup](#5-phase-1--environment-setup)
6. [Phase 2 — Data Preparation](#6-phase-2--data-preparation)
7. [Phase 3 — Core Logic Implementation](#7-phase-3--core-logic-implementation)
8. [Phase 4 — ML Pipeline](#8-phase-4--ml-pipeline)
9. [Phase 5 — Execution Engine Integration](#9-phase-5--execution-engine-integration)
10. [Phase 6 — Evaluation & Experiments](#10-phase-6--evaluation--experiments)
11. [Algorithm Pseudocode Reference](#11-algorithm-pseudocode-reference)
12. [Baseline vs. Learned Model: Experimental Instructions](#12-baseline-vs-learned-model-experimental-instructions)
13. [Risk Register & Mitigation](#13-risk-register--mitigation)
14. [References](#14-references)

---

## 1. Project Overview & Goals

### 1.1 Problem Statement

Modern applications execute mixed workloads over both relational stores (SQL) and graph/document stores (NoSQL). A single high-level HIFUN query may contain subexpressions that are best served by different runtimes — Spark SQL for joins and aggregations, and GraphFrames/GraphX for traversals and motif finding.

Manually specifying execution engines per subquery is error-prone. This system **automatically decomposes** a HIFUN query and **routes each subexpression** to the optimal engine using a trained ML classifier.

### 1.2 Core Research Questions

| # | Question |
|---|----------|
| RQ1 | Can a lightweight ML model (decision tree, XGBoost) reliably predict whether a HIFUN subexpression runs faster as SQL or graph? |
| RQ2 | Which features (cardinality, degree, selectivity, historical runtime) are most predictive? |
| RQ3 | How much improvement does learned routing provide over rule-based heuristics? |

### 1.3 Success Criteria

- **Correctness:** Results of routed queries match reference (naive in-memory) execution.
- **Latency:** Learned routing achieves ≥10% median latency improvement over rule-based baseline on mixed-workload queries.
- **Model inference time:** <10ms per subexpression (does not dominate planning overhead).
- **Reproducibility:** All experiments runnable from a single shell script with publicly available datasets.

---

## 2. System Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                        │
│             User submits JSON-DSL HIFUN Query                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  QUERY INTELLIGENCE LAYER                    │
│  ┌───────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │  Parser   │→ │  Decomposer    │→ │ Feature Extractor│   │
│  │ (AST Gen) │  │(Subexpressions)│  │ (Vector Builder) │   │
│  └───────────┘  └────────────────┘  └──────────────────┘   │
│                                              │               │
│                                    ┌─────────────────┐      │
│                                    │  ML Router Model │      │
│                                    │ (XGBoost/DTree)  │      │
│                                    └─────────────────┘      │
└─────────────────────────────────────────────────────────────┘
               │ SQL Route                    │ Graph Route
               ▼                              ▼
┌──────────────────────┐        ┌──────────────────────────┐
│   Spark SQL Engine   │        │  GraphFrames / GraphX    │
│  (Catalyst Optimizer)│        │  (BFS, Motif Finding)    │
└──────────────────────┘        └──────────────────────────┘
               │                              │
               └──────────────┬───────────────┘
                              ▼
                  ┌───────────────────────┐
                  │    Result Composer    │
                  │ (Join / Aggregate     │
                  │  Partial Results)     │
                  └───────────────────────┘
                              │
                  ┌───────────────────────┐
                  │      Final Output     │
                  └───────────────────────┘
                              │
                  ┌───────────────────────┐
                  │   Runtime Logger      │
                  │ (SQLite / JSON DB)    │  ←── Feeds retraining
                  └───────────────────────┘
```

---

## 3. HIFUN DSL Definition (JSON-Based Subset)

Because a full HIFUN parser is complex, this prototype uses a **JSON-based DSL** that captures the 4 essential operator types: `FILTER`, `MAP`, `JOIN`, and `TRAVERSAL`.

### 3.1 DSL Schema

```json
{
  "query_id": "string",
  "description": "human-readable label",
  "operations": [
    {
      "op_id": "string (unique per subexpression)",
      "type": "FILTER | MAP | JOIN | TRAVERSAL | AGGREGATE",
      "source": "table_name or graph_name",
      "fields": ["col1", "col2"],
      "predicate": {
        "column": "string",
        "operator": "= | > | < | >= | <= | IN | LIKE",
        "value": "scalar or list"
      },
      "join": {
        "right_source": "table_name",
        "left_key": "column",
        "right_key": "column",
        "join_type": "INNER | LEFT | RIGHT"
      },
      "traversal": {
        "start_vertex_filter": {"column": "string", "value": "scalar"},
        "edge_label": "string",
        "direction": "IN | OUT | BOTH",
        "max_hops": "integer",
        "return_fields": ["col1", "col2"]
      },
      "aggregate": {
        "group_by": ["col1"],
        "functions": [{"func": "SUM | COUNT | AVG | MAX | MIN", "column": "col"}]
      },
      "depends_on": ["op_id1", "op_id2"]
    }
  ]
}
```

### 3.2 DSL Examples

**Example A — Pure Relational Join Query**

```json
{
  "query_id": "q_tpch_001",
  "description": "TPC-H orders joined with customers filtered by region",
  "operations": [
    {
      "op_id": "s1",
      "type": "FILTER",
      "source": "customer",
      "fields": ["c_custkey", "c_name", "c_nationkey"],
      "predicate": {"column": "c_regionkey", "operator": "=", "value": 2},
      "depends_on": []
    },
    {
      "op_id": "s2",
      "type": "JOIN",
      "source": "orders",
      "fields": ["o_orderkey", "o_custkey", "o_totalprice"],
      "join": {
        "right_source": "s1",
        "left_key": "o_custkey",
        "right_key": "c_custkey",
        "join_type": "INNER"
      },
      "depends_on": ["s1"]
    },
    {
      "op_id": "s3",
      "type": "AGGREGATE",
      "source": "s2",
      "fields": ["c_name"],
      "aggregate": {
        "group_by": ["c_name"],
        "functions": [{"func": "SUM", "column": "o_totalprice"}]
      },
      "depends_on": ["s2"]
    }
  ]
}
```

**Example B — Mixed Traversal + Relational Query**

```json
{
  "query_id": "q_snb_002",
  "description": "Friends of a person (graph) joined with post counts (SQL)",
  "operations": [
    {
      "op_id": "s1",
      "type": "TRAVERSAL",
      "source": "social_graph",
      "traversal": {
        "start_vertex_filter": {"column": "person_id", "value": 123},
        "edge_label": "KNOWS",
        "direction": "BOTH",
        "max_hops": 2,
        "return_fields": ["person_id", "name"]
      },
      "depends_on": []
    },
    {
      "op_id": "s2",
      "type": "AGGREGATE",
      "source": "posts",
      "fields": ["creator_id"],
      "aggregate": {
        "group_by": ["creator_id"],
        "functions": [{"func": "COUNT", "column": "post_id"}]
      },
      "depends_on": []
    },
    {
      "op_id": "s3",
      "type": "JOIN",
      "source": "s1",
      "fields": ["person_id", "name", "post_count"],
      "join": {
        "right_source": "s2",
        "left_key": "person_id",
        "right_key": "creator_id",
        "join_type": "LEFT"
      },
      "depends_on": ["s1", "s2"]
    }
  ]
}
```

### 3.3 DSL Operator Type Reference

| Operator | Best Engine (Rule) | Key Feature Signals |
|----------|--------------------|---------------------|
| `FILTER` | SQL | High selectivity → SQL; low selectivity over graph → GRAPH |
| `MAP` | SQL | Always SQL unless on graph vertices |
| `JOIN` | SQL | Large cardinality → SQL |
| `TRAVERSAL` | GRAPH | High avg degree + low hops → GRAPH |
| `AGGREGATE` | SQL (usually) | Post-traversal aggregates may stay in GRAPH |

---

## 4. Proposed Directory Structure

```
hifun_router/
│
├── README.md                         # Project overview + quickstart
├── requirements.txt                  # Python dependencies
├── docker-compose.yml                # Spark + Jupyter environment
├── Makefile                          # Shortcut commands (setup, train, evaluate)
│
├── config/
│   ├── spark_config.py               # SparkSession settings (local/cluster)
│   ├── paths.py                      # Dataset and model paths
│   └── feature_schema.json           # Canonical feature vector definition
│
├── dsl/
│   ├── schema.json                   # JSON DSL schema (jsonschema format)
│   ├── validator.py                  # DSL schema validation
│   └── sample_queries/
│       ├── tpch_queries.json         # TPC-H sample DSL queries
│       ├── snb_queries.json          # SNB mixed queries
│       └── synthetic_queries.json    # Generated test queries
│
├── parser/
│   ├── __init__.py
│   ├── dsl_parser.py                 # JSON DSL → internal AST (Python dict tree)
│   └── ast_nodes.py                  # Dataclasses: QueryNode, FilterNode, JoinNode, etc.
│
├── decomposer/
│   ├── __init__.py
│   └── query_decomposer.py           # QueryDecomposer class — splits AST into subexpressions
│
├── features/
│   ├── __init__.py
│   ├── feature_extractor.py          # FeatureExtractor class — builds feature vectors
│   ├── stats_collector.py            # Collects cardinality, degree stats from Spark/Hive
│   └── historical_store.py           # SQLite-based runtime history lookup
│
├── model/
│   ├── __init__.py
│   ├── trainer.py                    # Training pipeline: load data, fit, CV, save
│   ├── predictor.py                  # ModelPredictor class — loads artifact, runs inference
│   ├── feature_importance.py         # SHAP analysis and ablation
│   └── artifacts/
│       ├── classifier_v1.pkl         # Trained XGBoost model
│       └── feature_schema_v1.json    # Feature names matching model artifact
│
├── execution/
│   ├── __init__.py
│   ├── sql_generator.py              # SQLGenerator — DSL subexpression → Spark SQL/DataFrame
│   ├── graph_generator.py            # GraphGenerator — DSL subexpression → GraphFrames code
│   └── result_composer.py            # ResultComposer — merges partial DataFrames
│
├── router/
│   ├── __init__.py
│   └── hybrid_router.py              # HybridRouter — orchestrates full query execution
│
├── data/
│   ├── raw/                          # Raw downloaded datasets
│   ├── parquet/                      # Preprocessed Parquet tables
│   ├── graphs/                       # Edge lists as Parquet (GraphFrames input)
│   └── stats/                        # Precomputed column stats JSON files
│
├── training_data/
│   ├── labeled_runs.csv              # (op_id, features..., actual_sql_ms, actual_graph_ms, label)
│   └── collection_script.py          # Runs both engines per subexpression to generate labels
│
├── experiments/
│   ├── run_baselines.py              # Runs always-SQL, always-GRAPH, rule-based
│   ├── run_learned.py                # Runs ML-routed execution
│   ├── ablation_study.py             # Feature ablation experiments
│   └── results/                      # Output CSVs + plots
│
├── tests/
│   ├── test_parser.py
│   ├── test_decomposer.py
│   ├── test_features.py
│   ├── test_correctness.py           # Compares ML-routed output vs reference executor
│   └── reference_executor.py         # Naive in-memory Python executor for correctness checks
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_feature_analysis.ipynb
    ├── 03_model_training.ipynb
    └── 04_results_visualization.ipynb
```

---

## 5. Phase 1 — Environment Setup

### 5.1 Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Core language |
| Apache Spark | 3.4+ | SQL + GraphFrames runtime |
| GraphFrames | 0.8.3+ | Graph processing on Spark |
| PySpark | 3.4+ | Python Spark bindings |
| scikit-learn | 1.4+ | Decision tree baseline model |
| XGBoost | 2.0+ | Primary ML classifier |
| SHAP | 0.44+ | Feature importance analysis |
| pandas | 2.0+ | Local data manipulation |
| SQLite3 | (stdlib) | Historical runtime store |
| jsonschema | 4.x | DSL query validation |
| Docker | 24+ | Reproducible environment (optional) |

### 5.2 Installation

```bash
# Clone repo
git clone https://github.com/your-org/hifun_router.git
cd hifun_router

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**

```
pyspark==3.4.2
graphframes==0.8.3
xgboost==2.0.3
scikit-learn==1.4.0
shap==0.44.0
pandas==2.1.4
numpy==1.26.3
jsonschema==4.21.1
matplotlib==3.8.2
seaborn==0.13.1
pyarrow==14.0.2
lightgbm==4.2.0
```

### 5.3 SparkSession Setup (`config/spark_config.py`)

```python
from pyspark.sql import SparkSession

def get_spark_session(app_name: str = "HIFUN_Router") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.4-s_2.12")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
```

---

## 6. Phase 2 — Data Preparation

### 6.1 Datasets to Use

| Dataset | Type | Use Case | Source |
|---------|------|----------|--------|
| TPC-H (SF 1–5) | Relational | Join-heavy SQL queries | tpc.org |
| LDBC SNB (Interactive) | Graph + Relational | Mixed traversal + SQL | ldbcouncil.org |
| IMDB Public Dump | Relational + Graph | Real-world relationships | imdb.com/interfaces |
| Synthetic Power-Law Graphs | Graph | Degree distribution control | NetworkX / SNAP |

### 6.2 Data Ingestion Scripts

**TPC-H → Parquet:**

```bash
# Generate TPC-H data using dbgen
cd tpch-kit && ./dbgen -s 1 -f
# Convert to Parquet via PySpark
python data/scripts/tpch_to_parquet.py --input ./tpch-kit --output data/parquet/tpch/
```

**LDBC SNB → Parquet + GraphFrames Edge List:**

```bash
python data/scripts/snb_to_parquet.py \
    --input data/raw/snb/ \
    --tables data/parquet/snb/ \
    --edges data/graphs/snb_edges.parquet
```

**Synthetic Graph Generator:**

```python
# data/scripts/generate_synthetic.py
import networkx as nx
import pandas as pd

def generate_powerlaw_graph(n_nodes, avg_degree, seed=42):
    G = nx.barabasi_albert_graph(n_nodes, avg_degree // 2, seed=seed)
    edges = pd.DataFrame(G.edges(), columns=["src", "dst"])
    vertices = pd.DataFrame({"id": list(G.nodes()), "attr1": range(n_nodes)})
    return vertices, edges

# Vary avg_degree in [2, 5, 10, 20, 50] to stress-test routing thresholds
```

### 6.3 Statistics Precomputation (`data/scripts/compute_stats.py`)

For each table and graph, precompute and save to `data/stats/<name>_stats.json`:

```python
def compute_table_stats(spark, table_name: str, parquet_path: str) -> dict:
    df = spark.read.parquet(parquet_path)
    stats = {
        "table_name": table_name,
        "row_count": df.count(),
        "column_count": len(df.columns),
        "columns": {}
    }
    for col in df.columns:
        stats["columns"][col] = {
            "null_count": df.filter(df[col].isNull()).count(),
            "distinct_count": df.select(col).distinct().count(),
            "min": df.agg({col: "min"}).collect()[0][0],
            "max": df.agg({col: "max"}).collect()[0][0],
        }
    return stats

def compute_graph_stats(spark, edge_parquet: str) -> dict:
    edges = spark.read.parquet(edge_parquet)
    degree_df = edges.groupBy("src").count()
    stats_row = degree_df.selectExpr(
        "avg(count) as avg_degree",
        "max(count) as max_degree",
        "stddev(count) as stddev_degree",
        "count(*) as vertex_count"
    ).collect()[0]
    return {
        "avg_degree": stats_row.avg_degree,
        "max_degree": stats_row.max_degree,
        "stddev_degree": stats_row.stddev_degree,
        "vertex_count": stats_row.vertex_count,
        "edge_count": edges.count()
    }
```

---

## 7. Phase 3 — Core Logic Implementation

### 7.1 Parser (`parser/dsl_parser.py`)

**Responsibilities:** Validate incoming JSON DSL query, convert to an internal Python AST (tree of `QueryNode` dataclasses).

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class QueryNode:
    op_id: str
    op_type: str                      # FILTER, MAP, JOIN, TRAVERSAL, AGGREGATE
    source: str
    fields: List[str]
    predicate: Optional[Dict] = None
    join: Optional[Dict] = None
    traversal: Optional[Dict] = None
    aggregate: Optional[Dict] = None
    depends_on: List[str] = field(default_factory=list)

class DSLParser:
    def parse(self, query_json: dict) -> List[QueryNode]:
        """
        1. Validate JSON against schema (jsonschema.validate).
        2. For each operation in query_json["operations"]:
           a. Instantiate a QueryNode dataclass.
           b. Append to ordered list.
        3. Return list of QueryNodes in dependency order (topological sort).
        """
        ...

    def _topological_sort(self, nodes: List[QueryNode]) -> List[QueryNode]:
        """
        Kahn's algorithm on depends_on edges.
        Ensures parent subexpressions are processed before children.
        """
        ...
```

### 7.2 QueryDecomposer (`decomposer/query_decomposer.py`)

**Responsibilities:** Takes the parsed list of `QueryNode`s and groups them into **candidate routing units** — contiguous subexpressions of the same logical type that can be dispatched independently or together.

#### Decomposition Rules

| Rule | Action |
|------|--------|
| A `TRAVERSAL` node is always its own candidate unit | Isolate for GRAPH routing |
| Contiguous `FILTER` + `JOIN` + `AGGREGATE` chains form one SQL candidate unit | Group for SQL routing |
| A `MAP` node is merged with its parent unit | Avoid fragmentation |
| Nodes with no shared dependencies are candidates to run in parallel | Mark `parallelizable=True` |

#### Step-by-Step Function Description

```
CLASS QueryDecomposer:

  METHOD decompose(nodes: List[QueryNode]) -> List[SubExpression]:

    STEP 1: Build dependency graph (adjacency list from depends_on).

    STEP 2: Identify TRAVERSAL nodes -> each becomes its own SubExpression.

    STEP 3: For non-TRAVERSAL nodes, apply contiguous grouping:
      - BFS from root nodes (no dependencies).
      - While current chain is FILTER/JOIN/MAP/AGGREGATE:
          -> Continue adding to current group.
      - When TRAVERSAL is encountered OR dependency from a TRAVERSAL result:
          -> Close current group as one SubExpression.
          -> Start new group after the TRAVERSAL.

    STEP 4: Assign each SubExpression:
      - sub_id: unique identifier
      - nodes: list of QueryNodes in this unit
      - primary_op_type: dominant operator (TRAVERSAL or RELATIONAL)
      - depends_on_subs: sub_ids this unit depends on
      - parallelizable: True if no shared upstream dependency

    STEP 5: Return List[SubExpression]
```

**`SubExpression` dataclass:**

```python
@dataclass
class SubExpression:
    sub_id: str
    nodes: List[QueryNode]
    primary_op_type: str          # "RELATIONAL" or "TRAVERSAL"
    depends_on_subs: List[str]
    parallelizable: bool
    estimated_output_rows: int = 0  # filled by FeatureExtractor
```

### 7.3 FeatureExtractor (`features/feature_extractor.py`)

**Responsibilities:** Given a `SubExpression` and precomputed table/graph statistics, build a fixed-length numeric feature vector for the ML classifier.

#### Complete Feature Vector Definition

The canonical feature vector has **22 features**, defined in `config/feature_schema.json`:

```json
{
  "features": [
    {"name": "op_count_filter",             "type": "int",   "description": "# FILTER nodes in subexpression"},
    {"name": "op_count_join",               "type": "int",   "description": "# JOIN nodes in subexpression"},
    {"name": "op_count_traversal",          "type": "int",   "description": "# TRAVERSAL nodes in subexpression"},
    {"name": "op_count_aggregate",          "type": "int",   "description": "# AGGREGATE nodes in subexpression"},
    {"name": "op_count_map",                "type": "int",   "description": "# MAP nodes in subexpression"},
    {"name": "ast_depth",                   "type": "int",   "description": "Max depth of AST subtree"},
    {"name": "has_traversal",               "type": "bool",  "description": "1 if any TRAVERSAL node present"},
    {"name": "max_hops",                    "type": "int",   "description": "Max traversal hops (0 if none)"},
    {"name": "input_cardinality_log",       "type": "float", "description": "log10(estimated input rows)"},
    {"name": "output_cardinality_log",      "type": "float", "description": "log10(estimated output rows)"},
    {"name": "selectivity",                 "type": "float", "description": "Fraction of rows passing filters [0,1]"},
    {"name": "avg_degree",                  "type": "float", "description": "Avg out-degree of source graph (0 if relational)"},
    {"name": "max_degree",                  "type": "float", "description": "Max out-degree of source graph (0 if relational)"},
    {"name": "degree_skew",                 "type": "float", "description": "stddev/avg of degree distribution (0 if relational)"},
    {"name": "num_projected_columns",       "type": "int",   "description": "# output columns in subexpression"},
    {"name": "has_index",                   "type": "bool",  "description": "1 if predicate column has index"},
    {"name": "join_fanout",                 "type": "float", "description": "Estimated output rows / input rows for join (1.0 if no join)"},
    {"name": "estimated_shuffle_bytes_log", "type": "float", "description": "log10(rows_out * avg_row_bytes) if SQL"},
    {"name": "estimated_traversal_ops",     "type": "float", "description": "start_vertices * avg_degree^max_hops (0 if SQL)"},
    {"name": "hist_avg_runtime_ms",         "type": "float", "description": "Avg runtime of similar past subexpressions (-1 if unknown)"},
    {"name": "hist_runtime_variance",       "type": "float", "description": "Variance of past runtimes (-1 if unknown)"},
    {"name": "num_tables_joined",           "type": "int",   "description": "# distinct tables referenced in subexpression"}
  ]
}
```

#### FeatureExtractor Step-by-Step Logic

```
CLASS FeatureExtractor:

  CONSTRUCTOR(stats_dir, history_db_path):
    - Load all JSON stats files from stats_dir into self.stats dict.
    - Connect to SQLite history DB.

  METHOD extract(sub_expr: SubExpression) -> np.ndarray:

    STEP 1 — Query Shape Features:
      op_count_filter    = count nodes where op_type == "FILTER"
      op_count_join      = count nodes where op_type == "JOIN"
      op_count_traversal = count nodes where op_type == "TRAVERSAL"
      op_count_aggregate = count nodes where op_type == "AGGREGATE"
      op_count_map       = count nodes where op_type == "MAP"
      ast_depth          = compute max depth by BFS from root node
      has_traversal      = 1 if op_count_traversal > 0 else 0
      max_hops           = max(node.traversal.max_hops for traversal nodes, default 0)

    STEP 2 — Data Statistics Features:
      source = sub_expr.nodes[0].source
      table_stats = self.stats.get(source, {})

      input_cardinality_log  = log10(table_stats["row_count"] + 1)
      selectivity            = compute_selectivity(sub_expr, table_stats)
      output_cardinality_log = log10(input_cardinality * selectivity + 1)
      avg_degree   = graph_stats["avg_degree"] if source is graph else 0.0
      max_degree   = graph_stats["max_degree"] if source is graph else 0.0
      degree_skew  = graph_stats["stddev_degree"] / avg_degree if avg_degree > 0 else 0.0
      num_projected_columns = len(sub_expr.nodes[-1].fields)
      has_index    = check stats for index on predicate column -> 1 or 0
      join_fanout  = estimated_output_rows / input_cardinality (JOIN nodes only)

    STEP 3 — Engine Cost Proxies:
      avg_row_bytes = num_projected_columns * 8   # rough estimate
      estimated_shuffle_bytes_log = log10(output_cardinality * avg_row_bytes + 1)
      IF has_traversal:
        start_vertices = apply_filter_selectivity(input_cardinality)
        estimated_traversal_ops = start_vertices * (avg_degree ** max_hops)
      ELSE:
        estimated_traversal_ops = 0.0
      num_tables_joined = count distinct source names in sub_expr.nodes

    STEP 4 — Historical Features:
      fingerprint = hash(sorted(op_types) + source_name)
      history_row = query SQLite WHERE fingerprint = fingerprint ORDER BY ts DESC
      IF history_row:
        hist_avg_runtime_ms    = history_row["avg_runtime_ms"]
        hist_runtime_variance  = history_row["variance_ms"]
      ELSE:
        hist_avg_runtime_ms = hist_runtime_variance = -1.0

    STEP 5 — Assemble vector:
      RETURN np.array([all 22 values in schema order], dtype=float32)
```

---

## 8. Phase 4 — ML Pipeline

### 8.1 Training Data Collection (`training_data/collection_script.py`)

For each query in the training set:

```
FOR each query JSON file in dsl/sample_queries/:
  1. Parse -> AST -> Decompose -> List of SubExpressions.
  2. FOR each SubExpression s:
       a. Execute s via SQLGenerator -> Spark SQL -> measure wall-clock ms.
       b. Execute s via GraphGenerator -> GraphFrames -> measure wall-clock ms.
       c. IF sql_ms < graph_ms: label = "SQL" ELSE label = "GRAPH".
       d. Extract feature vector via FeatureExtractor.
       e. Append row to training_data/labeled_runs.csv:
          (sub_id, query_id, dataset, ...22 features..., sql_ms, graph_ms, label)
```

**Columns in `labeled_runs.csv`:**

```
sub_id, query_id, dataset, op_count_filter, op_count_join, op_count_traversal,
op_count_aggregate, op_count_map, ast_depth, has_traversal, max_hops,
input_cardinality_log, output_cardinality_log, selectivity, avg_degree,
max_degree, degree_skew, num_projected_columns, has_index, join_fanout,
estimated_shuffle_bytes_log, estimated_traversal_ops, hist_avg_runtime_ms,
hist_runtime_variance, num_tables_joined, sql_runtime_ms, graph_runtime_ms, label
```

### 8.2 Model Training (`model/trainer.py`)

```python
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

FEATURE_COLS = [...]  # 22 feature names from schema

def train(labeled_data_path: str, model_out: str, cv_folds: int = 5):
    df = pd.read_csv(labeled_data_path)
    X = df[FEATURE_COLS].values
    y = (df["label"] == "GRAPH").astype(int).values  # 0=SQL, 1=GRAPH

    # Model 1: Decision Tree (interpretable baseline)
    dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10)
    dt_scores = cross_val_score(dt, X, y, cv=StratifiedKFold(cv_folds), scoring="f1")
    print(f"Decision Tree CV F1: {dt_scores.mean():.3f} +/- {dt_scores.std():.3f}")

    # Model 2: XGBoost (primary classifier)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        use_label_encoder=False, eval_metric="logloss"
    )
    xgb_scores = cross_val_score(xgb_clf, X, y, cv=StratifiedKFold(cv_folds), scoring="f1")
    print(f"XGBoost CV F1:       {xgb_scores.mean():.3f} +/- {xgb_scores.std():.3f}")

    # Train final model on all data
    xgb_clf.fit(X, y)
    joblib.dump(xgb_clf, model_out)
    print(f"Model saved to {model_out}")
```

### 8.3 Feature Importance Analysis (`model/feature_importance.py`)

```python
import shap
import matplotlib.pyplot as plt

def generate_shap_report(model, X_train, feature_names, output_dir):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.savefig(f"{output_dir}/shap_summary.pdf", bbox_inches="tight")
    plt.close()
```

### 8.4 Model Inference (`model/predictor.py`)

```python
import joblib
import numpy as np
import time

class ModelPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, feature_vector: np.ndarray) -> str:
        """Returns 'SQL' or 'GRAPH' in <5ms."""
        t0 = time.perf_counter()
        pred = self.model.predict(feature_vector.reshape(1, -1))[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        # Log inference time to ensure it stays <10ms
        assert elapsed_ms < 10, f"Inference too slow: {elapsed_ms:.1f}ms"
        return "GRAPH" if pred == 1 else "SQL"
```

---

## 9. Phase 5 — Execution Engine Integration

### 9.1 SQL Generator (`execution/sql_generator.py`)

Translates a `SubExpression` with label `SQL` into PySpark DataFrame operations:

```python
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

class SQLGenerator:
    def __init__(self, spark: SparkSession, parquet_dir: str):
        self.spark = spark
        self.parquet_dir = parquet_dir
        self._cache: dict = {}  # sub_id -> DataFrame (for composed results)

    def generate(self, sub_expr) -> DataFrame:
        df = None
        for node in sub_expr.nodes:
            if node.op_type == "FILTER":
                df = self._apply_filter(node, df)
            elif node.op_type == "JOIN":
                df = self._apply_join(node, df)
            elif node.op_type == "AGGREGATE":
                df = self._apply_aggregate(node, df)
            elif node.op_type == "MAP":
                df = df.select(node.fields)
        return df

    def _apply_filter(self, node, df) -> DataFrame:
        if df is None:
            df = self.spark.read.parquet(f"{self.parquet_dir}/{node.source}")
        p = node.predicate
        return df.filter(F.col(p["column"]).cast("string") == str(p["value"]))

    def _apply_join(self, node, df) -> DataFrame:
        right_source = node.join["right_source"]
        right_df = (self._cache.get(right_source) or
                    self.spark.read.parquet(f"{self.parquet_dir}/{right_source}"))
        return df.join(right_df,
                       df[node.join["left_key"]] == right_df[node.join["right_key"]],
                       how=node.join["join_type"].lower())

    def _apply_aggregate(self, node, df) -> DataFrame:
        agg_exprs = [
            getattr(F, fn["func"].lower())(fn["column"]).alias(f"{fn['func']}_{fn['column']}")
            for fn in node.aggregate["functions"]
        ]
        return df.groupBy(node.aggregate["group_by"]).agg(*agg_exprs)
```

### 9.2 Graph Generator (`execution/graph_generator.py`)

Translates a `SubExpression` with TRAVERSAL nodes into GraphFrames operations:

```python
from graphframes import GraphFrame
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

class GraphGenerator:
    def __init__(self, spark: SparkSession, graph_dir: str):
        self.spark = spark
        self.graph_dir = graph_dir

    def generate(self, sub_expr) -> DataFrame:
        for node in sub_expr.nodes:
            if node.op_type == "TRAVERSAL":
                return self._apply_traversal(node)
        raise ValueError("No TRAVERSAL node found in graph subexpression")

    def _apply_traversal(self, node) -> DataFrame:
        t = node.traversal
        vertices = self.spark.read.parquet(
            f"{self.graph_dir}/{node.source}_vertices.parquet")
        edges = self.spark.read.parquet(
            f"{self.graph_dir}/{node.source}_edges.parquet")
        gf = GraphFrame(vertices, edges)

        start_filter = t["start_vertex_filter"]
        result = gf.bfs(
            fromExpr=f"{start_filter['column']} = {start_filter['value']}",
            toExpr="id IS NOT NULL",
            edgeFilter=f"relationship = '{t['edge_label']}'",
            maxPathLength=t["max_hops"]
        )
        return result.select(t["return_fields"])
```

### 9.3 Result Composer (`execution/result_composer.py`)

```python
class ResultComposer:
    def compose(self, partial_results: dict, query_nodes: list) -> DataFrame:
        """
        Merges partial DataFrames from SQL and GRAPH engines
        according to the original depends_on relationship in the DSL.

        Algorithm:
        1. Build result map: {sub_id -> DataFrame}.
        2. For any JOIN node that spans sub_ids:
           a. Fetch left and right DataFrames from result map.
           b. Perform Spark join using join keys from DSL.
           c. Replace both entries in map with the composed result.
        3. Return final composed DataFrame.
        """
        ...
```

### 9.4 HybridRouter Orchestrator (`router/hybrid_router.py`)

```python
import time

class HybridRouter:
    def __init__(self, spark, predictor, sql_gen, graph_gen, composer,
                 feature_extractor, logger):
        self.spark = spark
        self.predictor = predictor
        self.sql_gen = sql_gen
        self.graph_gen = graph_gen
        self.composer = composer
        self.feature_extractor = feature_extractor
        self.logger = logger

    def execute(self, query_json: dict):
        # 1. Parse
        nodes = DSLParser().parse(query_json)
        # 2. Decompose
        sub_expressions = QueryDecomposer().decompose(nodes)
        # 3. Route + Execute
        partial_results = {}
        for sub in sub_expressions:
            fv = self.feature_extractor.extract(sub)
            engine = self.predictor.predict(fv)
            t0 = time.perf_counter()
            if engine == "SQL":
                df = self.sql_gen.generate(sub)
            else:
                df = self.graph_gen.generate(sub)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            partial_results[sub.sub_id] = df
            self.logger.log(sub, engine, fv, elapsed_ms)
        # 4. Compose
        return self.composer.compose(partial_results, nodes)
```

---

## 10. Phase 6 — Evaluation & Experiments

### 10.1 Metrics

| Metric | Measurement Method |
|--------|-------------------|
| Wall-clock latency (median, p95) | `time.perf_counter()` in HybridRouter |
| Total shuffle bytes | Spark REST API: `GET /api/v1/applications/{id}/stages` — `shuffleWriteBytes` |
| CPU time | Spark executor metrics via SparkContext listener |
| Model inference time | Logged in `ModelPredictor.predict()` |
| Routing accuracy | Fraction of subexpressions where model chose the faster engine |
| Correctness | Row count + SHA256 checksum vs reference executor |

### 10.2 Experimental Plan (3 Stages)

**Stage 1 — Microbenchmarks (Decision Boundary Mapping)**

Goal: Understand at what values of `avg_degree` and `selectivity` the routing decision flips.

```python
# experiments/microbenchmark.py
for avg_degree in [2, 5, 10, 20, 50]:
    for selectivity in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
        generate_synthetic_graph(avg_degree=avg_degree)
        run_single_traversal_subexpression(selectivity=selectivity)
        record(sql_ms, graph_ms)
# Plot heatmap: avg_degree vs selectivity -> winner engine
```

**Stage 2 — End-to-End Mixed Queries**

```python
# experiments/run_learned.py
for dataset in ["tpch_sf1", "snb_interactive", "imdb", "synthetic"]:
    for query in load_queries(dataset):
        result = HybridRouter().execute(query)
        latency_ms = measure()
        shuffle_bytes = measure()
        record(dataset, query_id, "learned", latency_ms, shuffle_bytes)
```

**Stage 3 — Robustness & Online Retraining**

- Evaluate a model trained on TPC-H applied to SNB (cross-dataset generalization).
- Measure accuracy degradation on out-of-distribution queries.
- Retrain with 20% new SNB-labeled samples; measure recovery in routing accuracy.

---

## 11. Algorithm Pseudocode Reference

### 11.1 End-to-End Execution Flow

```
FUNCTION run_query(query_json):
  nodes       <- DSLParser.parse(query_json)
  sub_exprs   <- QueryDecomposer.decompose(nodes)
  results     <- {}

  PARALLEL_GROUPS <- group sub_exprs by dependency level

  FOR each group G in PARALLEL_GROUPS (in topological order):
    PARALLEL FOR each sub IN G:
      features   <- FeatureExtractor.extract(sub)
      engine     <- ModelPredictor.predict(features)   // "SQL" or "GRAPH"
      IF engine == "SQL":
        df <- SQLGenerator.generate(sub)
      ELSE:
        df <- GraphGenerator.generate(sub)
      results[sub.sub_id] <- df
      RuntimeLogger.log(sub.sub_id, engine, features, elapsed_ms)

  final_df <- ResultComposer.compose(results, nodes)
  RETURN final_df
```

### 11.2 Selectivity Estimation

```
FUNCTION compute_selectivity(sub_expr, table_stats):
  IF no FILTER node in sub_expr:
    RETURN 1.0
  filter_node <- first FILTER node
  col_stats   <- table_stats["columns"][filter_node.predicate["column"]]
  op          <- filter_node.predicate["operator"]

  IF op == "=":
    RETURN 1.0 / col_stats["distinct_count"]
  ELIF op IN [">", "<", ">=", "<="]:
    col_range <- col_stats["max"] - col_stats["min"]
    pred_val  <- filter_node.predicate["value"]
    IF op IN [">", ">="]:
      RETURN (col_stats["max"] - pred_val) / col_range
    ELSE:
      RETURN (pred_val - col_stats["min"]) / col_range
  ELIF op == "IN":
    RETURN len(filter_node.predicate["value"]) / col_stats["distinct_count"]
  ELSE:
    RETURN 0.1   // conservative default
```

---

## 12. Baseline vs. Learned Model: Experimental Instructions

### 12.1 Baseline 1 — Always SQL

```bash
python experiments/run_baselines.py \
  --strategy always_sql \
  --queries dsl/sample_queries/ \
  --output experiments/results/always_sql.csv
```

Implementation: Override `HybridRouter` to always call `SQLGenerator`, bypassing the ML predictor entirely.

### 12.2 Baseline 2 — Always Graph

```bash
python experiments/run_baselines.py \
  --strategy always_graph \
  --queries dsl/sample_queries/ \
  --output experiments/results/always_graph.csv
```

### 12.3 Baseline 3 — Rule-Based Router

```python
# Rule: TRAVERSAL nodes -> GRAPH; everything else -> SQL
def rule_based_route(sub_expr) -> str:
    return "GRAPH" if sub_expr.primary_op_type == "TRAVERSAL" else "SQL"
```

```bash
python experiments/run_baselines.py \
  --strategy rule_based \
  --queries dsl/sample_queries/ \
  --output experiments/results/rule_based.csv
```

### 12.4 Learned Model Router

```bash
# Step 1: Train (if not already done)
python model/trainer.py \
  --data training_data/labeled_runs.csv \
  --output model/artifacts/classifier_v1.pkl

# Step 2: Evaluate
python experiments/run_learned.py \
  --model model/artifacts/classifier_v1.pkl \
  --queries dsl/sample_queries/ \
  --output experiments/results/learned_routing.csv
```

### 12.5 Results Comparison

```bash
python experiments/compare_results.py \
  --results experiments/results/ \
  --metrics latency_ms shuffle_bytes routing_accuracy \
  --output experiments/results/comparison_table.csv
```

**Expected Output Table Format:**

| Strategy | Median Latency (ms) | p95 Latency (ms) | Shuffle Bytes (MB) | Routing Accuracy |
|----------|--------------------|-----------------|--------------------|-----------------|
| Always SQL | baseline | baseline | baseline | — |
| Always Graph | — | — | — | — |
| Rule-Based | — | — | — | — |
| **Learned (XGBoost)** | **target: -15%** | **target: -10%** | **target: -20%** | **target: >85%** |

### 12.6 Ablation Study

```bash
python experiments/ablation_study.py \
  --model model/artifacts/classifier_v1.pkl \
  --data training_data/labeled_runs.csv \
  --drop_features avg_degree selectivity input_cardinality_log hist_avg_runtime_ms \
  --output experiments/results/ablation.csv
```

For each feature group: retrain model with that group removed, compare CV F1 vs the full 22-feature model. Report delta accuracy per removed group to identify the most critical features.

---

## 13. Risk Register & Mitigation

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|-----------|--------|---------------------|
| Labeling cost — running both engines per subexpression is slow | High | High | Sample 500–1000 subexpressions; use SF=1 for TPC-H during label collection |
| Poor cardinality estimates degrade feature quality | Medium | High | Add histogram-based sampling; use `approx_count_distinct` in Spark |
| GraphFrames incompatibility with Spark version | Medium | Medium | Pin versions in `requirements.txt`; validate setup in Docker first |
| Correctness failures in result composition | Medium | High | Unit tests with reference executor; validate row counts + SHA256 checksums |
| Model overfits to TPC-H, fails on SNB | Medium | Medium | Stratified sampling across datasets; online retraining stage (Stage 3) |
| BFS explodes for high-degree nodes in graph traversal | Low | High | Cap `max_hops <= 3`; add vertex count limit to GraphFrames BFS call |

---

## 14. References

1. **Spyratos, N. et al.** — *HIFUN: A High-Level Functional Query Language for Big Data Analytics.* (HIFUN foundational semantics — cite in Introduction and System Design.)

2. **Yu, X. et al.** — *Cost-Based or Learning-Based? A Hybrid Query Optimizer for Query Plan Selection.* PVLDB. (Hybrid learned optimizer context — cite in Related Work.)

3. **Anneser, C. et al.** — *Learned Query Optimization for Any SQL Database (AutoSteer).* (Learned SQL optimizer — cite in Related Work.)

4. **Li, Y. et al.** — *A Learned Cost Model for Big Data Query Processing / LOOPLINE.* (Deep cost model comparison — cite in Related Work.)

5. **Ye, J. et al.** — *LEAP: A Low-Cost Spark SQL Query Optimizer.* (Spark-based optimization — directly relevant, cite in Related Work.)

---

*End of Blueprint — Version 1.0*

> **Recommended Next Steps for the Team:**
> 1. Set up environment per Phase 1. Confirm Spark + GraphFrames run locally in Docker.
> 2. Ingest TPC-H SF=1 and SNB-small per the Phase 2 ingestion scripts.
> 3. Implement `DSLParser` + `QueryDecomposer` with full unit tests before touching ML code.
> 4. Run label collection on 10 sample queries to validate the pipeline before scaling up.
> 5. Train the first model iteration and verify that inference time stays under 10ms.
> 6. Run all four routing strategies and produce the comparison table for the paper.