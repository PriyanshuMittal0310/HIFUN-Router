# HIFUN Router — Hybrid SQL/Graph Query Routing System

> A dynamic query routing framework that parses structured DSL queries, decomposes them into dependency-aware DAGs, extracts a 22-dimensional feature vector, and routes execution to either SQL (Spark SQL / pandas) or Graph (GraphFrames) backends.

📂 **Repository:** [https://github.com/DataScience-ArtificialIntelligence/HybridSQLGraphQueryRoutingSystem](https://github.com/DataScience-ArtificialIntelligence/HybridSQLGraphQueryRoutingSystem)  
🎥 **2-Minute Demo Video:** [https://drive.google.com/file/d/1PmTvewqPUMJ_vOrPGdnUg_BhPUDZGSv3/view?usp=sharing](https://drive.google.com/file/d/1PmTvewqPUMJ_vOrPGdnUg_BhPUDZGSv3/view?usp=sharing) 
---

## Current Status (April 2026)

- Strict curated source is available and quality-gated:
    - `training_data/real_labeled_runs_strict_curated.csv`
    - `training_data/fixed_train_base_strict.csv`
    - `training_data/fixed_eval_set_strict.csv`
    - `training_data/dataset_quality_report_strict_runtime.json`
- Publishable strict runtime artifacts are generated under `experiments/results/`:
    - `relevance_eval_strict_runtime.json`
    - `dataset_shift_eval_strict_runtime.json`
    - `strict_robustness_eval_runtime.json`
    - `ablation_strict_runtime.json`
- Dashboard supports:
    - `strict` profile for publication-grade artifacts
    - `fast` profile for quick iteration artifacts (`*_fast_runtime.*`)

---

## 👥 Team Members

| Name | Department | Roll No. | Institute | Email |
|------|-----------|----------|-----------|-------|
| Piyush Prashant | Data Science & AI | 24BDS055 | IIIT Dharwad | 24bds055@iiitdwd.ac.in |
| Priyanshu Mittal | Data Science & AI | 24BDS058 | IIIT Dharwad | 24bds058@iiitdwd.ac.in |
| Harshitha M S | Data Science & AI | 24BDS038 | IIIT Dharwad | 24bds038@iiitdwd.ac.in |
| J. Sameer Karthikeya | Data Science & AI | 24BDS026 | IIIT Dharwad | 24bds026@iiitdwd.ac.in |

---

## 📁 Repository Structure

```
HIFUN-Router/
│
├── config/                          # Configuration files
│   ├── feature_schema.json          # 22-dimensional feature schema definition
│   ├── paths.py                     # Filesystem path constants
│   └── spark_config.py              # PySpark session & cluster configuration
│
├── data/
│   ├── graphs/                      # Pre-built graph parquet files (SNB, synthetic)
│   │   ├── snb/                     # LDBC Social Network Benchmark graph
│   │   │   ├── vertices.parquet
│   │   │   └── edges.parquet
│   │   └── synthetic/               # Synthetically generated graph data
│   ├── scripts/                     # Dataset ingestion & conversion scripts
│   │   ├── ldbc_snb_to_parquet.py
│   │   ├── ogb_to_parquet.py
│   │   ├── job_to_parquet.py
│   │   ├── tpcds_to_parquet.py
│   │   ├── generate_synthetic.py
│   │   └── compute_stats.py
│   └── stats/                       # JSON column statistics per table
│
├── dsl/
│   └── sample_queries/              # Example JSON DSL query files (TPC-H, SNB, etc.)
│
├── experiments/                     # Evaluation & benchmarking scripts
│   ├── relevance_evaluation.py      # Model vs baseline accuracy/F1
│   ├── dataset_shift_evaluation.py  # Cross-family domain shift analysis
│   ├── strict_robustness_evaluation.py  # Bootstrap CI + permutation tests
│   └── results/                     # JSON/MD output artifacts from evaluations
│
├── features/
│   └── feature_extractor.py         # Builds the R^22 feature vector per routing node
│
├── model/
│   ├── trainer.py                   # XGBoost/LogReg model training & cross-validation
│   ├── predictor.py                 # Inference wrapper for routing-time prediction
│   ├── feature_analysis.py          # VIF, correlation, collinearity analysis
│   ├── feature_importance.py        # SHAP-based feature importance plots
│   └── artifacts/                   # Saved model files & analysis outputs
│       ├── classifier_v1.pkl
│       ├── classifier_v1_dt.pkl
│       ├── feature_schema_v1.json
│       ├── training_results.json
│       └── analysis/                # SHAP bar/summary plots + values
│
├── notebooks/                       # Jupyter exploration & reporting notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_results_visualization.ipynb
│   └── project_report.ipynb         # Full project report notebook
│
├── parser/
│   ├── dsl_parser.py                # JSON DSL → AST with jsonschema validation
│   └── ast_nodes.py                 # Typed operation node representations
│
├── report/
│   ├── hifun_router_ieee_report.tex # LaTeX source for the IEEE-format paper
│   └── decision_boundary.pdf        # Decision boundary visualization
│
├── router/
│   ├── hybrid_router.py             # Core routing engine (rule + ML hybrid)
│   └── baselines.py                 # AlwaysSQL, AlwaysGRAPH, TraversalRule baselines
│
├── execution/
│   ├── sql_generator.py             # Translates plan nodes → Spark SQL / pandas
│   ├── graph_generator.py           # Translates plan nodes → GraphFrames operations
│   └── result_composer.py           # Merges cross-engine outputs into final result
│
├── training_data/                   # Curated train/eval CSV splits & generation scripts
│   ├── fixed_train_balanced_strict.csv
│   ├── fixed_eval_set_strict.csv
│   ├── fixed_split_manifest_strict.json
│   ├── real_labeled_runs_strict_curated.csv
│   ├── query_generator.py
│   ├── real_collection_script.py
│   ├── real_query_generator.py
│   ├── fix_dataset_splits.py
│   └── dataset_quality_gate.py
│
├── tests/                           # Unit + integration test suite (pytest)
│   ├── test_parser.py
│   ├── test_decomposer.py
│   ├── test_features.py
│   ├── test_model.py
│   ├── test_execution.py
│   ├── test_correctness.py
│   ├── test_experiments.py
│   └── reference_executor.py
│
├── decomposer/
│   └── query_decomposer.py          # Builds dependency DAG + schedulable blocks
│
├── streamlit_app.py                 # Interactive evaluation dashboard (demo UI)
├── test_setup.py                    # Environment sanity check
├── run_project_strict.sh            # End-to-end strict pipeline runner
├── Dockerfile                       # Container image definition
├── Makefile                         # Convenience command shortcuts
├── requirements.txt                 # Python dependency list
├── SETUP_AND_RUN.md                 # Detailed Linux setup guide
└── RUN_STREAMLIT_DEMO.md            # Demo recording script and guide
```

---

## ⚙️ Installation & Setup

### Prerequisites

Install system packages (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv openjdk-17-jdk \
                   docker.io docker-compose-plugin make
```

Enable Docker for your user:

```bash
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
newgrp docker
```

### 1. Clone the Repository

```bash
cd ~
git clone https://github.com/PriyanshuMittal0310/HIFUN-Router.git HIFUN-Router-clone
cd HIFUN-Router-clone
```

### 2. Create Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
export PYTHONPATH="$PWD"
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
mkdir -p /tmp/spark-events
export HIFUN_HISTORY_SERVER=/tmp/spark-events
```

### 4. Verify Environment

```bash
python test_setup.py
```

A successful run prints confirmation that Spark, GraphFrames, and the ML stack are importable.

---

## 🚀 Running the Project

### Quick Start (Recommended Strict Runner)

Use this for reproducible strict artifacts without manually chaining commands:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"
./run_project_strict.sh smoke
```

Use this for the full strict bundle:

```bash
./run_project_strict.sh all
```

### Quick Start (Manual)

If you want to run step-by-step manually:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"

python -m model.trainer
python -m experiments.relevance_evaluation
python -m experiments.dataset_shift_evaluation \
    --source training_data/real_labeled_runs_strict_curated.csv
pytest -q
```

---

### Full Data Pipeline (Rebuild from Raw Data)

#### Step 1 — Generate Datasets

```bash
# LDBC SNB via Docker datagen
mkdir -p data/raw/ldbc_snb
docker run --rm \
  --mount type=bind,source="$(pwd)/data/raw/ldbc_snb",target=/out \
  ldbc/datagen-standalone:latest \
  --parallelism 1 -- --format csv --scale-factor 1 --mode raw --output-dir /out

python -m data.scripts.ldbc_snb_to_parquet \
    --input data/raw/ldbc_snb \
    --parquet-dir data/parquet/snb \
    --graph-dir data/graphs/snb

# OGB graph dataset
python -m data.scripts.ogb_to_parquet \
    --dataset ogbn-arxiv --root data/raw/ogb --graph-dir data/graphs

# JOB/IMDB (if available)
python -m data.scripts.job_to_parquet \
    --input data/raw/job --output data/parquet/job

# TPC-DS (if available)
python -m data.scripts.tpcds_to_parquet \
    --input data/raw/tpcds --output data/parquet/tpcds
```

#### Step 2 — Compute Column Statistics

```bash
python -m data.scripts.compute_stats
```

#### Step 3 — Generate Query Workloads

```bash
python -m training_data.real_query_generator --scale aggressive --focus-mode all
```

#### Step 4 — Collect Runtime Labels

```bash
python -m training_data.real_collection_script \
  --queries_dir dsl/sample_queries \
  --output training_data/real_labeled_runs.csv \
  --n_warmup 2 --n_measure 3 --repeat 3
```

#### Step 5 — Create Train/Eval Splits

```bash
python -m training_data.fix_dataset_splits \
    --source training_data/real_labeled_runs.csv
```

#### Step 6 — Train the Router Model

```bash
python -m model.trainer
```

---

### Running Evaluations

```bash
# Relevance evaluation (accuracy, F1, PR-AUC)
python -m experiments.relevance_evaluation

# Domain shift evaluation (strict curated source)
python -m experiments.dataset_shift_evaluation \
    --source training_data/real_labeled_runs_strict_curated.csv

# Strict robustness (bootstrap CI + permutation tests)
python -m experiments.strict_robustness_evaluation \
    --train training_data/fixed_train_base_strict.csv \
    --eval training_data/fixed_eval_set_strict.csv \
    --transfer_source training_data/real_labeled_runs_strict_curated.csv \
    --n_bootstrap 1000 --n_perm_labels 100 --n_perm_features 20 \
    --out_json experiments/results/strict_robustness_eval_runtime.json \
    --out_md experiments/results/strict_robustness_eval_runtime.md
```

For broad all-data transfer analysis, you can still run:

```bash
python -m experiments.dataset_shift_evaluation \
    --source training_data/real_labeled_runs.csv
```

---

### Running Tests

```bash
# All tests
pytest -q

# Specific suites
pytest tests/test_execution.py -v
pytest tests/test_correctness.py -v
pytest tests/test_experiments.py -v
```

---

### Makefile Shortcuts

```bash
make help          # Show all available commands
make setup         # Set up environment
make test-env      # Verify environment
make data-all      # Run full dataset pipeline
make train         # Train routing model
make evaluate      # Run all evaluations
make test-all      # Run full test suite
make clean         # Remove generated outputs
```

---

### Optional: Docker Big-Data Stack (Spark + HDFS + YARN)

```bash
# Start all services
docker compose up -d
docker compose ps

# Monitor Spark master
docker compose logs -f spark-master

# Stop services
docker compose down
```

Service UIs once running:

| Service | URL |
|---------|-----|
| Spark Master | http://localhost:8080 |
| Spark Worker 1 | http://localhost:8081 |
| Spark Worker 2 | http://localhost:8082 |
| Spark History Server | http://localhost:18080 |
| HDFS NameNode | http://localhost:9870 |
| YARN Resource Manager | http://localhost:8088 |

---

## 🖥️ Launching the Demo Dashboard

The project includes an interactive Streamlit dashboard for exploring routing decisions and evaluation metrics.

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"

streamlit run streamlit_app.py --server.port 8501 --server.headless true
```

Open your browser at: **http://localhost:8501**

The dashboard features four tabs: **Dataset and Quality**, **Relevance Evaluation**, **Robustness Evaluation**, and **Cross-Dataset Generalization**. Use the sidebar to switch between `fast` (rapid iteration) and `strict` (publication-grade) run profiles.

---

## 🎥 2-Minute Demo Video

🔗 **Demo Link:** [https://drive.google.com/file/d/1PmTvewqPUMJ_vOrPGdnUg_BhPUDZGSv3/view?usp=sharing](https://drive.google.com/file/d/1PmTvewqPUMJ_vOrPGdnUg_BhPUDZGSv3/view?usp=sharing)

Also available:
- 4-minute detailed recording script: RUN_STREAMLIT_DEMO.md

---

## 📊 Key Results

| Policy | Accuracy | F1 | Precision | Recall |
|--------|----------|----|-----------|--------|
| AlwaysSQL | 0.5345 | 0.0000 | 0.0000 | 0.0000 |
| AlwaysGRAPH | 0.4655 | 0.6353 | 0.4655 | 1.0000 |
| **TraversalRule** | **0.9655** | **0.9643** | **0.9310** | **1.0000** |
| LogReg (Balanced) | 0.9655 | 0.9643 | 0.9310 | 1.0000 |
| XGBoost (Balanced) | 0.9655 | 0.9643 | 0.9310 | 1.0000 |

Evaluated on a strict template-disjoint split of **N=58** evaluation cases. Bootstrap 95% CI (F1): **[0.913, 1.000]**. Top permutation-importance feature: `op_count_traversal`.

See also:
- `experiments/results/relevance_eval_strict_runtime.json`
- `experiments/results/strict_robustness_eval_runtime.json`
- `experiments/results/dataset_shift_eval_strict_runtime.json`

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Query Parsing | Python, jsonschema |
| Distributed Compute | Apache Spark 3.4.2, PySpark |
| Graph Engine | GraphFrames 0.8.3 |
| Storage | Apache Parquet, HDFS |
| Cluster Management | Apache YARN |
| ML Models | XGBoost 2.0.3, LightGBM 4.2.0, scikit-learn 1.4.0 |
| Explainability | SHAP 0.44.0 |
| Dashboard | Streamlit 1.44.1 |
| Notebooks | JupyterLab 4.1.2 |
| Containerization | Docker Compose |
| Testing | pytest 7.4.4 |

---

## 🩺 Troubleshooting

**`JAVA_HOME is not set`**
```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
```

**`ModuleNotFoundError: No module named 'config'`**
```bash
cd ~/HIFUN-Router-clone
export PYTHONPATH="$PWD"
```

**Spark fails downloading packages on first run**
```bash
python test_setup.py   # retry — Spark caches JARs on second run
```

---

## 📚 References

1. L. Libkin et al., "HIFUN: A Framework for Hybrid Query Processing," *Journal of Intelligent Information Systems*, 2022.
2. M. Armbrust et al., "Spark SQL: Relational Data Processing in Spark," *SIGMOD*, 2015.
3. A. Dave et al., "GraphFrames: An Integrated API for Mixing Graph and Relational Queries," *GRADES*, 2016.
4. R. Marcus et al., "Neo: A Learned Query Optimizer," *VLDB*, 2019.
5. W. Hu et al., "Open Graph Benchmark: Datasets for Machine Learning on Graphs," *NeurIPS*, 2020.
6. LDBC Council, "LDBC Social Network Benchmark," 2015.

---

*IIIT Dharwad — Department of Data Science and AI — BDA Project, 2025-2026*