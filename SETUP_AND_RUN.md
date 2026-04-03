# HIFUN Router: Complete Setup and Run Guide (Linux)

This guide gives exact commands to set up and run the project end-to-end.

## 1) Prerequisites

Install OS packages (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv openjdk-17-jdk docker.io docker-compose-plugin make
```

Enable Docker for your user:

```bash
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
newgrp docker
```

## 2) Clone and Enter Project

```bash
cd ~
git clone <YOUR_REPO_URL> HIFUN-Router-clone
cd HIFUN-Router-clone
```

If the repository is already present:

```bash
cd ~/HIFUN-Router-clone
```

## 3) Python Environment Setup

```bash
cd ~/HIFUN-Router-clone
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

Set runtime environment variables for Spark + Python imports:

```bash
export PYTHONPATH="$PWD"
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
mkdir -p /tmp/spark-events
export HIFUN_HISTORY_SERVER=/tmp/spark-events
```

Verify environment:

```bash
python test_setup.py
```

## 4) Quick Start (Use Existing Included Training Artifacts)

Use this if you want to run immediately without regenerating datasets:

```bash
cd ~/HIFUN-Router-clone
source .venv/bin/activate
export PYTHONPATH="$PWD"
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
mkdir -p /tmp/spark-events
export HIFUN_HISTORY_SERVER=/tmp/spark-events

python -m model.trainer
python -m experiments.relevance_evaluation
python -m experiments.dataset_shift_evaluation --source training_data/real_labeled_runs.csv
pytest -q
```

## 5) Full Data Pipeline (Rebuild From Raw Data)

### 5.1 Generate / Convert Datasets

SNB raw generation via Docker datagen:

```bash
mkdir -p data/raw/ldbc_snb
docker run --rm \
  --mount type=bind,source="$(pwd)/data/raw/ldbc_snb",target=/out \
  ldbc/datagen-standalone:latest \
  --parallelism 1 -- --format csv --scale-factor 1 --mode raw --output-dir /out
```

Convert SNB raw CSV to parquet + graph:

```bash
python -m data.scripts.ldbc_snb_to_parquet --input data/raw/ldbc_snb --parquet-dir data/parquet/snb --graph-dir data/graphs/snb
```

Download + convert OGB graph dataset:

```bash
python -m data.scripts.ogb_to_parquet --dataset ogbn-arxiv --root data/raw/ogb --graph-dir data/graphs
```

Convert JOB/IMDB dumps (if available in data/raw/job):

```bash
python -m data.scripts.job_to_parquet --input data/raw/job --output data/parquet/job
```

Convert TPC-DS dsdgen output (if available in data/raw/tpcds):

```bash
python -m data.scripts.tpcds_to_parquet --input data/raw/tpcds --output data/parquet/tpcds
```

### 5.2 Compute Stats

```bash
python -m data.scripts.compute_stats
```

### 5.3 Generate Real Query Packs

```bash
python -m training_data.real_query_generator --scale aggressive --focus-mode all
```

Optional class-diversity focused generation:

```bash
python -m training_data.real_query_generator --scale balanced --focus-mode graph_win
python -m training_data.real_query_generator --scale balanced --focus-mode sql_win
```

### 5.4 Collect Real Runtime Labels

Standard run:

```bash
python -m training_data.real_collection_script \
  --queries_dir dsl/sample_queries \
  --output training_data/real_labeled_runs.csv \
  --n_warmup 2 --n_measure 3 --repeat 3
```

Strict real-only run with failure taxonomy report:

```bash
python -m training_data.real_collection_script \
  --queries_dir dsl/sample_queries \
  --output training_data/real_labeled_runs_strict.csv \
  --n_warmup 2 --n_measure 3 --repeat 3 \
  --strict_real_only \
  --failure_report training_data/real_collection_failures.json
```

### 5.5 Create Fixed Train/Eval Splits

```bash
python -m training_data.fix_dataset_splits --source training_data/real_labeled_runs.csv
```

### 5.6 Train Model

```bash
python -m model.trainer
```

## 6) Run Evaluation Scripts

Relevance evaluation:

```bash
python -m experiments.relevance_evaluation
```

Dataset-shift evaluation:

```bash
python -m experiments.dataset_shift_evaluation --source training_data/real_labeled_runs.csv
```

Expected outputs:

```bash
ls -lh experiments/results/relevance_eval.json experiments/results/relevance_eval.md
ls -lh experiments/results/dataset_shift_eval.json experiments/results/dataset_shift_eval.md
```

## 7) Run Router / Tests

Run all tests:

```bash
pytest -q
```

Run correctness-focused test suites:

```bash
pytest tests/test_execution.py -v
pytest tests/test_correctness.py -v
pytest tests/test_experiments.py -v
```

Run one sample query through the router using Makefile helper:

```bash
make run-query QUERY=dsl/sample_queries/tpch_queries.json
```

Run complete synthetic query pack:

```bash
make run-synthetic
```

## 8) Optional: Start Docker Big-Data Stack

Start all services:

```bash
docker compose up -d
docker compose ps
```

Watch Spark master logs:

```bash
docker compose logs -f spark-master
```

Stop services:

```bash
docker compose down
```

Useful UIs:

- Spark Master: http://localhost:8080
- Spark Worker 1: http://localhost:8081
- Spark Worker 2: http://localhost:8082
- Spark History Server: http://localhost:18080
- NameNode: http://localhost:9870
- YARN Resource Manager: http://localhost:8088

## 9) Makefile Command Reference

Show all commands:

```bash
make help
```

Common shortcuts:

```bash
make setup
make test-env
make data-all
make train
make evaluate
make test-all
```

## 10) Clean Generated Outputs

```bash
make clean
```

## 11) Troubleshooting

If you see `JAVA_HOME is not set`:

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
```

If you see `ModuleNotFoundError: No module named 'config'`:

```bash
cd ~/HIFUN-Router-clone
export PYTHONPATH="$PWD"
```

If Spark fails while downloading packages on first run, retry:

```bash
python test_setup.py
```
