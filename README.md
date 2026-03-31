## HIFUN Router

Hybrid query router for selecting SQL vs GRAPH execution paths from DSL sub-expressions.

This repository now supports a real-data-first training workflow with scalable query generation and measured runtime labels.

## Current Status (Phase 2)

- Real dataset ingestion scripts added for:
	- LDBC SNB (interactive raw data, parquet conversion, graph extraction)
	- OGB (ogbn-arxiv conversion to GraphFrames parquet)
	- JOB/IMDB (CSV/TSV to parquet)
	- TPC-DS (dsdgen .dat to parquet)
- Real query pack generator added with rigorous variants:
	- SNB traversal + mixed graph/relational workloads
	- SNB BI-style analytical templates
	- OGB traversal workloads
	- JOB multi-join SQL templates
	- TPC-DS analytical SQL templates
- Real label collection supports repeat runs and source availability checks.
- Large generated files are excluded from git (raw/parquet/graph outputs and runtime artifacts).

## Repository Structure

- `data/scripts/`: dataset download and conversion scripts
- `dsl/sample_queries/`: DSL query packs (templates + generated real variants)
- `training_data/`: query generation and real runtime label collection
- `features/`: feature extraction and statistics loaders
- `execution/`: Spark SQL and GraphFrames execution generators
- `model/`: trainer, predictor, and artifacts
- `experiments/`: evaluation and baseline experiments

## Datasets Used and How to Access

### 1) LDBC SNB (recommended primary graph+mixed workload)

- Why: strongest alignment with router features (`input_cardinality_log`, `avg_degree`, `max_degree`, `degree_skew`).
- Access path in project:
	- Raw output: `data/raw/ldbc_snb/`
	- Converted tables: `data/parquet/snb/`
	- Graph parquet: `data/graphs/snb/`
- Generation source:
	- Docker image: `ldbc/datagen-standalone:latest` (used in this workflow)

### 2) OGB (real graph topology)

- Why: natural power-law-like graph structures for GRAPH-class robustness.
- Supported dataset now: `ogbn-arxiv`
- Access path in project:
	- Download cache: `data/raw/ogb/`
	- Graph parquet: `data/graphs/ogbn_arxiv/`
- Source:
	- OGB Python package (`ogb`)

### 3) JOB / IMDB (real SQL-heavy joins)

- Why: realistic join fanout and multi-table SQL complexity.
- Input expected:
	- Place JOB table CSV/TSV files under `data/raw/job/`
- Converted output:
	- `data/parquet/job/`
- Source:
	- https://github.com/gregrahn/join-order-benchmark

### 4) TPC-DS (optional SQL diversity beyond TPC-H)

- Why: larger query variety and more dimensional analytical patterns.
- Input expected:
	- dsdgen outputs under `data/raw/tpcds/`
- Converted output:
	- `data/parquet/tpcds/`
- Source:
	- https://github.com/gregrahn/tpcds-kit

## Setup

### System prerequisites

- Docker (daemon running)
- Java 17 for Spark runtime
- Python 3.12+

### Python dependencies

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
```

If Java is unavailable system-wide, a user-space Java can be installed with:

```bash
python3 -m pip install --user --break-system-packages jdk4py==17.0.9.2
export JAVA_HOME="$(python3 - << 'PY'
import jdk4py
print(str(jdk4py.JAVA_HOME))
PY
)"
export PATH="$JAVA_HOME/bin:$PATH"
mkdir -p /tmp/spark-events
```

## Phase-by-Phase Pipeline

### Phase 1: Generate/convert real datasets

#### SNB raw generation

```bash
mkdir -p data/raw/ldbc_snb
sg docker -c 'docker run --rm \
	--mount type=bind,source="'"$(pwd)"'"/data/raw/ldbc_snb,target=/out \
	ldbc/datagen-standalone:latest \
	--parallelism 1 -- --format csv --scale-factor 1 --mode raw --output-dir /out'
```

#### SNB conversion to project schema/parquet

```bash
python3 data/scripts/ldbc_snb_to_parquet.py --input data/raw/ldbc_snb
```

#### OGB conversion

```bash
python3 data/scripts/ogb_to_parquet.py --dataset ogbn-arxiv
```

#### Optional JOB conversion

```bash
python3 data/scripts/job_to_parquet.py --input data/raw/job --output data/parquet/job
```

#### Optional TPC-DS conversion

```bash
python3 data/scripts/tpcds_to_parquet.py --input data/raw/tpcds --output data/parquet/tpcds
```

### Phase 2: Compute stats

```bash
python3 data/scripts/compute_stats.py
```

### Phase 3: Generate rigorous query packs and collect real labels

```bash
python3 training_data/real_query_generator.py --scale aggressive

python3 training_data/real_collection_script.py \
	--queries_dir dsl/sample_queries \
	--output training_data/real_labeled_runs.csv \
	--n_warmup 2 --n_measure 3 --repeat 3
```

### Phase 4: Retrain model

```bash
python3 model/trainer.py --labeled_data_path training_data/real_labeled_runs.csv
```

### Phase 5: Verification

```bash
python3 -m pytest -q
```

## Notes on Git and Large Files

- Generated data under `data/raw/`, `data/parquet/`, and parquet graph outputs are intentionally ignored.
- Runtime/training artifacts are also ignored (for example, `training_data/real_labeled_runs.csv`, model artifact binaries).
- Keep only scripts, configs, and query templates in git; regenerate data locally using this README.

## Key Scripts Added/Updated

- `data/scripts/ldbc_snb_to_parquet.py`
- `data/scripts/ogb_to_parquet.py`
- `data/scripts/job_to_parquet.py`
- `data/scripts/tpcds_to_parquet.py`
- `data/scripts/download_real_datasets.sh`
- `training_data/real_query_generator.py`
- `training_data/real_collection_script.py`
