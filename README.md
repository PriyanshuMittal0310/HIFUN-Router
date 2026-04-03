## HIFUN Router

Hybrid query router for selecting SQL vs GRAPH execution paths from DSL sub-expressions.

This repository now supports a real-data-first training workflow with scalable query generation and measured runtime labels.

## Current Status (April 2026)

### Publishable Strict Update (latest)

- Strict curated dataset created and quality-gated:
	- `training_data/real_labeled_runs_strict_curated.csv`
	- 288 rows total, `SQL` 151, `GRAPH` 137
	- GRAPH coverage across 2 datasets (`snb_real_queries`, `ogb_real_queries`)
	- All rows are real measurements (`label_source=real_measurement`)
- Strict fixed splits created:
	- `training_data/fixed_train_base_strict.csv`
	- `training_data/fixed_eval_set_strict.csv`
	- `training_data/fixed_eval_graph_only_strict.csv`
	- `training_data/fixed_split_manifest_strict.json`
	- strict split generation now uses query-disjoint mode (`split_mode=group`, `group_col=query_id`) to avoid train/eval query overlap
- Strict quality gate passes:
	- `training_data/dataset_quality_report_strict_curated.json`
- Strict publishable evaluation artifacts generated:
	- `experiments/results/relevance_eval_strict.json`
	- `experiments/results/relevance_eval_strict.md`
	- `experiments/results/ablation_strict.csv`
	- `experiments/results/ablation_strict_groups.csv`
	- `experiments/results/ablation_strict.json`
	- `experiments/results/strict_robustness_eval.json`
	- `experiments/results/strict_robustness_eval.md`

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

### Progress Summary

- Phase 1 complete: real data pipeline available for SNB, OGB, JOB, and TPC-DS.
- Phase 2 complete: statistics generation generalized for all discovered parquet/graph datasets.
- Phase 3 complete: real labeled runtime dataset generated from all configured real query packs.
- Class balancing utility applied and a balanced training CSV generated.
- Repository pushed with code/query/stat updates required for the above workflow.

### Latest Dataset Snapshot

- Primary measured labels:
	- `training_data/real_labeled_runs.csv`
	- Current composition: 738 rows
	- Datasets: `snb_real_queries` (432), `ogb_real_queries` (176), `snb_bi_real_queries` (96), `job_real_queries` (18), `tpcds_real_queries` (16)
	- Labels: `SQL` 734, `GRAPH` 4

- Balanced training labels (diagnostic only, not recommended for headline metrics):
	- `training_data/real_labeled_runs_balanced.csv`
	- Current composition: 1468 rows
	- Labels: `SQL` 734, `GRAPH` 734
	- Includes `resampled` column (`0` original row, `1` upsampled row)
	- Use only for debugging model plumbing; this file duplicates scarce GRAPH rows and can overfit.

- Balanced dataset summary:
	- `training_data/real_labeled_runs_balanced_summary.txt`

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

## One-Command Strict Runner (recommended)

If your workspace contains modified non-strict split artifacts, use the strict runner to avoid default-path failures:

```bash
./run_project_strict.sh all
```

Useful modes:

- `./run_project_strict.sh smoke` (fast validation: quality + relevance + robustness)
- `./run_project_strict.sh train`
- `./run_project_strict.sh relevance`
- `./run_project_strict.sh robustness`

This script pins strict inputs explicitly:

- `training_data/real_labeled_runs_strict_curated.csv`
- `training_data/fixed_train_base_strict.csv`
- `training_data/fixed_eval_set_strict.csv`

### System prerequisites

- Docker (daemon running)
- Java 17 for Spark runtime
- Python 3.12+

### Python dependencies

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
```

If `python3 -m venv` is unavailable, use:

```bash
python3 -m pip install --user --break-system-packages virtualenv
~/.local/bin/virtualenv .venv
. .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
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

# Optional: targeted workload mining for class diversity
python3 training_data/real_query_generator.py --scale balanced --focus-mode graph_win
python3 training_data/real_query_generator.py --scale balanced --focus-mode sql_win

# If needed, include SQL-heavy families even in graph focus mode
# python3 training_data/real_query_generator.py --scale balanced --focus-mode graph_win --include-sql-families-in-graph-focus

python3 training_data/real_collection_script.py \
	--queries_dir dsl/sample_queries \
	--output training_data/real_labeled_runs.csv \
	--n_warmup 2 --n_measure 3 --repeat 3

# Quality controls for strict real-only labels and failure taxonomy
python3 training_data/real_collection_script.py \
	--queries_dir dsl/sample_queries \
	--output training_data/real_labeled_runs_strict.csv \
	--strict_real_only \
	--failure_report training_data/real_collection_failures.json

# Create deterministic train/eval artifacts (leakage-safe balancing)
python3 training_data/fix_dataset_splits.py \
  --source training_data/real_labeled_runs.csv

# Optional debug-only mode for highly imbalanced early datasets
# python3 training_data/fix_dataset_splits.py --source training_data/real_labeled_runs.csv --allow_degenerate
```

### Phase 4: Retrain model

```bash
python3 model/trainer.py
```

Trainer defaults now prefer non-resampled fixed split artifacts:

1. `training_data/fixed_train_base.csv`
2. `training_data/fixed_train_balanced.csv`
3. `training_data/real_labeled_runs.csv`
4. `training_data/real_labeled_runs_balanced.csv`
5. `training_data/labeled_runs.csv`

### Phase 5: Verification

```bash
python3 -m pytest -q
```

### Phase 6: Relevance and Robustness Evaluation (new)

Use this step to benchmark learned routing against stronger baselines and a no-history ablation.

```bash
python3 experiments/relevance_evaluation.py
```

By default this uses:

- Train: `training_data/fixed_train_base.csv`
- Eval: `training_data/fixed_eval_set.csv` (or `training_data/fixed_eval_graph_only.csv` for GRAPH-focused stress tests)

Outputs:

- `experiments/results/relevance_eval.json`
- `experiments/results/relevance_eval.md`

This report includes:

- `AlwaysSQL`, `AlwaysGRAPH`, traversal rule, threshold baseline
- Logistic Regression (class-balanced)
- Decision Tree (class-balanced)
- XGBoost (class-balanced)
- `XGBoostNoHistory` ablation to detect leakage from historical runtime features
- PR-AUC and Brier calibration metrics for probabilistic models
- confidence-threshold sweep (coverage vs. quality)
- per-dataset confusion matrices for XGBoost

### Phase 7: Dataset-Shift Generalization (new)

Measure cross-dataset transfer quality (train on one dataset family, evaluate on another).

```bash
python3 experiments/dataset_shift_evaluation.py \
	--source training_data/real_labeled_runs.csv
```

Modes included in the report:

- `one_to_one`: train on one dataset family, evaluate on another
- `leave_one_out`: train on all datasets except one held-out dataset
- `grouped_domains`: train/evaluate across graph/mixed/sql domain groups

Outputs:

- `experiments/results/dataset_shift_eval.json`
- `experiments/results/dataset_shift_eval.md`

### Phase 8: Strict Robustness Evaluation (new)

Run confidence intervals, permutation-based sanity checks, and strict cross-dataset transfer from curated real labels:

```bash
python3 experiments/strict_robustness_evaluation.py \
	--train training_data/fixed_train_base_strict.csv \
	--eval training_data/fixed_eval_set_strict.csv \
	--transfer_source training_data/real_labeled_runs_strict_curated.csv
```

Outputs:

- `experiments/results/strict_robustness_eval.json`
- `experiments/results/strict_robustness_eval.md`

This report includes:

- bootstrap 95% confidence intervals for strict eval metrics
- label-permutation sanity distribution
- permutation importance (eval F1 drop per feature)
- cross-dataset transfer matrix on strict curated real labels

### Training Improvements (new)

- Trainer now prefers non-resampled datasets by default in this order:
	1. `training_data/fixed_train_base.csv`
	2. `training_data/fixed_train_balanced.csv`
	3. `training_data/real_labeled_runs.csv`
	4. `training_data/real_labeled_runs_balanced.csv`
	5. `training_data/labeled_runs.csv`
- Trainer now applies imbalance-aware learning by default:
	- Decision Tree uses `class_weight=balanced`
	- XGBoost uses `scale_pos_weight`
- Trainer now reports a stratified holdout evaluation split in addition to CV metrics.
- Split/training/evaluation scripts now fail fast on degenerate GRAPH-class support unless `--allow_degenerate` is provided.

### Validity Guardrails (critical)

- Do not report headline model quality when eval has only a handful of GRAPH rows.
- Default thresholds now require substantial GRAPH support in both train and eval.
- Upsampled balanced datasets are kept for diagnostics, not for defensible model claims.

Run the mandatory quality gate before any paper-facing metrics:

```bash
python3 -m training_data.dataset_quality_gate \
	--source training_data/real_labeled_runs.csv \
	--train training_data/fixed_train_base.csv \
	--eval training_data/fixed_eval_set.csv
```

Or via Makefile:

```bash
make quality-gate
make publish-eval
```

If quality gate fails, treat all downstream metrics as debug-only.

For strict publishable flow, run:

```bash
python3 -m training_data.fix_dataset_splits \
	--source training_data/real_labeled_runs_strict_curated.csv \
	--train_base_out training_data/fixed_train_base_strict.csv \
	--eval_out training_data/fixed_eval_set_strict.csv \
	--graph_eval_out training_data/fixed_eval_graph_only_strict.csv \
	--train_balanced_out training_data/fixed_train_balanced_strict.csv \
	--manifest_out training_data/fixed_split_manifest_strict.json \
	--allow_degenerate

python3 -m training_data.dataset_quality_gate \
	--source training_data/real_labeled_runs_strict_curated.csv \
	--train training_data/fixed_train_base_strict.csv \
	--eval training_data/fixed_eval_set_strict.csv \
	--out_json training_data/dataset_quality_report_strict_curated.json

python3 model/trainer.py \
	--data training_data/fixed_train_base_strict.csv \
	--model_out model/artifacts/classifier_strict.pkl \
	--min_graph_rows 100

python3 experiments/relevance_evaluation.py \
	--train training_data/fixed_train_base_strict.csv \
	--eval training_data/fixed_eval_set_strict.csv \
	--out_json experiments/results/relevance_eval_strict.json \
	--out_md experiments/results/relevance_eval_strict.md

python3 experiments/ablation_study.py \
	--data training_data/fixed_train_base_strict.csv \
	--output experiments/results/ablation_strict.csv
```

## Docker Snapshot

### Build image

```bash
docker build -t hifun-router:strict-latest .
```

### What is included in the Docker image

- Included:
	- Repository-tracked project code
	- Strict curated dataset and strict reports, including:
		- `training_data/real_labeled_runs_strict_curated.csv`
		- `training_data/dataset_quality_report_strict_curated.json`
		- `experiments/results/relevance_eval_strict.json`
		- `experiments/results/ablation_strict.json`
- Excluded (by `.dockerignore`):
	- `data/raw/`
	- `data/parquet/`
	- `data/graphs/`
	- temporary query folders under `training_data/tmp_queries*/`
	- debug result files (`*_debug.*`)

### Run image

```bash
docker run --rm hifun-router:strict-latest
```

### Share Docker image (offline)

```bash
docker save hifun-router:strict-latest -o hifun-router-strict-latest.tar
gzip -9 hifun-router-strict-latest.tar
```

Receiver side:

```bash
gunzip hifun-router-strict-latest.tar.gz
docker load -i hifun-router-strict-latest.tar
docker run --rm hifun-router:strict-latest
```

### Share Docker image (registry)

```bash
docker tag hifun-router:strict-latest <registry-user>/hifun-router:strict-latest
docker login
docker push <registry-user>/hifun-router:strict-latest
```

### Simulation Transparency

- Some older results in this repository were generated from heuristic/cost-model labels.
- Use explicit language in reports when presenting simulated timing or simulated labels.
- Treat simulation outputs as pipeline validation, not final evidence of engine-level performance.

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
