.PHONY: help setup data-tpch data-snb data-synthetic data-stats data-all validate-queries test clean collect-data train analyze report

PYTHON := python
SPARK_SUBMIT := spark-submit
VENV := source/bin/activate

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Phase 1: Setup ---

setup:  ## Install Python dependencies
	pip install -r requirements.txt

test-env:  ## Verify Spark + GraphFrames environment
	$(PYTHON) test_setup.py

# --- Phase 2: Data Preparation ---

data-tpch:  ## Convert TPC-H raw .tbl files to Parquet
	$(PYTHON) data/scripts/tpch_to_parquet.py \
		--input data/raw/tpch-kit/dbgen \
		--output data/parquet/tpch/

data-snb:  ## Ingest LDBC SNB data (or generate synthetic SNB if raw unavailable)
	$(PYTHON) data/scripts/snb_to_parquet.py \
		--input data/raw/snb \
		--tables data/parquet/snb/ \
		--edges data/graphs/snb_edges.parquet

data-synthetic:  ## Generate synthetic power-law graphs (default: avg_degree=10 only)
	$(PYTHON) data/scripts/generate_synthetic.py --default-only

data-synthetic-all:  ## Generate synthetic graphs for all degree variants [2,5,10,20,50]
	$(PYTHON) data/scripts/generate_synthetic.py --degrees 2,5,10,20,50

data-stats:  ## Precompute table and graph statistics
	$(PYTHON) data/scripts/compute_stats.py

data-all: data-tpch data-snb data-synthetic data-stats  ## Run full data preparation pipeline
	@echo "All data preparation complete."

validate-queries:  ## Validate all sample DSL queries against schema
	$(PYTHON) -c "\
import json, sys; \
sys.path.insert(0, '.'); \
from dsl.validator import validate_query_file; \
files = ['dsl/sample_queries/tpch_queries.json', 'dsl/sample_queries/snb_queries.json', 'dsl/sample_queries/synthetic_queries.json']; \
ok = True; \
[print(f'Validating {f}...') or (lambda r: (print(f'  ✅ Valid') if r['valid'] else (print(f'  ❌ Errors: {r[\"errors\"]}'), setattr(sys.modules[__name__], '_ok', False))))(validate_query_file(f)) for f in files]; \
print('All queries validated.' if ok else 'Some queries have errors.')"

# --- Phase 3+ (placeholders) ---

test:  ## Run all tests
	$(PYTHON) -m pytest tests/ -v

# --- Phase 4: ML Pipeline ---

collect-data:  ## Generate labeled training data from DSL queries
	$(PYTHON) -m training_data.collection_script

train: collect-data  ## Train ML classifier (Decision Tree + XGBoost)
	$(PYTHON) -m model.trainer

analyze: train  ## Run SHAP analysis and feature importance
	$(PYTHON) -m model.feature_importance

report:  ## Open project report notebook
	jupyter notebook notebooks/project_report.ipynb

# --- Phase 5: Execution Engine Integration ---

test-execution:  ## Run execution engine unit tests
	$(PYTHON) -m pytest tests/test_execution.py -v

test-correctness:  ## Run correctness tests (HybridRouter vs ReferenceExecutor)
	$(PYTHON) -m pytest tests/test_correctness.py -v

run-query:  ## Run a single HIFUN query file via HybridRouter (QUERY=path/to/query.json)
	$(PYTHON) -c "\
import json, sys; \
sys.path.insert(0, '.'); \
from router.hybrid_router import HybridRouter; \
router = HybridRouter(); \
with open('$(QUERY)') as f: queries = json.load(f); \
q = queries[0] if isinstance(queries, list) else queries; \
result = router.execute_query(q); \
print('Routing decisions:'); \
[print(f'  {d[\"sub_id\"]}: {d[\"engine\"]} (conf={d[\"confidence\"]:.3f})') for d in result['routing_decisions']]; \
print(f'Total time: {result[\"total_time_ms\"]:.1f}ms'); \
print(f'Result shape: {result[\"result\"].shape}'); \
print(result['result'].head())"

run-tpch:  ## Run all TPC-H queries via HybridRouter
	$(PYTHON) -c "\
import json, sys; \
sys.path.insert(0, '.'); \
from router.hybrid_router import HybridRouter; \
router = HybridRouter(); \
with open('dsl/sample_queries/tpch_queries.json') as f: queries = json.load(f); \
for q in queries: \
    r = router.execute_query(q); \
    engines = [d['engine'] for d in r['routing_decisions']]; \
    print(f\"{q['query_id']}: engines={engines}, time={r['total_time_ms']:.1f}ms, rows={len(r['result'])}\")"

run-synthetic:  ## Run all synthetic graph queries via HybridRouter
	$(PYTHON) -c "\
import json, sys; \
sys.path.insert(0, '.'); \
from router.hybrid_router import HybridRouter; \
router = HybridRouter(); \
with open('dsl/sample_queries/synthetic_queries.json') as f: queries = json.load(f); \
for q in queries: \
    r = router.execute_query(q); \
    engines = [d['engine'] for d in r['routing_decisions']]; \
    print(f\"{q['query_id']}: engines={engines}, time={r['total_time_ms']:.1f}ms, rows={len(r['result'])}\")"

pipeline: data-all collect-data train analyze  ## Run full pipeline: data → train → analyze
	@echo "Full pipeline complete."

test-all: test test-execution test-correctness test-experiments  ## Run ALL tests (phase 1-6)
	@echo "All tests passed."

# --- Phase 6: Evaluation & Experiments ---

test-experiments:  ## Run Phase 6 experiment tests
	$(PYTHON) -m pytest tests/test_experiments.py -v

baseline-sql:  ## Run Always-SQL baseline
	$(PYTHON) -m experiments.run_baselines --strategy always_sql

baseline-graph:  ## Run Always-Graph baseline
	$(PYTHON) -m experiments.run_baselines --strategy always_graph

baseline-rule:  ## Run Rule-Based baseline
	$(PYTHON) -m experiments.run_baselines --strategy rule_based

baselines: baseline-sql baseline-graph baseline-rule  ## Run all baselines
	@echo "All baselines complete."

learned:  ## Run ML-routed (learned) strategy
	$(PYTHON) -m experiments.run_learned

compare:  ## Compare all strategies and produce comparison table
	$(PYTHON) -m experiments.compare_results

ablation:  ## Run feature ablation study
	$(PYTHON) -m experiments.ablation_study

evaluate: baselines learned compare ablation  ## Run full Phase 6 evaluation pipeline
	@echo "Full evaluation pipeline complete."

clean:  ## Remove generated data (keeps raw data)
	rm -rf data/parquet/tpch data/parquet/snb
	rm -rf data/graphs/*.parquet
	rm -rf data/stats/*.json
	rm -rf model/artifacts/*.pkl model/artifacts/*.json
	rm -rf model/artifacts/analysis/
	rm -rf training_data/labeled_runs.csv
