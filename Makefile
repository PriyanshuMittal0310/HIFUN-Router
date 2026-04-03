.PHONY: help setup data-tpch data-snb data-synthetic data-stats data-all validate-queries test clean clean-local-results collect-data train analyze report quality-gate quality-gate-strict publish-eval publish-eval-strict publish-gate publish-gate-native status-snapshot data-tpch-check

PYTHON := python3
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

data-tpch-check:  ## Verify required TPCH .tbl inputs exist
	@test -f data/raw/tpch-kit/dbgen/customer.tbl || (echo "Missing data/raw/tpch-kit/dbgen/customer.tbl" && exit 1)
	@test -f data/raw/tpch-kit/dbgen/orders.tbl || (echo "Missing data/raw/tpch-kit/dbgen/orders.tbl" && exit 1)
	@echo "TPCH raw input files found."

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

ablation:  ## Run feature ablation study (repeated CV stability)
	$(PYTHON) -m experiments.ablation_study

evaluate: baselines learned compare ablation  ## Run full Phase 6 evaluation pipeline
	@echo "Full evaluation pipeline complete."

quality-gate:  ## Block invalid/degenerate datasets before reporting results
	$(PYTHON) -m training_data.dataset_quality_gate \
		--source training_data/real_labeled_runs.csv \
		--train training_data/fixed_train_base.csv \
		--eval training_data/fixed_eval_set.csv

quality-gate-strict:  ## Strict quality gate on curated real-measurement artifacts
	$(PYTHON) -m training_data.dataset_quality_gate \
		--source training_data/real_labeled_runs_strict_curated.csv \
		--train training_data/fixed_train_base_strict.csv \
		--eval training_data/fixed_eval_set_strict.csv \
		--out_json training_data/dataset_quality_report_strict_runtime.json

publish-eval:  ## Run publishable strict evaluation bundle
	$(PYTHON) -m training_data.fix_dataset_splits --source training_data/real_labeled_runs_strict_curated.csv --split_mode group --group_col query_id --train_base_out training_data/fixed_train_base_strict.csv --eval_out training_data/fixed_eval_set_strict.csv --graph_eval_out training_data/fixed_eval_graph_only_strict.csv --train_balanced_out training_data/fixed_train_balanced_strict.csv --manifest_out training_data/fixed_split_manifest_strict.json
	$(PYTHON) -m training_data.dataset_quality_gate --source training_data/real_labeled_runs_strict_curated.csv --train training_data/fixed_train_base_strict.csv --eval training_data/fixed_eval_set_strict.csv --out_json training_data/dataset_quality_report_strict_runtime.json
	$(PYTHON) -m model.trainer --data training_data/fixed_train_base_strict.csv
	$(PYTHON) -m experiments.relevance_evaluation --train training_data/fixed_train_base_strict.csv --eval training_data/fixed_eval_set_strict.csv --out_json experiments/results/relevance_eval_strict_runtime.json --out_md experiments/results/relevance_eval_strict_runtime.md
	$(PYTHON) -m experiments.ablation_study --data training_data/fixed_train_base_strict.csv --output experiments/results/ablation_strict_runtime.csv
	$(PYTHON) -m experiments.dataset_shift_evaluation --source training_data/real_labeled_runs_strict_curated.csv --out_json experiments/results/dataset_shift_eval_strict_runtime.json --out_md experiments/results/dataset_shift_eval_strict_runtime.md
	$(PYTHON) -m experiments.strict_robustness_evaluation --train training_data/fixed_train_base_strict.csv --eval training_data/fixed_eval_set_strict.csv --transfer_source training_data/real_labeled_runs_strict_curated.csv --out_json experiments/results/strict_robustness_eval_runtime.json --out_md experiments/results/strict_robustness_eval_runtime.md
	$(PYTHON) -m experiments.correctness_report --queries dsl/sample_queries --output experiments/results/correctness_report_runtime.csv
	$(PYTHON) -m experiments.publish_gate
	@echo "Publishable strict evaluation complete."

publish-eval-strict: publish-eval  ## Alias for strict publishable bundle

publish-gate:  ## Validate strict publish artifacts and thresholds
	$(PYTHON) -m experiments.publish_gate --min_max_feature_drop 0.005 --min_max_group_drop 0.005 --min_max_permutation_drop 0.05

publish-gate-native:  ## Validate strict publish artifacts and require native TPCH parquet
	$(PYTHON) -m experiments.publish_gate --require_native_tpch --min_max_feature_drop 0.005 --min_max_group_drop 0.005 --min_max_permutation_drop 0.05

status-snapshot:  ## Generate one-page strict runtime status checklist
	$(PYTHON) -m experiments.status_snapshot --output experiments/results/project_status_snapshot.md

correctness-native:  ## Run correctness report requiring native TPCH parquet
	$(PYTHON) -m experiments.correctness_report --queries dsl/sample_queries --output experiments/results/correctness_report_native_runtime.csv --require_native_tpch

clean:  ## Remove generated data (keeps raw data)
	rm -rf data/parquet/tpch data/parquet/snb
	rm -rf data/graphs/*.parquet
	rm -rf data/stats/*.json
	rm -rf model/artifacts/*.pkl model/artifacts/*.json
	rm -rf model/artifacts/analysis/
	rm -rf training_data/labeled_runs.csv

clean-local-results:  ## Remove local iterative result artifacts (keeps strict headline outputs)
	rm -f experiments/results/*_default_check.csv
	rm -f experiments/results/*_default_check.json
	rm -f experiments/results/*_default_check.md
	rm -f experiments/results/*_default_check_groups.csv
	rm -f experiments/results/*_strict_recheck.json
	rm -f experiments/results/*_strict_recheck.md
	rm -f experiments/results/correctness_report_runtime.csv
	rm -f experiments/results/ablation_debug_groups.csv
