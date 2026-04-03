#!/usr/bin/env bash
set -euo pipefail

# Always execute from repository root.
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-all}"

STRICT_SOURCE="training_data/real_labeled_runs_strict_curated.csv"
STRICT_TRAIN="training_data/fixed_train_base_strict.csv"
STRICT_EVAL="training_data/fixed_eval_set_strict.csv"
STRICT_EVAL_GRAPH="training_data/fixed_eval_graph_only_strict.csv"

run_quality_gate() {
  echo "[1/6] Running strict dataset quality gate"
  "$PYTHON_BIN" training_data/dataset_quality_gate.py \
    --source "$STRICT_SOURCE" \
    --train "$STRICT_TRAIN" \
    --eval "$STRICT_EVAL" \
    --out_json training_data/dataset_quality_report_strict_runtime.json
}

run_train() {
  echo "[2/6] Training model on strict train split"
  "$PYTHON_BIN" model/trainer.py --data "$STRICT_TRAIN"
}

run_relevance() {
  echo "[3/6] Running strict relevance evaluation"
  "$PYTHON_BIN" experiments/relevance_evaluation.py \
    --train "$STRICT_TRAIN" \
    --eval "$STRICT_EVAL" \
    --out_json experiments/results/relevance_eval_strict_runtime.json \
    --out_md experiments/results/relevance_eval_strict_runtime.md
}

run_ablation() {
  echo "[4/6] Running strict ablation study"
  "$PYTHON_BIN" experiments/ablation_study.py \
    --data "$STRICT_TRAIN" \
    --output experiments/results/ablation_strict_runtime.csv
}

run_shift() {
  echo "[5/6] Running strict dataset-shift evaluation"
  "$PYTHON_BIN" experiments/dataset_shift_evaluation.py \
    --source "$STRICT_SOURCE" \
    --out_json experiments/results/dataset_shift_eval_strict_runtime.json \
    --out_md experiments/results/dataset_shift_eval_strict_runtime.md
}

run_robustness() {
  echo "[6/6] Running strict robustness evaluation"
  "$PYTHON_BIN" experiments/strict_robustness_evaluation.py \
    --train "$STRICT_TRAIN" \
    --eval "$STRICT_EVAL" \
    --transfer_source "$STRICT_SOURCE" \
    --out_json experiments/results/strict_robustness_eval_runtime.json \
    --out_md experiments/results/strict_robustness_eval_runtime.md
}

run_correctness() {
  echo "[7/8] Running strict correctness report"
  "$PYTHON_BIN" experiments/correctness_report.py \
    --queries dsl/sample_queries \
    --output experiments/results/correctness_report_runtime.csv
}

run_publish_gate() {
  echo "[8/8] Validating strict publication gate"
  "$PYTHON_BIN" experiments/publish_gate.py
}

run_publish_gate_native() {
  echo "[8/8] Validating strict publication gate (native TPCH required)"
  "$PYTHON_BIN" experiments/publish_gate.py --require_native_tpch
}

case "$MODE" in
  all)
    run_quality_gate
    run_train
    run_relevance
    run_ablation
    run_shift
    run_robustness
    run_correctness
    run_publish_gate
    ;;
  quality)
    run_quality_gate
    ;;
  train)
    run_train
    ;;
  relevance)
    run_relevance
    ;;
  ablation)
    run_ablation
    ;;
  shift)
    run_shift
    ;;
  robustness)
    run_robustness
    ;;
  smoke)
    # Fast sanity path for CI/local quick checks.
    run_quality_gate
    run_relevance
    run_robustness
    run_correctness
    run_publish_gate
    ;;
  gate)
    run_publish_gate
    ;;
  gate-native)
    run_publish_gate_native
    ;;
  *)
    echo "Usage: ./run_project_strict.sh [all|quality|train|relevance|ablation|shift|robustness|smoke|gate|gate-native]"
    exit 1
    ;;
esac

echo "Done: strict pipeline mode '$MODE' completed successfully."
