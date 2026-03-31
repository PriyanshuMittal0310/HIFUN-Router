#!/bin/bash
# End-to-end helper for real datasets only (no synthetic data).
#
# Usage:
#   bash data/scripts/download_real_datasets.sh --snb-scale 1 --ogb ogbn-arxiv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SNB_SCALE="1"
OGB_DATASET="ogbn-arxiv"
SKIP_JOB="false"
SKIP_TPCDS="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --snb-scale)
      SNB_SCALE="$2"; shift 2 ;;
    --ogb)
      OGB_DATASET="$2"; shift 2 ;;
    --skip-job)
      SKIP_JOB="true"; shift ;;
    --skip-tpcds)
      SKIP_TPCDS="true"; shift ;;
    *)
      echo "Unknown argument: $1"; exit 1 ;;
  esac
done

cd "$PROJECT_ROOT"

echo "[1/7] Generating LDBC SNB SF=${SNB_SCALE}"
bash data/scripts/download_ldbc_snb.sh "$SNB_SCALE"

echo "[2/7] Converting SNB to Parquet + graph"
python3 data/scripts/ldbc_snb_to_parquet.py --input data/raw/ldbc_snb

echo "[3/7] Downloading and converting OGB dataset: ${OGB_DATASET}"
python3 data/scripts/ogb_to_parquet.py --dataset "$OGB_DATASET"

if [[ "$SKIP_JOB" != "true" ]]; then
  echo "[4/7] Converting JOB (if raw files exist in data/raw/job)"
  python3 data/scripts/job_to_parquet.py --input data/raw/job --output data/parquet/job || true
else
  echo "[4/7] Skipped JOB conversion"
fi

if [[ "$SKIP_TPCDS" != "true" ]]; then
  echo "[5/7] Converting TPC-DS (if raw files exist in data/raw/tpcds)"
  python3 data/scripts/tpcds_to_parquet.py --input data/raw/tpcds --output data/parquet/tpcds || true
else
  echo "[5/7] Skipped TPC-DS conversion"
fi

echo "[6/7] Generating rigorous real query packs"
python3 training_data/real_query_generator.py --scale aggressive

echo "[7/7] Computing dataset stats"
python3 data/scripts/compute_stats.py

echo "Done. Next: run real collection"
echo "python3 training_data/real_collection_script.py --queries_dir dsl/sample_queries --output training_data/real_labeled_runs.csv --n_warmup 2 --n_measure 3 --repeat 3"
