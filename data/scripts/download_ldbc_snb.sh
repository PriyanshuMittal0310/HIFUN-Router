#!/bin/bash
# Downloads and generates LDBC SNB Interactive data using the official datagen Docker image.
# Produces CSV files under data/raw/ldbc_snb/ that can be converted to Parquet
# via data/scripts/ldbc_snb_to_parquet.py
#
# Prerequisites: Docker installed and running
#
# Usage:
#   bash data/scripts/download_ldbc_snb.sh [scale_factor]
#
# Examples:
#   bash data/scripts/download_ldbc_snb.sh 1
#   LDBC_SCALE_FACTOR=3 bash data/scripts/download_ldbc_snb.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/data/raw/ldbc_snb"
SCALE_FACTOR="${1:-${LDBC_SCALE_FACTOR:-1}}"

echo "=== LDBC SNB Data Generation (SF=${SCALE_FACTOR}) ==="
echo "Output directory: $OUTPUT_DIR"

# Check Docker is available
if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker is required but not installed."
    echo "Install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Pulling LDBC datagen Docker image..."
docker pull ldbc/datagen:latest

echo "Generating SNB data at scale factor ${SCALE_FACTOR}..."
docker run --rm \
    -v "$OUTPUT_DIR:/output" \
    ldbc/datagen:latest \
    --scale-factor "$SCALE_FACTOR" \
    --output-dir /output \
    --format csv \
    --mode raw

echo ""
echo "LDBC SNB data generated at: $OUTPUT_DIR"
echo ""
echo "Next step: convert to Parquet with:"
echo "  python data/scripts/ldbc_snb_to_parquet.py --input $OUTPUT_DIR"
