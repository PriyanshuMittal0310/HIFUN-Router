#!/bin/bash
# Downloads and generates LDBC SNB Interactive SF=0.1 data using the official datagen Docker image.
# Produces CSV files under data/raw/ldbc_snb/ that can be converted to Parquet
# via data/scripts/ldbc_snb_to_parquet.py
#
# Prerequisites: Docker installed and running
#
# Usage:
#   bash data/scripts/download_ldbc_snb.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/data/raw/ldbc_snb"

echo "=== LDBC SNB Data Generation (SF=0.1) ==="
echo "Output directory: $OUTPUT_DIR"

# Check Docker is available
if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker is required but not installed."
    echo "Install Docker from https://docs.docker.com/get-docker/"
    echo ""
    echo "Alternatively, run the SNB-to-Parquet script which will fall back to"
    echo "synthetic SNB-like data generation:"
    echo "  python data/scripts/ldbc_snb_to_parquet.py --synthetic"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Pulling LDBC datagen Docker image..."
docker pull ldbc/datagen:latest

echo "Generating SF=0.1 data (approx 30K persons, 200K posts, 180K edges)..."
docker run --rm \
    -v "$OUTPUT_DIR:/output" \
    ldbc/datagen:latest \
    --scale-factor 0.1 \
    --output-dir /output \
    --format csv \
    --mode raw

echo ""
echo "LDBC SNB data generated at: $OUTPUT_DIR"
echo ""
echo "Next step: convert to Parquet with:"
echo "  python data/scripts/ldbc_snb_to_parquet.py --input $OUTPUT_DIR"
