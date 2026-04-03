#!/usr/bin/env bash
# data/scripts/generate_tpch_sf5.sh
# ─────────────────────────────────────────────────────────────────────────────
# Generate TPC-H data at Scale Factor 5 (~5 GB) and convert to Parquet.
#
# Prerequisites:
#   - tpch-kit already compiled in data/raw/tpch-kit/dbgen/
#   - Python environment activated  (source .venv/bin/activate)
#   - PySpark available              (pip install pyspark pyarrow)
#
# Usage:
#   bash data/scripts/generate_tpch_sf5.sh [SCALE_FACTOR]
#   bash data/scripts/generate_tpch_sf5.sh 5     # default
#   bash data/scripts/generate_tpch_sf5.sh 10    # larger scale
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Arguments ─────────────────────────────────────────────────────────────────
SF="${1:-5}"
DBGEN_DIR="${PROJECT_ROOT}/data/raw/tpch-kit/dbgen"
RAW_OUTPUT="${PROJECT_ROOT}/data/raw/tpch-kit/dbgen"
PARQUET_OUTPUT="${PROJECT_ROOT}/data/parquet/tpch"

echo "============================================================"
echo "  HIFUN Router — TPC-H Data Generation at SF=${SF}"
echo "  dbgen dir  : ${DBGEN_DIR}"
echo "  raw output : ${RAW_OUTPUT}"
echo "  parquet out: ${PARQUET_OUTPUT}"
echo "============================================================"

# ── Step 1: Verify dbgen is compiled ─────────────────────────────────────────
if [[ ! -x "${DBGEN_DIR}/dbgen" ]]; then
    echo "[INFO]  dbgen not found; attempting to compile ..."
    if [[ -d "${PROJECT_ROOT}/data/raw/tpch-kit" ]]; then
        cd "${PROJECT_ROOT}/data/raw/tpch-kit"
        make clean 2>/dev/null || true
        make DATABASE=SQLSERVER MACHINE=LINUX WORKLOAD=TPCH
        echo "[OK]    dbgen compiled"
    else
        echo "[ERROR] tpch-kit not found at data/raw/tpch-kit"
        echo "        Clone it with:"
        echo "          git clone https://github.com/gregrahn/tpch-kit.git data/raw/tpch-kit"
        exit 1
    fi
fi

# ── Step 2: Generate raw TPC-H data ──────────────────────────────────────────
mkdir -p "${RAW_OUTPUT}"
cd "${DBGEN_DIR}"

echo "[INFO]  Generating TPC-H SF=${SF} (this may take several minutes) ..."
./dbgen -s "${SF}" -f -T a -b dists.dss

# dbgen writes .tbl files in DBGEN_DIR.
echo "[OK]    Raw TBL files in: ${RAW_OUTPUT}"
ls -lh "${RAW_OUTPUT}"/*.tbl | awk '{print "       "$5, $9}'

# ── Step 3: Convert TBL → Parquet using PySpark ───────────────────────────────
echo "[INFO]  Converting TBL files to Parquet (SF=${SF}) ..."
cd "${PROJECT_ROOT}"

# Re-use the existing tpch_to_parquet.py if it supports --input/--output;
# otherwise run an inline PySpark conversion.
if python3 data/scripts/tpch_to_parquet.py --help 2>&1 | grep -q "\-\-input"; then
    python3 data/scripts/tpch_to_parquet.py \
        --input  "${RAW_OUTPUT}" \
    --output "${PARQUET_OUTPUT}"
else
    # Inline PySpark conversion
    python3 - <<PYEOF
import os, glob
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (SparkSession.builder
         .appName("TPCH_to_Parquet_SF${SF}")
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "32")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# TPC-H column schemas (pipe-delimited .tbl files)
SCHEMAS = {
    "customer": "c_custkey,c_name,c_address,c_nationkey,c_phone,c_acctbal,c_mktsegment,c_comment",
    "orders":   "o_orderkey,o_custkey,o_orderstatus,o_totalprice,o_orderdate,o_orderpriority,o_clerk,o_shippriority,o_comment",
    "lineitem": "l_orderkey,l_partkey,l_suppkey,l_linenumber,l_quantity,l_extendedprice,l_discount,l_tax,l_returnflag,l_linestatus,l_shipdate,l_commitdate,l_receiptdate,l_shipinstruct,l_shipmode,l_comment",
    "part":     "p_partkey,p_name,p_mfgr,p_brand,p_type,p_size,p_container,p_retailprice,p_comment",
    "supplier": "s_suppkey,s_name,s_address,s_nationkey,s_phone,s_acctbal,s_comment",
    "partsupp": "ps_partkey,ps_suppkey,ps_availqty,ps_supplycost,ps_comment",
    "nation":   "n_nationkey,n_name,n_regionkey,n_comment",
    "region":   "r_regionkey,r_name,r_comment",
}

raw_dir     = "${RAW_OUTPUT}"
parquet_dir = "${PARQUET_OUTPUT}"

for table, cols_str in SCHEMAS.items():
    tbl_path = os.path.join(raw_dir, f"{table}.tbl")
    if not os.path.exists(tbl_path):
        print(f"  [SKIP] {tbl_path} not found")
        continue
    cols = cols_str.split(",")
    df = (spark.read
          .option("sep", "|")
          .option("header", "false")
          .csv(tbl_path)
          .toDF(*cols, "_trailing"))  # dbgen adds trailing |
    df = df.drop("_trailing")
    # Cast numeric columns
    for c in df.columns:
        if any(c.endswith(s) for s in ("key", "qty", "quantity", "size", "priority", "shippriority")):
            df = df.withColumn(c, F.col(c).cast("long"))
        if any(c.endswith(s) for s in ("price", "bal", "discount", "tax", "cost")):
            df = df.withColumn(c, F.col(c).cast("double"))
    out_path = os.path.join(parquet_dir, table)
    df.write.mode("overwrite").parquet(out_path)
    print(f"  [OK] {table}: {df.count()} rows → {out_path}")

spark.stop()
print(f"Done. Parquet files at {parquet_dir}")
PYEOF
fi

echo "[OK]    Parquet files written to: ${PARQUET_OUTPUT}"
ls -lh "${PARQUET_OUTPUT}" 2>/dev/null || echo "(directory is empty or does not exist)"

echo ""
echo "============================================================"
echo "  TPC-H SF=${SF} generation complete."
echo "  Native TPCH parquet path ready for gate checks: ${PARQUET_OUTPUT}"
echo "============================================================"
