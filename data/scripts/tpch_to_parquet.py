import argparse
import os
import sys

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# Ensure Spark config can be imported from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.spark_config import get_spark_session
from config.paths import TPCH_RAW_DIR, TPCH_PARQUET_DIR


def _require_input(path: str, table: str) -> None:
    fpath = os.path.join(path, f"{table}.tbl")
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"Required TPCH input file missing: {fpath}. "
            "Generate TPC-H .tbl files first (dbgen), then retry."
        )

def main():
    parser = argparse.ArgumentParser(description="Convert TPCH .tbl files to parquet")
    parser.add_argument("--input", default=TPCH_RAW_DIR, help="Path to raw .tbl files")
    parser.add_argument("--output", default=TPCH_PARQUET_DIR, help="Path to save parquet files")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    spark = get_spark_session("TPCH_Ingestion")

    # Define schemas for the prototype queries
    customer_schema = StructType([
        StructField("c_custkey", IntegerType(), True), StructField("c_name", StringType(), True),
        StructField("c_address", StringType(), True), StructField("c_nationkey", IntegerType(), True),
        StructField("c_phone", StringType(), True), StructField("c_acctbal", DoubleType(), True),
        StructField("c_mktsegment", StringType(), True), StructField("c_comment", StringType(), True)
    ])

    orders_schema = StructType([
        StructField("o_orderkey", IntegerType(), True), StructField("o_custkey", IntegerType(), True),
        StructField("o_orderstatus", StringType(), True), StructField("o_totalprice", DoubleType(), True),
        StructField("o_orderdate", StringType(), True), StructField("o_orderpriority", StringType(), True),
        StructField("o_clerk", StringType(), True), StructField("o_shippriority", IntegerType(), True),
        StructField("o_comment", StringType(), True)
    ])

    tables = {"customer": customer_schema, "orders": orders_schema}

    for table_name, schema in tables.items():
        _require_input(args.input, table_name)
        print(f"Converting {table_name} to Parquet...")
        df = spark.read.csv(f"{args.input}/{table_name}.tbl", sep="|", schema=schema)
        df.write.mode("overwrite").parquet(f"{args.output}/{table_name}")
        print(f"  wrote {df.count()} rows -> {args.output}/{table_name}")
    
    print("TPC-H conversion complete.")
    spark.stop()

if __name__ == "__main__":
    main()
