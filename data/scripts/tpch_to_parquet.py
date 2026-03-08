import argparse
import os
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import sys

# Ensure Spark config can be imported from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.spark_config import get_spark_session

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw .tbl files")
    parser.add_argument("--output", required=True, help="Path to save Parquet files")
    args = parser.parse_args()

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
        print(f"Converting {table_name} to Parquet...")
        df = spark.read.csv(f"{args.input}/{table_name}.tbl", sep="|", schema=schema)
        # Drop the trailing null column caused by the trailing '|' in TPC-H files
        df = df.drop(df.columns[-1]) 
        df.write.mode("overwrite").parquet(f"{args.output}/{table_name}")
    
    print("TPC-H conversion complete.")
    spark.stop()

if __name__ == "__main__":
    main()
