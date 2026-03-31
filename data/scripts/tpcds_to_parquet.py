"""Convert TPC-DS .dat output files to Parquet.

Usage:
  python data/scripts/tpcds_to_parquet.py --input /output/tpcds_sf5 --output data/parquet/tpcds
"""

import argparse
import os
from pyspark.sql import functions as F

from config.spark_config import get_spark_session


def convert_tpcds_to_parquet(spark, input_dir: str, output_dir: str) -> None:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    converted = 0
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".dat"):
            continue

        table_name = os.path.splitext(fname)[0].lower()
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, table_name)

        print(f"Converting {fname} -> {table_name}")
        df = (
            spark.read
            .option("header", "false")
            .option("inferSchema", "true")
            .option("sep", "|")
            .csv(in_path)
        )

        # Drop trailing empty column when source rows end with '|'.
        if df.columns:
            last = df.columns[-1]
            if df.filter(F.col(last).isNotNull()).limit(1).count() == 0:
                df = df.drop(last)

        # Normalize column names to c0..cN when schema is unknown.
        renamed = df
        for i, c in enumerate(df.columns):
            renamed = renamed.withColumnRenamed(c, f"c{i}")

        renamed.write.mode("overwrite").parquet(out_path)
        print(f"  rows={renamed.count()} -> {out_path}")
        converted += 1

    if converted == 0:
        raise RuntimeError(
            f"No .dat files found under {input_dir}. "
            "Run dsdgen first to generate TPC-DS data."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TPC-DS .dat files to Parquet")
    parser.add_argument("--input", required=True, help="TPC-DS dsdgen output directory")
    parser.add_argument("--output", default="data/parquet/tpcds", help="Parquet output directory")
    args = parser.parse_args()

    spark = get_spark_session("TPCDS_To_Parquet")
    try:
        convert_tpcds_to_parquet(spark, args.input, args.output)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
