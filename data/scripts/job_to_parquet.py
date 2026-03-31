"""Convert Join Order Benchmark (JOB / IMDB) CSV files to Parquet.

Usage:
  python data/scripts/job_to_parquet.py --input data/raw/job --output data/parquet/job
"""

import argparse
import os
from pyspark.sql import functions as F

from config.spark_config import get_spark_session


def _guess_separator(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        line = f.readline()
    if line.count("\t") > line.count(","):
        return "\t"
    return ","


def convert_job_to_parquet(spark, input_dir: str, output_dir: str) -> None:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    converted = 0
    for fname in sorted(os.listdir(input_dir)):
        if not (fname.endswith(".csv") or fname.endswith(".tsv")):
            continue

        fpath = os.path.join(input_dir, fname)
        sep = _guess_separator(fpath)
        table_name = os.path.splitext(fname)[0].lower()

        print(f"Converting {fname} (sep={repr(sep)}) -> {table_name}")
        df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .option("sep", sep)
            .csv(fpath)
        )

        # Drop fully-null trailing column common in delimiter-terminated dumps.
        if df.columns:
            last = df.columns[-1]
            if df.filter(F.col(last).isNotNull()).limit(1).count() == 0:
                df = df.drop(last)

        out_path = os.path.join(output_dir, table_name)
        df.write.mode("overwrite").parquet(out_path)
        print(f"  rows={df.count()} -> {out_path}")
        converted += 1

    if converted == 0:
        raise RuntimeError(
            f"No CSV/TSV files found under {input_dir}. "
            "Place JOB IMDB table dumps there first."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JOB/IMDB dumps to Parquet")
    parser.add_argument("--input", default="data/raw/job", help="Directory with JOB CSV/TSV files")
    parser.add_argument("--output", default="data/parquet/job", help="Parquet output directory")
    args = parser.parse_args()

    spark = get_spark_session("JOB_To_Parquet")
    try:
        convert_job_to_parquet(spark, args.input, args.output)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
