import argparse
import os
import sys

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.spark_config import get_spark_session


def _edge_frame(df: DataFrame, table: str, left_col: str, right_col: str) -> DataFrame:
    src = F.concat(F.lit(f"{table}:{left_col}:"), F.col(left_col).cast("string"))
    dst = F.concat(F.lit(f"{table}:{right_col}:"), F.col(right_col).cast("string"))
    return (
        df.select(src.alias("src"), dst.alias("dst"))
        .filter(F.col("src").isNotNull() & F.col("dst").isNotNull())
        .withColumn("relationship", F.lit("KNOWS"))
        .dropna(subset=["src", "dst"])
        .distinct()
    )


def _build_graph(spark, parquet_root: str, graph_out_dir: str, graph_name: str, specs: list[tuple[str, str, str]]) -> None:
    edges = None
    for table, left_col, right_col in specs:
        tpath = os.path.join(parquet_root, table)
        if not os.path.exists(tpath):
            continue
        tdf = spark.read.parquet(tpath)
        if left_col not in tdf.columns or right_col not in tdf.columns:
            continue
        e = _edge_frame(tdf, table, left_col, right_col)
        edges = e if edges is None else edges.unionByName(e)

    if edges is None:
        raise RuntimeError(f"No edges generated for graph '{graph_name}' from root {parquet_root}")

    edges = edges.distinct()
    vertices = (
        edges.select(F.col("src").alias("id"))
        .unionByName(edges.select(F.col("dst").alias("id")))
        .distinct()
    )

    os.makedirs(graph_out_dir, exist_ok=True)
    v_out = os.path.join(graph_out_dir, f"{graph_name}_vertices.parquet")
    e_out = os.path.join(graph_out_dir, f"{graph_name}_edges.parquet")
    vertices.write.mode("overwrite").parquet(v_out)
    edges.write.mode("overwrite").parquet(e_out)

    print(f"Built graph {graph_name}: vertices={vertices.count()} edges={edges.count()}")
    print(f"  vertices -> {v_out}")
    print(f"  edges    -> {e_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build graph projections for real datasets")
    parser.add_argument("--job_parquet", default="data/parquet/job")
    parser.add_argument("--tpcds_parquet", default="data/parquet/tpcds")
    parser.add_argument("--graph_root", default="data/graphs")
    args = parser.parse_args()

    spark = get_spark_session("BuildDatasetGraphs")
    try:
        job_specs = [
            ("movie_keyword", "c0", "c1"),
            ("cast_info", "c0", "c1"),
            ("movie_companies", "c0", "c1"),
            ("movie_link", "c0", "c1"),
        ]
        tpcds_specs = [
            ("store_sales", "c0", "c1"),
            ("catalog_sales", "c0", "c1"),
            ("web_sales", "c0", "c1"),
        ]

        _build_graph(
            spark,
            parquet_root=args.job_parquet,
            graph_out_dir=os.path.join(args.graph_root, "job"),
            graph_name="job_real_queries",
            specs=job_specs,
        )
        _build_graph(
            spark,
            parquet_root=args.tpcds_parquet,
            graph_out_dir=os.path.join(args.graph_root, "tpcds"),
            graph_name="tpcds_real_queries",
            specs=tpcds_specs,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
