import json
import os
import sys
from pyspark.sql import functions as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.spark_config import get_spark_session

def compute_table_stats(spark, table_name: str, parquet_path: str) -> dict:
    df = spark.read.parquet(parquet_path)
    stats = {
        "table_name": table_name,
        "row_count": df.count(),
        "column_count": len(df.columns),
        "columns": {}
    }
    # Only compute stats for numeric keys in our prototype to save time
    numeric_cols = [c for c, t in df.dtypes if t in ('int', 'double')]
    for col in numeric_cols:
        distinct_count = df.select(col).distinct().count()
        # FIXED: Using PySpark native functions instead of a dictionary
        min_max = df.agg(F.min(col), F.max(col)).collect()[0]
        stats["columns"][col] = {
            "distinct_count": distinct_count,
            "min": min_max[0],
            "max": min_max[1],
        }
    return stats

def compute_graph_stats(spark, edge_parquet: str) -> dict:
    edges = spark.read.parquet(edge_parquet)
    degree_df = edges.groupBy("src").count()
    stats_row = degree_df.selectExpr(
        "avg(count) as avg_degree",
        "max(count) as max_degree",
        "stddev(count) as stddev_degree",
        "count(*) as vertex_count"
    ).collect()[0]
    return {
        "avg_degree": float(stats_row.avg_degree or 0),
        "max_degree": float(stats_row.max_degree or 0),
        "stddev_degree": float(stats_row.stddev_degree or 0),
        "vertex_count": int(stats_row.vertex_count or 0),
        "edge_count": edges.count()
    }

def main():
    spark = get_spark_session("Compute_Stats")

    os.makedirs("data/stats", exist_ok=True)

    # 1. Compute table stats for all datasets under data/parquet/*
    parquet_root = "data/parquet"
    if os.path.isdir(parquet_root):
        for dataset in sorted(os.listdir(parquet_root)):
            dataset_dir = os.path.join(parquet_root, dataset)
            if not os.path.isdir(dataset_dir):
                continue
            for table_name in sorted(os.listdir(dataset_dir)):
                table_path = os.path.join(dataset_dir, table_name)
                if not os.path.isdir(table_path):
                    continue
                print(f"Computing stats for table: {dataset}/{table_name}...")
                stats = compute_table_stats(spark, table_name, table_path)
                with open(f"data/stats/{table_name}_stats.json", "w") as f:
                    json.dump(stats, f, indent=2)

    # 2. Compute graph stats for *all* discovered edge parquet files
    graph_edge_candidates = []

    # Legacy flat layout: data/graphs/<name>_edges.parquet
    graphs_root = "data/graphs"
    if os.path.isdir(graphs_root):
        for fname in sorted(os.listdir(graphs_root)):
            if fname.endswith("_edges.parquet"):
                graph_edge_candidates.append((fname.replace("_edges.parquet", ""), os.path.join(graphs_root, fname)))

        # Nested layout: data/graphs/<graph>/<graph>_edges.parquet
        for dname in sorted(os.listdir(graphs_root)):
            subdir = os.path.join(graphs_root, dname)
            if not os.path.isdir(subdir):
                continue
            expected = os.path.join(subdir, f"{dname}_edges.parquet")
            if os.path.exists(expected):
                graph_edge_candidates.append((dname, expected))

    seen = set()
    for graph_name, edge_path in graph_edge_candidates:
        key = (graph_name, edge_path)
        if key in seen:
            continue
        seen.add(key)
        print(f"Computing stats for graph: {graph_name}...")
        gstats = compute_graph_stats(spark, edge_path)
        with open(f"data/stats/{graph_name}_graph_stats.json", "w") as f:
            json.dump(gstats, f, indent=2)

    print("Statistics precomputation complete.")
    spark.stop()

if __name__ == "__main__":
    main()
