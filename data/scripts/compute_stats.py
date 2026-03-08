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

    # 1. Compute TPC-H Customer Stats
    if os.path.exists("data/parquet/tpch/customer"):
        print("Computing stats for customer table...")
        cust_stats = compute_table_stats(spark, "customer", "data/parquet/tpch/customer")
        with open("data/stats/customer_stats.json", "w") as f:
            json.dump(cust_stats, f, indent=2)

    # 2. Compute TPC-H Orders Stats
    if os.path.exists("data/parquet/tpch/orders"):
        print("Computing stats for orders table...")
        orders_stats = compute_table_stats(spark, "orders", "data/parquet/tpch/orders")
        with open("data/stats/orders_stats.json", "w") as f:
            json.dump(orders_stats, f, indent=2)

    # 3. Compute Synthetic Graph Stats
    if os.path.exists("data/graphs/synthetic_edges.parquet"):
        print("Computing stats for synthetic graph...")
        graph_stats = compute_graph_stats(spark, "data/graphs/synthetic_edges.parquet")
        with open("data/stats/synthetic_graph_stats.json", "w") as f:
            json.dump(graph_stats, f, indent=2)

    # 4. Compute SNB Table Stats (if available)
    snb_tables_dir = "data/parquet/snb"
    if os.path.isdir(snb_tables_dir):
        for table_name in os.listdir(snb_tables_dir):
            table_path = os.path.join(snb_tables_dir, table_name)
            if os.path.isdir(table_path):
                print(f"Computing stats for SNB table: {table_name}...")
                stats = compute_table_stats(spark, table_name, table_path)
                with open(f"data/stats/{table_name}_stats.json", "w") as f:
                    json.dump(stats, f, indent=2)

    # 5. Compute SNB Graph Stats (if available)
    if os.path.exists("data/graphs/snb_edges.parquet"):
        print("Computing stats for SNB graph...")
        snb_graph_stats = compute_graph_stats(spark, "data/graphs/snb_edges.parquet")
        with open("data/stats/snb_graph_stats.json", "w") as f:
            json.dump(snb_graph_stats, f, indent=2)

    print("Statistics precomputation complete.")
    spark.stop()

if __name__ == "__main__":
    main()
