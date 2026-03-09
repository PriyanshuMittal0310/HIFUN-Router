"""Convert LDBC SNB data (CSV or synthetic fallback) to Parquet tables + graph edge list.

Supports two modes:
  1. Official LDBC data: reads pipe-delimited CSV files from the LDBC datagen output.
  2. Synthetic fallback: generates an SNB-like dataset when official data is unavailable.

Usage:
    # From official LDBC CSV:
    python data/scripts/ldbc_snb_to_parquet.py --input data/raw/ldbc_snb

    # Synthetic fallback:
    python data/scripts/ldbc_snb_to_parquet.py --synthetic

Output:
    data/parquet/snb/{person,post,comment}/   (Parquet tables for Spark SQL)
    data/graphs/snb_vertices.parquet          (Vertex list for GraphFrames)
    data/graphs/snb_edges.parquet             (Edge list for GraphFrames)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.spark_config import get_spark_session


# ── Official LDBC CSV conversion ────────────────────────────────────

def _find_csv(input_dir, pattern):
    """Locate an LDBC CSV file by pattern in subdirectories."""
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if pattern in f and f.endswith(".csv"):
                return os.path.join(root, f)
    return None


def convert_official_ldbc(spark, input_dir, parquet_dir, graph_dir):
    """Convert official LDBC datagen CSV to Parquet + edge list."""
    tables = {
        "person": "person_0_0.csv",
        "post": "post_0_0.csv",
        "comment": "comment_0_0.csv",
    }

    for name, filename in tables.items():
        path = _find_csv(input_dir, filename)
        if path is None:
            # Try alternative naming
            path = _find_csv(input_dir, name)
        if path is None:
            print(f"  WARNING: {filename} not found in {input_dir}, skipping {name}")
            continue

        df = spark.read.option("header", "true").option("sep", "|").csv(path)
        out = os.path.join(parquet_dir, "snb", name)
        df.write.mode("overwrite").parquet(out)
        print(f"  {name}: {df.count()} rows -> {out}")

    # Graph edges: person_knows_person
    knows_path = _find_csv(input_dir, "person_knows_person")
    if knows_path:
        from pyspark.sql import functions as F

        edges = (
            spark.read.option("header", "true").option("sep", "|").csv(knows_path)
        )
        # LDBC uses Person.id and Person.id.1 (or similar) as column names
        cols = edges.columns
        if len(cols) >= 2:
            edges = edges.withColumnRenamed(cols[0], "src").withColumnRenamed(
                cols[1], "dst"
            )
        edges = edges.select("src", "dst").withColumn(
            "relationship", F.lit("KNOWS")
        )

        # Make edges bidirectional
        reverse = edges.select(
            F.col("dst").alias("src"),
            F.col("src").alias("dst"),
            F.col("relationship"),
        )
        all_edges = edges.union(reverse).distinct()

        # Vertices from person table
        person_path = os.path.join(parquet_dir, "snb", "person")
        if os.path.exists(person_path):
            vertices = spark.read.parquet(person_path)
            id_col = "id" if "id" in vertices.columns else vertices.columns[0]
            vertices = vertices.select(F.col(id_col).alias("id"))
        else:
            # Derive vertices from edge endpoints
            src_ids = all_edges.select(F.col("src").alias("id"))
            dst_ids = all_edges.select(F.col("dst").alias("id"))
            vertices = src_ids.union(dst_ids).distinct()

        out_e = os.path.join(graph_dir, "snb_edges.parquet")
        out_v = os.path.join(graph_dir, "snb_vertices.parquet")
        all_edges.write.mode("overwrite").parquet(out_e)
        vertices.write.mode("overwrite").parquet(out_v)
        print(f"  SNB graph: {all_edges.count()} edges -> {out_e}")
        print(f"  SNB vertices: {vertices.count()} -> {out_v}")
    else:
        print("  WARNING: person_knows_person CSV not found; skipping graph edges")


# ── Synthetic fallback ──────────────────────────────────────────────

def generate_synthetic_snb(spark, parquet_dir, graph_dir, n_persons=5000,
                           n_posts=20000, n_comments=50000, n_edges=25000,
                           seed=42):
    """Generate a synthetic SNB-like dataset for prototyping."""
    rng = np.random.RandomState(seed)

    cities = ["Berlin", "London", "Paris", "New York", "Tokyo", "Mumbai", "Sydney"]
    countries = ["Germany", "UK", "France", "USA", "Japan", "India", "Australia"]

    # Person table
    persons = pd.DataFrame({
        "id": range(n_persons),
        "firstName": [f"Person_{i}" for i in range(n_persons)],
        "lastName": [f"Last_{i}" for i in range(n_persons)],
        "gender": rng.choice(["male", "female"], n_persons),
        "city": rng.choice(cities, n_persons),
        "country": rng.choice(countries, n_persons),
        "age": rng.randint(18, 70, n_persons),
    })
    person_sdf = spark.createDataFrame(persons)
    person_out = os.path.join(parquet_dir, "snb", "person")
    person_sdf.write.mode("overwrite").parquet(person_out)
    print(f"  person: {len(persons)} rows -> {person_out}")

    # Post table
    posts = pd.DataFrame({
        "id": range(n_posts),
        "creator_id": rng.randint(0, n_persons, n_posts),
        "content": [f"Post content {i}" for i in range(n_posts)],
        "length": rng.randint(10, 2000, n_posts),
        "language": rng.choice(["en", "de", "fr", "ja", "hi"], n_posts),
    })
    post_sdf = spark.createDataFrame(posts)
    post_out = os.path.join(parquet_dir, "snb", "post")
    post_sdf.write.mode("overwrite").parquet(post_out)
    print(f"  post: {len(posts)} rows -> {post_out}")

    # Comment table
    comments = pd.DataFrame({
        "id": range(n_comments),
        "creator_id": rng.randint(0, n_persons, n_comments),
        "content": [f"Comment {i}" for i in range(n_comments)],
        "length": rng.randint(5, 500, n_comments),
        "reply_of_post": rng.randint(0, n_posts, n_comments),
    })
    comment_sdf = spark.createDataFrame(comments)
    comment_out = os.path.join(parquet_dir, "snb", "comment")
    comment_sdf.write.mode("overwrite").parquet(comment_out)
    print(f"  comment: {len(comments)} rows -> {comment_out}")

    # KNOWS edges
    src = rng.randint(0, n_persons, n_edges)
    dst = rng.randint(0, n_persons, n_edges)
    mask = src != dst
    edges_pd = pd.DataFrame({
        "src": src[mask],
        "dst": dst[mask],
        "relationship": "KNOWS",
    }).drop_duplicates(subset=["src", "dst"])

    # Make bidirectional
    reverse = edges_pd.rename(columns={"src": "dst", "dst": "src"})[
        ["src", "dst", "relationship"]
    ]
    all_edges = pd.concat([edges_pd, reverse]).drop_duplicates(
        subset=["src", "dst"]
    )

    edges_sdf = spark.createDataFrame(all_edges)
    edges_out = os.path.join(graph_dir, "snb_edges.parquet")
    edges_sdf.write.mode("overwrite").parquet(edges_out)
    print(f"  SNB edges: {len(all_edges)} -> {edges_out}")

    vertices_sdf = spark.createDataFrame(
        pd.DataFrame({"id": range(n_persons)})
    )
    vertices_out = os.path.join(graph_dir, "snb_vertices.parquet")
    vertices_sdf.write.mode("overwrite").parquet(vertices_out)
    print(f"  SNB vertices: {n_persons} -> {vertices_out}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert LDBC SNB data to Parquet + graph edge list"
    )
    parser.add_argument(
        "--input", default="data/raw/ldbc_snb",
        help="Path to LDBC datagen CSV output directory",
    )
    parser.add_argument(
        "--parquet-dir", default="data/parquet",
        help="Parquet output directory",
    )
    parser.add_argument(
        "--graph-dir", default="data/graphs",
        help="Graph data output directory",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic SNB-like data instead of reading official LDBC CSVs",
    )
    args = parser.parse_args()

    os.makedirs(os.path.join(args.parquet_dir, "snb"), exist_ok=True)
    os.makedirs(args.graph_dir, exist_ok=True)

    spark = get_spark_session("LDBC_SNB_Convert")

    if args.synthetic or not os.path.isdir(args.input):
        if not args.synthetic:
            print(f"Official LDBC data not found at {args.input}, "
                  "falling back to synthetic generation.")
        print("Generating synthetic SNB-like dataset...")
        generate_synthetic_snb(spark, args.parquet_dir, args.graph_dir)
    else:
        print(f"Converting official LDBC SNB data from {args.input}...")
        convert_official_ldbc(spark, args.input, args.parquet_dir, args.graph_dir)

    spark.stop()
    print("\nDone. Run compute_stats.py to update statistics:")
    print("  python data/scripts/compute_stats.py")


if __name__ == "__main__":
    main()
