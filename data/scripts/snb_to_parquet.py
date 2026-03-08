"""LDBC SNB data ingestion: converts CSV/raw files to Parquet + GraphFrames edge list.

Usage:
    python data/scripts/snb_to_parquet.py \
        --input data/raw/snb/ \
        --tables data/parquet/snb/ \
        --edges data/graphs/snb_edges.parquet

If raw SNB data is not available, generates a synthetic SNB-like dataset for prototyping.
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.spark_config import get_spark_session


def _generate_synthetic_snb(n_persons=5000, n_posts=20000, n_comments=50000,
                            n_edges=25000, seed=42):
    """Generate a synthetic SNB-like dataset when raw data is unavailable."""
    rng = np.random.RandomState(seed)

    # Persons table
    cities = ["Berlin", "London", "Paris", "New York", "Tokyo", "Mumbai", "Sydney"]
    countries = ["Germany", "UK", "France", "USA", "Japan", "India", "Australia"]
    persons = pd.DataFrame({
        "person_id": range(n_persons),
        "name": [f"Person_{i}" for i in range(n_persons)],
        "city": rng.choice(cities, n_persons),
        "country": rng.choice(countries, n_persons),
        "age": rng.randint(18, 70, n_persons),
    })

    # KNOWS edges (social graph)
    src = rng.randint(0, n_persons, n_edges)
    dst = rng.randint(0, n_persons, n_edges)
    # Remove self-loops
    mask = src != dst
    edges = pd.DataFrame({"src": src[mask], "dst": dst[mask], "relationship": "KNOWS"})
    edges = edges.drop_duplicates(subset=["src", "dst"])

    # Posts table
    posts = pd.DataFrame({
        "post_id": range(n_posts),
        "creator_id": rng.randint(0, n_persons, n_posts),
        "content_length": rng.randint(10, 5000, n_posts),
        "creation_date": pd.date_range("2020-01-01", periods=n_posts, freq="h")
                          .strftime("%Y-%m-%d").tolist()[:n_posts],
    })

    # Comments table
    comments = pd.DataFrame({
        "comment_id": range(n_comments),
        "author_id": rng.randint(0, n_persons, n_comments),
        "post_id": rng.randint(0, n_posts, n_comments),
        "content_length": rng.randint(5, 500, n_comments),
    })

    # Messages (union-like view of posts + comments)
    messages = pd.DataFrame({
        "message_id": range(n_posts + n_comments),
        "sender_id": list(posts["creator_id"]) + list(comments["author_id"]),
        "content_length": list(posts["content_length"]) + list(comments["content_length"]),
    })

    # Works_at table
    companies = [f"Company_{i}" for i in range(200)]
    works_at = pd.DataFrame({
        "person_id": range(n_persons),
        "company_name": rng.choice(companies, n_persons),
    })

    return {
        "person": persons,
        "posts": posts,
        "comments": comments,
        "messages": messages,
        "works_at": works_at,
        "knows_edges": edges,
    }


def ingest_raw_snb(spark, input_dir, tables_dir, edges_path):
    """Ingest actual LDBC SNB CSV files."""
    csv_mappings = {
        "person": "person_0_0.csv",
        "posts": "post_0_0.csv",
        "comments": "comment_0_0.csv",
        "knows_edges": "person_knows_person_0_0.csv",
    }

    for table_name, filename in csv_mappings.items():
        csv_path = os.path.join(input_dir, filename)
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping {table_name}")
            continue
        print(f"Converting {table_name} from {csv_path}...")
        df = spark.read.csv(csv_path, header=True, inferSchema=True, sep="|")
        out_path = os.path.join(tables_dir, table_name)
        df.write.mode("overwrite").parquet(out_path)
        print(f"  -> Saved to {out_path}")

    # Edge list for GraphFrames
    knows_csv = os.path.join(input_dir, "person_knows_person_0_0.csv")
    if os.path.exists(knows_csv):
        edges = spark.read.csv(knows_csv, header=True, inferSchema=True, sep="|")
        col_names = edges.columns
        edges = edges.withColumnRenamed(col_names[0], "src").withColumnRenamed(col_names[1], "dst")
        edges.write.mode("overwrite").parquet(edges_path)
        print(f"Edge list saved to {edges_path}")


def ingest_synthetic_snb(spark, tables_dir, edges_path):
    """Generate and save synthetic SNB-like data."""
    print("Raw SNB data not found. Generating synthetic SNB-like dataset...")
    data = _generate_synthetic_snb()

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(os.path.dirname(edges_path), exist_ok=True)

    for table_name, pdf in data.items():
        if table_name == "knows_edges":
            # Save as GraphFrames edge list
            edf = spark.createDataFrame(pdf)
            edf.write.mode("overwrite").parquet(edges_path)
            print(f"  Edge list ({len(pdf)} edges) -> {edges_path}")
        else:
            sdf = spark.createDataFrame(pdf)
            out_path = os.path.join(tables_dir, table_name)
            sdf.write.mode("overwrite").parquet(out_path)
            print(f"  {table_name} ({len(pdf)} rows) -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="LDBC SNB data ingestion")
    parser.add_argument("--input", default="data/raw/snb",
                        help="Path to raw SNB CSV files")
    parser.add_argument("--tables", default="data/parquet/snb",
                        help="Output path for Parquet tables")
    parser.add_argument("--edges", default="data/graphs/snb_edges.parquet",
                        help="Output path for edge list Parquet")
    args = parser.parse_args()

    spark = get_spark_session("SNB_Ingestion")

    if os.path.isdir(args.input) and any(f.endswith(".csv") for f in os.listdir(args.input)):
        ingest_raw_snb(spark, args.input, args.tables, args.edges)
    else:
        ingest_synthetic_snb(spark, args.tables, args.edges)

    print("SNB ingestion complete.")
    spark.stop()


if __name__ == "__main__":
    main()
