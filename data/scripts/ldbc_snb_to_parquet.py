"""Convert official LDBC SNB CSV output to Parquet tables + graph edge list.

Reads pipe-delimited CSV files from the official LDBC datagen output.

Usage:
    # From official LDBC CSV:
    python data/scripts/ldbc_snb_to_parquet.py --input data/raw/ldbc_snb

Output:
    data/parquet/snb/{person,posts,comments,messages,works_at}/ (Parquet tables)
    data/graphs/snb/snb_vertices.parquet                        (Vertex list)
    data/graphs/snb/snb_edges.parquet                           (Edge list)
"""

import argparse
import os
import sys

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
    standalone_parquet_root = os.path.join(
        input_dir, "graphs", "parquet", "raw", "composite-merged-fk"
    )
    if os.path.isdir(standalone_parquet_root):
        _convert_standalone_layout(spark, standalone_parquet_root, parquet_dir, graph_dir)
        return

    tables = {
        "person": "person_0_0.csv",
        "posts": "post_0_0.csv",
        "comments": "comment_0_0.csv",
        "works_at": "person_workAt_organisation_0_0.csv",
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
        # Normalize common SNB key column names expected by DSL queries.
        if name == "posts":
            if "id" in df.columns:
                df = df.withColumnRenamed("id", "post_id")
            if "Person.id" in df.columns:
                df = df.withColumnRenamed("Person.id", "creator_id")
            if "length" in df.columns:
                df = df.withColumnRenamed("length", "content_length")
        elif name == "comments":
            if "id" in df.columns:
                df = df.withColumnRenamed("id", "comment_id")
            if "Person.id" in df.columns:
                df = df.withColumnRenamed("Person.id", "author_id")
            if "length" in df.columns:
                df = df.withColumnRenamed("length", "content_length")
        elif name == "works_at":
            cols = set(df.columns)
            if "Person.id" in cols:
                df = df.withColumnRenamed("Person.id", "person_id")
            if "Organisation.id" in cols:
                df = df.withColumnRenamed("Organisation.id", "organisation_id")

        out = os.path.join(parquet_dir, "snb", name)
        df.write.mode("overwrite").parquet(out)
        print(f"  {name}: {df.count()} rows -> {out}")

    # Build a convenience "messages" table from posts + comments.
    from pyspark.sql import functions as F
    posts_path = os.path.join(parquet_dir, "snb", "posts")
    comments_path = os.path.join(parquet_dir, "snb", "comments")
    if os.path.exists(posts_path) and os.path.exists(comments_path):
        posts_df = spark.read.parquet(posts_path)
        comments_df = spark.read.parquet(comments_path)

        posts_msg = posts_df.select(
            F.col("post_id").alias("message_id"),
            F.col("creator_id").alias("sender_id"),
            F.col("content_length"),
        )
        comments_msg = comments_df.select(
            F.col("comment_id").alias("message_id"),
            F.col("author_id").alias("sender_id"),
            F.col("content_length"),
        )
        messages = posts_msg.unionByName(comments_msg)
        out_messages = os.path.join(parquet_dir, "snb", "messages")
        messages.write.mode("overwrite").parquet(out_messages)
        print(f"  messages: {messages.count()} rows -> {out_messages}")

    # Graph edges: person_knows_person
    knows_path = _find_csv(input_dir, "person_knows_person")
    if knows_path:
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

        snb_graph_dir = os.path.join(graph_dir, "snb")
        os.makedirs(snb_graph_dir, exist_ok=True)
        out_e = os.path.join(snb_graph_dir, "snb_edges.parquet")
        out_v = os.path.join(snb_graph_dir, "snb_vertices.parquet")
        all_edges.write.mode("overwrite").parquet(out_e)
        vertices.write.mode("overwrite").parquet(out_v)
        print(f"  SNB graph: {all_edges.count()} edges -> {out_e}")
        print(f"  SNB vertices: {vertices.count()} -> {out_v}")
    else:
        print("  WARNING: person_knows_person CSV not found; skipping graph edges")


def _convert_standalone_layout(spark, standalone_root, parquet_dir, graph_dir):
    """Convert Spark standalone datagen parquet layout to project schema."""
    from pyspark.sql import functions as F

    dynamic_root = os.path.join(standalone_root, "dynamic")

    def _read(name):
        p = os.path.join(dynamic_root, name)
        return spark.read.parquet(p) if os.path.isdir(p) else None

    person_df = _read("Person")
    post_df = _read("Post")
    comment_df = _read("Comment")
    works_df = _read("Person_workAt_Company")
    knows_df = _read("Person_knows_Person")

    if person_df is not None:
        person_out = os.path.join(parquet_dir, "snb", "person")
        person_df.write.mode("overwrite").parquet(person_out)
        print(f"  person: {person_df.count()} rows -> {person_out}")

    if post_df is not None:
        posts = post_df
        if "id" in posts.columns:
            posts = posts.withColumnRenamed("id", "post_id")
        if "CreatorPersonId" in posts.columns:
            posts = posts.withColumnRenamed("CreatorPersonId", "creator_id")
        if "PersonId" in posts.columns:
            posts = posts.withColumnRenamed("PersonId", "creator_id")
        if "length" in posts.columns:
            posts = posts.withColumnRenamed("length", "content_length")
        post_out = os.path.join(parquet_dir, "snb", "posts")
        posts.write.mode("overwrite").parquet(post_out)
        print(f"  posts: {posts.count()} rows -> {post_out}")

    if comment_df is not None:
        comments = comment_df
        if "id" in comments.columns:
            comments = comments.withColumnRenamed("id", "comment_id")
        if "CreatorPersonId" in comments.columns:
            comments = comments.withColumnRenamed("CreatorPersonId", "author_id")
        if "PersonId" in comments.columns:
            comments = comments.withColumnRenamed("PersonId", "author_id")
        if "length" in comments.columns:
            comments = comments.withColumnRenamed("length", "content_length")
        comment_out = os.path.join(parquet_dir, "snb", "comments")
        comments.write.mode("overwrite").parquet(comment_out)
        print(f"  comments: {comments.count()} rows -> {comment_out}")

    if works_df is not None:
        works = works_df
        if "PersonId" in works.columns:
            works = works.withColumnRenamed("PersonId", "person_id")
        if "CompanyId" in works.columns:
            works = works.withColumnRenamed("CompanyId", "organisation_id")
        works_out = os.path.join(parquet_dir, "snb", "works_at")
        works.write.mode("overwrite").parquet(works_out)
        print(f"  works_at: {works.count()} rows -> {works_out}")

    posts_path = os.path.join(parquet_dir, "snb", "posts")
    comments_path = os.path.join(parquet_dir, "snb", "comments")
    if os.path.exists(posts_path) and os.path.exists(comments_path):
        posts_df = spark.read.parquet(posts_path)
        comments_df = spark.read.parquet(comments_path)
        messages = posts_df.select(
            F.col("post_id").alias("message_id"),
            F.col("creator_id").alias("sender_id"),
            F.col("content_length"),
        ).unionByName(
            comments_df.select(
                F.col("comment_id").alias("message_id"),
                F.col("author_id").alias("sender_id"),
                F.col("content_length"),
            )
        )
        msg_out = os.path.join(parquet_dir, "snb", "messages")
        messages.write.mode("overwrite").parquet(msg_out)
        print(f"  messages: {messages.count()} rows -> {msg_out}")

    if knows_df is not None:
        if "Person1Id" in knows_df.columns and "Person2Id" in knows_df.columns:
            knows_df = knows_df.withColumnRenamed("Person1Id", "src").withColumnRenamed("Person2Id", "dst")
        else:
            cols = knows_df.columns
            if len(cols) >= 2:
                knows_df = knows_df.withColumnRenamed(cols[0], "src").withColumnRenamed(cols[1], "dst")
        knows_df = knows_df.select("src", "dst").withColumn("relationship", F.lit("KNOWS"))
        reverse = knows_df.select(F.col("dst").alias("src"), F.col("src").alias("dst"), F.col("relationship"))
        edges = knows_df.unionByName(reverse).distinct()

        if person_df is not None and "id" in person_df.columns:
            vertices = person_df.select(F.col("id"))
        else:
            vertices = edges.select(F.col("src").alias("id")).unionByName(edges.select(F.col("dst").alias("id"))).distinct()

        snb_graph_dir = os.path.join(graph_dir, "snb")
        os.makedirs(snb_graph_dir, exist_ok=True)
        e_out = os.path.join(snb_graph_dir, "snb_edges.parquet")
        v_out = os.path.join(snb_graph_dir, "snb_vertices.parquet")
        edges.write.mode("overwrite").parquet(e_out)
        vertices.write.mode("overwrite").parquet(v_out)
        print(f"  SNB graph: {edges.count()} edges -> {e_out}")
        print(f"  SNB vertices: {vertices.count()} -> {v_out}")


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
    args = parser.parse_args()

    os.makedirs(os.path.join(args.parquet_dir, "snb"), exist_ok=True)
    os.makedirs(args.graph_dir, exist_ok=True)

    spark = get_spark_session("LDBC_SNB_Convert")

    if not os.path.isdir(args.input):
        raise FileNotFoundError(
            f"Official LDBC data not found at '{args.input}'. "
            "Run data/scripts/download_ldbc_snb.sh first."
        )

    print(f"Converting official LDBC SNB data from {args.input}...")
    convert_official_ldbc(spark, args.input, args.parquet_dir, args.graph_dir)

    spark.stop()
    print("\nDone. Run compute_stats.py to update statistics:")
    print("  python data/scripts/compute_stats.py")


if __name__ == "__main__":
    main()
