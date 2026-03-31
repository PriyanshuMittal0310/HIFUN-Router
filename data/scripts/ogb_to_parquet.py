"""Download OGB graph datasets and convert them to GraphFrames-friendly Parquet.

Usage:
  python data/scripts/ogb_to_parquet.py --dataset ogbn-arxiv

Output:
  data/graphs/ogbn_arxiv/ogbn_arxiv_vertices.parquet
  data/graphs/ogbn_arxiv/ogbn_arxiv_edges.parquet
"""

import argparse
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.spark_config import get_spark_session


def _safe_name(dataset: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", dataset).strip("_").lower()


def convert_ogb_dataset(spark, dataset: str, root: str, graph_dir: str) -> None:
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'ogb'. Install with: pip install ogb"
        ) from exc

    safe = _safe_name(dataset)
    output_dir = os.path.join(graph_dir, safe)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading OGB dataset: {dataset} (root={root})")
    dset = NodePropPredDataset(name=dataset, root=root)
    graph, _ = dset[0]

    num_nodes = int(graph["num_nodes"])
    edge_index = graph["edge_index"]

    edges = []
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        edges.append((int(src), int(dst), "KNOWS"))

    vertices = [(i,) for i in range(num_nodes)]

    v_df = spark.createDataFrame(vertices, ["id"])
    e_df = spark.createDataFrame(edges, ["src", "dst", "relationship"]).dropDuplicates(["src", "dst"])

    v_path = os.path.join(output_dir, f"{safe}_vertices.parquet")
    e_path = os.path.join(output_dir, f"{safe}_edges.parquet")

    v_df.write.mode("overwrite").parquet(v_path)
    e_df.write.mode("overwrite").parquet(e_path)

    print(f"Saved vertices: {v_df.count()} -> {v_path}")
    print(f"Saved edges: {e_df.count()} -> {e_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OGB dataset to GraphFrames Parquet")
    parser.add_argument("--dataset", default="ogbn-arxiv", help="OGB dataset name")
    parser.add_argument("--root", default="data/raw/ogb", help="OGB download cache root")
    parser.add_argument("--graph-dir", default="data/graphs", help="Graph output root")
    args = parser.parse_args()

    spark = get_spark_session("OGB_To_Parquet")
    try:
        convert_ogb_dataset(
            spark=spark,
            dataset=args.dataset,
            root=args.root,
            graph_dir=args.graph_dir,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
