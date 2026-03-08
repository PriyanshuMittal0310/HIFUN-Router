import networkx as nx
import pandas as pd
import json
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.spark_config import get_spark_session

def generate_powerlaw_graph(n_nodes, avg_degree, seed=42):
    print(f"Generating graph with {n_nodes} nodes and avg degree {avg_degree}...")
    # Barabasi-Albert model requires m >= 1
    m = max(1, avg_degree // 2)
    G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    
    edges_df = pd.DataFrame(G.edges(), columns=["src", "dst"])
    edges_df["relationship"] = "KNOWS"
    
    vertices_df = pd.DataFrame({"id": list(G.nodes()), "attr1": range(n_nodes)})
    return vertices_df, edges_df

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic power-law graphs")
    parser.add_argument("--n-nodes", type=int, default=10000,
                        help="Number of nodes per graph")
    parser.add_argument("--degrees", type=str, default="2,5,10,20,50",
                        help="Comma-separated list of avg degrees to generate")
    parser.add_argument("--output", default="data/graphs",
                        help="Output directory for Parquet files")
    parser.add_argument("--default-only", action="store_true",
                        help="Only generate the default graph (avg_degree=10)")
    args = parser.parse_args()

    spark = get_spark_session("Synthetic_Graph_Gen")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs("data/stats", exist_ok=True)

    if args.default_only:
        degree_list = [10]
    else:
        degree_list = [int(d.strip()) for d in args.degrees.split(",")]

    for avg_degree in degree_list:
        v_pd, e_pd = generate_powerlaw_graph(args.n_nodes, avg_degree)

        v_df = spark.createDataFrame(v_pd)
        e_df = spark.createDataFrame(e_pd)

        if avg_degree == 10:
            # Default graph uses the canonical name for backward compatibility
            v_path = f"{args.output}/synthetic_vertices.parquet"
            e_path = f"{args.output}/synthetic_edges.parquet"
            stats_path = "data/stats/synthetic_graph_stats.json"
        else:
            v_path = f"{args.output}/synthetic_deg{avg_degree}_vertices.parquet"
            e_path = f"{args.output}/synthetic_deg{avg_degree}_edges.parquet"
            stats_path = f"data/stats/synthetic_deg{avg_degree}_graph_stats.json"

        v_df.write.mode("overwrite").parquet(v_path)
        e_df.write.mode("overwrite").parquet(e_path)

        # Compute and save inline stats for this graph
        degree_series = e_pd.groupby("src").size()
        stats = {
            "avg_degree": float(degree_series.mean()) if len(degree_series) > 0 else 0.0,
            "max_degree": int(degree_series.max()) if len(degree_series) > 0 else 0,
            "stddev_degree": float(degree_series.std()) if len(degree_series) > 0 else 0.0,
            "vertex_count": len(v_pd),
            "edge_count": len(e_pd),
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"  Saved: vertices={v_path}, edges={e_path}, stats={stats_path}")

    print("Synthetic graph generation complete.")
    spark.stop()

if __name__ == "__main__":
    main()
