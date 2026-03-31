"""StatsCollector: loads precomputed table/graph statistics from JSON files."""

import json
import os
from typing import Dict, Optional


# Known graph source names (used to distinguish graph vs relational lookups)
_GRAPH_SOURCES = {"social_graph", "synthetic_graph"}


class StatsCollector:
    """Loads and provides access to precomputed table and graph statistics.

    Stats files are expected as JSON in the stats directory with names like:
      - customer_stats.json
      - orders_stats.json
      - synthetic_graph_stats.json
    """

    def __init__(self, stats_dir: str):
        self.stats_dir = stats_dir
        self._table_stats: Dict[str, dict] = {}
        self._graph_stats: Dict[str, dict] = {}
        self._load_all()

    def _load_all(self):
        """Load all JSON stats files from the stats directory."""
        if not os.path.isdir(self.stats_dir):
            return

        for filename in os.listdir(self.stats_dir):
            if not filename.endswith("_stats.json"):
                continue
            filepath = os.path.join(self.stats_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)

            # Determine if this is a graph stats file or table stats file
            name = filename.replace("_stats.json", "")
            if "avg_degree" in data:
                # Graph stats file
                self._graph_stats[name] = data
            else:
                # Table stats file
                self._table_stats[name] = data

    def get_table_stats(self, source_name: str) -> Optional[dict]:
        """Get stats for a relational table by name."""
        return self._table_stats.get(source_name)

    def get_graph_stats(self, source_name: str) -> Optional[dict]:
        """Get stats for a graph source by name.

        Tries exact match first, then looks for 'synthetic_graph' stats
        as default for any graph source.
        """
        # Direct match
        if source_name in self._graph_stats:
            return self._graph_stats[source_name]

        # Common file naming: <name>_graph_stats.json
        graph_key = f"{source_name}_graph"
        if graph_key in self._graph_stats:
            return self._graph_stats[graph_key]

        # Try normalized name
        normalized = source_name.replace("_graph", "").replace("graph", "synthetic")
        if normalized in self._graph_stats:
            return self._graph_stats[normalized]

        normalized_graph_key = f"{normalized}_graph"
        if normalized_graph_key in self._graph_stats:
            return self._graph_stats[normalized_graph_key]

        # Default: return the first available graph stats (often synthetic_graph)
        for key, stats in self._graph_stats.items():
            if "synthetic" in key:
                return stats

        # Return first available if any
        if self._graph_stats:
            return next(iter(self._graph_stats.values()))

        return None

    def is_graph_source(self, source_name: str) -> bool:
        """Determine if a source name refers to a graph (not a relational table)."""
        if source_name in _GRAPH_SOURCES:
            return True
        if source_name in self._graph_stats:
            return True
        if f"{source_name}_graph" in self._graph_stats:
            return True
        if "graph" in source_name.lower():
            return True
        return False

    def get_row_count(self, source_name: str) -> int:
        """Get the row count for a table, or vertex count for a graph."""
        table = self.get_table_stats(source_name)
        if table:
            return table.get("row_count", 0)
        graph = self.get_graph_stats(source_name)
        if graph:
            return graph.get("vertex_count", 0)
        return 0

    def get_column_stats(self, source_name: str, column_name: str) -> Optional[dict]:
        """Get per-column statistics (distinct_count, min, max) for a table column."""
        table = self.get_table_stats(source_name)
        if table:
            return table.get("columns", {}).get(column_name)
        return None

    @property
    def table_names(self) -> list:
        return list(self._table_stats.keys())

    @property
    def graph_names(self) -> list:
        return list(self._graph_stats.keys())
