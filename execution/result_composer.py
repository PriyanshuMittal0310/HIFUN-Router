"""ResultComposer: merges partial DataFrames from heterogeneous execution engines.

Given a mapping of {sub_id -> DataFrame} from SQL and GRAPH engines, composes
the final query result by resolving cross-engine dependencies and joins.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from parser.ast_nodes import SubExpression

logger = logging.getLogger(__name__)


class ResultComposer:
    """Merges partial results from SQL and GRAPH engines into a final result."""

    def __init__(self) -> None:
        self.result_map: Dict[str, pd.DataFrame] = {}

    def register_result(self, sub_id: str, df: pd.DataFrame) -> None:
        """Register a partial result from an engine execution."""
        self.result_map[sub_id] = df

    def compose(self, sub_expressions: List[SubExpression],
                merge_strategy: str = "auto") -> pd.DataFrame:
        """Compose final result from all partial DataFrames.

        Args:
            sub_expressions: ordered list of SubExpressions (topological order)
            merge_strategy: "auto" | "concat" | "join"
                auto: join if cross-engine dependencies exist, else concat last result

        Returns:
            Final composed DataFrame.
        """
        if not sub_expressions:
            return pd.DataFrame()

        if not self.result_map:
            return pd.DataFrame()

        # Build dependency graph
        dep_graph = {se.sub_id: se.depends_on_subs for se in sub_expressions}

        # Find sub-expressions that have cross-engine dependencies
        cross_deps = []
        for se in sub_expressions:
            for dep_id in se.depends_on_subs:
                if dep_id in self.result_map and se.sub_id in self.result_map:
                    cross_deps.append((dep_id, se.sub_id))

        if merge_strategy == "concat" or (merge_strategy == "auto" and not cross_deps):
            return self._compose_concat(sub_expressions)
        else:
            return self._compose_join(sub_expressions, cross_deps)

    def _compose_concat(self, sub_expressions: List[SubExpression]) -> pd.DataFrame:
        """Return the last sub-expression's result (most common for linear pipelines)."""
        # Return last registered result in topological order
        last_result = None
        for se in sub_expressions:
            if se.sub_id in self.result_map:
                last_result = self.result_map[se.sub_id]
        return last_result if last_result is not None else pd.DataFrame()

    def _compose_join(self, sub_expressions: List[SubExpression],
                      cross_deps: List[tuple]) -> pd.DataFrame:
        """Join cross-engine results on common columns."""
        if not cross_deps:
            return self._compose_concat(sub_expressions)

        # Start with the result that has no incoming cross-deps
        dep_targets = {dep[1] for dep in cross_deps}
        dep_sources = {dep[0] for dep in cross_deps}

        # Start from the root dependency
        roots = dep_sources - dep_targets
        if not roots:
            # Circular or all connected — pick first
            roots = dep_sources

        result = None
        joined_ids = set()

        for root_id in roots:
            if root_id in self.result_map:
                if result is None:
                    result = self.result_map[root_id]
                    joined_ids.add(root_id)
                else:
                    result = self._merge_dfs(result, self.result_map[root_id])
                    joined_ids.add(root_id)

        # Join remaining dependent results
        for src_id, tgt_id in cross_deps:
            if tgt_id not in joined_ids and tgt_id in self.result_map:
                result = self._merge_dfs(result, self.result_map[tgt_id])
                joined_ids.add(tgt_id)

        # Add any results not yet incorporated
        for se in sub_expressions:
            if se.sub_id not in joined_ids and se.sub_id in self.result_map:
                other = self.result_map[se.sub_id]
                result = self._merge_dfs(result, other)
                joined_ids.add(se.sub_id)

        return result if result is not None else pd.DataFrame()

    @staticmethod
    def _merge_dfs(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        """Merge two DataFrames on common columns, or concatenate if no common columns."""
        if left is None:
            return right
        if right is None:
            return left

        common_cols = list(set(left.columns) & set(right.columns))
        if common_cols:
            try:
                return pd.merge(left, right, on=common_cols, how="inner",
                                suffixes=("", "_composed"))
            except Exception:
                return pd.concat([left, right], ignore_index=True)
        else:
            # No common columns — cross join or concat
            return pd.concat([left, right], axis=1)

    def get_result(self, sub_id: str) -> Optional[pd.DataFrame]:
        """Get a specific sub-expression result."""
        return self.result_map.get(sub_id)

    def clear(self) -> None:
        """Clear all cached results."""
        self.result_map.clear()
