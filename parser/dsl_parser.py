"""DSL Parser: validates JSON DSL queries and converts them to QueryNode ASTs."""

from collections import deque
from typing import List, Dict

from dsl.validator import validate_query
from parser.ast_nodes import QueryNode


class DSLParser:
    """Parses a JSON DSL query dict into an ordered list of QueryNode objects."""

    def parse(self, query_json: dict) -> List[QueryNode]:
        """Validate JSON DSL and return QueryNodes in topological (dependency) order.

        Raises ValueError if the query is invalid.
        """
        errors = validate_query(query_json)
        if errors:
            raise ValueError(f"Invalid DSL query: {'; '.join(errors)}")

        nodes = []
        for op in query_json["operations"]:
            node = QueryNode(
                op_id=op["op_id"],
                op_type=op["type"],
                source=op["source"],
                fields=op.get("fields", []),
                predicate=op.get("predicate"),
                join=op.get("join"),
                traversal=op.get("traversal"),
                aggregate=op.get("aggregate"),
                depends_on=op.get("depends_on", []),
            )
            nodes.append(node)

        return self._topological_sort(nodes)

    def _topological_sort(self, nodes: List[QueryNode]) -> List[QueryNode]:
        """Kahn's algorithm — returns nodes in dependency order."""
        node_map: Dict[str, QueryNode] = {n.op_id: n for n in nodes}

        # Build in-degree counts and adjacency
        in_degree: Dict[str, int] = {n.op_id: 0 for n in nodes}
        dependents: Dict[str, List[str]] = {n.op_id: [] for n in nodes}

        for node in nodes:
            for dep in node.depends_on:
                if dep in node_map:
                    in_degree[node.op_id] += 1
                    dependents[dep].append(node.op_id)

        # Start with nodes that have no dependencies
        queue = deque(
            oid for oid, deg in in_degree.items() if deg == 0
        )
        sorted_nodes: List[QueryNode] = []

        while queue:
            oid = queue.popleft()
            sorted_nodes.append(node_map[oid])
            for child_id in dependents[oid]:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        if len(sorted_nodes) != len(nodes):
            raise ValueError("Dependency cycle detected in query operations")

        return sorted_nodes

    def parse_file(self, file_path: str) -> Dict[str, List[QueryNode]]:
        """Parse a JSON file containing one query or a list of queries.

        Returns a dict mapping query_id -> list of QueryNodes.
        """
        import json
        with open(file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            result = {}
            for query_json in data:
                qid = query_json.get("query_id", "unknown")
                result[qid] = self.parse(query_json)
            return result

        qid = data.get("query_id", "unknown")
        return {qid: self.parse(data)}
