"""QueryDecomposer: groups parsed QueryNodes into SubExpression routing units."""

from collections import deque
from typing import List, Dict, Set

from parser.ast_nodes import QueryNode, SubExpression

# Op types that naturally belong in SQL routing units
_RELATIONAL_OPS = {"FILTER", "MAP", "JOIN", "AGGREGATE"}


class QueryDecomposer:
    """Decomposes a parsed query AST into candidate routing units (SubExpressions).

    Decomposition rules (from project spec):
      - A TRAVERSAL node is always its own candidate unit (isolated for GRAPH routing).
      - Contiguous FILTER + JOIN + AGGREGATE chains form one SQL candidate unit.
      - A MAP node is merged with its parent unit (avoids fragmentation).
      - Nodes with no shared upstream dependencies are candidates to run in parallel.
    """

    def decompose(self, nodes: List[QueryNode]) -> List[SubExpression]:
        """Split a topologically-sorted list of QueryNodes into SubExpressions."""
        if not nodes:
            return []

        node_map: Dict[str, QueryNode] = {n.op_id: n for n in nodes}

        # --- STEP 1: Build dependency graph ---
        children: Dict[str, List[str]] = {n.op_id: [] for n in nodes}
        for node in nodes:
            for dep in node.depends_on:
                if dep in children:
                    children[dep].append(node.op_id)

        # --- STEP 2: Identify TRAVERSAL nodes — each is its own SubExpression ---
        traversal_ids: Set[str] = {n.op_id for n in nodes if n.op_type == "TRAVERSAL"}

        # --- STEP 3: Group contiguous relational chains via BFS ---
        # Track which group each node belongs to
        node_to_group: Dict[str, int] = {}
        groups: List[List[str]] = []  # group_index -> list of op_ids

        # First, assign every TRAVERSAL its own group
        for tid in traversal_ids:
            gid = len(groups)
            groups.append([tid])
            node_to_group[tid] = gid

        # BFS from root nodes to build relational groups
        roots = [n for n in nodes if not n.depends_on]
        visited: Set[str] = set()

        for root in roots:
            if root.op_id in visited:
                continue
            self._build_groups(
                root.op_id, node_map, children, traversal_ids,
                node_to_group, groups, visited,
            )

        # Handle any unvisited nodes (shouldn't happen with valid input)
        for node in nodes:
            if node.op_id not in node_to_group:
                gid = len(groups)
                groups.append([node.op_id])
                node_to_group[node.op_id] = gid

        # --- STEP 4: Build SubExpression objects ---
        sub_expressions: List[SubExpression] = []

        for gid, op_ids in enumerate(groups):
            group_nodes = [node_map[oid] for oid in op_ids]

            # Determine primary type
            has_trav = any(n.op_type == "TRAVERSAL" for n in group_nodes)
            primary_op_type = "TRAVERSAL" if has_trav else "RELATIONAL"

            # Find which other groups this group depends on
            dep_groups: Set[int] = set()
            for node in group_nodes:
                for dep_id in node.depends_on:
                    if dep_id in node_to_group:
                        dep_gid = node_to_group[dep_id]
                        if dep_gid != gid:
                            dep_groups.add(dep_gid)

            depends_on_subs = [f"sub_{dgid}" for dgid in sorted(dep_groups)]

            sub_expr = SubExpression(
                sub_id=f"sub_{gid}",
                nodes=group_nodes,
                primary_op_type=primary_op_type,
                depends_on_subs=depends_on_subs,
                parallelizable=False,  # set below
            )
            sub_expressions.append(sub_expr)

        # --- STEP 5: Mark parallelizable ---
        self._mark_parallelizable(sub_expressions)

        return sub_expressions

    def _build_groups(
        self,
        start_id: str,
        node_map: Dict[str, QueryNode],
        children: Dict[str, List[str]],
        traversal_ids: Set[str],
        node_to_group: Dict[str, int],
        groups: List[List[str]],
        visited: Set[str],
    ):
        """BFS to build contiguous relational groups, stopping at TRAVERSAL boundaries."""
        if start_id in visited:
            return
        if start_id in traversal_ids:
            # Already its own group; process children from here
            visited.add(start_id)
            for child_id in children.get(start_id, []):
                self._build_groups(
                    child_id, node_map, children, traversal_ids,
                    node_to_group, groups, visited,
                )
            return

        # Start a new relational group via BFS
        current_group_id = len(groups)
        group_ops: List[str] = []
        groups.append(group_ops)

        queue = deque([start_id])
        while queue:
            oid = queue.popleft()
            if oid in visited:
                continue
            node = node_map[oid]

            # If this node is a TRAVERSAL, it's already its own group; don't merge
            if oid in traversal_ids:
                continue

            # Check if this node depends on a different existing group that isn't
            # the current group (break the chain if dependency crosses groups)
            can_merge = True
            for dep_id in node.depends_on:
                if dep_id in node_to_group and node_to_group[dep_id] != current_group_id:
                    # This node depends on another group — it should still join
                    # the current chain if it's relational and the dep is resolved
                    pass
                if dep_id in traversal_ids and dep_id not in visited:
                    can_merge = False

            if not can_merge:
                continue

            visited.add(oid)
            group_ops.append(oid)
            node_to_group[oid] = current_group_id

            # Continue BFS to children that are relational ops
            for child_id in children.get(oid, []):
                if child_id not in visited and child_id not in traversal_ids:
                    # Check all dependencies of child are visited
                    child_node = node_map[child_id]
                    all_deps_ready = all(
                        d in visited or d not in node_map
                        for d in child_node.depends_on
                    )
                    if all_deps_ready and child_node.op_type in _RELATIONAL_OPS:
                        queue.append(child_id)

        # After relational chain, recurse into children that were skipped
        for oid in list(group_ops):
            for child_id in children.get(oid, []):
                if child_id not in visited:
                    self._build_groups(
                        child_id, node_map, children, traversal_ids,
                        node_to_group, groups, visited,
                    )

    def _mark_parallelizable(self, sub_expressions: List[SubExpression]):
        """Mark SubExpressions with no shared upstream dependencies as parallelizable."""
        # SubExpressions with no dependencies on other subs can run in parallel
        no_dep_subs = [se for se in sub_expressions if not se.depends_on_subs]
        for se in no_dep_subs:
            se.parallelizable = True

        # Also mark subs whose dependencies are all on independent branches
        if len(no_dep_subs) > 1:
            for se in no_dep_subs:
                se.parallelizable = True
