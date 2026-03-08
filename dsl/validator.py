"""DSL schema validation for HIFUN queries."""

import json
import os
from jsonschema import validate, ValidationError

_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")

_schema_cache = None


def _load_schema():
    global _schema_cache
    if _schema_cache is None:
        with open(_SCHEMA_PATH, "r") as f:
            _schema_cache = json.load(f)
    return _schema_cache


def validate_query(query: dict) -> list:
    """Validate a HIFUN DSL query against the JSON schema.

    Returns a list of error messages. Empty list means valid.
    """
    schema = _load_schema()
    errors = []

    # 1. JSON Schema validation
    try:
        validate(instance=query, schema=schema)
    except ValidationError as e:
        errors.append(f"Schema error: {e.message}")
        return errors

    # 2. Semantic validation
    operations = query.get("operations", [])
    op_ids = {op["op_id"] for op in operations}

    # Check for duplicate op_ids
    if len(op_ids) != len(operations):
        seen = set()
        for op in operations:
            if op["op_id"] in seen:
                errors.append(f"Duplicate op_id: {op['op_id']}")
            seen.add(op["op_id"])

    # Check depends_on references are valid
    for op in operations:
        for dep in op.get("depends_on", []):
            if dep not in op_ids:
                errors.append(
                    f"op_id '{op['op_id']}' depends on unknown op_id '{dep}'"
                )

    # Check for circular dependencies
    if not errors:
        cycle_err = _check_cycles(operations)
        if cycle_err:
            errors.append(cycle_err)

    # Check JOIN right_source references
    for op in operations:
        if op["type"] == "JOIN" and "join" in op:
            rs = op["join"]["right_source"]
            if rs not in op_ids:
                # It's a table name, not an op reference — that's fine
                pass

    return errors


def _check_cycles(operations: list) -> str | None:
    """Detect cycles in the dependency graph using DFS."""
    adj = {}
    for op in operations:
        adj[op["op_id"]] = op.get("depends_on", [])

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {oid: WHITE for oid in adj}

    def dfs(node):
        color[node] = GRAY
        for dep in adj.get(node, []):
            if dep not in color:
                continue
            if color[dep] == GRAY:
                return f"Circular dependency detected involving '{dep}'"
            if color[dep] == WHITE:
                result = dfs(dep)
                if result:
                    return result
        color[node] = BLACK
        return None

    for node in adj:
        if color[node] == WHITE:
            result = dfs(node)
            if result:
                return result
    return None


def validate_query_file(file_path: str) -> dict:
    """Validate a JSON DSL query file. Returns {valid: bool, errors: list, query: dict|None}."""
    try:
        with open(file_path, "r") as f:
            query = json.load(f)
    except json.JSONDecodeError as e:
        return {"valid": False, "errors": [f"Invalid JSON: {e}"], "query": None}

    # Handle files containing a list of queries
    if isinstance(query, list):
        all_errors = []
        for i, q in enumerate(query):
            errs = validate_query(q)
            for err in errs:
                all_errors.append(f"Query [{i}] ({q.get('query_id', '?')}): {err}")
        return {"valid": len(all_errors) == 0, "errors": all_errors, "query": query}

    errors = validate_query(query)
    return {"valid": len(errors) == 0, "errors": errors, "query": query}
