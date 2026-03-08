"""AST node dataclasses for the HIFUN DSL parser."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class QueryNode:
    """Represents a single operation in a parsed HIFUN query."""
    op_id: str
    op_type: str                      # FILTER, MAP, JOIN, TRAVERSAL, AGGREGATE
    source: str
    fields: List[str] = field(default_factory=list)
    predicate: Optional[Dict] = None
    join: Optional[Dict] = None
    traversal: Optional[Dict] = None
    aggregate: Optional[Dict] = None
    depends_on: List[str] = field(default_factory=list)


@dataclass
class SubExpression:
    """A routing unit — a group of QueryNodes dispatched to a single engine."""
    sub_id: str
    nodes: List[QueryNode]
    primary_op_type: str              # "RELATIONAL" or "TRAVERSAL"
    depends_on_subs: List[str]
    parallelizable: bool
    estimated_output_rows: int = 0    # filled by FeatureExtractor
