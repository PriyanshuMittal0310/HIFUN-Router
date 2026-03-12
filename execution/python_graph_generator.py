"""python_graph_generator — re-exports the pandas-based GraphGenerator for legacy compatibility.

Use this module when running unit tests or lightweight experiments that do not
require a SparkSession.  For production / paper experiments, use
execution.graphframes_generator.GraphFramesGenerator instead.
"""

# Re-export so callers can do:
#   from execution.python_graph_generator import GraphGenerator
from execution.graph_generator import GraphGenerator  # noqa: F401

__all__ = ["GraphGenerator"]
