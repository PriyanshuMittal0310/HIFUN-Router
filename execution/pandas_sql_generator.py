"""pandas_sql_generator — re-exports the pandas-based SQLGenerator for legacy compatibility.

Use this module when running unit tests or lightweight experiments that do not
require a SparkSession.  For production / paper experiments, use
execution.spark_sql_generator.SparkSQLGenerator instead.
"""

# Re-export so callers can do:
#   from execution.pandas_sql_generator import SQLGenerator
from execution.sql_generator import SQLGenerator  # noqa: F401

__all__ = ["SQLGenerator"]
