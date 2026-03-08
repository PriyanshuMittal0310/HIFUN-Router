"""HistoricalStore: SQLite-based runtime history for subexpression fingerprints."""

import hashlib
import sqlite3
import time
from typing import Optional, Tuple


class HistoricalStore:
    """Stores and retrieves historical runtime data for subexpression fingerprints.

    Used by the FeatureExtractor to provide hist_avg_runtime_ms and
    hist_runtime_variance features based on past execution data.
    """

    def __init__(self, db_path: str = "data/runtime_history.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS runtime_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL,
                engine TEXT NOT NULL,
                runtime_ms REAL NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fingerprint
            ON runtime_history (fingerprint)
        """)
        self._conn.commit()

    def record(self, fingerprint: str, engine: str, runtime_ms: float):
        """Record a runtime measurement for a subexpression fingerprint."""
        self._conn.execute(
            "INSERT INTO runtime_history (fingerprint, engine, runtime_ms, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (fingerprint, engine, runtime_ms, time.time()),
        )
        self._conn.commit()

    def lookup(self, fingerprint: str) -> Tuple[float, float]:
        """Look up historical avg runtime and variance for a fingerprint.

        Returns (avg_runtime_ms, variance_ms). Returns (-1.0, -1.0) if no history.
        """
        row = self._conn.execute(
            "SELECT AVG(runtime_ms) as avg_ms, "
            "       CASE WHEN COUNT(*) > 1 THEN "
            "         SUM((runtime_ms - sub.mean) * (runtime_ms - sub.mean)) / (COUNT(*) - 1) "
            "       ELSE 0.0 END as var_ms "
            "FROM runtime_history, "
            "     (SELECT AVG(runtime_ms) as mean FROM runtime_history WHERE fingerprint = ?) sub "
            "WHERE fingerprint = ?",
            (fingerprint, fingerprint),
        ).fetchone()

        if row and row["avg_ms"] is not None:
            return (float(row["avg_ms"]), float(row["var_ms"]))
        return (-1.0, -1.0)

    def close(self):
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass

    @staticmethod
    def compute_fingerprint(op_types: list, source_name: str) -> str:
        """Compute a stable fingerprint from sorted op types and source name."""
        key = "|".join(sorted(op_types)) + "|" + source_name
        return hashlib.sha256(key.encode()).hexdigest()[:16]
