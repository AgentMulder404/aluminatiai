"""
Local time-series database — SQLite-based rolling metric store.

Stores GPU metrics locally for historical queries without requiring
the cloud API. Auto-prunes data older than RETENTION_DAYS on startup.

CLI: aluminatiai query --metric power --gpu 0 --from DATE --to DATE

Schema:
    metrics(timestamp INTEGER, gpu_index INTEGER, metric TEXT, value REAL)
    CREATE INDEX idx_ts_gpu_metric ON metrics(timestamp, gpu_index, metric)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LocalTSDB:
    """SQLite time-series store with automatic retention pruning."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        retention_days: int = 7,
    ):
        if db_path is None:
            from config import DATA_DIR
            db_path = str(DATA_DIR / "metrics.db")

        self._path = db_path
        self._retention_days = retention_days
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Create tables and prune old data."""
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp INTEGER NOT NULL,
                gpu_index INTEGER NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ts_gpu_metric
            ON metrics(timestamp, gpu_index, metric)
        """)
        self._conn.commit()
        self._prune()

    def _prune(self) -> None:
        """Remove data older than retention period."""
        cutoff = int(time.time()) - (self._retention_days * 86400)
        cursor = self._conn.execute(
            "DELETE FROM metrics WHERE timestamp < ?", (cutoff,)
        )
        if cursor.rowcount > 0:
            self._conn.commit()
            logger.info("TSDB: pruned %d rows older than %d days",
                       cursor.rowcount, self._retention_days)

    def insert(self, gpu_index: int, metrics: dict[str, float]) -> None:
        """Insert metric values for one GPU at the current timestamp."""
        if not self._conn:
            return
        ts = int(time.time())
        rows = [(ts, gpu_index, k, v) for k, v in metrics.items() if v is not None]
        self._conn.executemany(
            "INSERT INTO metrics (timestamp, gpu_index, metric, value) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def insert_batch(self, gpu_metrics: list) -> None:
        """Insert a batch of GPUMetrics objects."""
        if not self._conn:
            return
        ts = int(time.time())
        rows = []
        for m in gpu_metrics:
            rows.append((ts, m.gpu_index, "power_w", m.power_draw_w))
            rows.append((ts, m.gpu_index, "utilization_pct", m.utilization_gpu_pct))
            rows.append((ts, m.gpu_index, "temperature_c", m.temperature_c))
            rows.append((ts, m.gpu_index, "memory_used_mb", m.memory_used_mb))
            if m.energy_delta_j:
                rows.append((ts, m.gpu_index, "energy_j", m.energy_delta_j))
        self._conn.executemany(
            "INSERT INTO metrics (timestamp, gpu_index, metric, value) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def query(
        self,
        metric: str,
        gpu_index: int = 0,
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None,
        limit: int = 10000,
    ) -> list[tuple[int, float]]:
        """Query metric values. Returns list of (timestamp, value) tuples."""
        if not self._conn:
            return []

        sql = "SELECT timestamp, value FROM metrics WHERE metric = ? AND gpu_index = ?"
        params: list = [metric, gpu_index]

        if from_ts:
            sql += " AND timestamp >= ?"
            params.append(from_ts)
        if to_ts:
            sql += " AND timestamp <= ?"
            params.append(to_ts)

        sql += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        return self._conn.execute(sql, params).fetchall()

    def stats(self) -> dict:
        """Return DB statistics."""
        if not self._conn:
            return {}
        row_count = self._conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        size_bytes = os.path.getsize(self._path) if os.path.exists(self._path) else 0
        return {
            "rows": row_count,
            "size_mb": round(size_bytes / 1_048_576, 2),
            "path": self._path,
            "retention_days": self._retention_days,
        }

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query local GPU metrics history")
    parser.add_argument("--metric", required=True,
                        choices=["power_w", "utilization_pct", "temperature_c", "memory_used_mb", "energy_j"],
                        help="Metric to query")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--from", dest="from_date", help="Start date (ISO format)")
    parser.add_argument("--to", dest="to_date", help="End date (ISO format)")
    parser.add_argument("--format", dest="fmt", choices=["json", "csv"], default="csv")
    parser.add_argument("--limit", type=int, default=10000, help="Max rows")
    return parser


def run_query(args: argparse.Namespace) -> int:
    from datetime import datetime

    from_ts = None
    to_ts = None
    if args.from_date:
        from_ts = int(datetime.fromisoformat(args.from_date).timestamp())
    if args.to_date:
        to_ts = int(datetime.fromisoformat(args.to_date).timestamp())

    db = LocalTSDB()
    results = db.query(
        metric=args.metric,
        gpu_index=args.gpu,
        from_ts=from_ts,
        to_ts=to_ts,
        limit=args.limit,
    )

    if not results:
        print(f"No data found for metric={args.metric} gpu={args.gpu}")
        db.close()
        return 0

    if args.fmt == "csv":
        print("timestamp,value")
        for ts, val in results:
            print(f"{datetime.fromtimestamp(ts).isoformat()},{val}")
    else:
        data = [{"timestamp": ts, "value": val} for ts, val in results]
        print(json.dumps(data, indent=2))

    stats = db.stats()
    if stats:
        import sys
        print(f"\n# DB: {stats.get('rows', 0)} rows, {stats.get('size_mb', 0)} MB", file=sys.stderr)

    db.close()
    return 0
