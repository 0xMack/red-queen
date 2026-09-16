import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Protocol, TypedDict


class RunInfo(TypedDict):
    run_id: str
    config: dict[str, Any]
    status: str  # "running" | "paused" | "completed" | "failed"
    created_at: float
    updated_at: float
    summary: dict[str, Any] | None


class RunRegistry(Protocol):
    def create_run(self, config: dict[str, Any]) -> str: ...
    def update_status(self, run_id: str, status: str) -> None: ...
    def set_summary(self, run_id: str, summary: dict[str, Any]) -> None: ...
    def get_run(self, run_id: str) -> RunInfo: ...
    def list_runs(self) -> list[RunInfo]: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    config TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    summary TEXT
);
"""


class SqliteRunRegistry:
    """Local-dev RunRegistry backed by a single SQLite file (WAL mode: one writer, many readers)."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def create_run(self, config: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex
        now = time.time()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO runs (run_id, config, status, created_at, updated_at, summary) "
                "VALUES (?, ?, ?, ?, ?, NULL)",
                (run_id, json.dumps(config), "running", now, now),
            )
            conn.commit()
        finally:
            conn.close()
        return run_id

    def update_status(self, run_id: str, status: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
                (status, time.time(), run_id),
            )
            conn.commit()
        finally:
            conn.close()

    def set_summary(self, run_id: str, summary: dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE runs SET summary = ?, updated_at = ? WHERE run_id = ?",
                (json.dumps(summary), time.time(), run_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_run(self, run_id: str) -> RunInfo:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(f"no such run: {run_id}")
        return _row_to_run_info(row)

    def list_runs(self) -> list[RunInfo]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        finally:
            conn.close()
        return [_row_to_run_info(row) for row in rows]


def _row_to_run_info(row: sqlite3.Row) -> RunInfo:
    return RunInfo(
        run_id=row["run_id"],
        config=json.loads(row["config"]),
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        summary=json.loads(row["summary"]) if row["summary"] is not None else None,
    )
