"""Leaderboard evaluations (docs/design/0007).

An evaluation is one entrant (a run's champion, a fixed baseline, later a human) measured under one
versioned protocol (e.g. "snake.score.v1": fixed held-out seeds, step cap, metric = game score).
Kept separate from runs/metrics on purpose: training fitness is never used for ranking, and one
champion can be re-evaluated under a new protocol without touching its run.

`metrics` is a free-form dict of measurement groups (quality / inference / training), same
flexibility as RunInfo.config -- the protocol version, not a schema migration, is what says two
records are comparable.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

EntrantKind = Literal["champion", "baseline", "human"]


class EvaluationRecord(BaseModel):
    game: str = Field(..., min_length=1)
    protocol: str = Field(..., min_length=1, description="Versioned evaluation protocol, e.g. 'snake.score.v1'.")
    entrant_id: str = Field(..., min_length=1, description="Stable id, e.g. 'run:<run_id>' or 'baseline:greedy'.")
    entrant_kind: EntrantKind
    label: str = Field(..., description="Human-readable entrant name.")
    interface: str = Field(..., description="Interface id the entrant played under (docs/design/0007).")
    run_id: str | None = None
    champion_ref: str | None = None
    created_at: float = Field(..., gt=0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)


class EvaluationStore(Protocol):
    def put(self, record: EvaluationRecord) -> None:
        """Insert, or replace the existing record for the same (protocol, entrant_id)."""
        ...

    def list(self, game: str, protocol: str | None = None) -> list[EvaluationRecord]: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
    protocol TEXT NOT NULL,
    entrant_id TEXT NOT NULL,
    game TEXT NOT NULL,
    record TEXT NOT NULL,
    PRIMARY KEY (protocol, entrant_id)
);
"""


class SqliteEvaluationStore:
    """Local-dev EvaluationStore: one SQLite file (evaluations.db, beside runs.db)."""

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
        return conn

    def put(self, record: EvaluationRecord) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO evaluations (protocol, entrant_id, game, record) VALUES (?, ?, ?, ?)",
                (record.protocol, record.entrant_id, record.game, record.model_dump_json()),
            )
            conn.commit()
        finally:
            conn.close()

    def list(self, game: str, protocol: str | None = None) -> list[EvaluationRecord]:
        conn = self._connect()
        try:
            if protocol is None:
                rows = conn.execute("SELECT record FROM evaluations WHERE game = ?", (game,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT record FROM evaluations WHERE game = ? AND protocol = ?", (game, protocol)
                ).fetchall()
        finally:
            conn.close()
        return [EvaluationRecord.model_validate(json.loads(row[0])) for row in rows]
