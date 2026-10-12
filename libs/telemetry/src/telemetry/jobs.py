"""Jobs and experiments (docs/design/0018, decision 2): what has been asked for, and how far it got.

A job is one payload (a train, evaluate or publish spec) on its way through the scheduler: `queued` (submitted),
`running` (claimed, a worker process started), then `succeeded`, `failed` or `cancelled`. The API records and serves
jobs; only the scheduler claims and runs them, and it alone moves one to `running` (atomically, so two schedulers
can't both take it). A cancel on a queued job cancels it outright; on a running one it sets `cancel_requested` for
the scheduler to act on. An experiment is a named spec whose fan-out is jobs with that `parent`.

Local-dev store: one SQLite file (`jobs.db`, beside `runs.db`), the same pattern as the run registry.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

JobKind = Literal["train", "evaluate", "publish"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
FINISHED: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})


class Job(BaseModel):
    job_id: str = Field(..., min_length=1)
    kind: JobKind
    spec: dict[str, Any] = Field(..., description="The payload as submitted (validated on submission).")
    status: JobStatus
    parent: str | None = Field(None, description="The experiment (or pipeline) this job is part of, if any.")
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    heartbeat_at: float | None = Field(None, description="The scheduler's last sign of the worker being alive.")
    cancel_requested: bool = False
    pid: int | None = None
    log: str | None = Field(None, description="The worker's log file, relative to the data directory.")
    result: dict[str, Any] | None = Field(None, description="What the job produced: a run id, a summary, a report.")
    error: str | None = None


class Experiment(BaseModel):
    name: str = Field(..., min_length=1)
    spec: dict[str, Any]
    created_at: float
    updated_at: float = Field(..., description="Last (re)submission: a failed cell is retried once this is newer.")


class JobStore(Protocol):
    def submit(self, kind: JobKind, spec: dict[str, Any], parent: str | None = None) -> Job: ...
    def get(self, job_id: str) -> Job: ...
    def list(
        self, status: JobStatus | None = None, kind: JobKind | None = None, parent: str | None = None
    ) -> list[Job]: ...
    def claim_next(self, kinds: set[JobKind] | None = None) -> Job | None: ...
    def update(self, job_id: str, **fields: Any) -> Job: ...
    def request_cancel(self, job_id: str) -> Job: ...
    def put_experiment(self, name: str, spec: dict[str, Any]) -> Experiment: ...
    def get_experiment(self, name: str) -> Experiment: ...
    def list_experiments(self) -> list[Experiment]: ...


_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        parent TEXT,
        created_at REAL NOT NULL,
        job TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, created_at)",
    """CREATE TABLE IF NOT EXISTS experiments (
        name TEXT PRIMARY KEY,
        spec TEXT NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )""",
]
UPDATABLE = frozenset(
    {"status", "started_at", "finished_at", "heartbeat_at", "cancel_requested", "pid", "log", "result", "error"}
)


class SqliteJobStore:
    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            for statement in _SCHEMA:
                conn.execute(statement)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, isolation_level=None)  # explicit transactions below
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _write(self, conn: sqlite3.Connection, job: Job) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO jobs (job_id, kind, status, parent, created_at, job) VALUES (?, ?, ?, ?, ?, ?)",
            (job.job_id, job.kind, job.status, job.parent, job.created_at, job.model_dump_json()),
        )

    def submit(self, kind: JobKind, spec: dict[str, Any], parent: str | None = None) -> Job:
        job = Job(job_id=uuid.uuid4().hex, kind=kind, spec=spec, status="queued", parent=parent, created_at=time.time())
        conn = self._connect()
        try:
            self._write(conn, job)
        finally:
            conn.close()
        return job

    def get(self, job_id: str) -> Job:
        conn = self._connect()
        try:
            row = conn.execute("SELECT job FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(f"no such job: {job_id}")
        return Job.model_validate_json(row["job"])

    def list(
        self, status: JobStatus | None = None, kind: JobKind | None = None, parent: str | None = None
    ) -> list[Job]:
        clauses, values = [], []
        for column, value in (("status", status), ("kind", kind), ("parent", parent)):
            if value is not None:
                clauses.append(f"{column} = ?")
                values.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        conn = self._connect()
        try:
            rows = conn.execute(f"SELECT job FROM jobs{where} ORDER BY created_at", values).fetchall()
        finally:
            conn.close()
        return [Job.model_validate_json(r["job"]) for r in rows]

    def claim_next(self, kinds: set[JobKind] | None = None) -> Job | None:
        """The oldest queued job (of `kinds`), moved to `running` in one transaction -- or None."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")  # take the write lock before reading: no two claimers see the same row
            rows = conn.execute("SELECT job FROM jobs WHERE status = 'queued' ORDER BY created_at").fetchall()
            for row in rows:
                job = Job.model_validate_json(row["job"])
                if kinds is None or job.kind in kinds:
                    now = time.time()
                    job = job.model_copy(update={"status": "running", "started_at": now, "heartbeat_at": now})
                    self._write(conn, job)
                    conn.execute("COMMIT")
                    return job
            conn.execute("COMMIT")
            return None
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def update(self, job_id: str, **fields: Any) -> Job:
        unknown = set(fields) - UPDATABLE
        if unknown:
            raise ValueError(f"not updatable: {sorted(unknown)}")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT job FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                raise KeyError(f"no such job: {job_id}")
            job = Job.model_validate({**json.loads(row["job"]), **fields})
            if job.status in FINISHED and job.finished_at is None:
                job = job.model_copy(update={"finished_at": time.time()})
            self._write(conn, job)
            conn.execute("COMMIT")
            return job
        finally:
            conn.close()

    def request_cancel(self, job_id: str) -> Job:
        """A queued job is cancelled at once; a running one is marked for the scheduler to stop; a finished one is
        left as it is."""
        job = self.get(job_id)
        if job.status == "queued":
            return self.update(job_id, status="cancelled")
        if job.status == "running":
            return self.update(job_id, cancel_requested=True)
        return job

    def put_experiment(self, name: str, spec: dict[str, Any]) -> Experiment:
        now = time.time()
        conn = self._connect()
        try:
            existing = conn.execute("SELECT created_at FROM experiments WHERE name = ?", (name,)).fetchone()
            # Resubmitting keeps when it began (an experiment is resumed, not restarted) and retries failed cells.
            created_at = existing["created_at"] if existing is not None else now
            conn.execute(
                "INSERT OR REPLACE INTO experiments (name, spec, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (name, json.dumps(spec), created_at, now),
            )
        finally:
            conn.close()
        return Experiment(name=name, spec=spec, created_at=created_at, updated_at=now)

    def get_experiment(self, name: str) -> Experiment:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM experiments WHERE name = ?", (name,)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(f"no such experiment: {name}")
        return _experiment(row)

    def list_experiments(self) -> list[Experiment]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
        finally:
            conn.close()
        return [_experiment(r) for r in rows]


def _experiment(row: sqlite3.Row) -> Experiment:
    return Experiment(
        name=row["name"], spec=json.loads(row["spec"]), created_at=row["created_at"], updated_at=row["updated_at"]
    )
