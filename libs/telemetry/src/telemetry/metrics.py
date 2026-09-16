import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from telemetry.types import GenerationStats


class MetricsSink(Protocol):
    def record_generation(self, stats: GenerationStats) -> None: ...


class MetricsSource(Protocol):
    def history(self, run_id: str, since_generation: int = 0) -> list[GenerationStats]: ...

    def subscribe(self, run_id: str, since_generation: int = 0) -> Iterator[GenerationStats]: ...


class FileMetricsStore:
    """Local-dev MetricsSink + MetricsSource backed by one JSON-lines file per run.

    `subscribe` polls the file rather than using OS-level file watching, which is fine at
    once-per-generation write frequency and keeps this dependency-free/cross-platform. It never
    returns on its own — it's a live tail, so the caller decides when to stop iterating (e.g. once
    the run's status, tracked separately in RunRegistry, is no longer "running").
    """

    def __init__(self, base_dir: str | Path, poll_interval: float = 0.5):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._poll_interval = poll_interval

    def _path(self, run_id: str) -> Path:
        return self._base_dir / f"{run_id}.jsonl"

    def record_generation(self, stats: GenerationStats) -> None:
        path = self._path(stats["run_id"])
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
            f.flush()

    def _read_all(self, path: Path) -> list[GenerationStats]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def history(self, run_id: str, since_generation: int = 0) -> list[GenerationStats]:
        return [
            stats
            for stats in self._read_all(self._path(run_id))
            if stats["generation"] >= since_generation
        ]

    def subscribe(self, run_id: str, since_generation: int = 0) -> Iterator[GenerationStats]:
        path = self._path(run_id)
        last_generation = since_generation - 1
        while True:
            for stats in self._read_all(path):
                if stats["generation"] > last_generation:
                    last_generation = stats["generation"]
                    yield stats
            time.sleep(self._poll_interval)
