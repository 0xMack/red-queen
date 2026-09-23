import threading
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

    `history` keeps what it has parsed per run and, since a job only ever appends, reads just the lines added since
    the last call -- so a long-lived reader (the API) re-serving 100+ runs pays for parsing each line once, not on
    every request. A file that shrank (rewritten by hand) is parsed again from the start.
    """

    def __init__(self, base_dir: str | Path, poll_interval: float = 0.5):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._poll_interval = poll_interval
        self._parsed: dict[Path, tuple[int, list[GenerationStats]]] = {}  # path -> (bytes parsed, stats)
        self._lock = threading.Lock()  # FastAPI serves sync routes from a thread pool

    def _path(self, run_id: str) -> Path:
        return self._base_dir / f"{run_id}.jsonl"

    def record_generation(self, stats: GenerationStats) -> None:
        path = self._path(stats.run_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(stats.model_dump_json() + "\n")
            f.flush()

    def _read_from(self, path: Path, offset: int = 0) -> tuple[list[GenerationStats], int]:
        """Parses the complete lines after byte `offset`; returns them and the offset just past the last one.

        A job appends one line per generation, so a reader can catch a line mid-write -- anything after
        the last newline is left for the next read instead of failing to parse.
        """
        if not path.exists():
            return [], offset
        with path.open("rb") as f:
            f.seek(offset)
            chunk = f.read()
        complete = chunk[: chunk.rfind(b"\n") + 1]
        stats = [GenerationStats.model_validate_json(line) for line in complete.splitlines() if line.strip()]
        return stats, offset + len(complete)

    def history(self, run_id: str, since_generation: int = 0) -> list[GenerationStats]:
        path = self._path(run_id)
        with self._lock:
            offset, parsed = self._parsed.get(path, (0, []))
            size = path.stat().st_size if path.exists() else 0
            if size < offset:  # rewritten, not appended to: start over
                offset, parsed = 0, []
            if size > offset:
                new, offset = self._read_from(path, offset)
                parsed = parsed + new
                self._parsed[path] = (offset, parsed)
        return [s for s in parsed if s.generation >= since_generation]

    def subscribe(self, run_id: str, since_generation: int = 0) -> Iterator[GenerationStats]:
        # Tails by byte offset, so each poll parses only what was appended since the last one.
        path = self._path(run_id)
        offset = 0
        last_generation = since_generation - 1
        while True:
            stats, offset = self._read_from(path, offset)
            for s in stats:
                if s.generation > last_generation:
                    last_generation = s.generation
                    yield s
            time.sleep(self._poll_interval)
