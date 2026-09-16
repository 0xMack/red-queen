from pathlib import Path
from typing import Protocol


class ArtifactStore(Protocol):
    def put_program(self, ref: str, program: bytes) -> None: ...
    def get_program(self, ref: str) -> bytes: ...
    def put_trace(self, ref: str, trace: bytes) -> None: ...
    def get_trace(self, ref: str) -> bytes: ...


class FileArtifactStore:
    """Local-dev ArtifactStore backed by plain files, one per (kind, ref)."""

    def __init__(self, base_dir: str | Path):
        self._base_dir = Path(base_dir)
        (self._base_dir / "programs").mkdir(parents=True, exist_ok=True)
        (self._base_dir / "traces").mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, ref: str) -> Path:
        return self._base_dir / kind / ref

    def put_program(self, ref: str, program: bytes) -> None:
        self._path("programs", ref).write_bytes(program)

    def get_program(self, ref: str) -> bytes:
        return self._path("programs", ref).read_bytes()

    def put_trace(self, ref: str, trace: bytes) -> None:
        self._path("traces", ref).write_bytes(trace)

    def get_trace(self, ref: str) -> bytes:
        return self._path("traces", ref).read_bytes()
