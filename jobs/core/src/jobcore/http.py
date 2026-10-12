"""The HTTP sink (docs/design/0018, decision 3): telemetry's four store Protocols, implemented over the backend's API --
for a worker that can't share the data directory (a cluster pod). A job holding it can't tell it from the local
sink: `recorded_run()`, the control callback and every adapter call the same methods.

Errors keep their local meaning: an unknown run is a KeyError (as `SqliteRunRegistry.get_run`), an unknown artifact a
FileNotFoundError (as `FileArtifactStore`). Every request is synchronous and raises on any other failure -- a training
run that can't record should stop, not carry on unrecorded.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import httpx2
from telemetry import EvaluationRecord, GenerationStats, RunInfo, RunStatus, Stores

# The artifact store is flat (ref -> bytes); its routes are grouped under a run for REST readability only, and the
# backend ignores that segment -- a ref like `<run_id>-population` names no run of its own.
ANY_RUN = "_"


def _check(response: httpx2.Response, missing: type[Exception] | None = None, what: str = "") -> httpx2.Response:
    if missing is not None and response.status_code == 404:
        raise missing(what or response.text)
    response.raise_for_status()
    return response


class HttpRunRegistry:
    def __init__(self, client: httpx2.Client):
        self._client = client

    def create_run(self, config: dict[str, Any]) -> str:
        return _check(self._client.post("/runs", json={"config": config})).json()["run_id"]

    def _patch(self, run_id: str, body: dict[str, Any]) -> RunInfo:
        response = self._client.patch(f"/runs/{run_id}", json=body)
        return RunInfo.model_validate(_check(response, KeyError, f"no such run: {run_id}").json())

    def update_status(self, run_id: str, status: RunStatus) -> None:
        self._patch(run_id, {"status": status})

    def set_summary(self, run_id: str, summary: dict[str, Any]) -> None:
        self._patch(run_id, {"summary": summary})

    def update_config(self, run_id: str, patch: dict[str, Any]) -> RunInfo:
        return self._patch(run_id, {"config": patch})

    def get_run(self, run_id: str) -> RunInfo:
        response = self._client.get(f"/runs/{run_id}")
        return RunInfo.model_validate(_check(response, KeyError, f"no such run: {run_id}").json())

    def list_runs(self) -> list[RunInfo]:
        return [RunInfo.model_validate(r) for r in _check(self._client.get("/runs")).json()]


class HttpMetricsStore:
    def __init__(self, client: httpx2.Client, poll_interval: float = 0.5):
        self._client = client
        self._poll_interval = poll_interval

    def record_generation(self, stats: GenerationStats) -> None:
        _check(self._client.post(f"/runs/{stats.run_id}/metrics", json=[stats.model_dump(mode="json")]))

    def history(self, run_id: str, since_generation: int = 0) -> list[GenerationStats]:
        response = self._client.get(f"/runs/{run_id}/metrics/history", params={"since_generation": since_generation})
        return [GenerationStats.model_validate(s) for s in _check(response).json()]

    def subscribe(self, run_id: str, since_generation: int = 0) -> Iterator[GenerationStats]:
        """Polls the history, as the backend's own stream does; never ends (a caller stops consuming)."""
        next_generation = since_generation
        while True:
            for stats in self.history(run_id, next_generation):
                next_generation = max(next_generation, stats.generation + 1)
                yield stats
            time.sleep(self._poll_interval)


class HttpArtifactStore:
    def __init__(self, client: httpx2.Client):
        self._client = client

    def _url(self, ref: str) -> str:
        return f"/runs/{ANY_RUN}/artifacts/{ref}"

    def _put(self, ref: str, data: bytes, kind: str) -> None:
        _check(self._client.put(self._url(ref), params={"kind": kind}, content=data))

    def _get(self, ref: str, kind: str) -> bytes:
        response = self._client.get(self._url(ref), params={"kind": kind})
        return _check(response, FileNotFoundError, f"no such {kind} artifact: {ref}").content

    def put_program(self, ref: str, program: bytes) -> None:
        self._put(ref, program, "program")

    def get_program(self, ref: str) -> bytes:
        return self._get(ref, "program")

    def put_trace(self, ref: str, trace: bytes) -> None:
        self._put(ref, trace, "trace")

    def get_trace(self, ref: str) -> bytes:
        return self._get(ref, "trace")


class HttpEvaluationStore:
    def __init__(self, client: httpx2.Client):
        self._client = client

    def put(self, record: EvaluationRecord) -> None:
        _check(self._client.put(f"/games/{record.game}/evaluations", json=record.model_dump(mode="json")))

    def list(self, game: str, protocol: str | None = None) -> list[EvaluationRecord]:
        params = {"protocol": protocol} if protocol else {}
        response = self._client.get(f"/games/{game}/leaderboard", params=params)
        return [EvaluationRecord.model_validate(r) for r in _check(response).json()]

    def prune(self, game: str, protocol: str, keep: set[str]) -> list[str]:
        body = {"protocol": protocol, "keep": sorted(keep)}
        return _check(self._client.post(f"/games/{game}/evaluations/prune", json=body)).json()["removed"]


def http_sink(target: str | httpx2.Client, timeout: float = 60.0) -> Stores:
    """The HTTP sink for a backend at `target` (a base URL), or over an existing client (tests pass the backend's own
    TestClient)."""
    client = target if isinstance(target, httpx2.Client) else httpx2.Client(base_url=target, timeout=timeout)
    return Stores(
        registry=HttpRunRegistry(client),
        metrics=HttpMetricsStore(client),
        artifacts=HttpArtifactStore(client),
        evaluations=HttpEvaluationStore(client),
    )
