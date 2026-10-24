"""The API, as the CLI calls it: requests, errors as `ApiError`, run-id prefixes, and following a job to its end.

Any object with httpx's `request(method, url, params=, json=)` works as the transport -- an `httpx2.Client` against a
live server, or the backend's own `TestClient` in tests.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from typing import Any

DEFAULT_URL = "http://localhost:8000"
FINISHED = frozenset({"succeeded", "failed", "cancelled"})


class ApiError(Exception):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


class Client:
    def __init__(self, base_url: str | None = None, transport: Any = None, timeout: float = 30.0):
        self.base_url = (base_url or os.environ.get("REDQUEEN_API_URL") or DEFAULT_URL).rstrip("/")
        if transport is None:
            import httpx2

            transport = httpx2.Client(base_url=self.base_url, timeout=timeout)
        self._http = transport

    def request(self, method: str, path: str, *, params: dict[str, Any] | None = None, json: Any = None) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        try:
            response = self._http.request(method, path, params=params or None, json=json)
        except Exception as e:  # a connection error, whatever the transport calls it
            if type(e).__name__ in {"ConnectError", "ConnectTimeout"}:
                raise ApiError(0, f"can't reach the API at {self.base_url} (is the backend running?)") from None
            raise
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(response.status_code, detail)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    def get(self, path: str, **params: Any) -> Any:
        return self.request("GET", path, params=params)

    def post(self, path: str, json: Any = None, **params: Any) -> Any:
        return self.request("POST", path, params=params, json=json)

    def delete(self, path: str) -> Any:
        return self.request("DELETE", path)

    # --- composite calls ----------------------------------------------------------------------------------------

    def resolve_run(self, prefix: str) -> str:
        """A run id from a unique prefix (or the whole id)."""
        matches = [r["run_id"] for r in self.get("/runs") if r["run_id"].startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ApiError(404, f"no run starts with {prefix!r}")
        raise ApiError(400, f"{prefix!r} matches {len(matches)} runs: {', '.join(m[:12] for m in matches[:5])} ...")

    def leaderboard(self, game: str, protocol: str | None = None) -> list[dict[str, Any]]:
        """A game's leaderboard records under `protocol`, or under its newest protocol if none is given
        ("checkers.versus.v10" is newer than "v9") -- the API returns every protocol's, which aren't comparable."""
        records = self.get(f"/games/{game}/leaderboard", protocol=protocol)
        if protocol is None and records:
            latest = max({r["protocol"] for r in records}, key=_natural)
            records = [r for r in records if r["protocol"] == latest]
        return records

    def follow_job(self, job_id: str, write: Callable[[str], None], poll: float = 1.0) -> dict[str, Any]:
        """Streams a job's log (a pipeline's: each stage's, in turn) until it finishes; returns the finished job."""
        offsets: dict[str, int] = {}
        announced: set[str] = set()
        while True:
            job = self.get(f"/jobs/{job_id}")
            followed = [job]
            if job["kind"] == "pipeline":
                followed = self.get("/jobs", parent=job_id)
                for stage in followed:
                    if stage["job_id"] not in announced:
                        announced.add(stage["job_id"])
                        write(f"--- stage job {stage['job_id'][:8]} ({stage['kind']})\n")
            for each in followed:
                log = self.get(f"/jobs/{each['job_id']}/logs", offset=offsets.get(each["job_id"], 0))
                offsets[each["job_id"]] = log["offset"]
                if log["text"]:
                    write(log["text"])
            if job["status"] in FINISHED:
                return job
            time.sleep(poll)


def _natural(text: str) -> list[Any]:
    """A sort key that orders the numbers in `text` by value: v9 < v10."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]
