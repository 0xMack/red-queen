"""What every command shares: the API client, `--json`, and turning spec files and `--set` overrides into payloads."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
import yaml

from redqueen_cli.client import ApiError, Client
from redqueen_cli.output import emit


@dataclass
class State:
    api_url: str | None = None
    as_json: bool = False
    transport: Any = None  # tests hand in the backend's TestClient
    _client: Client | None = field(default=None, repr=False)

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(self.api_url, self.transport)
        return self._client


def state(ctx: typer.Context) -> State:
    return ctx.ensure_object(State)


def load_spec(path: Path, sets: list[str] | None = None) -> dict[str, Any]:
    """A spec file (.yaml/.yml/.json) with `key.path=value` overrides applied (values parsed as YAML)."""
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if path.suffix in (".yaml", ".yml") else json.loads(text)
    if not isinstance(data, dict):
        raise typer.BadParameter(f"{path} isn't a spec (a mapping)")
    return apply_sets(data, sets or [])


def apply_sets(data: dict[str, Any], sets: list[str]) -> dict[str, Any]:
    for item in sets:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise typer.BadParameter(f"--set takes key.path=value, not {item!r}")
        *parents, last = key.split(".")
        node = data
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise typer.BadParameter(f"--set {key}: {part} isn't a mapping")
        node[last] = yaml.safe_load(raw)
    return data


def submit_job(ctx: typer.Context, spec: dict[str, Any], wait: bool) -> None:
    """POST /jobs; with `--wait`, follow its log to the end and exit with its outcome."""
    st = state(ctx)
    job = st.client.post("/jobs", json=spec)
    if not wait:
        emit(job, st.as_json, lambda j: typer.echo(f"queued {j['kind']} job {j['job_id']}"))
        return
    if not st.as_json:
        typer.echo(f"queued {job['kind']} job {job['job_id']}; following (Ctrl-C stops following, not the job)")
    finished = st.client.follow_job(job["job_id"], (lambda _t: None) if st.as_json else _write)
    emit(finished, st.as_json, _outcome)
    if finished["status"] != "succeeded":
        raise typer.Exit(1)


def _write(text: str) -> None:
    typer.echo(text, nl=False)


def _outcome(job: dict[str, Any]) -> None:
    typer.echo(f"job {job['job_id']} {job['status']}" + (f": {job['error']}" if job.get("error") else ""))
    if job.get("result"):
        typer.echo(json.dumps(job["result"], indent=2))


__all__ = ["ApiError", "State", "apply_sets", "load_spec", "state", "submit_job"]
