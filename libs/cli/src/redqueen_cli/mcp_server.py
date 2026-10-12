"""`redqueen mcp serve` (docs/design/0018, decision 6): the CLI's API layer as MCP tools, so an agent drives the project
the way a person drives the CLI.

Each tool is one or two API calls on the same `Client` the commands use, returning the API's JSON. Long work is never one
blocking call: `submit_*` returns a job, then `get_job` / `job_logs` (by byte offset, to follow a log) and `cancel_job`.
Specs are the API's own payloads (`schemas` lists them, with each algorithm's, protocol's and target's params); the API
validates every one and a tool call fails with its reason. Needs the `mcp` extra: `uv sync --all-packages --all-extras`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, Literal

from redqueen_cli.client import ApiError, Client
from redqueen_cli.state import apply_sets

INSTRUCTIONS = """Red Queen: from-scratch evolutionary and reinforcement learning against purpose-built games.
Runs are training runs (curves, champions); jobs are work the scheduler runs (train, evaluate, publish, pipeline).
Start work with a submit_* tool, then poll get_job / job_logs (pass the returned offset back to follow a log);
nothing blocks. Call `schemas` first to see which algorithms, protocols and publish targets exist and their params.
Checked-in specs live under experiments/ (experiments, pipelines/, schedules/): submit_spec_file submits one.
Run and job ids may be given as unique prefixes."""


def _trim(record: dict[str, Any]) -> dict[str, Any]:
    """A leaderboard record without its per-game score list (200+ numbers an agent doesn't need to rank entrants)."""
    quality = record.get("metrics", {}).get("quality")
    if isinstance(quality, dict) and "scores" in quality:
        record = {
            **record,
            "metrics": {**record["metrics"], "quality": {k: v for k, v in quality.items() if k != "scores"}},
        }
    return record


def build_server(client: Client, root: Path | None = None) -> Any:
    """The MCP server over `client`. `root` is where relative spec paths resolve (default: the working directory)."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    server = MCPServer(name="redqueen", instructions=INSTRUCTIONS)
    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    acting = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)

    def tool(annotations: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def register(fn: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(fn)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                try:
                    return fn(*args, **kwargs)
                except ApiError as e:  # an anticipated failure: the model reads the API's reason (any other
                    # exception is a crash, which the SDK reports only as "Error executing tool")
                    raise ToolError(f"{e.status}: {e.detail}" if e.status else str(e.detail)) from None

            server.add_tool(wrapped, annotations=annotations)
            return wrapped

        return register

    def job_id(prefix: str) -> str:
        matches = [j["job_id"] for j in client.get("/jobs") if j["job_id"].startswith(prefix)]
        if len(matches) != 1:
            raise ApiError(404 if not matches else 400, f"{prefix!r} matches {len(matches)} jobs")
        return matches[0]

    # --- what exists ------------------------------------------------------------------------------------------------

    @tool(read_only)
    def schemas(
        section: Literal["kinds", "algorithms", "protocols", "publish_targets"] | None = None,
    ) -> dict[str, Any]:
        """What the API accepts: spec kinds, and every algorithm, protocol and publish target with its params' JSON
        Schema. Without a section: just the names in each."""
        data = client.get("/specs/schemas")
        if section is None:
            return {key: sorted(value) for key, value in data.items()}
        return data[section]

    # --- runs -------------------------------------------------------------------------------------------------------

    @tool(read_only)
    def list_runs(
        game: str | None = None, status: str | None = None, experiment: str | None = None, limit: int = 30
    ) -> list[dict[str, Any]]:
        """Training runs, newest first, each with its config, status, generations so far and best fitness."""
        summaries = {s["run_id"]: s for s in client.get("/runs/summaries", trend_points=1)}
        rows = []
        for run in client.get("/runs"):
            c = run["config"]
            if (game and c.get("game") != game) or (status and run["status"] != status):
                continue
            if experiment and c.get("experiment") != experiment:
                continue
            s = summaries.get(run["run_id"], {})
            rows.append({**run, "generations": s.get("generations"), "best_fitness": s.get("best_fitness")})
        return rows[:limit] if limit else rows

    @tool(read_only)
    def get_run(run: str) -> dict[str, Any]:
        """One run (an id or unique prefix): config, status, summary."""
        return client.get(f"/runs/{client.resolve_run(run)}")

    @tool(read_only)
    def run_history(run: str, since_generation: int = 0) -> list[dict[str, Any]]:
        """A run's recorded generations (or RL iterations): best/mean fitness, diversity, held-out score, champion ref."""
        return client.get(f"/runs/{client.resolve_run(run)}/metrics/history", since_generation=since_generation or None)

    @tool(acting)
    def control_run(run: str, action: Literal["pause", "resume", "step"]) -> dict[str, Any]:
        """Pauses a running run between generations, resumes it, or lets a paused one train exactly one generation."""
        run_id = client.resolve_run(run)
        client.post(f"/runs/{run_id}/control", json={"action": action})
        return {"run_id": run_id, "action": action}

    # --- leaderboards and models ------------------------------------------------------------------------------------

    @tool(read_only)
    def leaderboard(game: str, protocol: str | None = None) -> list[dict[str, Any]]:
        """A game's leaderboard records, best first (quality.mean is the score; per-game scores omitted)."""
        records = client.get(f"/games/{game}/leaderboard", protocol=protocol)
        records.sort(key=lambda r: r.get("metrics", {}).get("quality", {}).get("mean", float("-inf")), reverse=True)
        return [_trim(r) for r in records]

    @tool(read_only)
    def model_catalog(game: str) -> dict[str, Any]:
        """A game's published model packages and their variants."""
        return client.get(f"/games/{game}/models")

    # --- jobs -------------------------------------------------------------------------------------------------------

    @tool(acting)
    def submit_job(spec: dict[str, Any]) -> dict[str, Any]:
        """Submits any job spec -- `kind` train (default), evaluate, publish or pipeline; see `schemas` -- and returns
        the queued job. The scheduler runs it; follow it with get_job / job_logs."""
        return client.post("/jobs", json=spec)

    @tool(acting)
    def train(
        algorithm: str,
        game: str,
        budget: dict[str, int],
        params: dict[str, Any] | None = None,
        interface: str | None = None,
        seed: int = 0,
        held_out_every: int | None = None,
        init_run: str | None = None,
    ) -> dict[str, Any]:
        """Trains one run (a train job): e.g. algorithm td_lambda, game checkers, budget {"iterations": 200},
        params {"agent": {"hidden": 64}}. `init_run` continues that run's final network."""
        spec: dict[str, Any] = {"kind": "train", "algorithm": algorithm, "game": game, "budget": budget, "seed": seed}
        spec["params"] = params or {}
        if interface:
            spec["interface"] = interface
        if held_out_every:
            spec["held_out_every"] = held_out_every
        if init_run:
            spec["init_from"] = {"run": client.resolve_run(init_run)}
        return client.post("/jobs", json=spec)

    @tool(acting)
    def evaluate(protocol: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Runs an evaluation protocol (a job): a leaderboard (snake.score.v2, checkers.versus.v2, bandit.skill.v1) or a
        report (checkers.sprt with params {"a": ..., "b": ...}, experiment.* with {"experiment": NAME})."""
        return client.post("/jobs", json={"kind": "evaluate", "protocol": protocol, "params": params or {}})

    @tool(acting)
    def publish(target: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Runs a publish target (a job): model packages (snake, tinylm, scale-test) or a frontend data file
        (frontend.ballot, frontend.pbt-history, frontend.rl-curves, frontend.rl-recording)."""
        return client.post("/jobs", json={"kind": "publish", "target": target, "params": params or {}})

    @tool(acting)
    def submit_spec_file(path: str, overrides: list[str] | None = None) -> dict[str, Any]:
        """Submits a checked-in spec file (experiments/*.yaml, experiments/pipelines/*.yaml, experiments/schedules/*.yaml,
        or any job spec) to the right endpoint, with `key.path=value` overrides. Returns what the API stored."""
        import yaml

        file = Path(path) if Path(path).is_absolute() else (root or Path.cwd()) / path
        spec = apply_sets(yaml.safe_load(file.read_text(encoding="utf-8")), overrides or [])
        endpoint = {"experiment": "/experiments", "schedule": "/schedules"}.get(spec.get("kind", "train"), "/jobs")
        return client.post(endpoint, json=spec)

    @tool(read_only)
    def list_jobs(
        status: str | None = None, kind: str | None = None, parent: str | None = None, limit: int = 30
    ) -> list[dict[str, Any]]:
        """Jobs, newest first. `parent`: an experiment name, a pipeline's job id, or `schedule:<name>`."""
        jobs = list(reversed(client.get("/jobs", status=status, kind=kind, parent=parent)))
        return jobs[:limit] if limit else jobs

    @tool(read_only)
    def get_job(job: str) -> dict[str, Any]:
        """One job: spec, status (queued, running, succeeded, failed, cancelled), result or error. A pipeline's stage
        jobs are list_jobs(parent=<its id>)."""
        return client.get(f"/jobs/{job_id(job)}")

    @tool(read_only)
    def job_logs(job: str, offset: int = 0) -> dict[str, Any]:
        """A job's worker log from byte `offset`: {text, offset (pass it back next time), finished}."""
        return client.get(f"/jobs/{job_id(job)}/logs", offset=offset)

    @tool(ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False))
    def cancel_job(job: str) -> dict[str, Any]:
        """Cancels a job: a queued one at once, a running one when the scheduler stops its worker (its run is marked
        failed). A pipeline's running stage is cancelled with it."""
        return client.post(f"/jobs/{job_id(job)}/cancel")

    # --- experiments and schedules ----------------------------------------------------------------------------------

    @tool(acting)
    def submit_experiment(spec: dict[str, Any]) -> dict[str, Any]:
        """Stores (or resubmits, resuming and retrying failed cells) an experiment spec; returns its arm x seed matrix.
        The scheduler trains every cell, then runs its report."""
        return client.post("/experiments", json=spec)

    @tool(read_only)
    def experiment_status(name: str) -> dict[str, Any]:
        """An experiment's arm x seed matrix: each cell completed (its run), a job's status, or pending."""
        return client.get(f"/experiments/{name}")

    @tool(read_only)
    def experiment_report(name: str, variant: Literal["", "h2h", "h2h-full"] = "") -> dict[str, Any]:
        """An experiment's report, once its report job has run (`h2h`/`h2h-full`: the head-to-head ones)."""
        return client.get(f"/experiments/{name}{'-' + variant if variant else ''}/report")

    @tool(read_only)
    def list_schedules() -> list[dict[str, Any]]:
        """Stored schedules: their triggers, the job each submits, and when each last fired."""
        return client.get("/schedules")

    @tool(acting)
    def submit_schedule(spec: dict[str, Any]) -> dict[str, Any]:
        """Stores (or replaces) a schedule: `every` (30m, 6h, 1d) and/or `on_run_completed` {game}, and `submit` (a job
        spec)."""
        return client.post("/schedules", json=spec)

    @tool(ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False))
    def delete_schedule(name: str) -> dict[str, Any]:
        """Stops a schedule firing (the jobs it submitted are kept)."""
        client.delete(f"/schedules/{name}")
        return {"deleted": name}

    return server
