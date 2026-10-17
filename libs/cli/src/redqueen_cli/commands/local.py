"""`redqueen export` and `redqueen dev`: the commands that produce files in the repo.

`export` writes the frontend data files the Learn chapters draw. Each one is a publisher job (`frontend.*` targets),
so it goes through the API like everything else, and the file lands in the checkout the *scheduler* runs from. `dev`
is the only local-only group: it runs the repo's own scripts (builds, code generation, benchmarks, the dev servers)
with this interpreter, from the repo root, and imports none of them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import typer

from redqueen_cli.state import state, submit_job

export = typer.Typer(help="Frontend data files from real results (publisher jobs; commit what they write).")
dev = typer.Typer(help="Local development: builds, code generation, benchmarks, the dev servers.")

WaitOpt = Annotated[bool, typer.Option("--wait/--no-wait", help="Follow the job until it ends (the default).")]


def _publish(ctx: typer.Context, target: str, params: dict, wait: bool) -> None:
    params = {k: v for k, v in params.items() if v is not None}
    submit_job(ctx, {"kind": "publish", "target": target, "params": params}, wait)


@export.command()
def ballot(ctx: typer.Context, wait: WaitOpt = True) -> None:
    """apps/frontend/app/data/checkersBallot.ts: every 3-ply Checkers opening and which are the ballot."""
    _publish(ctx, "frontend.ballot", {}, wait)


@export.command("pbt-history")
def pbt_history(
    ctx: typer.Context,
    experiment: str = "selfplay-v4",
    arm: str = "g-pbt",
    seed: int = 0,
    wait: WaitOpt = True,
) -> None:
    """apps/frontend/app/data/pbtHistory.ts: one PBT run's population history."""
    _publish(ctx, "frontend.pbt-history", {"experiment": experiment, "arm": arm, "seed": seed}, wait)


@export.command("rl-curves")
def rl_curves(
    ctx: typer.Context,
    names: Annotated[list[str] | None, typer.Argument(help="dqn-snake, pg-snake (default: both).")] = None,
    wait: WaitOpt = True,
) -> None:
    """apps/frontend/app/data/recordings/<name>.json: chosen RL experiment runs' curves."""
    _publish(ctx, "frontend.rl-curves", {"names": names or None}, wait)


@export.command("rl-recording")
def rl_recording(
    ctx: typer.Context,
    run: Annotated[str, typer.Argument(help="A tabular RL run (an id or a unique prefix).")],
    label: Annotated[str | None, typer.Option(help="How the chapter describes it.")] = None,
    wait: WaitOpt = True,
) -> None:
    """apps/frontend/app/data/recordings/q-learning-snake.json: a Q-table run, the chapter's no-WASM fallback."""
    _publish(ctx, "frontend.rl-recording", {"run": state(ctx).client.resolve_run(run), "label": label}, wait)


# --- dev ------------------------------------------------------------------------------------------------------------


def repo_root() -> Path:
    """The workspace root: the nearest directory up from here with the uv workspace's pyproject."""
    for directory in [Path.cwd(), *Path.cwd().parents]:
        pyproject = directory / "pyproject.toml"
        if pyproject.exists() and "[tool.uv.workspace]" in pyproject.read_text(encoding="utf-8"):
            return directory
    raise typer.BadParameter("not inside the red-queen repo (no workspace pyproject.toml above here)")


def _run(*command: str | Path, cwd: Path | None = None) -> None:
    root = repo_root()
    typer.echo(f"$ {' '.join(str(c) for c in command)}", err=True)
    code = subprocess.call([str(c) for c in command], cwd=cwd or root)
    if code:
        raise typer.Exit(code)


def _python(script: str, *args: str) -> None:
    _run(sys.executable, repo_root() / script, *args)


@dev.command("build-wasm")
def build_wasm(
    which: Annotated[list[str] | None, typer.Argument(help="games, rl, or both (the default).")] = None,
) -> None:
    """Rebuilds the browser's WASM copies of the Rust cores (after a change under rust/; their tests say when)."""
    for name in which or ["games", "rl"]:
        if name not in ("games", "rl"):
            raise typer.BadParameter(f"games or rl, not {name!r}")
        _python(f"libs/{name}/scripts/build_wasm.py")


@dev.command()
def openapi() -> None:
    """Regenerates the API's OpenAPI snapshot and the frontend's types from it (after any API change)."""
    _python("apis/backend/scripts/export_openapi.py")
    _run(_pnpm(), "gen:api-types", cwd=repo_root() / "apps" / "frontend")


@dev.command("bench-rl")
def bench_rl(args: Annotated[list[str] | None, typer.Argument(help="Passed to the benchmark.")] = None) -> None:
    """The native half of /dev/rl's speed numbers."""
    _python("libs/rl/scripts/benchmark.py", *(args or []))


@dev.command()
def screenshots(run: Annotated[str, typer.Argument(help="A trained Snake run id.")]) -> None:
    """Recaptures the app's real screenshots (needs `redqueen dev up` and Playwright's browser)."""
    _run(shutil.which("uv") or "uv", "run", "--with", "playwright", "python",
         "apps/frontend/scripts/capture_screenshots.py", run)  # fmt: skip


WRANGLER = "wrangler@4"


@dev.command()
def deploy(
    data: Annotated[bool, typer.Option("--data/--no-data", help="Upload the snapshot and ONNX Runtime to R2.")] = True,
    site: Annotated[bool, typer.Option("--site/--no-site", help="Build the site and deploy it to Pages.")] = True,
    snapshot: Annotated[
        Path | None, typer.Option(help="`redqueen publish site`'s output (default: <data dir>/site).")
    ] = None,
    project: Annotated[str, typer.Option(help="The Cloudflare Pages project.")] = "redqueen",
    prune: Annotated[bool, typer.Option(help="Delete bucket objects the new snapshot no longer lists.")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Build, and say what would upload; deploy nothing.")
    ] = False,
) -> None:
    """Publishes redqueen.mackinations.ca (docs/design/0019; one-time setup: docs/deploying.md).

    In order: build the static site (`pnpm generate:static`), upload the data (the snapshot from `redqueen publish
    site`, plus ONNX Runtime) to R2, then deploy the site to Cloudflare Pages. Data goes first, so a new site never asks
    for files that aren't there yet. Credentials come from the environment: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID and
    R2_SECRET_ACCESS_KEY for R2; CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID (or a one-time `wrangler login`) for
    Pages."""
    root = repo_root()
    frontend = root / "apps" / "frontend"
    if site or not (frontend / ".output" / "ort").exists():
        _run(_pnpm(), "generate:static", cwd=frontend)  # the site, and the ONNX Runtime files the data host serves
    if data:
        data_dir = Path(os.environ.get("REDQUEEN_DATA_DIR") or os.environ.get("REDQUEEN_RUN_DATA_DIR") or root / "data")
        command: list[str | Path] = [shutil.which("uv") or "uv", "run", "--script"]
        command += [
            root / "jobs" / "publisher" / "scripts" / "deploy_r2.py",
            "--snapshot",
            snapshot or data_dir / "site",
        ]
        command += ["--ort", frontend / ".output" / "ort"]
        _run(*command, *(["--prune"] if prune else []), *(["--dry-run"] if dry_run else []))
    if site:
        if dry_run:
            typer.echo(f"dry run: would deploy {frontend / '.output' / 'public'} to Pages project {project}", err=True)
            return
        _run(_pnpm(), "dlx", WRANGLER, "pages", "deploy", ".output/public", "--project-name", project,
             "--branch", "main", cwd=frontend)  # fmt: skip


@dev.command()
def up(
    frontend: Annotated[bool, typer.Option(help="Also the Nuxt dev server (port 3000).")] = True,
    workers: Annotated[int, typer.Option(help="The scheduler's worker limit.")] = 1,
) -> None:
    """The backend (port 8000), the scheduler and the frontend, together; Ctrl-C stops all three."""
    import signal

    root = repo_root()
    # stopped by CTRL_BREAK/SIGTERM as by Ctrl-C: either way, it stops what it started
    signal.signal(signal.SIGBREAK if sys.platform == "win32" else signal.SIGTERM, _interrupt)
    commands = {
        "backend": [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--app-dir",
            "apis/backend/src",
            "--port",
            "8000",
        ],
        "scheduler": [sys.executable, "-m", "scheduler", "--workers", str(workers)],
    }
    if frontend:
        commands["frontend"] = [_pnpm(), "--dir", "apps/frontend", "dev"]
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    processes = {name: subprocess.Popen(cmd, cwd=root, creationflags=flags) for name, cmd in commands.items()}
    typer.echo(f"up: {', '.join(processes)} -- Ctrl-C stops them", err=True)
    try:
        while all(p.poll() is None for p in processes.values()):
            time.sleep(0.5)
        gone = [n for n, p in processes.items() if p.poll() is not None]
        typer.echo(f"{', '.join(gone)} exited; stopping the rest", err=True)
    except KeyboardInterrupt:
        pass
    finally:
        _stop(processes)


def _interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def _stop(processes: dict[str, subprocess.Popen]) -> None:
    import signal

    for process in processes.values():
        if process.poll() is None:
            # never Ctrl-C a child (docs: a console-less child often never gets CTRL_C_EVENT); the scheduler turns
            # CTRL_BREAK/SIGTERM into stopping its workers cleanly
            process.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM)
    deadline = time.monotonic() + 15
    for process in processes.values():
        try:
            process.wait(max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()


def _pnpm() -> str:
    found = shutil.which("pnpm")
    if found is None:
        raise typer.BadParameter("pnpm isn't on PATH (the frontend needs it)")
    return found


__all__ = ["dev", "export"]
