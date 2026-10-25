"""The API as static files, for hosting the site with no server (docs/design/0019).

Every GET route the frontend reads is fetched *through this app itself* (an in-process TestClient over the local stores)
and written at its own path plus `.json` -- `/runs/{id}/metrics/history` -> `api/runs/{id}/metrics/history.json` -- so
a file is, byte for byte, what the API would have answered. The frontend's static mode (`NUXT_PUBLIC_API_MODE=static`)
asks for exactly those paths. What a static host can't do is done ahead of time or left out:

- the on-demand package export (`POST .../package`) runs here for each Snake run's latest champion, and its response is
  written to `.../package.json`;
- of the artifact store (every generation's champion: gigabytes), only the champions a page asks for are copied: each
  run's latest, and each one a leaderboard ranks;
- runs still `running`/`paused` are left out (a static copy would look frozen), and nothing live is exported (the
  metrics stream, run control, jobs, experiments, schedules, the write endpoints);
- of the model store, only what a catalog or an exported package refers to is copied, and catalogs' `base_url` is the
  public models URL. The `scale-test` catalog (~340 MB, for the unlinked /dev/inference page) is left out unless asked.

Nothing about the machine leaves it. Every `hardware` block (the training-cost meter's CPU model, core count, OS and
Python version: a fingerprint of this computer) is reduced to its `engine` and a generic `hardware_class` -- the only
transformation of an API answer -- and before anything is kept, every file is checked for this machine's identifiers
(home directory, username, hostname, the repo's path, the CPU's name): one found fails the export (`PrivacyError`),
naming the file. Local records keep the full detail.

`index.json` lists every file with its sha256, so a deploy uploads only what changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from modelpack import LocalModelStore
from telemetry import models_dir, workspace_root

LIVE = frozenset({"running", "paused"})
GAMES = ("snake", "checkers", "bandit", "copycat")  # the game pages: a leaderboard (and interfaces) each
SKIPPED_CATALOGS = frozenset({"scale-test"})
# What a published `hardware` block says instead of the machine's own description (the game page's cost caption reads
# "Timings are from <hardware_class> -- only compare them within the same hardware class").
PUBLIC_HARDWARE_CLASS = "one desktop computer"


class PrivacyError(ValueError):
    """A file about to be published contains one of this machine's identifiers."""


def redact(value: Any) -> Any:
    """`value` with every `hardware` block reduced to its `engine` (rust / pure-python: what the code ran on, which a
    reader needs to compare timings) and the generic `hardware_class`."""
    if isinstance(value, dict):
        return {
            k: (
                {**({"engine": v["engine"]} if "engine" in v else {}), "hardware_class": PUBLIC_HARDWARE_CLASS}
                if k == "hardware" and isinstance(v, dict)
                else redact(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def redact_json(data: bytes) -> bytes:
    """An API answer as published: unchanged bytes if there's nothing to redact, else re-serialized as the API
    serializes (FastAPI: compact, UTF-8, not ASCII-escaped). Not JSON (a linear-GP champion is a plain-text program):
    unchanged -- the privacy check still reads it."""
    try:
        value = json.loads(data)
    except ValueError:
        return data
    redacted = redact(value)
    if redacted == value:
        return data
    return json.dumps(redacted, separators=(",", ":"), ensure_ascii=False).encode()


def local_identifiers() -> list[str]:
    """What would identify this machine or its owner, as it would appear in a file: the home directory and the repo's
    path (both slash styles, and JSON-escaped), the username, the hostname, the CPU's name. Read from the machine at
    export time, so nothing personal is written into the code."""
    from arena.costs import hardware_fingerprint

    found: set[str] = set()
    for path in (Path.home(), workspace_root()):
        text = str(path)
        found |= {text, text.replace("\\", "/"), text.replace("\\", "\\\\")}
    for word in (Path.home().name, socket.gethostname(), hardware_fingerprint().get("cpu", "")):
        if len(word) >= 4 and word.lower() not in {"user", "root", "unknown", "localhost"}:
            found.add(word)
    return sorted(found)


def check_private(root: Path, files: list[str], identifiers: list[str] | None = None) -> None:
    """Raises PrivacyError if any of `files` (under `root`) contains one of `identifiers` (case-insensitive)."""
    needles = [n.lower().encode() for n in (identifiers if identifiers is not None else local_identifiers())]
    leaks = []
    for path in files:
        data = (root / path).read_bytes().lower()
        leaks += [f"{path}: {needle.decode()!r}" for needle in needles if needle in data]
    if leaks:
        listing = "\n  ".join(leaks[:20])
        raise PrivacyError(f"{len(leaks)} local identifier(s) in the site's files:\n  {listing}")


@contextmanager
def _env(name: str, value: str) -> Iterator[None]:
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


class _Writer:
    def __init__(self, root: Path):
        self.root = root
        self.files: dict[str, str] = {}

    def write(self, path: str, data: bytes) -> None:
        if path.startswith("api/"):
            data = redact_json(data)
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.files[path] = hashlib.sha256(data).hexdigest()

    def json(self, path: str, value: Any) -> None:
        """A filtered response, re-serialized as the API serializes (FastAPI: compact, UTF-8, not ASCII-escaped)."""
        self.write(path, json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())


def export(
    out: Path, models_base_url: str, *, include_scale_test: bool = False, package: bool = True
) -> dict[str, Any]:
    """Writes the static site's data under `out` (`api/...`, `models/...`, `index.json`); returns a summary."""
    from fastapi.testclient import TestClient

    from backend.dependencies import _job_store, _model_store, _stores
    from backend.main import app

    if out.exists():
        shutil.rmtree(out)  # a snapshot, not an accumulation: a run left out now is gone from the copy too
    writer = _Writer(out)
    for cached in (_stores, _job_store, _model_store):
        cached.cache_clear()  # the stores of the data directory as it is now
    with _env("REDQUEEN_MODELS_BASE_URL", models_base_url.rstrip("/")), TestClient(app) as http:

        def get(path: str, **params: Any) -> Any:
            response = http.get(path, params=params or None)
            response.raise_for_status()
            return response

        runs = [r for r in get("/runs").json() if r["status"] not in LIVE]
        published = {r["run_id"] for r in runs}
        writer.json("api/runs.json", runs)
        writer.json("api/runs/summaries.json", [s for s in get("/runs/summaries").json() if s["run_id"] in published])

        champions: set[tuple[str, str]] = set()
        for run in runs:
            run_id = run["run_id"]
            writer.write(f"api/runs/{run_id}.json", get(f"/runs/{run_id}").content)
            response = get(f"/runs/{run_id}/metrics/history")
            writer.write(f"api/runs/{run_id}/metrics/history.json", response.content)
            history = response.json()
            latest = next((h["champion_ref"] for h in reversed(history) if h.get("champion_ref")), None)
            if latest:
                champions.add((run_id, latest))

        for game in GAMES:
            response = get(f"/games/{game}/leaderboard")
            writer.write(f"api/games/{game}/leaderboard.json", response.content)
            board = response.json()
            interfaces = http.get(f"/games/{game}/interfaces")
            if interfaces.status_code == 200:  # 404: a game with no representations to choose (Copy Cat)
                writer.write(f"api/games/{game}/interfaces.json", interfaces.content)
            champions |= {
                (r["run_id"], r["champion_ref"])
                for r in board
                if r.get("run_id") in published and r.get("champion_ref")
            }

        response = get("/games/copycat/targets")
        writer.write("api/games/copycat/targets.json", response.content)
        for target in response.json():
            seed = target["seed"]
            writer.write(f"api/games/copycat/targets/{seed}.json", get(f"/games/copycat/targets/{seed}").content)

        games = {r["run_id"]: r["config"].get("game") for r in runs}
        packages: set[str] = set()
        for run_id, ref in sorted(champions):
            writer.write(f"api/runs/{run_id}/artifacts/{ref}.json", get(f"/runs/{run_id}/artifacts/{ref}").content)
            if games.get(run_id) == "checkers":
                brain = http.get(f"/runs/{run_id}/artifacts/{ref}/brain")
                if brain.status_code == 200:  # 422: not a network (nothing a page would build a player from)
                    writer.write(f"api/runs/{run_id}/artifacts/{ref}/brain.json", brain.content)
            if package and games.get(run_id) == "snake":
                exported = http.post(f"/runs/{run_id}/artifacts/{ref}/package")
                if exported.status_code == 200:  # 422: a champion kind modelpack can't export
                    writer.write(f"api/runs/{run_id}/artifacts/{ref}/package.json", exported.content)
                    packages.add(exported.json()["package_id"])

        store = LocalModelStore(models_dir())
        catalogs = sorted(p.stem for p in (models_dir() / "catalog").glob("*.json"))
        for game in catalogs:
            if game in SKIPPED_CATALOGS and not include_scale_test:
                continue
            response = get(f"/games/{game}/models")
            writer.write(f"api/games/{game}/models.json", response.content)
            packages |= {e["package_id"] for e in response.json()["entries"]}

    blobs = 0
    for package_id in sorted(packages):
        writer.write(f"models/manifests/{package_id}.json", store.manifest_path(package_id).read_bytes())
        for blob in store.manifest(package_id).blobs():
            if f"models/blobs/{blob.sha256}" not in writer.files:
                writer.write(f"models/blobs/{blob.sha256}", store.read_blob(blob.sha256))
                blobs += 1

    index = dict(sorted(writer.files.items()))
    try:
        check_private(out, list(index))
    except PrivacyError:
        shutil.rmtree(out)  # never leave a leaking copy where a deploy could pick it up
        raise
    (out / "index.json").write_bytes(json.dumps(index, indent=0).encode())
    summary = {
        "out": str(out),
        "runs": len(runs),
        "champions": len(champions),
        "packages": len(packages),
        "blobs": blobs,
        "files": len(index),
        "bytes": sum((out / p).stat().st_size for p in index),
    }
    print(
        f"static site: {summary['runs']} runs, {summary['champions']} champions, {summary['packages']} packages, "
        f"{summary['files']} files ({summary['bytes'] / 1e6:.1f} MB) in {out}",
        flush=True,
    )
    return summary
