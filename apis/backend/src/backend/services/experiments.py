"""Experiments as the API offers them: stored once their spec validates, and read back as an arm x seed matrix.

Where each cell stands is derived, not stored: a *completed* run tagged with the experiment, arm and seed is done
(however it was trained -- the old scripts' runs count); otherwise the newest job of the experiment for that arm and
seed says how far it got; otherwise it is pending. So the matrix is right for experiments run before the job API
existed, and stays right if the scheduler is restarted.
"""

from __future__ import annotations

import json
from typing import Any

from jobcore.specs import ExperimentSpec
from telemetry import Experiment, JobStore, RunRegistry, data_dir

from backend.services.jobs import InvalidSpec


def put(store: JobStore, spec: dict[str, Any]) -> Experiment:
    try:
        parsed = ExperimentSpec.model_validate({**spec, "kind": "experiment"})
        parsed.resolve()
    except ValueError as e:
        raise InvalidSpec(str(e)) from None
    return store.put_experiment(parsed.name, parsed.model_dump(mode="json"))


def status(experiment: Experiment, store: JobStore, registry: RunRegistry) -> dict[str, Any]:
    spec = ExperimentSpec.model_validate(experiment.spec)
    runs = {
        (r.config.get("arm"), r.config.get("rng_seed")): r
        for r in registry.list_runs()  # newest first: the dict keeps the oldest completed run, as the scripts did
        if r.config.get("experiment") == spec.name and r.status == "completed"
    }
    jobs = {}
    for job in store.list(parent=spec.name):  # oldest first: the newest job for a cell wins
        tags = job.spec.get("tags", {})
        jobs[(tags.get("arm"), job.spec.get("seed"))] = job
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    done = 0
    expanded = spec.expand()
    for arm, seed, _train in expanded:
        run, job = runs.get((arm, seed)), jobs.get((arm, seed))
        if run is not None:
            cell = {"status": "completed", "run_id": run.run_id}
            done += 1
        elif job is not None:
            cell = {"status": job.status, "job_id": job.job_id}
        else:
            cell = {"status": "pending"}
        cells.setdefault(arm, {})[str(seed)] = cell
    return {
        "name": spec.name,
        "created_at": experiment.created_at,
        "seeds": spec.seeds,
        "arms": cells,
        "baselines": {name: arm.baseline for name, arm in spec.arms.items() if arm.baseline},
        "done": done,
        "total": len(expanded),
        "report": (data_dir() / "experiments" / f"{spec.name}.json").exists(),
    }


def report(name: str) -> dict[str, Any]:
    """The experiment's report (`<data dir>/experiments/<name>.json`), as its report job wrote it."""
    path = data_dir() / "experiments" / f"{name}.json"
    if not path.exists():
        raise KeyError(f"no report for {name} yet")
    return json.loads(path.read_text(encoding="utf-8"))
