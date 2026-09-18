"""Pause/resume control for a running evolve() call, via RunRegistry's existing RunStatus.

Lives in jobs/, not libs/telemetry or libs/evolve, for the same reason telemetry-wiring callbacks
already do (see baseline_gp_run.py's make_telemetry_callback): it's integration glue between an
algorithm (zero telemetry dependency) and telemetry (zero algorithm dependency), not a concern of
either package on its own. See docs/design/0005 "Control API".
"""

from __future__ import annotations

import time

from evolve import GenerationCallback, GenerationSummary
from telemetry import RunRegistry


def make_control_callback(
    registry: RunRegistry, run_id: str, poll_interval: float = 0.2
) -> GenerationCallback:
    """Returns an on_generation callback that blocks while the run's status is "paused".

    apis/backend/routers/runs.py's control endpoint flips status between "running" and "paused" via
    this same RunRegistry -- that's the only coordination needed between the (separate) API process
    and this (separate) training-job process; no new shared state, no new IPC primitive. "step" is
    implemented entirely on the API side (resume, wait for exactly one new generation to be
    recorded, re-pause) -- this callback only ever needs to understand two states.
    """

    def on_generation(_summary: GenerationSummary) -> None:
        while registry.get_run(run_id).status == "paused":
            time.sleep(poll_interval)

    return on_generation
