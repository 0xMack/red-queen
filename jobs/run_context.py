"""One training run's lifecycle, shared by every jobs/*_run.py script.

Every training job does the same things around its algorithm: open the telemetry stores, create a run
with its config, wire the pause/resume control callback (excluding paused time from the cost meter),
record a summary, and mark the run `completed` -- or `failed`, if it dies. The last part is the one
that matters: a run left `running` after a crash shows as live on the runs page forever. Each script
used to repeat all of it, and only some remembered `failed`; `recorded_run()` does it once.

A hard kill can't be caught -- a run killed that way still needs marking by hand.

Where the data lives is `telemetry.data_dir()` (`REDQUEEN_DATA_DIR`), read at call time: jobs/tests/conftest.py points
every test at a temporary directory that way.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arena.costs import TrainingCostMeter
from control import make_control_callback
from evolve import GenerationCallback
from telemetry import (
    FileArtifactStore,
    FileMetricsStore,
    GenerationStats,
    SqliteRunRegistry,
    Stores,
    data_dir,
    open_stores,
)


def experiments_dir() -> Path:
    """Where experiment reports are written (gitignored data, not results: those are cited in docs/design)."""
    return data_dir() / "experiments"


@dataclass(frozen=True)
class RecordedRun:
    """A run in progress: its id, its stores, and the pieces every job wires the same way."""

    run_id: str
    stores: Stores

    @property
    def registry(self) -> SqliteRunRegistry:
        return self.stores.registry

    @property
    def metrics(self) -> FileMetricsStore:
        return self.stores.metrics

    @property
    def artifacts(self) -> FileArtifactStore:
        return self.stores.artifacts

    def control_callback(self, cost: TrainingCostMeter | None = None) -> GenerationCallback:
        """The pause/resume callback (jobs/control.py); with `cost`, time spent paused isn't counted."""
        callback = make_control_callback(self.registry, self.run_id)
        return cost.excluding_pauses(callback) if cost is not None else callback

    def history(self) -> list[GenerationStats]:
        return self.metrics.history(self.run_id)

    def set_summary(self, summary: dict[str, Any]) -> None:
        self.registry.set_summary(self.run_id, summary)

    def set_training_summary(self, cost: TrainingCostMeter) -> list[GenerationStats]:
        """The summary every evolved run records (final best fitness, held-out score, measured cost);
        returns the run's history, which the caller usually wants next."""
        history = self.history()
        self.set_summary(
            {
                "best_fitness": history[-1].best_fitness,
                "held_out_score": history[-1].held_out_score,
                "cost": cost.summary(),
            }
        )
        return history


@contextmanager
def recorded_run(config: dict[str, Any], stores: Stores | None = None) -> Iterator[RecordedRun]:
    """Creates a run with `config`, yields it, and marks it `completed` when the block finishes or
    `failed` if it raises (including Ctrl-C), re-raising."""
    stores = stores or open_stores()
    run = RecordedRun(run_id=stores.registry.create_run(config=config), stores=stores)
    print(f"run_id={run.run_id}", flush=True)
    try:
        yield run
    except BaseException:
        stores.registry.update_status(run.run_id, "failed")
        raise
    stores.registry.update_status(run.run_id, "completed")
