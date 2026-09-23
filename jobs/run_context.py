"""One training run's lifecycle, shared by every jobs/*_run.py script.

Every training job does the same things around its algorithm: open the telemetry stores, create a run
with its config, wire the pause/resume control callback (excluding paused time from the cost meter),
record a summary, and mark the run `completed` -- or `failed`, if it dies. The last part is the one
that matters: a run left `running` after a crash shows as live on the runs page forever. Each script
used to repeat all of it, and only some remembered `failed`; `recorded_run()` does it once.

A hard kill can't be caught -- a run killed that way still needs marking by hand.

Tests point jobs at a temporary directory with `monkeypatch.setattr(run_context, "RUN_DATA_DIR", tmp_path)`;
`TelemetryStores.open()` reads it at call time for exactly that reason.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from control import make_control_callback
from costs import TrainingCostMeter
from evolve import GenerationCallback
from telemetry import (
    FileArtifactStore,
    FileMetricsStore,
    GenerationStats,
    SqliteRunRegistry,
)

# The same override apis/backend honours (settings.py), so a job and the API can agree on a scratch directory.
RUN_DATA_DIR = Path(os.environ.get("REDQUEEN_RUN_DATA_DIR") or Path(__file__).parent / "run-data")


class TelemetryStores(NamedTuple):
    registry: SqliteRunRegistry
    metrics: FileMetricsStore
    artifacts: FileArtifactStore

    @classmethod
    def open(cls, data_dir: Path | None = None) -> TelemetryStores:
        data_dir = RUN_DATA_DIR if data_dir is None else data_dir
        return cls(
            registry=SqliteRunRegistry(data_dir / "runs.db"),
            metrics=FileMetricsStore(data_dir / "metrics"),
            artifacts=FileArtifactStore(data_dir / "artifacts"),
        )


@dataclass(frozen=True)
class RecordedRun:
    """A run in progress: its id, its stores, and the pieces every job wires the same way."""

    run_id: str
    stores: TelemetryStores

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
def recorded_run(config: dict[str, Any], stores: TelemetryStores | None = None) -> Iterator[RecordedRun]:
    """Creates a run with `config`, yields it, and marks it `completed` when the block finishes or
    `failed` if it raises (including Ctrl-C), re-raising."""
    stores = stores or TelemetryStores.open()
    run = RecordedRun(run_id=stores.registry.create_run(config=config), stores=stores)
    print(f"run_id={run.run_id}", flush=True)
    try:
        yield run
    except BaseException:
        stores.registry.update_status(run.run_id, "failed")
        raise
    stores.registry.update_status(run.run_id, "completed")
