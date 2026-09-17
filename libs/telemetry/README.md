# telemetry

Interfaces (and local-dev implementations) for observing evolving/training populations: run
metadata, a cheap per-generation metrics stream, and an on-demand store for heavier artifacts
(programs, execution traces). See
[docs/design/0002-realtime-visualization-architecture.md](../../docs/design/0002-realtime-visualization-architecture.md)
for the reasoning behind the split.

## Contents

- `RunRegistry` (`registry.py`) — run metadata (config, status, timestamps), as the pydantic model
  `RunInfo`. SQLite-backed.
- `MetricsSink` / `MetricsSource` (`metrics.py`) — append-only per-generation stats, as the
  pydantic model `GenerationStats`, with backfill-then-live `subscribe()`. File-backed (JSON Lines).
- `ArtifactStore` (`artifacts.py`) — on-demand key/value store for programs and traces (plain
  bytes — no schema at this layer). File-backed.

All three are defined as `typing.Protocol`s so a hosted/scaled backend (e.g. Redis-backed) can be
swapped in later without changing any code that depends on them. The local implementations here
are the default for single-user, single-machine runs. `RunInfo` and `GenerationStats` are pydantic
models with `Field(..., description=...)` and validation on every field (see
[../../docs/CODING_GUIDELINES.md](../../docs/CODING_GUIDELINES.md)) — constructing one with bad
data (e.g. a negative `generation`) raises immediately instead of failing silently or downstream.

Part of the repo's `uv` workspace (see [AGENTS.md](../../AGENTS.md)) — `uv sync --all-packages`
from the repo root installs this alongside every other `libs/*` package into one shared `.venv`.

## Usage

```python
from telemetry import FileMetricsStore, GenerationStats, SqliteRunRegistry

registry = SqliteRunRegistry("runs.db")
run_id = registry.create_run(config={"population_size": 32})

metrics = FileMetricsStore("./run-data")
metrics.record_generation(GenerationStats(
    run_id=run_id, island_id=None, generation=0, timestamp=time.time(),
    best_fitness=0.9, mean_fitness=0.5, worst_fitness=0.1, diversity=0.3,
    champion_ref="gen0-champion",
))

for stats in metrics.subscribe(run_id):
    print(stats.generation, stats.best_fitness)
```
