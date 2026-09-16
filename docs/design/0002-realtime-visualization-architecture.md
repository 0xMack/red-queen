# 0002 — Connecting evolving populations to real-time visualization

Status: **Draft** — architecture direction agreed in discussion, implementation starting with
local-only backends.
Relates to: [0001](0001-fast-cpp-gp-pybind11.md)

## Context

Doc 0001 established that the GP engine's hot loop stays lean (no tracing/logging in the
per-instruction path), with cheap per-generation stats always collected and heavier
introspection (`trace()`) available on demand for a specific individual. This doc is about what
happens to that data once it leaves the engine: how it's stored, how a frontend gets it in near
real time, and how this scales from "one run on my laptop" to "a hosted demo" to "evolution
distributed across my workstation and homelab cluster."

## Guiding analog: TensorBoard, not a push-based dashboard

TensorBoard's trainer writes event files to a logdir and knows nothing about any viewer; a
separate process reads that logdir and serves the UI. MLflow's tracking server and W&B follow the
same run-registry shape. We're copying the **decoupled, log-then-serve** version of this, not a
direct push model (trainer → API → browser with no store in between) — an evolution run stalling
because a websocket backed up, or losing all history because nobody was watching, is not
acceptable, and decoupling is what lets a run be inspected after the fact or from multiple
viewers.

## Components

Three separate interfaces, not one — they have different reliability requirements and access
patterns, and conflating them is the main way this design would go wrong.

### 1. `RunRegistry` — metadata about runs

Low-frequency, relational: run id, config (representation, fitness evaluator, population size,
island topology if any), status (running/paused/completed/failed), timestamps, final summary.
SQLite-appropriate — this is the "list runs" / "get run config" surface.

### 2. `MetricsSink` / `MetricsSource` — cheap, frequent, observational stream

Written once per generation by the Python orchestration layer, *after* it has fitness results
back from the C++ hot loop — never from inside it. Small enough that JSON and a once-per-
generation network call are both fine regardless of backend.

```python
class GenerationStats(TypedDict):
    run_id: str
    island_id: str | None      # None if single-population
    generation: int
    timestamp: float
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity: float
    champion_ref: str          # pointer into ArtifactStore, not the program itself

class MetricsSink(Protocol):
    def record_generation(self, stats: GenerationStats) -> None: ...

class MetricsSource(Protocol):
    def history(self, run_id: str, since_generation: int = 0) -> list[GenerationStats]: ...
    def subscribe(self, run_id: str, since_generation: int = 0) -> Iterator[GenerationStats]: ...
```

`subscribe(since_generation=...)` exists so a viewer opening the dashboard mid-run gets backfill
then live continuation — same pattern as Kafka consumer offsets or SSE's `Last-Event-ID`. Handling
this from the start means reconnects (laptop sleep, flaky wifi mid-demo) show no gaps instead of
needing a special case later.

**Reliability rule:** `record_generation` is fire-and-forget. If the sink is unavailable, log a
warning and keep evolving — observability must never be able to kill a multi-hour run.

### 3. `ArtifactStore` — heavy, on-demand data

Full programs and `trace()` output: larger, written rarely (only for individuals someone actually
cares about — a generation's champion, or one selected in the UI), read on demand rather than
streamed.

```python
class ArtifactStore(Protocol):
    def put_program(self, ref: str, program: bytes) -> None: ...
    def get_program(self, ref: str) -> bytes: ...
    def put_trace(self, ref: str, trace: bytes) -> None: ...
    def get_trace(self, ref: str) -> bytes: ...
```

Plain key-value, no pub/sub needed. Doesn't need to evolve in lockstep with the metrics stream's
backend.

### Explicitly excluded: island migration

Migration of individuals between islands (see "Distributed compute" below) is **not** part of
`MetricsSink`. It's part of the algorithm's actual state — losing a migrant is a correctness issue,
losing a metrics event isn't — so it needs different delivery guarantees and will get its own
interface when island model is built, even if it ends up sharing the same broker at runtime.

## Data store tradeoffs

| Backend | Pros | Cons |
|---|---|---|
| Flat files (jsonlines) | Zero infra, trivial to write, human-inspectable, matches TensorBoard directly | No native pub/sub (API server must poll/tail); multiple API replicas need a shared volume |
| SQLite | Zero infra, real queries — good fit for `RunRegistry`; WAL mode suits our one-writer/many-readers pattern | Same replica/shared-volume limitation as files; no pub/sub |
| Redis (streams) | Native pub/sub — any API replica can subscribe with no polling; the only option built for a horizontally-scaled hosted demo | Real external dependency; durability needs deliberate config; overkill for single-user local runs |

Decision: don't hard-commit to one. `RunRegistry`, `MetricsSink`/`Source`, and `ArtifactStore` are
defined as `Protocol`s precisely so the local-dev default (files + SQLite) and a future hosted
default (Redis-backed, or Postgres + pub/sub) are swappable without touching the trainer or API
code that depends on them — same pattern as `FitnessEvaluator` in doc 0001.

## Streaming layer: SSE, with a control API alongside it

SSE (not WebSocket) for the metrics stream itself — it's one-directional (server → client) and
that's all the live dashboard needs. Run *control* (pause, step a generation, request a trace on a
running individual) is a separate, ordinary REST/RPC surface against the run — it doesn't need to
share a transport with the observational stream. This keeps "watching" and "controlling" as
separate concerns with separate failure modes (a dropped SSE connection shouldn't affect your
ability to pause a run, and vice versa).

## Frontend

Vue 3 SPA, reactive to the SSE stream. Two views fall out of the two-tier data split:

- **Live/historical dashboard** — fitness-over-time, diversity — fed by the metrics
  stream + `history()` backfill.
- **Detail view** — a specific individual's program structure / trace step-through — fed by
  on-demand REST calls to `ArtifactStore` when something is selected, not by the stream.

## Distributed compute

Island-model GA is the right frame for parallelizing this, and pays off before any networking is
involved:

- **Single machine, multi-threaded**: islands = threads/processes, migration = in-memory queue.
  Do this first — validates migration mechanics (topology, rate) without network variables.
- **Workstation + homelab MicroK8s**: once migration and metrics both go through a network broker
  instead of shared memory/filesystem, an island doesn't care where it runs — a laptop process and
  a cluster Pod are just two peers pointed at the same broker.
- **Multiple k8s clusters**: same island code, harder networking (broker reachable across cluster
  boundaries). Deployment problem, not an island-model design problem — deferred until
  workstation+single-cluster is working.

Worth keeping distinct: **parallel fitness evaluation within one population** (farm individuals to
workers, gather scores — pure speed, no migration) vs. **island-model parallelism** (semi-
independent populations that occasionally cross-pollinate — changes exploration dynamics, not just
speed). They compose, but one is a performance optimization and the other is an algorithmic
choice.

## Open questions

- Exact migration channel interface and delivery guarantees — deferred until island model is
  actually built.
- Whether `RunRegistry` and `ArtifactStore` end up sharing a SQLite file locally, or stay separate
  from day one.
- At what point (data volume / need for multiple viewers) the Redis-backed implementation actually
  gets built, vs. staying hypothetical.

## Incremental plan

1. Scaffold `RunRegistry` / `MetricsSink` / `MetricsSource` / `ArtifactStore` as `Protocol`s with
   local file/SQLite-backed implementations (`libs/telemetry`). No API server or frontend yet.
2. Wire the Python orchestration loop (once the evolution loop from doc 0001 exists) to write
   through `MetricsSink` once per generation.
3. Build a minimal API server exposing `RunRegistry`/`MetricsSource`/`ArtifactStore` over
   REST + SSE.
4. Build the Vue 3 dashboard against that API.
5. Redis-backed implementations + island-model migration channel, once there's a concrete need
   (multiple concurrent runs, a hosted demo, or actual distributed islands).
