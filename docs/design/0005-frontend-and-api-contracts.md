# 0005 — Frontend, API contracts, and endpoint definitions

Status: **Draft** — prepared for review, not yet implemented.
Relates to: [0002](0002-realtime-visualization-architecture.md) (original architecture proposal),
[0003](0003-algorithm-landscape-and-roadmap.md) (games plan — `apps/`/`apis/` deliberately deferred
until now)

## Context

Doc 0002 proposed the shape of this (decoupled log-then-serve, `RunRegistry`/
`MetricsSink`+`Source`/`ArtifactStore`, SSE for the observational stream, a separate control API,
a Vue 3 frontend) before `libs/telemetry` or any game existed. Doc 0003 then deliberately deferred
building `apps/`/`apis/` until more than one `Environment` existed to design the rendering contract
against — that's now true (`games.reach1d`, `games.snake`, both exposing `render_state()`). This
doc makes doc 0002's proposal concrete: an actual tech stack with pinned versions, actual pydantic
schemas, an actual endpoint table, and an actual frontend architecture — plus one thing doc 0002
didn't anticipate at all: using WebAssembly for client-side game rendering/simulation.

## Tech stack (versions as of September 2026 — verify against source before pinning; AI-summarized
search results can be slightly stale or approximate)

| Piece | Choice | Version | Why |
|---|---|---|---|
| API framework | FastAPI | 0.141.x | Already pydantic-native; every response model can be a `telemetry`/`evolve` pydantic model directly, no schema duplication |
| Data models | Pydantic | 2.13.x | Already this repo's standard (docs/CODING_GUIDELINES.md) — this doc adds no new modeling convention, just a new consumer of the existing one |
| SSE | `sse-starlette` | latest | FastAPI has no first-party SSE support; raw `StreamingResponse` is missing the mandatory `Cache-Control: no-cache` header, disconnect detection, and can silently buffer under some ASGI setups. `sse-starlette`'s `EventSourceResponse` handles all three. |
| Frontend framework | Nuxt | 4.5.x | Nuxt 3 reached end-of-life July 31, 2026 — 4 is the only currently-supported line, not just the newer one |
| UI framework | Vue | 3.5.x (stable) | Not the 3.6 release-candidate line (in RC as of this writing) — flagging as a choice worth your input: stable foundation vs. tracking the RC |
| State | Pinia | 4.x | Note: Pinia 4 is ESM-only and requires `@vue/devtools-api` installed alongside it — a real breaking change from 3.x, not just a version bump |
| Styling | Tailwind CSS | 4.3.x | |
| Client-side game execution | WebAssembly | — | See the dedicated section below — this is the one open architectural question in this doc, not just a version pin |

[FastAPI](https://pypi.org/project/fastapi/) ·
[Pydantic](https://github.com/pydantic/pydantic/releases) ·
[Nuxt](https://nuxt.com/blog/v4) ·
[Vue](https://vuejs.org/about/releases) ·
[Pinia](https://github.com/vuejs/pinia/releases) ·
[Tailwind CSS](https://github.com/tailwindlabs/tailwindcss/releases) ·
[sse-starlette](https://github.com/sysid/sse-starlette) ·
[Pyodide](https://pyodide.org/)

## Architecture overview

```
apis/          FastAPI app -- reads/writes libs/telemetry directly, no new persistence layer
  routers/
    runs.py       RunRegistry + MetricsSource + ArtifactStore endpoints
    games.py      live game sessions (new -- see below)
  schemas.py     API-only models (RunRegistry/MetricsSource/ArtifactStore models are reused as-is)
  main.py

apps/          Nuxt 4 app
  pages/
    index.vue          run list
    runs/[id].vue       one run's live metrics (SSE) + champion trace
    play/[game].vue     watch/play a game (WASM-rendered)
  stores/            Pinia: useRunsStore, useMetricsStream, useGameSession
  wasm/              compiled game modules (see WebAssembly section)
```

Dependency direction stays consistent with every other decision in this repo: `apis/` depends on
`libs/telemetry`, `libs/evolve`, and `libs/games`; none of those know `apis/` exists. `apps/` only
ever talks to `apis/` over HTTP/SSE — it never imports Python.

## API contracts

### Reusing existing pydantic models directly

`telemetry.GenerationStats` and `telemetry.RunInfo` are already pydantic models with full
`Field(..., description=...)` and validation (docs/CODING_GUIDELINES.md's first standard, from
early in this repo) — FastAPI can use them **directly** as `response_model`s. No parallel
"API schema" needs to be hand-written and kept in sync; this is the concrete payoff of a decision
made long before an API existed.

### New models this layer needs

```python
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ControlAction(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    STEP = "step"


class ControlRequest(BaseModel):
    action: ControlAction = Field(..., description="Control action to apply to a running evolve() call.")


class GameSessionCreate(BaseModel):
    game: str = Field(..., description="Registered game name, e.g. 'snake' or 'reach1d'.")
    seed: int | None = Field(default=None, description="Fixed seed for reproducibility; random if omitted.")


class GameSessionState(BaseModel):
    session_id: str = Field(..., min_length=1)
    game: str
    render_state: dict[str, Any] = Field(..., description="Exactly the game's own render_state() output.")
    reward: float
    done: bool
    step: int = Field(..., ge=0)


class ActionRequest(BaseModel):
    action: float = Field(..., description="Action value; shape/range depends on the game's action space.")


class TrajectoryArtifact(BaseModel):
    """A full episode's states/actions/rewards -- doc 0003 flagged 'ArtifactStore needs a
    trajectory artifact type' as a future extension; this is that, now concrete."""

    run_id: str | None = Field(default=None, description="Originating run, if this came from training rather than a live/human session.")
    game: str
    seed: int
    states: list[dict[str, Any]] = Field(..., description="render_state() output at every step.")
    actions: list[float]
    rewards: list[float]
```

### Endpoints

| Method | Path | Returns | Notes |
|---|---|---|---|
| GET | `/runs` | `list[RunInfo]` | `RunRegistry.list_runs()` |
| GET | `/runs/{run_id}` | `RunInfo` | `RunRegistry.get_run()` |
| GET | `/runs/{run_id}/metrics/history` | `list[GenerationStats]` | `?since_generation=N`; `MetricsSource.history()` |
| GET | `/runs/{run_id}/metrics/stream` | SSE of `GenerationStats` | backfill-then-live, `MetricsSource.subscribe()` (see adaptation note below) |
| GET | `/runs/{run_id}/artifacts/{ref}` | raw bytes | `ArtifactStore.get_program`/`get_trace` |
| POST | `/runs/{run_id}/control` | 202 | `ControlRequest` — pause/resume/step (see open question) |
| POST | `/games/{game}/sessions` | `GameSessionState` | `GameSessionCreate` — start a live session |
| POST | `/games/{game}/sessions/{id}/actions` | `GameSessionState` | `ActionRequest` — one step, human or scripted |
| GET | `/games/{game}/sessions/{id}/stream` | SSE of `GameSessionState` | for an agent playing continuously, or replay |
| GET | `/games/{game}/sessions/{id}/trajectory` | `TrajectoryArtifact` | full recorded episode, e.g. for WASM client-side replay or as training data (doc 0003 phase 6/7) |

### A real integration detail worth flagging now, not discovering later

`MetricsSource.subscribe()` (`libs/telemetry`) is a **synchronous** generator that polls with
`time.sleep()`. FastAPI/`sse-starlette` want an **async** generator. The adaptation is
straightforward (wrap the sync generator in a thread via `anyio.to_thread` or rewrite the polling
loop with `asyncio.sleep`), but it's real work, not a detail to wave away — `libs/telemetry` was
deliberately built with zero dependency on this layer (docs/design/0002), so it doesn't and
shouldn't know about async/FastAPI itself; the adaptation belongs in `apis/`, not in `telemetry`.

## WebAssembly for game rendering — the actual open question in this doc

Two honest options for what "WebAssembly for game rendering" means concretely:

1. **Pyodide**: compile/run the actual `libs/games` Python code in the browser via Pyodide (CPython
   compiled to WASM). Reuses the exact simulation code already written and tested — `games.snake`
   and `games.reach1d` behave identically client-side and server-side, zero risk of drift between
   two implementations. Cost: a multi-MB runtime download, and Pyodide's own startup overhead —
   probably fine for a local single-user tool, worth confirming before committing.
2. **A from-scratch lightweight reimplementation** (Rust or AssemblyScript, compiled to WASM):
   smaller payload, likely faster, but now the same game logic exists in two languages that must be
   kept in sync — a real, ongoing maintenance cost and a real correctness risk (the two could
   silently diverge) for games this small.

**Recommendation: Pyodide**, specifically because the games here are deliberately tiny (that's
what made them cheap to evolve against in the first place) — the payload-size argument against
Pyodide matters far more for a large app than for occasionally loading one interactive demo page,
and "the client runs the identical, already-tested simulation" is a correctness property worth
paying for. This is the one recommendation in this doc I'd most want checked against your own
intuition before it's built, since it trades off differently depending on how snappy the play
experience needs to feel.

What this enables concretely: a human playing against a live agent doesn't need a network
round-trip per frame — the browser steps its own Pyodide-run copy of the simulation from an action
immediately, and only syncs with the server (submitting the resulting trajectory) once per episode
via `POST /games/{game}/sessions/{id}/trajectory` or similar. Replay of a stored `TrajectoryArtifact`
also becomes cheap: ship the seed + action sequence, not every frame's full state, and let the
client re-simulate deterministically (every game here already resets deterministically from a
seed, per docs/design/0003's fixed-benchmark-environments principle).

## Frontend architecture

- **Pinia stores**: `useRunsStore` (list + selected run), `useMetricsStream` (owns the
  `EventSource` connection to `/runs/{id}/metrics/stream`, exposes reactive `GenerationStats[]`),
  `useGameSession` (owns a Pyodide worker instance + the current session's state).
- **Pages**: a run list, a run detail page (live chart, backed by `useMetricsStream`, same shape as
  the notebooks' matplotlib plots but live), and a game page that boots a Pyodide worker and either
  replays a `TrajectoryArtifact` or accepts live human input.
- **Tailwind**: utility-first, no separate component library decision made here — deferred until
  there's an actual page to style.
- Nothing here is prescriptive about visual design — this doc is about the data contract between
  frontend and backend, not the UI itself.

## Open questions

- **WebAssembly approach** (above) — the one thing in this doc most worth your review before any
  code gets written, since it shapes both `apps/wasm/` and how much of `libs/games` needs touching.
- **Control API scope**: `POST /runs/{run_id}/control` (pause/resume/step) needs `evolve()` itself
  to support cooperative pausing. The cheapest path is extending the existing `on_generation`
  callback mechanism (docs/design/0001) to check a shared flag each generation and block if paused
  — no change to `evolve()`'s core contract, reusing the exact decoupling point already designed
  for observability. Worth confirming this is actually wanted before building it — doc 0003's games
  plan treated this as a "whenever `apps/`/`apis/` work starts" item, not a hard requirement of the
  first version.
- **Auth/deployment**: out of scope for this doc — assumed local, single-user, no auth, same as
  everything else in this repo so far.

## Incremental plan

1. FastAPI skeleton in `apis/` wired directly to existing `libs/telemetry` backends — no new
   persistence, just endpoints over what already exists. `/runs`, `/runs/{id}`,
   `/runs/{id}/metrics/history` first (plain REST, no SSE yet — prove the read path before adding
   streaming).
2. Add `/runs/{id}/metrics/stream` (SSE), including the sync-to-async adaptation noted above.
3. Nuxt 4 skeleton in `apps/`, Pinia + Tailwind scaffolding, one page: run list → run detail with a
   live chart. This is the first true end-to-end vertical slice.
4. Game viewing via `render_state()` rendered in plain Canvas/SVG (no WASM yet) — prove the game
   data contract before adding Pyodide's complexity on top of it.
5. Pyodide-based client-side simulation, once the JS-rendered version has proven the contract.
6. Control API (pause/step), only once actually wanted.
