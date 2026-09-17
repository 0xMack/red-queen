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
| Client-side game execution | WebAssembly (Pyodide) | latest | **Decided.** See the dedicated section below for why. |

[FastAPI](https://pypi.org/project/fastapi/) ·
[Pydantic](https://github.com/pydantic/pydantic/releases) ·
[Nuxt](https://nuxt.com/blog/v4) ·
[Vue](https://vuejs.org/about/releases) ·
[Pinia](https://github.com/vuejs/pinia/releases) ·
[Tailwind CSS](https://github.com/tailwindlabs/tailwindcss/releases) ·
[sse-starlette](https://github.com/sysid/sse-starlette) ·
[Pyodide](https://pyodide.org/)

## Architecture overview

### `apis/` and `apps/` are containers of independent modules, like `libs/` — but start with one each

Per feedback on the draft: `apis/` and `apps/` shouldn't each be *one* service/deployment forever —
they should hold *modules*, the same convention `libs/<name>/` already establishes (each one
self-contained: its own `pyproject.toml`/`package.json`, own tests, independently deployable), so
adding a second API service or a second UI deployment later is "add another directory," not a
refactor. A first revision of this doc jumped straight to *two* of each (split along
training/observability vs. games) — corrected per review: **start with one API module and one UI
module**, and get the *separation-readiness* from how each one is organized internally, not from
having multiple of them on day one.

```
apis/
  README.md                index of API services (one today)
  backend/                  the one API service -- runs, metrics, artifacts, control, game sessions
    pyproject.toml
    src/backend/
      main.py
      routers/
        runs.py             training/observability endpoints
        games.py            game session/trajectory endpoints
      schemas.py
    tests/
      test_runs.py
      test_games.py

apps/
  README.md                index of UI deployments (one today)
  frontend/                 the one UI deployment -- run dashboard + game arcade pages
    package.json
    pages/
      index.vue             run list
      runs/[id].vue          one run's live metrics
      play/[game].vue        watch/play a game (Pyodide)
    stores/
      useRunsStore.ts
      useMetricsStream.ts
      useGameSession.ts
    wasm/                    Pyodide bootstrapping, shared game-loader glue
```

The separation-readiness is in the internal layout, not the directory count: `routers/games.py` +
its slice of `schemas.py` + `tests/test_games.py` are already the exact set of files that would
move into a new `apis/games_api/` package if/when games ever earns its own service — a mechanical
extraction, not a redesign, same idea on the frontend side with `useGameSession.ts` and
`play/[game].vue`. Splitting happens when something actually forces it (different deploy cadence,
different scaling needs, a second team) — not speculatively now.

Once `apis/backend/pyproject.toml` exists, the uv workspace `members` glob (currently just
`["libs/*"]`, per AGENTS.md) needs `"apis/*"` added, exactly as already anticipated there.

Dependency direction stays consistent with every other decision in this repo: `apis/backend`
depends on `libs/telemetry`, `libs/evolve`, and `libs/games`; none of those know it exists.
`apps/frontend` only ever talks to `apis/backend` over HTTP/SSE — it never imports Python.

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

All served by the one `apis/backend` service, grouped by router below (`routers/runs.py`,
`routers/games.py`) — the grouping is exactly the seam a future split would cut along, but there's
one service, one process, one `pyproject.toml` for now.

**`routers/runs.py`**

| Method | Path | Returns | Notes |
|---|---|---|---|
| GET | `/runs` | `list[RunInfo]` | `RunRegistry.list_runs()` |
| GET | `/runs/{run_id}` | `RunInfo` | `RunRegistry.get_run()` |
| GET | `/runs/{run_id}/metrics/history` | `list[GenerationStats]` | `?since_generation=N`; `MetricsSource.history()` |
| GET | `/runs/{run_id}/metrics/stream` | SSE of `GenerationStats` | backfill-then-live, `MetricsSource.subscribe()` (see adaptation note below) |
| GET | `/runs/{run_id}/artifacts/{ref}` | raw bytes | `ArtifactStore.get_program`/`get_trace` |
| POST | `/runs/{run_id}/control` | 202 | `ControlRequest` — pause/resume/step (see open question) |

**`routers/games.py`**

| Method | Path | Returns | Notes |
|---|---|---|---|
| POST | `/games/{game}/sessions` | `GameSessionState` | `GameSessionCreate` — start a live session |
| POST | `/games/{game}/sessions/{id}/actions` | `GameSessionState` | `ActionRequest` — one step, human or scripted; not on the interaction's critical path once gameplay moves client-side (see "Real-time player interaction" below) — used for recording/archival, not driving the loop |
| GET | `/games/{game}/sessions/{id}/trajectory` | `TrajectoryArtifact` | full recorded episode — the seed + action sequence a client re-simulates locally, or training data (doc 0003 phase 6/7) |

Note what's *not* here relative to the original draft: a per-frame `GET .../stream` for game state.
Working through real-time interaction (below) concluded gameplay itself doesn't need server
round-trips at all once Pyodide is in the picture — removed rather than left in as an unused,
untested "just in case" endpoint.

### A real integration detail worth flagging now, not discovering later

`MetricsSource.subscribe()` (`libs/telemetry`) is a **synchronous** generator that polls with
`time.sleep()`. FastAPI/`sse-starlette` want an **async** generator. The adaptation is
straightforward (wrap the sync generator in a thread via `anyio.to_thread` or rewrite the polling
loop with `asyncio.sleep`), but it's real work, not a detail to wave away — `libs/telemetry` was
deliberately built with zero dependency on this layer (docs/design/0002), so it doesn't and
shouldn't know about async/FastAPI itself; the adaptation belongs in `apis/`, not in `telemetry`.

## WebAssembly for game rendering — decided: Pyodide

Confirmed. Compile/run the actual `libs/games` Python code in the browser via Pyodide (CPython
compiled to WASM), rather than a from-scratch Rust/AssemblyScript reimplementation. Reuses the
exact simulation code already written and tested — `games.snake` and `games.reach1d` behave
identically client-side and server-side, zero risk of the two silently diverging, which matters
more for correctness here than the alternative's smaller payload/faster startup would save, given
how tiny these games already are.

What this enables concretely: a human playing against a live agent doesn't need a network
round-trip per frame — the browser steps its own Pyodide-run copy of the simulation from an action
immediately, and only syncs with the server (submitting the resulting trajectory) once per episode
via `POST /games/{game}/sessions/{id}/trajectory` or similar. Replay of a stored `TrajectoryArtifact`
also becomes cheap: ship the seed + action sequence, not every frame's full state, and let the
client re-simulate deterministically (every game here already resets deterministically from a
seed, per docs/design/0003's fixed-benchmark-environments principle). The next section works out
just how far this reasoning goes.

## Real-time player interaction

Worth working through explicitly rather than assuming SSE (one-directional, server → client)
"just handles" it — a player's keypress needs a response *now*, and that doesn't sound like an SSE
shape at first glance. It turns out there are three genuinely different interaction modes here, and
only one of them actually needs the server involved per-step at all:

1. **Human plays a game.** Fully client-side after the initial page load. Pyodide steps the game
   the instant a key is pressed — no round trip, no server involvement in the interaction loop
   itself. `POST .../actions` (if called at all) or `POST .../trajectory` happens once, after the
   episode ends, purely for recording — off the interaction's critical path entirely.
2. **Watching an already-trained policy play.** Same answer as (1). Ship the policy's serialized
   weights/program to the client once, alongside the game; `WeightVector.forward()` /
   `LinearProgram.run()` are both pure NumPy/Python and both run under Pyodide exactly like the game
   logic does. Zero server calls per frame — the whole episode plays out client-side.
3. **Watching training's *current* best individual play, live, as training progresses.** This is
   the one case that genuinely needs the server — but only once per *generation*, not once per
   *frame*. The existing metrics SSE stream (`routers/runs.py`) already announces every new
   `GenerationStats`, which already carries a `champion_ref`; the client fetches that one artifact
   when it changes (`GET /runs/{id}/artifacts/{ref}`) and re-runs it locally via the same Pyodide
   mechanism as (2). This reuses infrastructure that exists for a different reason (the metrics
   stream) rather than inventing a new one.

This also validates rather than contradicts the original SSE-only decision: nothing above needs a
genuinely bidirectional, low-latency transport, because gameplay execution itself moved client-side
— a WebSocket would be solving a problem this design doesn't have.

**Explicitly out of scope, not silently ignored**: all of this assumes a single trusted local user.
Server-authoritative gameplay — needed if this ever became adversarial/competitive, or needed
multiple simultaneous viewers watching one genuinely synchronized shared session — is a different,
harder problem this design does not solve. That would need the server to be the source of truth
with a real bidirectional transport (WebSocket), not client-side Pyodide execution. Worth
remembering if multiplayer or shared-viewing ever becomes an actual goal, not something to design
for speculatively now.

## Frontend architecture

All in the one `apps/frontend` deployment for now:

- **Pinia stores**: `useRunsStore` (list + selected run), `useMetricsStream` (owns the
  `EventSource` connection to `/runs/{id}/metrics/stream`, exposes reactive `GenerationStats[]`),
  `useGameSession` (owns a Pyodide worker instance + the current session's state, per the real-time
  interaction design above — the worker runs both the game and, when watching an agent, its
  policy). Each store only ever talks to its own router's endpoints, which is exactly what keeps a
  future split mechanical.
- **Pages**: a run list (`index.vue`), a run detail page with a live chart — the same shape as the
  notebooks' matplotlib plots, but live — and a game page (`play/[game].vue`) that boots the
  Pyodide worker and either replays a `TrajectoryArtifact`, accepts live human input, or polls for
  a live-updating champion (interaction mode 3, above).
- **Tailwind**: utility-first, no separate component library decision made here — deferred until
  there's an actual page to style.
- Nothing here is prescriptive about visual design — this doc is about the data contract between
  frontend and backend, not the UI itself.

## Open questions

- **Control API scope**: `POST /runs/{run_id}/control` (pause/resume/step) needs `evolve()` itself
  to support cooperative pausing. The cheapest path is extending the existing `on_generation`
  callback mechanism (docs/design/0001) to check a shared flag each generation and block if paused
  — no change to `evolve()`'s core contract, reusing the exact decoupling point already designed
  for observability. Confirmed as wanted; not yet built — doc 0003's games plan treated this as a
  "whenever `apps/`/`apis/` work starts" item, and that point has now arrived.
- **Auth/deployment**: confirmed out of scope — assumed local, single-user, no auth, same as
  everything else in this repo so far.

## Incremental plan

1. `apis/backend` skeleton (`routers/runs.py` first), wired directly to existing `libs/telemetry`
   backends — no new persistence, just endpoints over what already exists. `/runs`, `/runs/{id}`,
   `/runs/{id}/metrics/history` first (plain REST, no SSE yet — prove the read path before adding
   streaming). Add `"apis/*"` to the uv workspace `members` glob at this point.
2. Add `/runs/{id}/metrics/stream` (SSE), including the sync-to-async adaptation noted above.
3. `apps/frontend` skeleton (Nuxt 4 + Pinia + Tailwind), one page: run list → run detail with a
   live chart. This is the first true end-to-end vertical slice.
4. `routers/games.py` + game viewing via `render_state()` rendered in plain Canvas/SVG (no Pyodide
   yet) — prove the game data contract before adding Pyodide's complexity.
5. Pyodide-based client-side simulation (interaction modes 1 and 2 above), once the JS-rendered
   version has proven the contract.
6. Interaction mode 3 (watching the live current-best champion) — depends on (2) and (5) both
   existing.
7. Control API (pause/step) in `routers/runs.py`, extending `evolve()`'s `on_generation` mechanism.
