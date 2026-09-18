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
   locally, driven by a client-side tick loop (worked example below) — no round trip, no server
   involvement in the interaction loop itself. `POST .../actions` (if called at all) or
   `POST .../trajectory` happens once, after the episode ends, purely for recording — off the
   interaction's critical path entirely.
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

### Worked example: Snake's keyboard controls

Concrete, since "runs client-side" glosses over real mechanics worth pinning down before they're
needed. Snake specifically doesn't move once per keypress — it moves on a fixed tick, and a
keypress just sets the pending direction for the next tick, the same as every Snake implementation
(true of the arcade version of the game, not something this architecture adds):

- **A JS timer drives movement, not the keypress.** `setInterval(tick, 120)` (~120ms per grid-move)
  is the actual clock. `keydown` doesn't move the snake — it updates a `pendingAction` variable that
  whichever tick fires next will consume. Fast enough that a keypress feels instant to the player,
  but mechanically it's "queued, applied within one tick" — which is also what stops a rapid
  double-tap from causing a double-move.
- **Arrow keys are absolute; `Snake.step()` is relative** (`-1`/`0`/`1`, left/straight/right) — a
  real translation the client has to do, not something `games.snake` handles. It needs the snake's
  *current heading* (from the last state) to turn "player pressed Up, snake is heading right" into
  "that's a left turn." This conversion lives in the JS/Pyodide glue, never in `games.snake` itself.
- **Pyodide runs in a Web Worker, not the main thread**, so a Python-side hiccup or Pyodide's call
  overhead can never jank input handling or rendering. The worker owns the `Snake` instance *and*
  the tick timer entirely; the main thread only forwards `keydown` events in and redraws the canvas
  from whatever `render_state()` comes back out:

  ```
  keydown (Up/Down/Left/Right)
    -> main thread: translate to a relative turn using the last-known heading
    -> postMessage({ type: "input", action }) to the worker
    -> worker: pendingAction = action

  worker's tick timer (every ~120ms):
    -> snake.step(pendingAction)
    -> postMessage({ type: "state", render_state, reward, done }) to the main thread
    -> main thread: redraw the canvas
  ```

  Neither side blocks the other. This is also why it's genuinely real-time: there's no network hop
  anywhere in that loop, only in-browser message passing and a local Python call — Pyodide's call
  overhead and `postMessage` marshalling are both sub-millisecond for something this small, negligible
  against a 120ms tick. The server only re-enters the picture once, at episode end, for the
  trajectory upload.
- **Phasing**: build the main-thread version first (simpler — proves the tick-loop and
  action-translation logic work at all) and move Pyodide into a Worker as the next step once that's
  confirmed, same incremental style as everything else in this doc.

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

1. ✅ `apis/backend` skeleton (`routers/runs.py` first), wired directly to existing
   `libs/telemetry` backends — no new persistence, just endpoints over what already exists.
   `/runs`, `/runs/{id}`, `/runs/{id}/metrics/history` first (plain REST). Added `"apis/*"` to the
   uv workspace `members` glob at this point.
2. ✅ Added `/runs/{id}/metrics/stream` (SSE) and `/runs/{id}/artifacts/{ref}`, including the
   sync-to-async adaptation noted above (`anyio.to_thread.run_sync(..., abandon_on_cancel=True)`
   over `FileMetricsStore.subscribe()`). Tested at the generator level rather than through a live
   `TestClient` request — Starlette's TestClient doesn't reliably simulate a mid-stream client
   disconnect, so a full request through it hangs waiting for a generation that never arrives; see
   `apis/backend/README.md` and `apis/backend/tests/test_runs.py`. Verified against real data from
   `jobs/baseline_gp_run.py` with a live `uvicorn` process, not just tests.
3. ✅ `apps/frontend` skeleton (Nuxt 4.5, Pinia 4, Tailwind CSS 4 via `@tailwindcss/vite`): run
   list (`/`) → run detail (`/runs/[id]`) with a live chart. The first true end-to-end vertical
   slice — verified with a live backend + a headless-browser (Playwright) check, not just SSR HTML
   inspection: navigating to a run detail page actually opens the SSE connection, backfills real
   generations, and updates the chart with zero console errors. Chart is a hand-rolled SVG
   polyline, not a charting library, to keep the dependency list to exactly what's named above. No
   frontend test suite yet — a gap noted in `apps/frontend/README.md`, not silently skipped.
4. ✅ `routers/games.py` (session create/action/trajectory, server-side `Environment` in an
   in-memory `GameSessionStore`) + `apps/frontend`'s `/play/[game].vue` rendering `render_state()`
   via SVG (no Pyodide yet) — proved the game data contract, including a real wrinkle:
   `render_state()`'s `cells` uses tuple keys, not valid JSON as-is, fixed by flattening to a list
   at the API boundary (`game_sessions.json_safe_render_state()`, see
   `docs/CODING_GUIDELINES.md`). Controls are Snake's native relative action space directly
   (&larr;/&rarr;), not the absolute-direction/client-tracked-heading scheme below — that's specific
   to step 5, where the simulation actually moves client-side. Verified with a live backend + a
   headless-browser check that actually sends keyboard input and watches the snake's path change,
   die, and restart.
5. Pyodide-based client-side simulation, once the JS-rendered version proved the contract:
   - ✅ Main-thread Pyodide, interaction mode 1 (human play): `/play/[game].vue` loads Pyodide (CDN,
     `usePyodideGames.ts`) and runs the actual `games.snake.Snake` class in-browser, ticked by a
     local `setInterval` — no backend round trip per tick, verified by asserting zero requests to
     `apis/backend` occur during play. Controls translate absolute arrow keys to Snake's relative
     action space via a client-tracked heading, matching the worked example above. Real bug hit and
     fixed: `render_state()`'s tuple-keyed `cells` dict makes `pyProxy.toJs()` raise
     `ConversionError` (a JS `Map` key restriction, not a JSON one, but the same underlying shape
     problem doc 0005 step 4 already hit and fixed on the JSON side) — fixed the same way, by
     flattening on the Python side before crossing into JS (see `docs/CODING_GUIDELINES.md`).
   - Not yet done: interaction mode 2 (watching a finished policy) — there's no serialized
     Snake-playing policy artifact anywhere in the repo yet to load (the neuroevolution runs against
     `games.snake` in `notebooks/0005-neuroevolution-snake.ipynb` never went through
     `telemetry`/`ArtifactStore`), so building this now would have no real data to point at. Revisit
     once a Snake-training job writes champions to `ArtifactStore` the way `jobs/baseline_gp_run.py`
     does for symbolic regression.
   - Not yet done: moving the simulation into a Web Worker (main-thread proved the tick-loop and
     keypress-to-action translation work; the Worker move is a separate, still-pending step).
6. Interaction mode 3 (watching the live current-best champion) — depends on (2) and (5) both
   existing.
7. Control API (pause/step) in `routers/runs.py`, extending `evolve()`'s `on_generation` mechanism.
