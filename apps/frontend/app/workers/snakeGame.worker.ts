/// <reference lib="webworker" />

// Runs games.snake.Snake entirely inside this worker, in one of two modes (doc 0005 steps 5-6):
//
// - "play": a human steers via keydown-translated relative actions (doc 0005's worked example).
// - "watch": a loaded trained network (evolve.neuro.WeightVector or evolve.neat.NeatGenome) decides every action instead (interaction
//   modes 2/3) -- the run detail / watch page feeds it a champion's serialized weights, fetched
//   from apis/backend's existing artifact endpoint (no backend changes needed there: a serialized
//   WeightVector is just opaque bytes to ArtifactStore, same as a LinearProgram's repr()).
//
// Either way, Pyodide's WASM execution and the tick timer live off the main thread, so heavy
// Python work can never jank the page. The main thread only sends input/policy messages in and
// redraws from the state messages this worker posts back.

import type { PyodideInterface } from "pyodide"
import type { PyProxy } from "pyodide/ffi"

declare const self: DedicatedWorkerGlobalScope

interface RenderState {
  width: number
  height: number
  cells: { x: number; y: number; label: string }[]
  score: number
  alive: boolean
}

// What drives the snake in watch mode: a trained network (policyJson, a serialized
// evolve.neuro.WeightVector or evolve.neat.NeatGenome) or a named games.baselines entry -- either way with the interface
// (docs/design/0007: observer + action adapter) it plays under. Absent interfaceId = the game's
// default observer.
interface PolicySpec {
  policyJson?: string
  baseline?: string
  interfaceId?: string
}

type InboundMessage =
  // A start with neither policyJson nor baseline is play mode (a human steers).
  | ({ type: "start" } & PolicySpec)
  | { type: "input"; action: number }
  | { type: "restart" }
  | ({ type: "load_policy" } & PolicySpec)
  | { type: "stop" }
  | { type: "set_speed"; intervalMs: number }
  // Load the Python runtime without starting anything (answers "ready") -- lets a page run a
  // countdown only once the ~10s cold load is done, instead of mid-game.
  | { type: "warmup" }

type OutboundMessage =
  | { type: "ready" }
  // observation: watch mode only -- the 11 features the policy will decide its *next* move from, so
  // the page can visualize the network's live activations without a second Python round trip.
  | { type: "state"; renderState: RenderState; reward: number; done: boolean; step: number; observation?: number[] }
  | { type: "error"; message: string }

// Pinned to match the `pyodide` npm package version exactly -- the JS loader and the CDN-hosted
// runtime files (wasm binary, stdlib zip) must be the same version.
const PYODIDE_VERSION = "314.0.7"
const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`
const DEFAULT_TICK_INTERVAL_MS = 110
// Adjustable (watch-mode speed control): clamped so a bad message can't spin the worker.
let tickIntervalMs = DEFAULT_TICK_INTERVAL_MS

let pyodide: PyodideInterface | null = null
let mode: "play" | "watch" = "play"
let snake: PyProxy | null = null
let policy: PyProxy | null = null
let currentObservation: unknown = null // watch mode only; a plain JS array fed back into Python

let renderStateForJs: PyProxy | null = null
let watchTick: PyProxy | null = null

let newSnake: PyProxy | null = null
let makePolicyFn: PyProxy | null = null

let pendingAction = 0 // play mode only
let stepCount = 0
let timer: ReturnType<typeof setInterval> | null = null

// Watch mode: a finished run has no more champions coming, so an episode ending would otherwise
// just freeze the page forever. Auto-replay the same policy (a fresh Snake seed each time, for
// variety) after a short pause -- superseded immediately if a genuinely new champion arrives
// first (loadPolicy() clears any pending timer before doing anything else).
let lastSpec: PolicySpec | null = null
let watchRestartTimer: ReturnType<typeof setTimeout> | null = null
const WATCH_RESTART_DELAY_MS = 1000

function post(message: OutboundMessage) {
  self.postMessage(message)
}

function stopTicking() {
  if (timer !== null) {
    clearInterval(timer)
    timer = null
  }
}

// render_state_for_js (Python bootstrap below) flattens render_state()'s tuple-keyed `cells` dict
// before it crosses into JS -- a Python tuple isn't a valid key for the Map toJs() would otherwise
// try to build (pyodide.ffi.ConversionError), the same underlying problem apis/backend hit on the
// JSON boundary (see docs/CODING_GUIDELINES.md).
function readRenderState(): RenderState {
  const stateProxy = renderStateForJs!(snake)
  const state = stateProxy.toJs({ dict_converter: Object.fromEntries }) as RenderState
  stateProxy.destroy()
  return state
}

async function loadPackageSource(target: PyodideInterface, pkg: string) {
  const response = await fetch(`/api/py-source/${pkg}`)
  const files = (await response.json()) as Record<string, string>
  target.FS.mkdirTree(`/py/${pkg}`)
  for (const [name, content] of Object.entries(files)) {
    target.FS.writeFile(`/py/${pkg}/${name}`, content)
  }
}

function playTick() {
  // Explicit destroy() on every PyProxy this loop creates -- it runs every 150ms for as long as
  // the worker is alive, so relying on Pyodide's FinalizationRegistry safety net alone would grow
  // the WASM heap needlessly instead of freeing it immediately.
  const resultProxy = snake!.step(pendingAction)
  const [, reward, doneNow] = resultProxy.toJs() as [unknown, number, boolean]
  resultProxy.destroy()

  pendingAction = 0
  stepCount += 1
  if (doneNow) stopTicking()
  post({ type: "state", renderState: readRenderState(), reward, done: doneNow, step: stepCount })
}

function watchTickOnce() {
  if (!policy) return
  // watch_tick (Python bootstrap) does action-decision + step + render-state-read in one call, to
  // avoid a JS<->Python round trip per sub-step; see its docstring for why.
  const resultProxy = watchTick!(snake, policy, currentObservation)
  const result = resultProxy.toJs({ dict_converter: Object.fromEntries }) as {
    observation: unknown
    reward: number
    done: boolean
    render_state: RenderState
  }
  resultProxy.destroy()

  currentObservation = result.observation
  stepCount += 1
  if (result.done) stopTicking()
  post({
    type: "state",
    renderState: result.render_state,
    reward: result.reward,
    done: result.done,
    step: stepCount,
    observation: result.observation as number[],
  })

  if (result.done && lastSpec) {
    const specToReplay = lastSpec
    watchRestartTimer = setTimeout(() => {
      watchRestartTimer = null
      if (mode === "watch") loadPolicy(specToReplay)
    }, WATCH_RESTART_DELAY_MS)
  }
}

function tick() {
  if (!snake) return
  if (mode === "watch") watchTickOnce()
  else playTick()
}

// This worker can be reused across page navigations (see app/composables/useSnakeWorker.ts) rather
// than recreated per visit, so a second/third "start" message must NOT pay Pyodide's multi-MB CDN
// load again. Memoizing the setup promise makes ensurePyodideReady() safe to call every time
// initialize() runs -- first call does the real work, every later call is a no-op await.
let pyodideReady: Promise<void> | null = null

function ensurePyodideReady(): Promise<void> {
  pyodideReady ??= (async () => {
    const { loadPyodide } = await import("pyodide")
    pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX_URL })

    // evolve needs its whole package (not just neuro.py): evolve/__init__.py eagerly imports every
    // submodule, and all of them are pure stdlib -- verified before wiring this up, no
    // numpy/micropip needed. games likewise needs no external dependencies.
    await loadPackageSource(pyodide, "games")
    await loadPackageSource(pyodide, "evolve")

    pyodide.runPython(`
import random
import sys

if "/py" not in sys.path:
    sys.path.insert(0, "/py")

from evolve.networks import network_from_json
from games import baselines as _baselines
from games import interfaces as _interfaces
from games.snake import Snake

# Pyodide bridge glue, not part of libs/games or libs/evolve themselves -- same
# boundary-flattening apis/backend/src/backend/game_sessions.json_safe_render_state() does for the
# JSON boundary.
def render_state_for_js(env):
    state = dict(env.render_state())
    cells = state.get("cells")
    if isinstance(cells, dict):
        state["cells"] = [{"x": x, "y": y, "label": label} for (x, y), label in cells.items()]
    return state

# The interface (docs/design/0007) the current watch-mode policy was trained under: its observer
# decides what the policy sees, its action adapter how outputs become a move. None = the game's
# default observer (play mode, or a caller that didn't say).
_current = {"interface": None}

def new_snake(interface_id=None):
    seed = random.randint(0, 2**31 - 1)
    if interface_id:
        _current["interface"] = _interfaces.get(interface_id)
        return _current["interface"].make_game(seed=seed)
    _current["interface"] = None
    return Snake(seed=seed)

class _NetworkPolicy:
    """A trained network (a fixed-topology WeightVector or an evolved NEAT graph -- both expose
    forward()), decoded through the current interface's action adapter."""
    def __init__(self, weights):
        self.weights = weights
    def decide(self, observation):
        outputs = self.weights.forward(list(observation))
        interface = _current["interface"]
        if interface is not None:
            return interface.action.decode(outputs)
        # No interface given: relative3.v1's argmax convention, what every pre-0007 champion used.
        best_index = max(range(len(outputs)), key=lambda i: outputs[i])
        return best_index - 1

class _BaselinePolicy:
    """A games.baselines entry -- the same code the evaluation job scores."""
    def __init__(self, fn):
        self.fn = fn
    def decide(self, observation):
        return self.fn(list(observation))

def make_policy(policy_json=None, baseline=None):
    if baseline:
        return _BaselinePolicy(_baselines.get("snake", baseline).factory(random.randint(0, 2**31 - 1)))
    return _NetworkPolicy(network_from_json(policy_json))

def watch_tick(env, policy, observation):
    action = policy.decide(observation)
    next_observation, reward, done = env.step(action)
    return {
        "observation": next_observation,
        "reward": reward,
        "done": done,
        "render_state": render_state_for_js(env),
    }
`)
    renderStateForJs = pyodide.globals.get("render_state_for_js")
    watchTick = pyodide.globals.get("watch_tick")
    newSnake = pyodide.globals.get("new_snake")
    makePolicyFn = pyodide.globals.get("make_policy")
  })()
  return pyodideReady
}

async function initialize(spec: PolicySpec) {
  await ensurePyodideReady()

  // A worker reused across page visits may already have a session from a previous page (a
  // different mode, a different policy, a pending auto-replay timer) -- clear all of it before
  // starting the newly requested one.
  stopTicking()
  if (watchRestartTimer !== null) {
    clearTimeout(watchRestartTimer)
    watchRestartTimer = null
  }
  policy?.destroy()
  policy = null
  lastSpec = null

  post({ type: "ready" })

  if (spec.policyJson || spec.baseline) {
    await loadPolicy(spec) // creates its own fresh Snake
  } else {
    mode = "play"
    snake?.destroy()
    snake = newSnake!()
    stepCount = 0
    post({ type: "state", renderState: readRenderState(), reward: 0, done: false, step: 0 })
    timer = setInterval(tick, tickIntervalMs)
  }
}

async function loadPolicy(spec: PolicySpec) {
  // Await setup, don't just check `pyodide`: on a live run a new champion can arrive (SSE) while the
  // first start() is still mid-setup -- pyodide exists but make_policy isn't bound yet.
  await ensurePyodideReady()
  stopTicking()
  if (watchRestartTimer !== null) {
    clearTimeout(watchRestartTimer)
    watchRestartTimer = null
  }

  const newPolicy = makePolicyFn!(spec.policyJson ?? null, spec.baseline ?? null)
  policy?.destroy()
  policy = newPolicy
  lastSpec = spec
  mode = "watch"

  // A fresh Snake instance (new random seed), not just snake.reset() on the same one -- otherwise
  // every auto-replay (and every genuinely new champion) would face the identical food sequence.
  snake?.destroy()
  snake = newSnake!(spec.interfaceId ?? null)
  const resetProxy = snake.reset()
  currentObservation = resetProxy.toJs()
  resetProxy.destroy()

  stepCount = 0
  post({
    type: "state",
    renderState: readRenderState(),
    reward: 0,
    done: false,
    step: 0,
    observation: currentObservation as number[],
  })
  timer = setInterval(tick, tickIntervalMs)
}

function restart() {
  if (!snake || mode !== "play") return
  stopTicking()
  // A fresh Snake instance (new random seed), not snake.reset() on the same one -- otherwise
  // "Play again" would always replay the identical food sequence.
  snake.destroy()
  snake = newSnake!()
  pendingAction = 0
  stepCount = 0
  post({ type: "state", renderState: readRenderState(), reward: 0, done: false, step: 0 })
  timer = setInterval(tick, tickIntervalMs)
}

self.onmessage = (event: MessageEvent<InboundMessage>) => {
  const message = event.data
  if (message.type === "start") {
    initialize(message).catch((e) =>
      post({ type: "error", message: e instanceof Error ? e.message : String(e) }),
    )
  } else if (message.type === "input") {
    if (mode === "play") pendingAction = message.action
  } else if (message.type === "restart") {
    restart()
  } else if (message.type === "load_policy") {
    loadPolicy(message).catch((e) =>
      post({ type: "error", message: e instanceof Error ? e.message : String(e) }),
    )
  } else if (message.type === "warmup") {
    ensurePyodideReady()
      .then(() => post({ type: "ready" }))
      .catch((e) => post({ type: "error", message: e instanceof Error ? e.message : String(e) }))
  } else if (message.type === "set_speed") {
    tickIntervalMs = Math.min(1000, Math.max(20, message.intervalMs))
    if (timer !== null) {
      stopTicking()
      timer = setInterval(tick, tickIntervalMs)
    }
  } else if (message.type === "stop") {
    // Pauses without tearing anything down -- used when a page using a reused worker (see
    // app/composables/useSnakeWorker.ts) unmounts, so it doesn't keep ticking/posting messages
    // into the void until the next page sends a fresh "start". Also cancels any pending watch
    // auto-replay, which would otherwise fire after nobody's listening.
    stopTicking()
    if (watchRestartTimer !== null) {
      clearTimeout(watchRestartTimer)
      watchRestartTimer = null
    }
  }
}
