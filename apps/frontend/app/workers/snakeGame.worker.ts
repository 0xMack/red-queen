/// <reference lib="webworker" />

// Runs games.snake.Snake entirely inside this worker (doc 0005 step 5, Web Worker phase) --
// Pyodide's WASM execution and the tick timer both happen off the main thread, so heavy Python
// work can never jank the page. The main thread (app/pages/play/[game].vue) only translates
// keypresses into relative actions and redraws from the state messages this worker posts back --
// see doc 0005's worked example for the message protocol implemented below. Supersedes the
// previous main-thread-Pyodide version (app/composables/usePyodideGames.ts, now removed) once the
// main-thread phase had proven the tick-loop and keypress-to-action translation worked at all.

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

type InboundMessage = { type: "start" } | { type: "input"; action: number } | { type: "restart" }

type OutboundMessage =
  | { type: "ready" }
  | { type: "state"; renderState: RenderState; reward: number; done: boolean; step: number }
  | { type: "error"; message: string }

// Pinned to match the `pyodide` npm package version exactly -- the JS loader and the CDN-hosted
// runtime files (wasm binary, stdlib zip) must be the same version.
const PYODIDE_VERSION = "314.0.7"
const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`
const TICK_INTERVAL_MS = 150

let pyodide: PyodideInterface | null = null
let snake: PyProxy | null = null
let renderStateForJs: PyProxy | null = null
let pendingAction = 0
let stepCount = 0
let timer: ReturnType<typeof setInterval> | null = null

function post(message: OutboundMessage) {
  self.postMessage(message)
}

function stopTicking() {
  if (timer !== null) {
    clearInterval(timer)
    timer = null
  }
}

// render_state_for_js (defined in the Python bootstrap below) flattens render_state()'s
// tuple-keyed `cells` dict before it crosses into JS -- a Python tuple isn't a valid key for the
// Map toJs() would otherwise try to build (pyodide.ffi.ConversionError), the same underlying
// problem apis/backend hit on the JSON boundary (see docs/CODING_GUIDELINES.md).
function readRenderState(): RenderState {
  const stateProxy = renderStateForJs!(snake)
  const state = stateProxy.toJs({ dict_converter: Object.fromEntries }) as RenderState
  stateProxy.destroy()
  return state
}

function tick() {
  if (!snake) return
  // Explicit destroy() on every PyProxy this loop creates -- it runs every 150ms for as long as
  // the worker is alive, so relying on Pyodide's FinalizationRegistry safety net alone would grow
  // the WASM heap needlessly instead of freeing it immediately.
  const resultProxy = snake.step(pendingAction)
  const [, reward, doneNow] = resultProxy.toJs() as [unknown, number, boolean]
  resultProxy.destroy()

  pendingAction = 0
  stepCount += 1
  if (doneNow) stopTicking()
  post({ type: "state", renderState: readRenderState(), reward, done: doneNow, step: stepCount })
}

async function initialize() {
  const { loadPyodide } = await import("pyodide")
  pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX_URL })

  const response = await fetch("/api/py-games")
  const files = (await response.json()) as Record<string, string>
  pyodide.FS.mkdirTree("/py/games")
  for (const [name, content] of Object.entries(files)) {
    pyodide.FS.writeFile(`/py/games/${name}`, content)
  }
  pyodide.runPython(`
import sys
if "/py" not in sys.path:
    sys.path.insert(0, "/py")

# Pyodide bridge glue, not part of libs/games itself -- same boundary-flattening
# apis/backend/src/backend/game_sessions.json_safe_render_state() does for the JSON boundary.
def render_state_for_js(env):
    state = dict(env.render_state())
    cells = state.get("cells")
    if isinstance(cells, dict):
        state["cells"] = [{"x": x, "y": y, "label": label} for (x, y), label in cells.items()]
    return state
`)
  renderStateForJs = pyodide.globals.get("render_state_for_js")

  snake = pyodide.runPython(`
import random
from games.snake import Snake
Snake(seed=random.randint(0, 2**31 - 1))
`)
  stepCount = 0
  post({ type: "ready" })
  post({ type: "state", renderState: readRenderState(), reward: 0, done: false, step: 0 })
  timer = setInterval(tick, TICK_INTERVAL_MS)
}

function restart() {
  if (!snake) return
  stopTicking()
  const obsProxy = snake.reset()
  obsProxy.destroy()
  pendingAction = 0
  stepCount = 0
  post({ type: "state", renderState: readRenderState(), reward: 0, done: false, step: 0 })
  timer = setInterval(tick, TICK_INTERVAL_MS)
}

self.onmessage = (event: MessageEvent<InboundMessage>) => {
  const message = event.data
  if (message.type === "start") {
    initialize().catch((e) =>
      post({ type: "error", message: e instanceof Error ? e.message : String(e) }),
    )
  } else if (message.type === "input") {
    pendingAction = message.action
  } else if (message.type === "restart") {
    restart()
  }
}
