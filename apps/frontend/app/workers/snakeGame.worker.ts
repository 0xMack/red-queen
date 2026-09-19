/// <reference lib="webworker" />

// One Snake session, entirely off the main thread (docs/design/0009): the game is the Rust game core
// compiled to WebAssembly (app/wasm/games -- the same rules, observers and baselines training runs
// through PyO3), and a trained model runs in ONNX Runtime Web (app/inference/runtime.ts) right next to
// it, so a decision is observation -> ORT -> action with no postMessage and no Python in between.
//
// Modes:
// - "play": a human steers via keydown-translated relative actions.
// - "watch": a model package (a published champion, or an unpublished one exported on demand) or a
//   games.baselines entry decides every action.
//
// The main thread only sends input/policy messages in and redraws from the state messages this worker
// posts back.

import init, { decodeRelative3, greedyDecide, RandomPolicy, SnakeGame } from "~/wasm/games/games.js"
import gamesWasmUrl from "~/wasm/games/games_bg.wasm?url"
import { loadModel, type LoadedModel, type LoadProgress } from "~/inference/runtime"
import type { ModelSpec } from "~/types/modelpack"

declare const self: DedicatedWorkerGlobalScope

interface RenderState {
  width: number
  height: number
  cells: { x: number; y: number; label: string }[]
  score: number
  alive: boolean
}

// What drives the snake in watch mode: a model package or a named baseline -- either way with the
// interface (docs/design/0007: observer + action adapter) it plays under. Absent interfaceId = the
// game's default (features.v1 + relative3.v1).
interface PolicySpec {
  baseline?: string
  model?: ModelSpec
  interfaceId?: string
}

type InboundMessage =
  // A start with neither model nor baseline is play mode (a human steers).
  | ({ type: "start" } & PolicySpec)
  | { type: "input"; action: number }
  | { type: "restart" }
  | ({ type: "load_policy" } & PolicySpec)
  | { type: "stop" }
  | { type: "set_speed"; intervalMs: number }
  // Load the game runtime without starting anything (answers "ready").
  | { type: "warmup" }

type OutboundMessage =
  | { type: "ready" }
  // observation: watch mode only -- what the policy will decide its *next* move from, so the page can
  // visualize the network's live activations without asking again.
  | { type: "state"; renderState: RenderState; reward: number; done: boolean; step: number; observation?: number[] }
  | { type: "error"; message: string }
  | { type: "model_progress"; progress: LoadProgress }
  | { type: "model_loaded"; backend: string; variantId: string; loadMs: number; selfTest: LoadedModel["selfTest"] }
  // A package failed to load or self-test on this device: the page remembers it and falls back.
  | { type: "model_error"; message: string; packageId: string; variantId: string; backend: string }

const BOARD = { width: 10, height: 10 }
const DEFAULT_TICK_INTERVAL_MS = 110
const LABELS = ["body", "head", "food"] as const
let tickIntervalMs = DEFAULT_TICK_INTERVAL_MS

let mode: "play" | "watch" = "play"
let game: SnakeGame | null = null
let observerId = "features.v1"
let pendingAction = 0 // play mode only
let stepCount = 0
let timer: ReturnType<typeof setInterval> | null = null

// Watch mode: exactly one of these decides.
let model: LoadedModel | null = null
let modelKey: string | null = null // package/variant/backend -- an auto-replay must not reload anything
let baseline: ((observation: Float64Array) => number) | null = null
let decisionInFlight = false

// Watch mode auto-replays the same policy on a fresh seed after a short pause (a finished run has no
// more champions coming, so an episode ending would otherwise freeze the page).
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

function clearRestart() {
  if (watchRestartTimer !== null) {
    clearTimeout(watchRestartTimer)
    watchRestartTimer = null
  }
}

let ready: Promise<void> | null = null
function ensureReady(): Promise<void> {
  ready ??= init({ module_or_path: gamesWasmUrl }).then(() => undefined)
  return ready
}

// "snake/features.v1+relative3.v1" -> the observer the game core encodes for the model.
function observerFor(interfaceId: string | undefined): string {
  if (!interfaceId) return "features.v1"
  const [gameName, rest] = interfaceId.split("/")
  const [observer, adapter] = (rest ?? "").split("+")
  if (gameName !== "snake" || !observer) throw new Error(`not a Snake interface: ${interfaceId}`)
  if (adapter !== "relative3.v1") throw new Error(`no action adapter ${adapter} in the game core`)
  return observer
}

function newGame(): SnakeGame {
  game?.free()
  // A fresh seed per game, so replays and restarts face new food sequences.
  return new SnakeGame(BOARD.width, BOARD.height, Math.floor(Math.random() * 2 ** 32), observerId)
}

function renderState(): RenderState {
  const flat = game!.cells()
  const cells = []
  for (let i = 0; i < flat.length; i += 3) cells.push({ x: flat[i]!, y: flat[i + 1]!, label: LABELS[flat[i + 2]!]! })
  return { width: game!.width, height: game!.height, cells, score: game!.score, alive: !game!.done }
}

function postState(reward: number, observation?: Float64Array) {
  post({
    type: "state",
    renderState: renderState(),
    reward,
    done: game!.done,
    step: stepCount,
    observation: observation ? Array.from(observation) : undefined,
  })
}

function playTick() {
  const reward = game!.step(pendingAction)
  pendingAction = 0
  stepCount += 1
  if (game!.done) stopTicking()
  postState(reward)
}

async function watchTick() {
  if (decisionInFlight || !game) return
  decisionInFlight = true
  try {
    const current = game
    const observation = current.observation()
    let action: number
    if (model) {
      // ORT's run() is async; a slow model skips ticks rather than stacking decisions.
      const decidedBy = model
      const [outputs] = await decidedBy.run([Array.from(observation)])
      if (decidedBy !== model || current !== game || timer === null) return // replaced or stopped meanwhile
      action = decodeRelative3(Float64Array.from(outputs!))
    } else if (baseline) {
      action = baseline(observation)
    } else {
      return
    }
    const reward = current.step(action)
    stepCount += 1
    const done = current.done
    if (done) stopTicking()
    postState(reward, done ? observation : current.observation())
    if (done && lastSpec) {
      const replay = lastSpec
      watchRestartTimer = setTimeout(() => {
        watchRestartTimer = null
        if (mode === "watch") void loadPolicy(replay)
      }, WATCH_RESTART_DELAY_MS)
    }
  } catch (e) {
    stopTicking()
    post({ type: "error", message: e instanceof Error ? e.message : String(e) })
  } finally {
    decisionInFlight = false
  }
}

function tick() {
  if (!game) return
  if (mode === "watch") void watchTick()
  else playTick()
}

function startTicking() {
  stopTicking()
  timer = setInterval(tick, tickIntervalMs)
}

async function releaseModel() {
  const previous = model
  model = null
  modelKey = null
  await previous?.release()
}

async function loadPolicy(spec: PolicySpec) {
  await ensureReady()
  stopTicking()
  clearRestart()
  observerId = observerFor(spec.interfaceId)

  if (spec.model) {
    const key = `${spec.model.manifest.package_id}/${spec.model.variantId}/${spec.model.backend}`
    if (key !== modelKey) {
      await releaseModel()
      try {
        model = await loadModel(spec.model, (progress) => post({ type: "model_progress", progress }))
      } catch (e) {
        post({
          type: "model_error",
          message: e instanceof Error ? e.message : String(e),
          packageId: spec.model.manifest.package_id,
          variantId: spec.model.variantId,
          backend: spec.model.backend,
        })
        return
      }
      modelKey = key
      post({ type: "model_loaded", backend: model.backend, variantId: model.variantId, loadMs: model.loadMs, selfTest: model.selfTest })
    }
    baseline = null
  } else if (spec.baseline) {
    await releaseModel()
    if (spec.baseline === "greedy") baseline = (observation) => greedyDecide(observation)
    else if (spec.baseline === "random") {
      const policy = new RandomPolicy(Math.floor(Math.random() * 2 ** 32))
      baseline = (observation) => policy.decide(observation)
    } else throw new Error(`unknown Snake baseline: ${spec.baseline}`)
  }

  lastSpec = spec
  mode = "watch"
  game = newGame()
  stepCount = 0
  postState(0, game.observation())
  startTicking()
}

async function initialize(spec: PolicySpec) {
  await ensureReady()
  // The worker is reused across page visits: clear whatever the previous page left running.
  stopTicking()
  clearRestart()
  lastSpec = null
  post({ type: "ready" })

  if (spec.model || spec.baseline) {
    await loadPolicy(spec)
  } else {
    mode = "play"
    baseline = null
    observerId = "features.v1"
    game = newGame()
    pendingAction = 0
    stepCount = 0
    postState(0)
    startTicking()
  }
}

function restart() {
  if (!game || mode !== "play") return
  game = newGame()
  pendingAction = 0
  stepCount = 0
  postState(0)
  startTicking()
}

function reportError(e: unknown) {
  post({ type: "error", message: e instanceof Error ? e.message : String(e) })
}

self.onmessage = (event: MessageEvent<InboundMessage>) => {
  const message = event.data
  if (message.type === "start") {
    initialize(message).catch(reportError)
  } else if (message.type === "input") {
    if (mode === "play") pendingAction = message.action
  } else if (message.type === "restart") {
    restart()
  } else if (message.type === "load_policy") {
    loadPolicy(message).catch(reportError)
  } else if (message.type === "warmup") {
    ensureReady()
      .then(() => post({ type: "ready" }))
      .catch(reportError)
  } else if (message.type === "set_speed") {
    tickIntervalMs = Math.min(1000, Math.max(20, message.intervalMs))
    if (timer !== null) startTicking()
  } else if (message.type === "stop") {
    // Pause without tearing anything down: the worker is shared across pages (useSnakeWorker.ts).
    stopTicking()
    clearRestart()
  }
}
