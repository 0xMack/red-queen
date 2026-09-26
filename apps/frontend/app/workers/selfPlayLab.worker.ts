// Checkers self-play, live (docs/design/0010 Phase 4): the Rust TD(λ) loop the training job runs, compiled to
// WebAssembly, learning Checkers from games against itself in the reader's browser. Training runs in ticks; every
// `evalEvery` games the network, searching `depth` plies, plays fixed opponents at equal depth (`material-N`) and
// with no search (`random`), and the current network is posted so the page can play it. A worker, so the page stays
// responsive.
import init, { SelfPlayTrainer } from "~/wasm/rl/rl.js"
import rlWasmUrl from "~/wasm/rl/rl_bg.wasm?url"

export interface SelfPlayConfig {
  params: string // "name=value,..." (libs/rl/rust/envs/src/selfplay.rs)
  seed: number
  budget: number // games
  evalEvery: number // games
  depth: number // plies the network searches when it's evaluated (and when you play it)
}

export type SelfPlayCommand = { type: "start"; config: SelfPlayConfig } | { type: "pause" } | { type: "resume" } | { type: "stop" }

export type SelfPlayMessage =
  | { type: "progress"; games: number; drawRate: number; meanPlies: number; loss: number; epsilon: number }
  // one evaluation point: points per game (win 1, draw 1/2) against material search at the network's depth, and
  // against a random mover
  | { type: "curve"; games: number; material: number; random: number }
  | { type: "network"; games: number; snapshot: string }
  | { type: "done"; games: number }
  | { type: "error"; message: string }

const TICK_MS = 50
const GAMES_PER_TICK = 40 // ~40 ms of training on a desktop (~1,000 games/s)
const EVAL_GAMES = 20
const EVAL_SEED = 50_000 // never a training game: self-play draws its own randomness

let ready: Promise<void> | null = null
let trainer: SelfPlayTrainer | null = null
let config: SelfPlayConfig | null = null
let games = 0 // played so far
let nextEval = 0
let running = false
let timer: ReturnType<typeof setInterval> | null = null

const post = (message: SelfPlayMessage) => self.postMessage(message)

function evaluate() {
  const material = trainer!.pointsAgainst(`material-${config!.depth}`, config!.depth, EVAL_GAMES, EVAL_SEED)
  const random = trainer!.pointsAgainst("random", 1, EVAL_GAMES, EVAL_SEED)
  post({ type: "curve", games, material, random })
  post({ type: "network", games, snapshot: trainer!.snapshot() })
}

function tick() {
  if (!running || !trainer || !config) return
  try {
    const played = trainer.train(Math.min(GAMES_PER_TICK, nextEval - games, config.budget - games))
    games = played.total_games
    post({
      type: "progress",
      games,
      drawRate: played.games ? played.draws / played.games : 0,
      meanPlies: played.mean_plies,
      loss: played.loss,
      epsilon: played.epsilon,
    })
    played.free()
    if (games >= nextEval) {
      evaluate()
      nextEval += config.evalEvery
    }
    if (games >= config.budget) {
      running = false
      post({ type: "done", games })
    }
  } catch (e) {
    running = false
    post({ type: "error", message: e instanceof Error ? e.message : String(e) })
  }
}

self.onmessage = async (event: MessageEvent<SelfPlayCommand>) => {
  const command = event.data
  try {
    ready ??= init({ module_or_path: rlWasmUrl }).then(() => undefined)
    await ready
    if (command.type === "start") {
      trainer?.free()
      config = command.config
      trainer = new SelfPlayTrainer(config.seed, config.params)
      games = 0
      evaluate() // the untrained network: where the curve starts
      nextEval = config.evalEvery
      running = true
      timer ??= setInterval(tick, TICK_MS)
    } else if (command.type === "pause") {
      running = false
    } else if (command.type === "resume") {
      if (trainer && config && games < config.budget) running = true
    } else if (command.type === "stop") {
      running = false
      if (timer) clearInterval(timer)
      timer = null
    }
  } catch (e) {
    post({ type: "error", message: e instanceof Error ? e.message : String(e) })
  }
}
