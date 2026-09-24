// Live tabular Q-learning for the Learn chapter (docs/design/0010 Phase 1b): the RL core compiled to WebAssembly --
// the same Rust `Trainer` the training jobs run -- learning Snake in the reader's browser. Training runs in ticks
// sized to the chosen speed; every EVAL_EVERY steps the greedy policy is scored on a fixed set of unseen games (the
// learning curve); the Q-table is streamed to the page; and a demo game plays the *current* greedy policy at a
// watchable pace, reporting which table row it's in. A worker, so none of this ever blocks the page.
import init, { DemoGame, Trainer } from "~/wasm/rl/rl.js"
import rlWasmUrl from "~/wasm/rl/rl_bg.wasm?url"

export interface LabConfig {
  algorithm: "q_learning" | "sarsa"
  alpha: number
  gamma: number
  epsilonDecaySteps: number
  nStep: number
  optimistic: boolean // initial Q 2 and epsilon 0.02: optimism does the exploring
  reward: "shaped" | "sparse"
  seed: number
  budget: number // stop after this many environment steps
}

export type LabCommand =
  | { type: "start"; config: LabConfig }
  | { type: "pause" }
  | { type: "resume" }
  | { type: "speed"; stepsPerSecond: number }
  | { type: "stop" }

export type LabMessage =
  | {
      type: "progress"
      totalSteps: number
      totalEpisodes: number
      epsilon: number
      statesVisited: number
      recentReturn: number | null
      running: boolean
    }
  | { type: "curve"; steps: number; score: number }
  | { type: "table"; values: Float64Array; visits: Uint32Array }
  | { type: "frame"; cells: Int32Array; score: number; row: number; action: number; done: boolean }
  | { type: "done"; totalSteps: number }
  | { type: "error"; message: string }

const OBSERVER = "features.v1"
const ENV = `snake/${OBSERVER}+relative3.v1`
const EVAL_EVERY = 25_000
// Games the learning curve is scored on: unseen in training (the monitor range jobs use), fixed, so the curve moves
// only when the policy does.
const EVAL_SEEDS = new Uint32Array(Array.from({ length: 30 }, (_, i) => 20_000 + i))
const EVAL_CAP = 1000
const TICK_MS = 40
const FRAME_MS = 90
const TABLE_MS = 250

let ready: Promise<void> | null = null
let trainer: Trainer | null = null
let demo: DemoGame | null = null
let demoSeed = 20_100
let running = false
let stepsPerSecond = 40_000
let budget = 1_000_000
let nextEval = 0
let returns: number[] = []
let timers: ReturnType<typeof setInterval>[] = []

const post = (message: LabMessage, transfer: Transferable[] = []) => self.postMessage(message, { transfer })

function params(config: LabConfig): string {
  const pairs: [string, number][] = [
    ["alpha", config.alpha],
    ["gamma", config.gamma],
    ["epsilon_decay_steps", config.epsilonDecaySteps],
    ["n_step", config.nStep],
  ]
  if (config.optimistic) pairs.push(["initial_q", 2], ["epsilon_start", 0.02], ["epsilon_end", 0.02])
  return pairs.map(([k, v]) => `${k}=${v}`).join(",")
}

function evaluate(steps: number) {
  const scores = trainer!.evaluate(EVAL_SEEDS, EVAL_CAP)
  post({ type: "curve", steps, score: scores.reduce((a, b) => a + b, 0) / scores.length })
}

function tick() {
  if (!running || !trainer) return
  try {
    const want = Math.max(1, Math.round((stepsPerSecond * TICK_MS) / 1000))
    let done = 0
    let last: { totalEpisodes: number; epsilon: number; statesVisited: number } | null = null
    while (running && done < want) {
      // Train up to the next evaluation point (or the budget), then score the greedy policy there.
      const chunk = Math.min(want - done, nextEval - trainer.totalSteps, budget - trainer.totalSteps)
      const p = trainer.train(chunk)
      if (p.episodes > 0) returns.push(p.mean_return)
      last = { totalEpisodes: p.total_episodes, epsilon: p.epsilon, statesVisited: p.states_visited }
      p.free()
      done += chunk
      if (trainer.totalSteps >= nextEval) {
        evaluate(trainer.totalSteps)
        nextEval += EVAL_EVERY
      }
      if (trainer.totalSteps >= budget) {
        running = false
        post({ type: "done", totalSteps: trainer.totalSteps })
      }
    }
    returns = returns.slice(-50)
    if (last) {
      post({
        type: "progress",
        totalSteps: trainer.totalSteps,
        ...last,
        recentReturn: returns.length ? returns.reduce((a, b) => a + b, 0) / returns.length : null,
        running,
      })
    }
  } catch (e) {
    running = false
    post({ type: "error", message: e instanceof Error ? e.message : String(e) })
  }
}

function sendTable() {
  if (!trainer) return
  const values = trainer.qValues()
  const visits = trainer.visits()
  post({ type: "table", values, visits }, [values.buffer, visits.buffer])
}

function playFrame() {
  if (!trainer) return
  if (!demo || demo.done) {
    demo?.free()
    demo = new DemoGame(demoSeed++, OBSERVER)
  }
  const observation = demo.observation()
  const row = trainer.row(observation)
  const action = trainer.greedyAction(observation)
  demo.step(action)
  const cells = new Int32Array(demo.cells())
  post({ type: "frame", cells, score: demo.score, row, action, done: demo.done }, [cells.buffer])
}

function stopTimers() {
  timers.forEach(clearInterval)
  timers = []
}

self.onmessage = async (event: MessageEvent<LabCommand>) => {
  const command = event.data
  try {
    ready ??= init({ module_or_path: rlWasmUrl }).then(() => undefined)
    await ready
    if (command.type === "start") {
      stopTimers()
      trainer?.free()
      demo?.free()
      demo = null
      const { config } = command
      trainer = new Trainer(config.algorithm, ENV, config.seed, params(config), config.reward)
      budget = config.budget
      evaluate(0) // the untrained policy: where the curve starts
      nextEval = EVAL_EVERY
      returns = []
      demoSeed = 20_100
      running = true
      timers = [setInterval(tick, TICK_MS), setInterval(playFrame, FRAME_MS), setInterval(sendTable, TABLE_MS)]
      sendTable()
    } else if (command.type === "pause") {
      running = false
    } else if (command.type === "resume") {
      if (trainer && trainer.totalSteps < budget) running = true
    } else if (command.type === "speed") {
      stepsPerSecond = command.stepsPerSecond
    } else if (command.type === "stop") {
      running = false
      stopTimers()
    }
  } catch (e) {
    post({ type: "error", message: e instanceof Error ? e.message : String(e) })
  }
}
