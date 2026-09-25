// Live reinforcement learning for the Learn chapters (docs/design/0010 Phases 1b, 2b, 3b): the RL core compiled to
// WebAssembly -- the same Rust `Trainer` the training jobs run -- learning in the reader's browser, for any algorithm
// (a Q-table, a DQN, a policy gradient) and environment (Snake under any observer, Reach1D) it knows. Training runs in
// ticks sized to the chosen speed; every `evalEvery` steps the greedy policy is scored on a fixed set of unseen games
// (the learning curve); a tabular agent's table is streamed to the page; and a demo plays the *current* greedy policy
// at a watchable pace -- a Snake board with the values or probabilities it's choosing between, or (any other
// environment) the observation and action, step by step. A worker, so none of this ever blocks the page.
import init, { DemoEnv, DemoGame, Trainer } from "~/wasm/rl/rl.js"
import rlWasmUrl from "~/wasm/rl/rl_bg.wasm?url"

export interface LabConfig {
  algorithm: string // q_learning, sarsa, dqn, reinforce, a2c, ppo (libs/rl/rust/core/src/agent.rs ALGORITHMS)
  env: string // a Snake interface id (snake/<observer>+relative3.v1), or reach1d
  params: string // the algorithm's parameters as "name=value,..." ("" for the defaults)
  reward: "shaped" | "sparse"
  seed: number
  budget: number // stop after this many environment steps
  evalEvery: number
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
  // one evaluation point: the held-out score, (a DQN) its mean Q(s, a) and loss since the last point, and the
  // policy's entropy (nats) now
  | { type: "curve"; steps: number; score: number; qMean: number | null; tdLoss: number | null; entropy: number | null }
  | { type: "table"; values: Float64Array; visits: Uint32Array }
  | { type: "frame"; cells: Int32Array; score: number; row: number; action: number; values: number[]; done: boolean }
  // a non-Snake environment's demo step: what it observed, the action taken, the policy's values or distribution
  | { type: "track"; step: number; observation: number[]; action: number; values: number[]; done: boolean }
  | { type: "done"; totalSteps: number }
  | { type: "error"; message: string }

// Games the learning curve is scored on: unseen in training (the monitor range jobs use), fixed, so the curve moves
// only when the policy does.
const EVAL_SEEDS = new Uint32Array(Array.from({ length: 30 }, (_, i) => 20_000 + i))
const EVAL_CAP = 1000
const TRACK_STEPS = 200 // a non-Snake demo episode (Reach1D never ends by itself)
const TICK_MS = 40
const FRAME_MS = 90
const TABLE_MS = 250

let ready: Promise<void> | null = null
let trainer: Trainer | null = null
let config: LabConfig | null = null
let demo: DemoGame | null = null
let track: DemoEnv | null = null
let trackStep = 0
let demoSeed = 20_100
let running = false
let stepsPerSecond = 40_000
let nextEval = 0
let returns: number[] = []
// a DQN's per-call mean Q and loss, averaged (weighted by updates) between evaluation points
let qSum = 0
let lossSum = 0
let lastUpdates = 0
let lastEntropy = Number.NaN
let updatesSinceEval = 0
let timers: ReturnType<typeof setInterval>[] = []

const post = (message: LabMessage, transfer: Transferable[] = []) => self.postMessage(message, { transfer })
const finite = (x: number) => (Number.isFinite(x) ? x : null)

function evaluate(steps: number) {
  const scores = trainer!.evaluate(EVAL_SEEDS, snakeObserver(config!.env) ? EVAL_CAP : TRACK_STEPS)
  const n = updatesSinceEval
  post({
    type: "curve",
    steps,
    score: scores.reduce((a, b) => a + b, 0) / scores.length,
    qMean: n > 0 ? qSum / n : null,
    tdLoss: n > 0 ? lossSum / n : null,
    entropy: finite(lastEntropy),
  })
  qSum = lossSum = updatesSinceEval = 0
}

function tick() {
  if (!running || !trainer || !config) return
  try {
    const want = Math.max(1, Math.round((stepsPerSecond * TICK_MS) / 1000))
    let done = 0
    let last: { totalEpisodes: number; epsilon: number; statesVisited: number } | null = null
    while (running && done < want) {
      // Train up to the next evaluation point (or the budget), then score the greedy policy there.
      const chunk = Math.min(want - done, nextEval - trainer.totalSteps, config.budget - trainer.totalSteps)
      const p = trainer.train(chunk)
      if (p.episodes > 0) returns.push(p.mean_return)
      if (Number.isFinite(p.updates)) {
        const updates = p.updates - lastUpdates
        lastUpdates = p.updates
        // a diverging network's Q can overflow to Infinity: keep it (the chart shows it leaving), but not NaN
        if (updates > 0 && !Number.isNaN(p.q_mean)) {
          qSum += p.q_mean * updates
          lossSum += p.td_loss * updates
          updatesSinceEval += updates
        }
      }
      last = { totalEpisodes: p.total_episodes, epsilon: p.epsilon, statesVisited: p.states_visited }
      lastEntropy = p.entropy
      p.free()
      done += chunk
      if (trainer.totalSteps >= nextEval) {
        evaluate(trainer.totalSteps)
        nextEval += config.evalEvery
      }
      if (trainer.totalSteps >= config.budget) {
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
  if (values.length === 0) return // not a tabular agent
  const visits = trainer.visits()
  post({ type: "table", values, visits }, [values.buffer, visits.buffer])
}

const snakeObserver = (env: string) => (env.startsWith("snake/") ? env.slice("snake/".length).split("+")[0]! : null)

function playTrack() {
  if (!trainer || !config) return
  if (!track || track.done || trackStep >= TRACK_STEPS) {
    track?.free()
    track = new DemoEnv(config.env, demoSeed++)
    trackStep = 0
  }
  const observation = track.observation()
  const values = Array.from(trainer.actionValues(observation))
  const action = trainer.greedyValue(observation)
  track.step(action)
  trackStep += 1
  post({ type: "track", step: trackStep, observation: Array.from(observation), action, values, done: track.done || trackStep >= TRACK_STEPS })
}

function playFrame() {
  if (!trainer || !config) return
  const observer = snakeObserver(config.env)
  if (!observer) return playTrack()
  if (!demo || demo.done) {
    demo?.free()
    demo = new DemoGame(demoSeed++, observer)
  }
  const observation = demo.observation()
  const row = trainer.row(observation)
  const values = Array.from(trainer.actionValues(observation))
  const action = trainer.greedyAction(observation)
  demo.step(action)
  const cells = new Int32Array(demo.cells())
  post({ type: "frame", cells, score: demo.score, row, action, values, done: demo.done }, [cells.buffer])
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
      track?.free()
      track = null
      config = command.config
      trainer = new Trainer(config.algorithm, config.env, config.seed, config.params, config.reward)
      qSum = lossSum = lastUpdates = updatesSinceEval = 0
      evaluate(0) // the untrained policy: where the curve starts
      nextEval = config.evalEvery
      returns = []
      demoSeed = 20_100
      running = true
      timers = [setInterval(tick, TICK_MS), setInterval(playFrame, FRAME_MS), setInterval(sendTable, TABLE_MS)]
      sendTable()
    } else if (command.type === "pause") {
      running = false
    } else if (command.type === "resume") {
      if (trainer && config && trainer.totalSteps < config.budget) running = true
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
