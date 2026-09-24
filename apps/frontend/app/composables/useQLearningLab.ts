import type { LabCommand, LabConfig, LabMessage } from "~/workers/rlLab.worker"
import type { RenderState } from "~/types/games"

// The Q-learning chapter's live lab (docs/design/0010 Phase 1b): owns the training worker (workers/rlLab.worker.ts)
// and turns its messages into reactive state -- progress, the learning curve, the Q-table, the demo game's board.
// The worker is created on first start and ended with the component, never at module load (SSR, and pages that
// don't show the lab never pay for it).

export const DEFAULT_LAB_CONFIG: LabConfig = {
  algorithm: "q_learning",
  alpha: 0.1,
  gamma: 0.95,
  epsilonDecaySteps: 100_000,
  nStep: 1,
  optimistic: false,
  reward: "shaped",
  seed: 0,
  budget: 1_000_000,
}

export const LAB_SPEEDS = [
  { label: "slow", stepsPerSecond: 5_000 },
  { label: "normal", stepsPerSecond: 40_000 },
  { label: "fast", stepsPerSecond: 400_000 },
] as const

const LABELS = ["body", "head", "food"] as const

export function useQLearningLab() {
  const status = ref<"idle" | "running" | "paused" | "done" | "error">("idle")
  const error = ref<string | null>(null)
  const totalSteps = ref(0)
  const totalEpisodes = ref(0)
  const epsilon = ref<number | null>(null)
  const statesVisited = ref(0)
  const recentReturn = ref<number | null>(null)
  // [env steps, mean held-out score] -- shallow: replaced, never mutated in place
  const curve = shallowRef<[number, number][]>([])
  const values = shallowRef<Float64Array | null>(null)
  const visits = shallowRef<Uint32Array | null>(null)
  const board = shallowRef<RenderState | null>(null)
  const row = ref<number | null>(null)
  const action = ref<number | null>(null)
  const speed = ref<number>(LAB_SPEEDS[1].stepsPerSecond)

  let worker: Worker | null = null
  const send = (command: LabCommand) => worker?.postMessage(command)

  function onMessage(event: MessageEvent<LabMessage>) {
    const m = event.data
    if (m.type === "progress") {
      totalSteps.value = m.totalSteps
      totalEpisodes.value = m.totalEpisodes
      epsilon.value = Number.isFinite(m.epsilon) ? m.epsilon : null
      statesVisited.value = Number.isFinite(m.statesVisited) ? m.statesVisited : 0
      recentReturn.value = m.recentReturn
    } else if (m.type === "curve") {
      curve.value = [...curve.value, [m.steps, m.score]]
    } else if (m.type === "table") {
      values.value = m.values
      visits.value = m.visits
    } else if (m.type === "frame") {
      const cells = []
      for (let i = 0; i < m.cells.length; i += 3) cells.push({ x: m.cells[i]!, y: m.cells[i + 1]!, label: LABELS[m.cells[i + 2]!]! })
      board.value = { width: 10, height: 10, cells, score: m.score, alive: !m.done }
      row.value = m.row
      action.value = m.action
    } else if (m.type === "done") {
      totalSteps.value = m.totalSteps
      status.value = "done"
    } else {
      error.value = m.message
      status.value = "error"
    }
  }

  function start(config: LabConfig) {
    if (!worker) {
      worker = new Worker(new URL("~/workers/rlLab.worker.ts", import.meta.url), { type: "module" })
      worker.onmessage = onMessage
      send({ type: "speed", stepsPerSecond: speed.value })
    }
    curve.value = []
    values.value = null
    visits.value = null
    totalSteps.value = 0
    totalEpisodes.value = 0
    recentReturn.value = null
    error.value = null
    status.value = "running"
    send({ type: "start", config: { ...config } })
  }

  function pause() {
    send({ type: "pause" })
    status.value = "paused"
  }

  function resume() {
    send({ type: "resume" })
    status.value = "running"
  }

  watch(speed, (stepsPerSecond) => send({ type: "speed", stepsPerSecond }))

  onUnmounted(() => {
    send({ type: "stop" })
    worker?.terminate()
    worker = null
  })

  return {
    status,
    error,
    totalSteps,
    totalEpisodes,
    epsilon,
    statesVisited,
    recentReturn,
    curve,
    values,
    visits,
    board,
    row,
    action,
    speed,
    start,
    pause,
    resume,
  }
}
