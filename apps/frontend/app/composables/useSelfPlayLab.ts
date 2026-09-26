import type { SelfPlayCommand, SelfPlayConfig, SelfPlayMessage } from "~/workers/selfPlayLab.worker"

// The self-play chapter's live lab (docs/design/0010 Phase 4): owns the training worker (workers/selfPlayLab.worker.ts)
// and turns its messages into reactive state -- progress, the learning curve against fixed opponents, and the latest
// network (as `WeightVector` JSON, what the Checkers stage plays). Created on first start, ended with the component.

export type { SelfPlayConfig }

export interface SelfPlayPoint {
  games: number
  material: number
  random: number
}

export function useSelfPlayLab() {
  const status = ref<"idle" | "running" | "paused" | "done" | "error">("idle")
  const error = ref<string | null>(null)
  const games = ref(0)
  const drawRate = ref<number | null>(null)
  const meanPlies = ref<number | null>(null)
  const loss = ref<number | null>(null)
  const points = shallowRef<SelfPlayPoint[]>([])
  // the latest network: { weights, layer_sizes } and after how many games
  const network = shallowRef<{ games: number; weights: number[]; layerSizes: number[] } | null>(null)

  let worker: Worker | null = null
  const send = (command: SelfPlayCommand) => worker?.postMessage(command)

  function onMessage(event: MessageEvent<SelfPlayMessage>) {
    const m = event.data
    if (m.type === "progress") {
      games.value = m.games
      drawRate.value = m.drawRate
      meanPlies.value = m.meanPlies
      loss.value = m.loss
    } else if (m.type === "curve") {
      points.value = [...points.value, { games: m.games, material: m.material, random: m.random }]
    } else if (m.type === "network") {
      const parsed = JSON.parse(m.snapshot) as { weights: number[]; layer_sizes: number[] }
      network.value = { games: m.games, weights: parsed.weights, layerSizes: parsed.layer_sizes }
    } else if (m.type === "done") {
      games.value = m.games
      status.value = "done"
    } else {
      error.value = m.message
      status.value = "error"
    }
  }

  function start(config: SelfPlayConfig) {
    if (!worker) {
      worker = new Worker(new URL("~/workers/selfPlayLab.worker.ts", import.meta.url), { type: "module" })
      worker.onmessage = onMessage
    }
    points.value = []
    network.value = null
    games.value = 0
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

  onUnmounted(() => {
    send({ type: "stop" })
    worker?.terminate()
    worker = null
  })

  return { status, error, games, drawRate, meanPlies, loss, points, network, start, pause, resume }
}
