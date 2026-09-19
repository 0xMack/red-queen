import type { LoadProgress } from "~/inference/runtime"
import type { RenderState } from "~/types/games"
import type { ModelSpec } from "~/types/modelpack"

// The worker-session lifecycle/message-handling shared by every way this app plays or watches
// Snake -- extracted from what used to be duplicated between play/[game].vue and
// watch/[runId].vue, and now also the base for LiveSnakeDemo.vue and WatchChampion.vue (embedded in
// the landing page, Learn chapters, and the run detail page). Built on the shared worker singleton
// (useSnakeWorker.ts); this composable owns per-consumer reactive state and message parsing, not
// the worker itself.
export const DEFAULT_TICK_MS = 110

// What drives the snake in watch mode (mirrors the worker's PolicySpec): a model package run by ONNX
// Runtime (docs/design/0009) or a games.baselines name -- plus the interface (docs/design/0007) it
// plays under. No policy at all = play mode, a human steers.
export interface PolicySpec {
  baseline?: string
  model?: ModelSpec
  interfaceId?: string
}

export interface ModelStatus {
  backend: string
  variantId: string
  loadMs: number
  selfTest: { samples: number; maxAbsError: number; tolerance: number } | null
}

export interface ModelFailure {
  message: string
  packageId: string
  variantId: string
  backend: string
}

export function useSnakeSession() {
  const loading = ref(true)
  const error = ref<string | null>(null)
  const renderState = ref<RenderState | null>(null)
  const reward = ref(0)
  const done = ref(false)
  const stepCount = ref(0)
  // Watch mode only: what the policy sees right now (see the worker's OutboundMessage docs).
  const observation = ref<number[] | null>(null)

  // Per-session episode stats -- the worker auto-replays in watch mode, so these accumulate across
  // episodes until resetStats() (e.g. a different champion is loaded).
  const episodes = ref(0)
  const episodeScores = ref<number[]>([])
  const bestScore = computed(() => (episodeScores.value.length ? Math.max(...episodeScores.value) : 0))
  const meanScore = computed(() =>
    episodeScores.value.length ? episodeScores.value.reduce((a, b) => a + b, 0) / episodeScores.value.length : 0,
  )
  const tickMs = ref(DEFAULT_TICK_MS)
  // Model-package mode only.
  const modelProgress = ref<LoadProgress | null>(null)
  const modelStatus = ref<ModelStatus | null>(null)
  let onModelError: ((failure: ModelFailure) => void) | null = null

  let worker: Worker | null = null

  function attach(): Worker {
    // A shared worker, not one created (and its runtimes loaded) fresh per consumer -- see
    // useSnakeWorker.ts. Assigning onmessage here replaces whatever the previously active
    // consumer attached, which is all the "detach" a single-worker,
    // one-visible-consumer-at-a-time app needs.
    worker = getSnakeWorker()
    worker.onmessage = (event: MessageEvent) => {
      const message = event.data
      if (message.type === "ready") {
        loading.value = false
      } else if (message.type === "state") {
        if (message.done && !done.value && message.step > 0) {
          episodes.value += 1
          episodeScores.value = [...episodeScores.value.slice(-49), message.renderState.score]
        }
        renderState.value = message.renderState
        reward.value = message.reward
        done.value = message.done
        stepCount.value = message.step
        observation.value = message.observation ?? null
      } else if (message.type === "model_progress") {
        modelProgress.value = message.progress
      } else if (message.type === "model_loaded") {
        modelProgress.value = null
        modelStatus.value = { backend: message.backend, variantId: message.variantId, loadMs: message.loadMs, selfTest: message.selfTest }
      } else if (message.type === "model_error") {
        modelProgress.value = null
        modelStatus.value = null
        if (onModelError) onModelError(message)
        else {
          error.value = message.message
          loading.value = false
        }
      } else if (message.type === "error") {
        error.value = message.message
        loading.value = false
      }
    }
    return worker
  }

  function start(spec: PolicySpec = {}) {
    loading.value = true
    error.value = null
    const w = attach()
    // The worker is shared across pages -- re-assert this session's speed, don't inherit the last one.
    w.postMessage({ type: "set_speed", intervalMs: tickMs.value })
    w.postMessage({ type: "start", ...spec })
  }

  // Load the game runtime without starting a game; `loading` flips false when it's ready.
  function warmup() {
    loading.value = true
    error.value = null
    attach().postMessage({ type: "warmup" })
  }

  function loadPolicy(spec: PolicySpec) {
    worker?.postMessage({ type: "load_policy", ...spec })
  }

  function sendInput(action: number) {
    worker?.postMessage({ type: "input", action })
  }

  function restart() {
    worker?.postMessage({ type: "restart" })
  }

  function setSpeed(intervalMs: number) {
    tickMs.value = intervalMs
    worker?.postMessage({ type: "set_speed", intervalMs })
  }

  function handleModelErrors(handler: (failure: ModelFailure) => void) {
    onModelError = handler
  }

  function resetStats() {
    episodes.value = 0
    episodeScores.value = []
  }

  function stop() {
    // Not terminate() -- the worker is shared and may be reused next. "stop" just pauses its tick
    // loop so it doesn't keep running (or posting messages into the void) while unused.
    worker?.postMessage({ type: "stop" })
    worker = null
  }

  return {
    loading,
    error,
    renderState,
    reward,
    done,
    stepCount,
    observation,
    episodes,
    episodeScores,
    bestScore,
    meanScore,
    tickMs,
    modelProgress,
    modelStatus,
    handleModelErrors,
    start,
    warmup,
    loadPolicy,
    sendInput,
    restart,
    setSpeed,
    resetStats,
    stop,
  }
}
