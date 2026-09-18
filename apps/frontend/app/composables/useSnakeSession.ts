import type { RenderState } from "~/types/games"

// The worker-session lifecycle/message-handling shared by every way this app plays or watches
// Snake -- extracted from what used to be duplicated between play/[game].vue and
// watch/[runId].vue, and now also the base for LiveSnakeDemo.vue (embedded in Learn/landing
// pages). Built on the shared worker singleton (useSnakeWorker.ts); this composable owns per-
// consumer reactive state and message parsing, not the worker itself.
export function useSnakeSession() {
  const loading = ref(true)
  const error = ref<string | null>(null)
  const renderState = ref<RenderState | null>(null)
  const reward = ref(0)
  const done = ref(false)
  const stepCount = ref(0)

  let worker: Worker | null = null

  function attach(): Worker {
    // A shared worker, not one created (and Pyodide-loaded) fresh per consumer -- see
    // useSnakeWorker.ts. Assigning onmessage here replaces whatever the previously active
    // consumer attached, which is all the "detach" a single-worker,
    // one-visible-consumer-at-a-time app needs.
    worker = getSnakeWorker()
    worker.onmessage = (event: MessageEvent) => {
      const message = event.data
      if (message.type === "ready") {
        loading.value = false
      } else if (message.type === "state") {
        renderState.value = message.renderState
        reward.value = message.reward
        done.value = message.done
        stepCount.value = message.step
      } else if (message.type === "error") {
        error.value = message.message
        loading.value = false
      }
    }
    return worker
  }

  function start(policyJson?: string) {
    loading.value = true
    error.value = null
    const w = attach()
    w.postMessage(policyJson ? { type: "start", policyJson } : { type: "start" })
  }

  function loadPolicy(policyJson: string) {
    worker?.postMessage({ type: "load_policy", policyJson })
  }

  function sendInput(action: number) {
    worker?.postMessage({ type: "input", action })
  }

  function restart() {
    worker?.postMessage({ type: "restart" })
  }

  function stop() {
    // Not terminate() -- the worker is shared and may be reused next. "stop" just pauses its tick
    // loop so it doesn't keep running (or posting messages into the void) while unused.
    worker?.postMessage({ type: "stop" })
    worker = null
  }

  return { loading, error, renderState, reward, done, stepCount, start, loadPolicy, sendInput, restart, stop }
}
