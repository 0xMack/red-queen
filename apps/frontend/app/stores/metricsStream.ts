import { defineStore } from "pinia"
import type { GenerationStats } from "~/types/telemetry"

// Backfill-then-live, matching apis/backend's /runs/{id}/metrics/{history,stream} split (doc 0005):
// fetch what already happened, then open one EventSource for what happens next. A page navigating
// to a different run calls start() again, which tears down any previous subscription first -- only
// one run is ever watched at a time in this skeleton.
export const useMetricsStreamStore = defineStore("metricsStream", () => {
  const runId = ref<string | null>(null)
  const history = ref<GenerationStats[]>([])
  const connected = ref(false)
  const error = ref<string | null>(null)

  let source: EventSource | null = null

  function stop() {
    source?.close()
    source = null
    connected.value = false
  }

  async function start(id: string) {
    stop()
    runId.value = id
    history.value = []
    error.value = null

    const config = useRuntimeConfig()
    try {
      history.value = await $fetch<GenerationStats[]>(`/runs/${id}/metrics/history`, {
        baseURL: config.public.apiBase,
      })
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      return
    }

    const lastSeen = history.value.at(-1)
    const sinceGeneration = lastSeen ? lastSeen.generation + 1 : 0
    const url = `${config.public.apiBase}/runs/${id}/metrics/stream?since_generation=${sinceGeneration}`

    source = new EventSource(url)
    // Event name matches apis/backend/src/backend/routers/runs.py's
    // yield {"event": "generation", "data": ...} -- not the default "message" event.
    source.addEventListener("generation", (event) => {
      history.value.push(JSON.parse((event as MessageEvent).data) as GenerationStats)
    })
    source.onopen = () => {
      connected.value = true
    }
    source.onerror = () => {
      connected.value = false
    }
  }

  return { runId, history, connected, error, start, stop }
})
