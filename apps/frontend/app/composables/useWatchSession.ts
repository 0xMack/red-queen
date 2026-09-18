import type { RunInfo } from "~/types/telemetry"

// The /watch/[runId].vue page's session, also usable by LiveSnakeDemo.vue in "watch" mode. Modes
// 2 (a finished run) and 3 (a still-training run's live current-best) turn out to be the same
// mechanism: reuse useMetricsStreamStore and reload whichever champion is latest whenever it
// changes -- see doc 0005 step 6.
export function useWatchSession(runId: string) {
  const session = useSnakeSession()
  const metricsStream = useMetricsStreamStore()
  const config = useRuntimeConfig()

  const run = ref<RunInfo | null>(null)
  const lastLoadedRef = ref<string | null>(null)
  let workerStarted = false

  async function loadLatestChampionIfNew() {
    const latest = metricsStream.history.at(-1)
    if (!latest || latest.champion_ref === lastLoadedRef.value) return

    let policyJson: string
    try {
      policyJson = await $fetch<string>(`/runs/${runId}/artifacts/${latest.champion_ref}`, {
        baseURL: config.public.apiBase,
        responseType: "text",
      })
    } catch (e) {
      session.error.value = e instanceof Error ? e.message : String(e)
      return // lastLoadedRef stays unset for this ref, so a retry (or the next SSE tick) tries again
    }
    lastLoadedRef.value = latest.champion_ref

    if (!workerStarted) {
      workerStarted = true
      session.start(policyJson)
    } else {
      session.loadPolicy(policyJson)
    }
  }

  watch(() => metricsStream.history.length, loadLatestChampionIfNew)

  async function start() {
    session.loading.value = true
    session.error.value = null

    try {
      run.value = await $fetch<RunInfo>(`/runs/${runId}`, { baseURL: config.public.apiBase })
    } catch (e) {
      session.error.value = e instanceof Error ? e.message : String(e)
      session.loading.value = false
      return
    }

    if (run.value.config?.game !== "snake") {
      session.error.value = `run ${runId} isn't a watchable game run (config.game = ${String(run.value.config?.game ?? "unset")})`
      session.loading.value = false
      return
    }

    // metricsStream.start() always resets history to [] before repopulating it, so the watch()
    // above fires again on retry even if the champion it lands on ends up being the same one.
    await metricsStream.start(runId)
  }

  function retry() {
    workerStarted = false
    lastLoadedRef.value = null
    start()
  }

  onUnmounted(() => {
    metricsStream.stop()
    session.stop()
  })

  return { ...session, run, lastLoadedRef, start, retry }
}
