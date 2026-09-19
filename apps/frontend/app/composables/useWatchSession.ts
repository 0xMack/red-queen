import type { RunInfo } from "~/types/telemetry"

export interface LoadedPolicy {
  ref: string
  interfaceId: string | null
  generation: number | null
  weights: number[]
  layerSizes: number[]
}

// Watch a trained Snake policy play. Modes 2 (a finished run) and 3 (a still-training run's live
// current-best) turn out to be the same mechanism: reuse useMetricsStreamStore and reload whichever
// champion is latest whenever it changes -- see doc 0005 step 6.
//
// `manageStream: false` is for a page that already owns the metrics stream for the same run (the
// run detail page shows the chart from it too) -- this composable then only reads it, rather than
// restarting/stopping it. `pin(generation)` switches from "follow the latest champion" to a fixed
// generation's champion (every generation's champion is stored as an artifact); `pin(null)` goes
// back to following.
export function useWatchSession(runId: string, options: { manageStream?: boolean } = {}) {
  const manageStream = options.manageStream ?? true
  const session = useSnakeSession()
  const metricsStream = useMetricsStreamStore()
  const config = useRuntimeConfig()

  const run = ref<RunInfo | null>(null)
  const lastLoadedRef = ref<string | null>(null)
  const policy = ref<LoadedPolicy | null>(null)
  const pinnedGeneration = ref<number | null>(null)
  let workerStarted = false
  let active = false

  const targetStats = computed(() => {
    const history = metricsStream.runId === runId ? metricsStream.history : []
    if (pinnedGeneration.value === null) return history.at(-1) ?? null
    return history.find((h) => h.generation === pinnedGeneration.value) ?? null
  })

  async function loadTargetChampionIfNew() {
    // Wait for the run config too: it names the interface the champion must run under, and on the
    // run page (manageStream: false) the metrics history can arrive before it.
    if (!active || !run.value) return
    const target = targetStats.value
    if (!target || target.champion_ref === lastLoadedRef.value) return

    let policyJson: string
    try {
      policyJson = await $fetch<string>(`/runs/${runId}/artifacts/${target.champion_ref}`, {
        baseURL: config.public.apiBase,
        responseType: "text",
      })
    } catch (e) {
      session.error.value = e instanceof Error ? e.message : String(e)
      session.loading.value = false
      return // lastLoadedRef stays unset for this ref, so a retry (or the next SSE tick) tries again
    }
    lastLoadedRef.value = target.champion_ref
    // docs/design/0007: run the champion under the interface its run recorded (backfilled for
    // pre-0007 runs by jobs/backfill_interfaces.py). Absent -> the worker's default.
    const interfaceId = typeof run.value?.config?.interface === "string" ? run.value.config.interface : undefined

    try {
      const parsed = JSON.parse(policyJson) as { weights: number[]; layer_sizes: number[] }
      policy.value = {
        ref: target.champion_ref,
        interfaceId: interfaceId ?? null,
        generation: target.generation,
        weights: parsed.weights,
        layerSizes: parsed.layer_sizes,
      }
    } catch {
      policy.value = null // the worker will report the real error if the artifact is unusable
    }

    session.resetStats()
    if (!workerStarted) {
      workerStarted = true
      session.start({ policyJson, interfaceId })
    } else {
      session.loadPolicy({ policyJson, interfaceId })
    }
  }

  watch(targetStats, loadTargetChampionIfNew)

  async function start() {
    active = true
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

    if (!manageStream) {
      await loadTargetChampionIfNew()
      return
    }

    // metricsStream.start() always resets history to [] before repopulating it, so the watch()
    // above fires again on retry even if the champion it lands on ends up being the same one.
    await metricsStream.start(runId)
  }

  function pin(generation: number | null) {
    pinnedGeneration.value = generation
  }

  function retry() {
    workerStarted = false
    lastLoadedRef.value = null
    start()
  }

  onUnmounted(() => {
    active = false
    if (manageStream) metricsStream.stop()
    session.stop()
  })

  return { ...session, run, lastLoadedRef, policy, pinnedGeneration, targetStats, start, retry, pin }
}
