import type { Availability } from "~/composables/useModelCatalog"
import type { Match } from "~/inference/match"
import type { ModelFailure } from "~/composables/useSnakeSession"
import type { ModelManifest, ModelSpec } from "~/types/modelpack"
import type { RunInfo } from "~/types/telemetry"
import { genomeFromJson, type Genome, type GenomeJson } from "~/utils/neat"

// A champion is either a fixed-topology network (`weights` + `layerSizes`, evolve.neuro.WeightVector) or an
// evolved graph (`genome`, evolve.neat.NeatGenome) -- `kind` says which; the unused half is empty/null.
export interface LoadedPolicy {
  ref: string
  interfaceId: string | null
  generation: number | null
  kind: "weights" | "neat"
  weights: number[]
  layerSizes: number[]
  genome: Genome | null
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
//
// `packaged` (docs/design/0009): when this run's champion is published as a model package, the page
// passes its availability here and the champion runs through ONNX Runtime instead of the Pyodide
// forward pass -- unless this device can't run it, in which case `unsupported` says why and nothing
// plays. The JSON artifact is still fetched, but only to *draw* the network (and for generations
// that were never published: pinned ones, or a still-training run's latest).
export interface WatchOptions {
  manageStream?: boolean
  // undefined = the catalog is still loading (wait, don't start the Python path and switch later);
  // null = this champion isn't published.
  packaged?: () => Availability | null | undefined
  onModelFailure?: (failure: ModelFailure) => void
}

export function useWatchSession(runId: string, options: WatchOptions = {}) {
  const manageStream = options.manageStream ?? true
  const session = useSnakeSession()
  const metricsStream = useMetricsStreamStore()
  const config = useRuntimeConfig()

  const run = ref<RunInfo | null>(null)
  const lastLoadedRef = ref<string | null>(null)
  const policy = ref<LoadedPolicy | null>(null)
  const pinnedGeneration = ref<number | null>(null)
  // Package mode: why this device can't run it (nothing plays), or a big download awaiting a click.
  const unsupported = ref<Extract<Match, { ok: false }> | null>(null)
  const awaitingConfirmation = ref<number | null>(null) // bytes
  const downloadConfirmed = ref(false)
  const runtime = ref<"onnxruntime" | "python" | null>(null)
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

    const packaged = options.packaged ? options.packaged() : null
    if (packaged === undefined) return // catalog still loading; the watch on `packaged` retries
    const usePackage = packaged !== null && target.champion_ref === packaged.entry.champion_ref
    unsupported.value = null
    awaitingConfirmation.value = null
    let model: ModelSpec | undefined
    if (usePackage) {
      const match = packaged.match
      if (!match || !packaged.manifest) return // still probing; the watch on `packaged` retries
      if (!match.ok) {
        unsupported.value = match
        session.loading.value = false
        // Nothing may keep playing behind the explanation (a previous variant, or the Python path).
        if (workerStarted) {
          session.stop()
          workerStarted = false
          runtime.value = null
        }
        return
      }
      if (match.needsConfirmation && !downloadConfirmed.value) {
        awaitingConfirmation.value = match.variant.requirements.download_bytes
        session.loading.value = false
        return
      }
      // A plain copy: the manifest lives in reactive state, and a Vue proxy can't cross postMessage.
      const manifest = JSON.parse(JSON.stringify(packaged.manifest)) as ModelManifest
      model = { baseUrl: packaged.baseUrl, manifest, variantId: match.variant.id, backend: match.backend }
    }

    let policyJson = ""
    try {
      policyJson = await $fetch<string>(`/runs/${runId}/artifacts/${target.champion_ref}`, {
        baseURL: config.public.apiBase,
        responseType: "text",
      })
    } catch (e) {
      // Fatal only when the artifact is what plays; a package plays without it (no diagram, then).
      if (!model) {
        session.error.value = e instanceof Error ? e.message : String(e)
        session.loading.value = false
        return // lastLoadedRef stays unset for this ref, so a retry (or the next SSE tick) tries again
      }
    }
    lastLoadedRef.value = target.champion_ref
    // docs/design/0007: run the champion under the interface its run recorded (backfilled for
    // pre-0007 runs by jobs/backfill_interfaces.py). Absent -> the worker's default.
    const interfaceId = typeof run.value?.config?.interface === "string" ? run.value.config.interface : undefined

    try {
      const parsed = JSON.parse(policyJson) as { type?: string; weights?: number[]; layer_sizes?: number[] }
      const base = { ref: target.champion_ref, interfaceId: interfaceId ?? null, generation: target.generation }
      policy.value =
        parsed.type === "neat"
          ? { ...base, kind: "neat", weights: [], layerSizes: [], genome: genomeFromJson(parsed as unknown as GenomeJson) }
          : { ...base, kind: "weights", weights: parsed.weights!, layerSizes: parsed.layer_sizes!, genome: null }
    } catch {
      policy.value = null // the worker will report the real error if the artifact is unusable
    }

    session.resetStats()
    runtime.value = model ? "onnxruntime" : "python"
    const spec = model ? { model, interfaceId } : { policyJson, interfaceId }
    if (!workerStarted) {
      workerStarted = true
      session.start(spec)
    } else {
      session.loadPolicy(spec)
    }
  }

  watch(targetStats, loadTargetChampionIfNew)
  // A package's device match can change after the first attempt: manifests/probe arrive, or a load
  // fails here and is remembered, so the next variant/backend (or an explanation) takes over.
  // Keyed by the decision, not the object: availability is recomputed wholesale, and an unchanged
  // choice must not restart the game.
  watch(
    () => {
      const packaged = options.packaged?.()
      if (packaged === undefined) return options.packaged ? "loading" : "none"
      if (packaged === null) return "none"
      const match = packaged.match
      if (!match) return "pending"
      return match.ok ? `${match.variant.id}/${match.backend}` : `no:${match.summary}`
    },
    () => {
      if (!active) return
      lastLoadedRef.value = null
      loadTargetChampionIfNew()
    },
  )
  session.handleModelErrors((failure) => {
    lastLoadedRef.value = null
    options.onModelFailure?.(failure)
  })

  function confirmDownload() {
    downloadConfirmed.value = true
    lastLoadedRef.value = null
    loadTargetChampionIfNew()
  }

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

  return {
    ...session,
    run,
    lastLoadedRef,
    policy,
    pinnedGeneration,
    targetStats,
    unsupported,
    awaitingConfirmation,
    runtime,
    start,
    retry,
    pin,
    confirmDownload,
  }
}
