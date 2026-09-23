import type { Availability } from "~/composables/useModelCatalog"
import type { ModelFailure } from "~/composables/useSnakeSession"
import { probeDevice } from "~/inference/device"
import { matchVariant, rememberFailure, rememberedFailure, type Match } from "~/inference/match"
import type { Backend, CatalogEntry, ModelManifest, ModelSpec } from "~/types/modelpack"
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
// Every champion plays as a model package in ONNX Runtime (docs/design/0009): `packaged` is the
// published one when the page has it (the game page's catalog); anything else -- a still-training
// run's latest champion, a pinned generation -- is exported on demand by the backend. If this device
// can't run it, `unsupported` says why and nothing plays. The JSON artifact is still fetched, but
// only to *draw* the network.
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
  const api = useApi()

  const run = ref<RunInfo | null>(null)
  const lastLoadedRef = ref<string | null>(null)
  const policy = ref<LoadedPolicy | null>(null)
  const pinnedGeneration = ref<number | null>(null)
  // Package mode: why this device can't run it (nothing plays), or a big download awaiting a click.
  const unsupported = ref<Extract<Match, { ok: false }> | null>(null)
  // The backend can't package this champion at all (a 422 from the on-demand export: a kind of champion modelpack
  // doesn't know, e.g. an RL agent before docs/design/0010 Phase 1). Its reason, shown instead of an error with a
  // Retry that could never succeed.
  const unexportable = ref<string | null>(null)
  const awaitingConfirmation = ref<number | null>(null) // bytes
  const downloadConfirmed = ref(false)
  // Where the package that's playing came from: the game's published catalog, or exported on demand.
  const source = ref<"published" | "on-demand" | null>(null)
  const currentPackage = ref<Availability | null>(null)
  let workerStarted = false
  let active = false

  const targetStats = computed(() => {
    const history = metricsStream.runId === runId ? metricsStream.history : []
    if (pinnedGeneration.value === null) return history.at(-1) ?? null
    return history.find((h) => h.generation === pinnedGeneration.value) ?? null
  })

  // A champion that was never published (a still-training run's latest, a pinned generation) is
  // exported on demand by the backend -- the same package format, so it runs the same way.
  async function onDemand(ref: string): Promise<Availability> {
    const exported = await api.fetch<{ base_url: string; package_id: string; variants: string[] }>(
      `/runs/${runId}/artifacts/${ref}/package`,
      { method: "POST" },
    )
    const [manifest, profile] = await Promise.all([
      $fetch<ModelManifest>(`${exported.base_url}/manifests/${exported.package_id}.json`),
      probeDevice(),
    ])
    const entry: CatalogEntry = {
      entrant_id: `run:${runId}`,
      package_id: exported.package_id,
      label: manifest.label,
      interface: null,
      run_id: runId,
      champion_ref: ref,
      variants: exported.variants,
      download_bytes: {},
    }
    return {
      entry,
      manifest,
      baseUrl: exported.base_url,
      match: matchVariant(manifest, exported.variants, profile, rememberedFailure(exported.package_id)),
    }
  }

  async function loadTargetChampionIfNew() {
    // Wait for the run config too: it names the interface the champion must run under, and on the
    // run page (manageStream: false) the metrics history can arrive before it.
    if (!active || !run.value) return
    const target = targetStats.value
    if (!target || target.champion_ref === lastLoadedRef.value) return

    const packaged = options.packaged ? options.packaged() : null
    if (packaged === undefined) return // catalog still loading; the watch on `packaged` retries
    unsupported.value = null
    unexportable.value = null
    awaitingConfirmation.value = null

    let available: Availability
    if (packaged !== null && target.champion_ref === packaged.entry.champion_ref) {
      if (!packaged.match || !packaged.manifest) return // still probing; the watch on `packaged` retries
      available = packaged
      source.value = "published"
    } else {
      try {
        available = await onDemand(target.champion_ref)
      } catch (e) {
        session.loading.value = false
        const failure = e as { status?: number; data?: { detail?: unknown } }
        if (failure.status === 422) {
          // Permanent for this champion: say why, and don't ask again on every SSE tick.
          const detail = typeof failure.data?.detail === "string" ? failure.data.detail : null
          unexportable.value = detail ?? "The backend can't package this champion to play in a browser."
          lastLoadedRef.value = target.champion_ref
          return
        }
        session.error.value = e instanceof Error ? e.message : String(e)
        return // lastLoadedRef stays unset for this ref, so a retry (or the next SSE tick) tries again
      }
      source.value = "on-demand"
    }
    currentPackage.value = available

    const match = available.match!
    if (!match.ok) {
      unsupported.value = match
      session.loading.value = false
      // Nothing may keep playing behind the explanation (a previous variant or champion).
      if (workerStarted) {
        session.stop()
        workerStarted = false
      }
      return
    }
    if (match.needsConfirmation && !downloadConfirmed.value) {
      awaitingConfirmation.value = match.variant.requirements.download_bytes
      session.loading.value = false
      return
    }
    // A plain copy: the manifest may live in reactive state, and a Vue proxy can't cross postMessage.
    const manifest = JSON.parse(JSON.stringify(available.manifest)) as ModelManifest
    const model: ModelSpec = { baseUrl: available.baseUrl, manifest, variantId: match.variant.id, backend: match.backend }

    lastLoadedRef.value = target.champion_ref
    // docs/design/0007: run the champion under the interface its run recorded (backfilled for
    // pre-0007 runs by jobs/backfill_interfaces.py). Absent -> the worker's default.
    const interfaceId = typeof run.value?.config?.interface === "string" ? run.value.config.interface : undefined

    // The trained network itself, only to *draw* it (its live activations): the package is what plays.
    api.fetch<string>(`/runs/${runId}/artifacts/${target.champion_ref}`, { responseType: "text" })
      .then((text) => {
        const parsed = JSON.parse(text) as { type?: string; weights?: number[]; layer_sizes?: number[] }
        const base = { ref: target.champion_ref, interfaceId: interfaceId ?? null, generation: target.generation }
        policy.value =
          parsed.type === "neat"
            ? { ...base, kind: "neat", weights: [], layerSizes: [], genome: genomeFromJson(parsed as unknown as GenomeJson) }
            : { ...base, kind: "weights", weights: parsed.weights!, layerSizes: parsed.layer_sizes!, genome: null }
      })
      .catch(() => {
        policy.value = null // no diagram; the game plays regardless
      })

    session.resetStats()
    const spec = { model, interfaceId }
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
  // A variant that fails to load or self-test here is remembered for this device, and the next
  // candidate (or an explanation) takes over: the page's catalog re-matches (published champions),
  // or this session does (on-demand ones).
  session.handleModelErrors((failure) => {
    rememberFailure(failure.packageId, failure.variantId, failure.backend as Backend, failure.message)
    lastLoadedRef.value = null
    options.onModelFailure?.(failure)
    if (source.value === "on-demand") loadTargetChampionIfNew()
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
      run.value = await api.fetch<RunInfo>(`/runs/${runId}`)
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
    unexportable,
    awaitingConfirmation,
    source,
    currentPackage,
    start,
    retry,
    pin,
    confirmDownload,
  }
}
