<script setup lang="ts">
import type { RunInfo } from "~/types/telemetry"
import type { RenderState } from "~/types/games"

// Interaction modes 2 and 3 in one page (doc 0005 step 6): watch a trained policy play, driven
// entirely client-side by the same Web Worker /play/[game].vue uses, just fed a loaded
// evolve.neuro.WeightVector instead of keyboard input. Mode 2 (a finished run) and mode 3 (a
// still-training run's live current-best) turn out to be the same mechanism: reuse
// useMetricsStreamStore (the run-detail page's own store) and re-load whichever champion is
// latest whenever it changes. For a completed run that's just once; for a running one, every time
// a new GenerationStats arrives over SSE -- no separate code path needed for either case.
const route = useRoute()
const runId = route.params.runId as string
const config = useRuntimeConfig()

const run = ref<RunInfo | null>(null)
const loading = ref(true)
const error = ref<string | null>(null)

const renderState = ref<RenderState | null>(null)
const reward = ref(0)
const done = ref(false)
const stepCount = ref(0)

const metricsStream = useMetricsStreamStore()

let worker: Worker | null = null
let workerStarted = false
let lastLoadedRef: string | null = null

function setupWorker() {
  // A shared worker, not one created (and Pyodide-loaded) fresh per visit -- see
  // app/composables/useSnakeWorker.ts.
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
}

async function loadLatestChampionIfNew() {
  const latest = metricsStream.history.at(-1)
  if (!latest || latest.champion_ref === lastLoadedRef) return

  let policyJson: string
  try {
    policyJson = await $fetch<string>(`/runs/${runId}/artifacts/${latest.champion_ref}`, {
      baseURL: config.public.apiBase,
      responseType: "text",
    })
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    return // lastLoadedRef stays unset for this ref, so a retry (or the next SSE tick) tries again
  }
  lastLoadedRef = latest.champion_ref

  if (!workerStarted) {
    workerStarted = true
    setupWorker()
    worker!.postMessage({ type: "start", policyJson })
  } else {
    worker!.postMessage({ type: "load_policy", policyJson })
  }
}

watch(() => metricsStream.history.length, loadLatestChampionIfNew)

async function startWatching() {
  loading.value = true
  error.value = null

  try {
    run.value = await $fetch<RunInfo>(`/runs/${runId}`, { baseURL: config.public.apiBase })
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    loading.value = false
    return
  }

  if (run.value.config?.game !== "snake") {
    error.value = `run ${runId} isn't a watchable game run (config.game = ${String(run.value.config?.game ?? "unset")})`
    loading.value = false
    return
  }

  // metricsStream.start() always resets history to [] before repopulating it, so the watch()
  // above fires again on retry even if the champion it lands on ends up being the same one.
  await metricsStream.start(runId)
}

function retry() {
  workerStarted = false
  lastLoadedRef = null
  startWatching()
}

onMounted(startWatching)
onUnmounted(() => {
  metricsStream.stop()
  // Not terminate() -- the worker is shared and may be reused by the next page (see
  // app/composables/useSnakeWorker.ts). "stop" just pauses its tick loop.
  worker?.postMessage({ type: "stop" })
  worker = null
})
</script>

<template>
  <main class="mx-auto max-w-2xl p-6">
    <NuxtLink :to="`/runs/${runId}`" class="text-sm text-slate-500 hover:underline">&larr; run detail</NuxtLink>
    <h1 class="mt-2 text-2xl font-semibold text-slate-900">Watching <span class="font-mono text-lg">{{ runId }}</span></h1>
    <p class="mt-1 text-sm text-slate-500">
      A trained policy plays automatically -- no controls. Re-loads the current-best champion
      whenever a new one is recorded.
    </p>

    <p v-if="loading" class="mt-6 text-slate-500">Loading Python runtime...</p>
    <div v-else-if="error" class="mt-4">
      <p class="text-red-600">{{ error }}</p>
      <button
        class="mt-2 rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
        @click="retry"
      >
        Retry
      </button>
    </div>

    <template v-else-if="renderState">
      <GridBoard :state="renderState" class="mt-4" />

      <div class="mt-4 flex items-center gap-6 text-sm">
        <span>score: <span class="font-mono">{{ renderState.score }}</span></span>
        <span>step: <span class="font-mono">{{ stepCount }}</span></span>
        <span>reward: <span class="font-mono">{{ reward.toFixed(2) }}</span></span>
        <span v-if="lastLoadedRef" class="font-mono text-xs text-slate-400">{{ lastLoadedRef }}</span>
      </div>

      <p v-if="done" class="mt-4 text-slate-600">
        Episode ended -- will restart automatically once a new champion is recorded.
      </p>
    </template>
  </main>
</template>
