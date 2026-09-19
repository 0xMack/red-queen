<script setup lang="ts">
// Embeddable "watch a trained champion play" panel: the board, playback speed, episode stats, and
// the champion's network with live activations. Used by the landing page (manageStream: true --
// it owns the metrics stream) and the run detail page (manageStream: false -- that page already
// streams the same run for its charts, and pins generations by clicking the chart).
//
// Laid out with a container query, not viewport breakpoints: the same component sits in a wide
// hero column and a narrower run-page column, and should adapt to the space it's actually given.
import { SNAKE_INPUT_LABELS, SNAKE_OUTPUT_LABELS } from "~/utils/snakePolicy"

// Plays either a run's champion (runId) or a fixed baseline (baseline + its interface) -- every kind
// of leaderboard entrant. Which one is decided at setup: key this component by entrant so switching
// remounts it (the Pyodide worker itself is shared and stays warm).
const props = withDefaults(
  defineProps<{
    runId?: string
    baseline?: { name: string; interface: string; description?: string }
    manageStream?: boolean
    pinnedGeneration?: number | null
    showNetwork?: boolean
  }>(),
  { runId: undefined, baseline: undefined, manageStream: true, pinnedGeneration: null, showNetwork: true },
)
const emit = defineEmits<{ unpin: [] }>()

const session = props.baseline
  ? useBaselineSession(props.baseline.name, props.baseline.interface)
  : useWatchSession(props.runId!, { manageStream: props.manageStream })
const {
  loading,
  error,
  renderState,
  stepCount,
  observation,
  episodes,
  episodeScores,
  bestScore,
  meanScore,
  tickMs,
  policy,
  targetStats,
  pinnedGeneration: sessionPinned,
} = session

watch(
  () => props.pinnedGeneration,
  (g) => session.pin(g),
  { immediate: true },
)

onMounted(session.start)

const SPEEDS = [
  { label: "½×", ms: 220 },
  { label: "1×", ms: 110 },
  { label: "2×", ms: 55 },
  { label: "4×", ms: 28 },
]

const activations = computed(() => {
  if (!policy.value || !observation.value) return null
  if (observation.value.length !== policy.value.layerSizes[0]) return null
  return forwardActivations(policy.value.weights, policy.value.layerSizes, observation.value)
})

const loadingMessage = computed(() =>
  targetStats.value || props.baseline ? "Starting the Python runtime (first load ~10s)…" : "Fetching the champion…",
)
</script>

<template>
  <div class="@container">
    <div class="grid gap-5 @3xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
      <!-- Board (capped when stacked, so a narrow column doesn't get a giant board) -->
      <div class="relative mx-auto w-full max-w-[520px] @3xl:max-w-none">
        <div v-if="error" class="card flex aspect-square flex-col items-center justify-center gap-3 p-6 text-center">
          <p class="text-sm text-queen-300">{{ error }}</p>
          <button class="btn-ghost btn-sm" @click="session.retry">Retry</button>
        </div>
        <div
          v-else-if="loading || !renderState"
          class="card relative flex aspect-square flex-col items-center justify-center gap-3 overflow-hidden"
        >
          <div class="absolute inset-0 animate-pulse bg-[linear-gradient(110deg,transparent_30%,rgb(255_255_255/0.03)_50%,transparent_70%)]" />
          <BrandMark class="size-10 animate-pulse" />
          <p class="text-sm text-fg-muted">{{ loadingMessage }}</p>
        </div>
        <template v-else>
          <GridBoard :state="renderState" :tick-ms="tickMs" />
          <div data-board-overlay class="pointer-events-none absolute inset-x-3 top-3 flex items-start justify-between gap-2">
            <span class="chip border-line-strong bg-bg/80 backdrop-blur">
              <span class="size-1.5 rounded-full bg-gold-400" />
              <template v-if="baseline">baseline · {{ baseline.name }}</template>
              <template v-else>{{ sessionPinned === null ? "latest champion" : "pinned" }} · gen {{ policy?.generation ?? "?" }}</template>
            </span>
            <span class="num rounded-md bg-bg/80 px-2 py-0.5 text-sm font-semibold text-life-300 backdrop-blur">
              {{ renderState.score }} 🍎
            </span>
          </div>
        </template>
      </div>

      <!-- Details -->
      <div class="flex min-w-0 flex-col gap-4">
        <div class="grid grid-cols-3 gap-2">
          <div class="rounded-lg border border-line bg-sunken px-3 py-2">
            <p class="text-[10px] tracking-wide text-fg-subtle uppercase">step</p>
            <p class="num text-lg text-fg">{{ stepCount }}</p>
          </div>
          <div class="rounded-lg border border-line bg-sunken px-3 py-2">
            <p class="text-[10px] tracking-wide text-fg-subtle uppercase">best score</p>
            <p class="num text-lg text-life-300">{{ bestScore }}</p>
          </div>
          <div class="rounded-lg border border-line bg-sunken px-3 py-2">
            <p class="text-[10px] tracking-wide text-fg-subtle uppercase">avg · {{ episodes }} ep</p>
            <p class="num text-lg text-fg">{{ meanScore.toFixed(1) }}</p>
          </div>
        </div>

        <div>
          <div class="mb-1.5 flex items-center justify-between text-[11px] text-fg-subtle">
            <span>score per episode</span>
            <span v-if="episodeScores.length === 0">first episode in progress…</span>
          </div>
          <div class="flex h-10 items-end gap-0.5 rounded-lg border border-line bg-sunken px-1.5 py-1">
            <div
              v-for="(s, i) in episodeScores.slice(-40)"
              :key="i"
              class="min-w-1 flex-1 rounded-sm bg-life-400/70"
              :style="{ height: `${Math.max(6, (s / Math.max(1, bestScore)) * 100)}%` }"
              :title="`episode score ${s}`"
            />
          </div>
        </div>

        <div class="flex flex-wrap items-center gap-2">
          <span class="text-[11px] text-fg-subtle">speed</span>
          <div class="flex rounded-lg border border-line bg-sunken p-0.5">
            <button
              v-for="s in SPEEDS"
              :key="s.ms"
              class="num rounded-md px-2.5 py-1 text-xs transition"
              :class="tickMs === s.ms ? 'bg-raised text-fg shadow' : 'text-fg-subtle hover:text-fg'"
              @click="session.setSpeed(s.ms)"
            >
              {{ s.label }}
            </button>
          </div>
          <button v-if="sessionPinned !== null" class="btn-ghost btn-sm ml-auto" @click="emit('unpin')">
            Follow latest champion
          </button>
        </div>

        <div v-if="baseline" class="rounded-lg border border-line bg-sunken p-4 text-sm">
          <p class="text-[11px] tracking-wide text-fg-subtle uppercase">How it decides</p>
          <p class="mt-1.5 text-fg-muted">{{ baseline.description ?? "A fixed, hand-written policy." }}</p>
          <p class="mt-2 text-xs text-fg-subtle">
            No training, no weights -- a reference point. If a learned model can't beat this, it hasn't
            learned much.
          </p>
        </div>

        <div v-if="showNetwork && policy" class="rounded-lg border border-line bg-sunken p-3">
          <div class="mb-2 flex items-center justify-between text-[11px] text-fg-subtle">
            <span>champion network · {{ policy.layerSizes.join(" → ") }} · {{ policy.weights.length }} weights</span>
            <span class="flex gap-2">
              <span class="text-life-400">+ excite</span><span class="text-queen-300">− inhibit</span>
            </span>
          </div>
          <NetworkDiagram
            :weights="policy.weights"
            :layer-sizes="policy.layerSizes"
            :activations="activations"
            :input-labels="policy.layerSizes[0] === SNAKE_INPUT_LABELS.length ? SNAKE_INPUT_LABELS : []"
            :output-labels="policy.layerSizes.at(-1) === SNAKE_OUTPUT_LABELS.length ? SNAKE_OUTPUT_LABELS : []"
          />
        </div>
      </div>
    </div>
  </div>
</template>
