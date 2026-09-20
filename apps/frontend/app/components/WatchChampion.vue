<script setup lang="ts">
// Embeddable "watch a trained champion play" panel: the board, playback speed, episode stats, and
// the champion's network with live activations. Used by the landing page (manageStream: true --
// it owns the metrics stream) and the run detail page (manageStream: false -- that page already
// streams the same run for its charts, and pins generations by clicking the chart).
//
// Laid out with a container query, not viewport breakpoints: the same component sits in a wide
// hero column and a narrower run-page column, and should adapt to the space it's actually given.
import type { Availability } from "~/composables/useModelCatalog"
import type { ModelFailure } from "~/composables/useSnakeSession"
import { formatBytes } from "~/inference/match"
import { SNAKE_INPUT_LABELS, SNAKE_OUTPUT_LABELS } from "~/utils/snakePolicy"
import { activations as neatActivations, complexity as neatComplexity } from "~/utils/neat"

// Plays either a run's champion (runId) or a fixed baseline (baseline + its interface) -- every kind
// of leaderboard entrant. Which one is decided at setup: key this component by entrant so switching
// remounts it (the session worker itself is shared and stays warm). A champion plays as a model
// package in ONNX Runtime (docs/design/0009) -- `packaged` when the page has the published one,
// otherwise exported on demand -- or the panel explains why this device can't run it.
const props = withDefaults(
  defineProps<{
    runId?: string
    baseline?: { name: string; interface: string; description?: string }
    manageStream?: boolean
    pinnedGeneration?: number | null
    showNetwork?: boolean
    packaged?: Availability | null
    // The page is still loading its model catalog: wait for it rather than start the Python path.
    packagedLoading?: boolean
  }>(),
  {
    runId: undefined,
    baseline: undefined,
    manageStream: true,
    pinnedGeneration: null,
    showNetwork: true,
    packaged: null,
    packagedLoading: false,
  },
)
const emit = defineEmits<{ unpin: []; modelFailed: [failure: ModelFailure]; retryFailed: [] }>()

const session = props.baseline
  ? useBaselineSession(props.baseline.name, props.baseline.interface)
  : useWatchSession(props.runId!, {
      manageStream: props.manageStream,
      packaged: () => (props.packagedLoading ? undefined : props.packaged),
      onModelFailure: (failure) => emit("modelFailed", failure),
    })
const {
  loading,
  error,
  renderState,
  stepCount,
  observation,
  done: gameOver,
  episodes,
  episodeScores,
  bestScore,
  meanScore,
  tickMs,
  policy,
  targetStats,
  pinnedGeneration: sessionPinned,
  unsupported,
  awaitingConfirmation,
  source,
  currentPackage,
  modelProgress,
  modelStatus,
} = session

watch(
  () => props.pinnedGeneration,
  (g) => session.pin(g),
  { immediate: true },
)

onMounted(session.start)

const activations = computed(() => {
  if (!policy.value || !observation.value || policy.value.kind !== "weights") return null
  if (observation.value.length !== policy.value.layerSizes[0]) return null
  return forwardActivations(policy.value.weights, policy.value.layerSizes, observation.value)
})

// An evolved graph is drawn (and lit up) by NeatDiagram, with node activations keyed by node id.
const neatLive = computed(() => {
  const genome = policy.value?.genome
  if (!genome || !observation.value || observation.value.length !== genome.numInputs) return null
  return neatActivations(genome, observation.value)
})
const neatSize = computed(() => (policy.value?.genome ? neatComplexity(policy.value.genome) : null))

const loadingMessage = computed(() => {
  if (props.packagedLoading || (props.packaged && !props.packaged.match)) return "Checking what this device can run…"
  return targetStats.value || props.baseline ? "Starting the game…" : "Fetching the champion…"
})

const progressLabel = computed(() => {
  const p = modelProgress.value
  if (!p) return null
  if (p.stage === "runtime") return "Loading ONNX Runtime…"
  if (p.stage === "download") return `Downloading model · ${formatBytes(p.loaded ?? 0)} / ${formatBytes(p.total ?? 0)}`
  if (p.stage === "compile") return "Compiling the model…"
  return "Checking it against the reference…"
})

// The variant actually running, for the runtime panel (what was published vs. what this device got).
const runningVariant = computed(() =>
  currentPackage.value?.manifest?.variants.find((v) => v.id === modelStatus.value?.variantId) ?? null,
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
          v-else-if="unsupported"
          data-unsupported
          class="card flex aspect-square flex-col items-center justify-center gap-3 p-6 text-center"
        >
          <p class="eyebrow">Can't run on this device</p>
          <p class="max-w-sm text-sm text-fg">{{ unsupported.summary }}</p>
          <ul v-if="unsupported.rejected.length > 1" class="max-w-sm space-y-1 text-left text-xs text-fg-subtle">
            <li v-for="(r, i) in unsupported.rejected" :key="i">
              <span class="font-mono">{{ r.variant }}{{ r.backend ? ` · ${r.backend}` : "" }}</span> — {{ r.message }}
            </li>
          </ul>
          <p class="max-w-sm text-xs text-fg-subtle">
            Every other entrant this device can run is still watchable from the leaderboard.
          </p>
          <button
            v-if="unsupported.rejected.some((r) => r.code === 'failed-before')"
            class="btn-ghost btn-sm"
            @click="emit('retryFailed')"
          >
            Try again anyway
          </button>
        </div>
        <div
          v-else-if="awaitingConfirmation !== null"
          class="card flex aspect-square flex-col items-center justify-center gap-3 p-6 text-center"
        >
          <p class="text-sm text-fg">This model is a {{ formatBytes(awaitingConfirmation) }} download.</p>
          <p class="max-w-sm text-xs text-fg-subtle">It runs entirely in your browser and is cached after the first time.</p>
          <button class="btn-ghost btn-sm" @click="session.confirmDownload">Download and watch</button>
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
          <BoardResult v-if="gameOver && renderState" title="Game over" :sub="`${renderState.score} 🍎`" tone="draw" />
          <PlaybackControls
            class="mt-3"
            :paused="session.paused.value"
            :speed="session.speed.value"
            new-game-label="Replay"
            @update:paused="session.setPaused"
            @update:speed="session.setSpeedMultiplier"
            @new-game="session.restart"
          />
          <div
            v-if="progressLabel"
            class="absolute inset-0 flex items-center justify-center rounded-xl bg-bg/70 text-sm text-fg-muted backdrop-blur-sm"
          >
            {{ progressLabel }}
          </div>
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

        <div v-if="sessionPinned !== null" class="flex justify-end">
          <button class="btn-ghost btn-sm" @click="emit('unpin')">Follow latest champion</button>
        </div>

        <div v-if="!baseline && source" data-runtime class="rounded-lg border border-line bg-sunken px-3 py-2 text-xs">
          <template v-if="modelStatus">
            <p class="text-fg">
              Running in your browser · ONNX Runtime ·
              <span class="font-mono">{{ modelStatus.variantId }}</span> on
              <span class="font-mono">{{ modelStatus.backend }}</span>
            </p>
            <p class="mt-0.5 text-fg-subtle">
              {{ formatBytes(runningVariant?.requirements.download_bytes ?? 0) }} · loaded in {{ Math.round(modelStatus.loadMs) }} ms
              <template v-if="modelStatus.selfTest">
                · self-test max |Δ| {{ modelStatus.selfTest.maxAbsError.toExponential(1) }} on {{ modelStatus.selfTest.samples }} samples
              </template>
              <template v-if="runningVariant?.parity.action_agreement != null">
                · {{ (runningVariant.parity.action_agreement * 100).toFixed(runningVariant.parity.action_agreement === 1 ? 0 : 2) }}% same
                moves as the trained model
              </template>
              <template v-else-if="source === 'on-demand'">
                · exported on demand (not a published package, so not checked move-for-move)
              </template>
            </p>
          </template>
          <p v-else class="text-fg-subtle">Loading the model package…</p>
        </div>

        <div v-if="baseline" class="rounded-lg border border-line bg-sunken p-4 text-sm">
          <p class="text-[11px] tracking-wide text-fg-subtle uppercase">How it decides</p>
          <p class="mt-1.5 text-fg-muted">{{ baseline.description ?? "A fixed, hand-written policy." }}</p>
          <p class="mt-2 text-xs text-fg-subtle">
            No training, no weights -- a reference point. If a learned model can't beat this, it hasn't
            learned much.
          </p>
        </div>

        <div v-if="showNetwork && policy?.genome" class="rounded-lg border border-line bg-sunken p-3">
          <div class="mb-2 flex items-center justify-between text-[11px] text-fg-subtle">
            <span>evolved network · {{ neatSize?.hidden }} hidden nodes · {{ neatSize?.connections }} connections</span>
            <span class="flex gap-2">
              <span class="text-life-400">+ excite</span><span class="text-queen-300">− inhibit</span>
            </span>
          </div>
          <NeatDiagram
            :genome="policy.genome"
            :activations="neatLive"
            :input-labels="policy.genome.numInputs === SNAKE_INPUT_LABELS.length ? SNAKE_INPUT_LABELS : []"
            :output-labels="policy.genome.numOutputs === SNAKE_OUTPUT_LABELS.length ? SNAKE_OUTPUT_LABELS : []"
          />
        </div>
        <div v-else-if="showNetwork && policy" class="rounded-lg border border-line bg-sunken p-3">
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
