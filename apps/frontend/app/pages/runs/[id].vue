<script setup lang="ts">
import type { EvaluationRecord } from "~/types/leaderboard"
import type { RunInfo } from "~/types/telemetry"

const route = useRoute()
const runId = route.params.id as string
const api = useApi()

const metricsStream = useMetricsStreamStore()
const run = ref<RunInfo | null>(null)
const runError = ref<string | null>(null)

async function loadRun() {
  try {
    run.value = await api.fetch<RunInfo>(`/runs/${runId}`)
  } catch (e) {
    runError.value = e instanceof Error ? e.message : String(e)
  }
}

// This run's leaderboard standing (docs/design/0007) -- the held-out game score, which is what
// training fitness is *not*. Best-effort: no evaluation (or an older backend) just hides it.
const standing = ref<{ record: EvaluationRecord; rank: number; of: number } | null>(null)
const greedyMean = ref<number | null>(null) // reference line for the held-out chart
async function loadStanding() {
  const game = run.value?.config?.game
  if (game !== "snake") return // only Snake has a leaderboard (docs/design/0007); versus games have none yet
  try {
    const board = await api.fetch<EvaluationRecord[]>(`/games/${game}/leaderboard`)
    greedyMean.value = board.find((r) => r.entrant_id === "baseline:greedy")?.metrics.quality.mean ?? null
    const mine = board.find((r) => r.run_id === runId)
    if (!mine) return
    const peers = board.filter((r) => r.protocol === mine.protocol)
    standing.value = { record: mine, rank: peers.indexOf(mine) + 1, of: peers.length }
  } catch {
    standing.value = null
  }
}

onMounted(() => {
  metricsStream.start(runId)
  loadRun().then(loadStanding)
})
onUnmounted(() => metricsStream.stop())

const meta = computed(() => (run.value ? describeRun(run.value) : null))
const terms = computed(
  () => meta.value?.terms ?? { unit: "generation", fitness: "fitness", diversity: "diversity", diversityTitle: "Population diversity", diversityNote: "", learner: "population" },
)
useHead({ title: () => meta.value?.title ?? "Run" })

const history = computed(() => metricsStream.history)
const first = computed(() => history.value[0])
const latest = computed(() => history.value.at(-1))
const bestEver = computed(() =>
  history.value.reduce<(typeof history.value)[number] | null>((b, h) => (!b || h.best_fitness > b.best_fitness ? h : b), null),
)

const now = useNow(5)

const stale = computed(() => (run.value ? isStale(run.value, now.value) : false))
const elapsed = computed(() => {
  if (!run.value) return 0
  const end = run.value.status === "running" && !stale.value ? now.value : Math.max(run.value.updated_at, latest.value?.timestamp ?? 0)
  return end - run.value.created_at
})
const pace = computed(() => {
  const h = history.value
  if (h.length < 2) return null
  const span = h.at(-1)!.timestamp - h[0]!.timestamp
  return span > 0 ? ((h.length - 1) / span) * 60 : null
})
const progress = computed(() =>
  meta.value?.targetGenerations && latest.value ? Math.min(1, (latest.value.generation + 1) / meta.value.targetGenerations) : null,
)

// Generation pinning (watchable runs): click the chart or a table row to watch that generation's
// champion; the WatchChampion panel's "follow latest" clears it.
const pinned = ref<number | null>(null)
const markerGeneration = computed(() => pinned.value ?? (meta.value?.watchable ? latest.value?.generation ?? null : null))

// Training fitness vs. the champion's score on unseen games, at the generations the job checked --
// where the two diverge (fitness up, held-out down) the population is memorizing its training games.
const heldOut = computed(() => history.value.filter((h) => h.held_out_score != null))
const monitorGames = computed(() => {
  const range = run.value?.config?.monitor_seeds
  return Array.isArray(range) && range.length === 2 ? Number(range[1]) - Number(range[0]) + 1 : "unseen"
})
const heldOutPeak = computed(() =>
  heldOut.value.reduce<(typeof heldOut.value)[number] | null>((b, h) => (!b || h.held_out_score! > b.held_out_score! ? h : b), null),
)
const heldOutSeries = computed(() => {
  const series = [
    { key: "held", label: "game score, unseen games", color: palette.life400, values: heldOut.value.map((h) => h.held_out_score!), width: 2.5 },
    { key: "fit", label: `best training ${terms.value.fitness}`, color: palette.queen400, values: heldOut.value.map((h) => h.best_fitness), width: 1.5, dashed: true },
  ]
  if (greedyMean.value !== null) {
    series.push({ key: "greedy", label: "greedy baseline", color: palette.gold400, values: heldOut.value.map(() => greedyMean.value!), width: 1, dashed: true })
  }
  return series
})

// Algorithm-specific curves (telemetry's `extras`) -- today NEAT's: how many species the population split into,
// and how large the champion's evolved structure is. Empty for runs whose algorithm reports none.
const structure = computed(() => history.value.filter((h) => h.extras && "champion_connections" in h.extras))
const structureSeries = computed(() => [
  { key: "hidden", label: "champion hidden nodes", color: palette.signal400, values: structure.value.map((h) => h.extras!.champion_hidden_nodes ?? 0), width: 2.5 },
  { key: "conns", label: "champion connections", color: palette.life400, values: structure.value.map((h) => h.extras!.champion_connections ?? 0), width: 1.5 },
])
const speciesSeries = computed(() => [
  { key: "species", label: "species", color: palette.gold400, values: structure.value.map((h) => h.extras!.species ?? 0), width: 2.5 },
])

const configEntries = computed(() => {
  const entries: [string, string][] = Object.entries(run.value?.config ?? {}).map(([k, v]) => [k, displayValue(v)])
  if (meta.value?.parameterCount) entries.push(["parameters (derived)", String(meta.value.parameterCount)])
  return entries
})
const summaryEntries = computed(() => Object.entries(run.value?.summary ?? {}).map(([k, v]) => [k, displayValue(v)] as const))

function displayValue(value: unknown): string {
  if (Array.isArray(value)) return value.join(" → ")
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : formatFitness(value, 4)
  if (value !== null && typeof value === "object") return JSON.stringify(value)
  return String(value)
}

// Pause/resume/step (POST /runs/{id}/control) -- only offered while a job is actually listening.
const controlBusy = ref(false)
const controlError = ref<string | null>(null)
const controllable = computed(() => run.value && !stale.value && (run.value.status === "running" || run.value.status === "paused"))
async function control(action: "pause" | "resume" | "step") {
  controlBusy.value = true
  controlError.value = null
  try {
    await api.fetch(`/runs/${runId}/control`, { method: "POST", body: { action } })
    await loadRun()
  } catch (e) {
    controlError.value = e instanceof Error ? e.message : String(e)
  } finally {
    controlBusy.value = false
  }
}

const copied = ref(false)
async function copyId() {
  try {
    await navigator.clipboard.writeText(runId)
    copied.value = true
    setTimeout(() => (copied.value = false), 1200)
  } catch {
    // Not worth an error state -- the id is visible to select by hand.
  }
}
</script>

<template>
  <main class="mx-auto max-w-[1600px] px-4 pt-8 pb-10 sm:px-6 lg:px-8">
    <!-- Header -->
    <UiSectionHeader :level="1" :eyebrow="meta?.representationLabel ?? 'Training run'" :title="meta?.title ?? 'Run'" :back="{ to: '/runs', label: 'Runs' }">
      <div class="flex flex-wrap items-center gap-2 text-sm">
        <StatusBadge v-if="run" :status="run.status" :stale="stale" />
        <UiBadge :tone="metricsStream.connected ? 'life' : 'neutral'" :pulse="metricsStream.connected" dot :title="metricsStream.connected ? 'Receiving new generations over SSE' : 'Stream not connected'">
          {{ metricsStream.connected ? "stream live" : "stream idle" }}
        </UiBadge>
        <button class="chip transition hover:text-fg" title="Copy run id" @click="copyId">{{ copied ? "copied!" : shortId(runId) }}</button>
        <span v-if="typeof run?.config?.interface === 'string'" class="chip" title="The representation this run's models were trained under (docs/design/0007)">
          {{ run.config.interface }}
        </span>
        <span v-if="run" class="text-fg-subtle">started {{ formatTimestamp(run.created_at) }} · ran {{ formatDuration(elapsed) }}</span>
        <span v-if="meta?.note" class="text-fg-subtle italic">· {{ meta.note }}</span>
      </div>
      <template v-if="controllable" #actions>
        <button v-if="run?.status === 'running'" class="btn-ghost btn-sm" :disabled="controlBusy" @click="control('pause')">❚❚ Pause</button>
        <template v-else>
          <button class="btn-primary btn-sm" :disabled="controlBusy" @click="control('resume')">▶ Resume</button>
          <button class="btn-ghost btn-sm" :disabled="controlBusy" @click="control('step')">Step one {{ terms.unit }}</button>
        </template>
        <p v-if="controlError" class="text-xs text-queen-300">{{ controlError }}</p>
      </template>
    </UiSectionHeader>

    <UiPanel v-if="metricsStream.error || runError" class="mt-8 border-queen-500/40 text-queen-200">{{ metricsStream.error ?? runError }}</UiPanel>

    <template v-else>
      <!-- Stat tiles -->
      <div class="mt-10 grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
        <StatTile :label="capitalize(terms.unit)" :value="latest ? latest.generation : '--'">
          <template #hint>
            <span v-if="meta?.targetGenerations">of {{ meta.targetGenerations }}</span>
            <span v-else>{{ history.length }} recorded</span>
          </template>
          <div v-if="progress !== null" class="mt-2 h-[3px] overflow-hidden rounded-full bg-raised">
            <div class="h-full rounded-full bg-queen-400 transition-all" :style="{ width: `${progress * 100}%` }" />
          </div>
        </StatTile>
        <StatTile
          :label="`Best ${terms.fitness}`"
          tone="queen"
          :value="formatFitness(latest?.best_fitness, 3)"
          :hint="latest && first ? `${formatSigned(latest.best_fitness - first.best_fitness)} since ${terms.unit} 0` : undefined"
        />
        <StatTile :label="`Mean ${terms.fitness}`" :value="formatFitness(latest?.mean_fitness, 3)" :hint="latest ? `worst ${formatFitness(latest.worst_fitness, 2)}` : undefined" />
        <StatTile
          :label="capitalize(terms.diversity)"
          :value="formatFitness(latest?.diversity, 3)"
          :hint="first && latest ? `${formatSigned(latest.diversity - first.diversity, 3)} since ${terms.unit} 0` : undefined"
        />
        <StatTile label="Best ever" tone="gold" :value="formatFitness(bestEver?.best_fitness, 3)" :hint="bestEver ? `at ${terms.unit} ${bestEver.generation}` : undefined" />
        <StatTile label="Pace" :value="pace ? `${pace.toFixed(pace < 10 ? 1 : 0)}/min` : '--'" :hint="meta?.populationSize ? `population ${meta.populationSize}` : `${terms.unit}s per minute`" />
      </div>

      <RunStanding v-if="standing" class="mt-3" :record="standing.record" :rank="standing.rank" :of="standing.of" />

      <!-- Charts + champion -->
      <div class="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <div class="flex min-w-0 flex-col gap-6">
          <UiPanel :title="`${capitalize(terms.fitness)} over ${terms.unit}s`">
            <template v-if="meta?.watchable" #actions><span class="text-xs text-fg-subtle">Click the chart to watch that {{ terms.unit }}'s champion.</span></template>
            <FitnessChart
              :x-label="terms.unit"
              :history="history"
              :height="320"
              :marker="markerGeneration"
              :clickable="meta?.watchable ?? false"
              @select="pinned = $event"
            />
          </UiPanel>

          <UiPanel v-if="heldOut.length" :title="`Training ${terms.fitness} vs. real game score`">
            <template v-if="heldOutPeak" #actions>
              <span class="text-xs text-fg-subtle">
                latest <span class="num text-fg">{{ heldOut.at(-1)!.held_out_score!.toFixed(2) }}</span> · peak
                <span class="num text-life-300">{{ heldOutPeak.held_out_score!.toFixed(2) }}</span> at {{ terms.unit }} {{ heldOutPeak.generation }}
              </span>
            </template>
            <p class="text-xs leading-relaxed text-fg-subtle">
              The champion's mean score on {{ monitorGames }} games it never trained on, checked every
              {{ run?.config?.held_out_every ?? "N" }} {{ terms.unit }}s. If training {{ terms.fitness }} keeps rising while this
              falls, the {{ terms.learner }} is memorizing its training games instead of learning the game.
            </p>
            <UiLegend class="mt-3" :series="heldOutSeries" />
            <LineChart :x-label="terms.unit" class="mt-2" :x="heldOut.map((h) => h.generation)" :series="heldOutSeries" :height="220" :format="(v: number) => v.toFixed(1)" />
          </UiPanel>

          <UiPanel v-if="structure.length" title="Evolved structure">
            <p class="text-xs leading-relaxed text-fg-subtle">
              NEAT starts every network with no hidden nodes and adds structure only when it pays. Left: how large the current
              champion's graph is. Right: how many species the population is split into -- the mechanism that lets a new
              structure survive long enough to be tuned.
            </p>
            <div class="mt-4 grid gap-4 md:grid-cols-2">
              <div>
                <UiLegend :series="structureSeries" />
                <LineChart :x-label="terms.unit" class="mt-2" :x="structure.map((h) => h.generation)" :series="structureSeries" :height="180" :format="(v: number) => v.toFixed(0)" />
              </div>
              <div>
                <UiLegend :series="speciesSeries" />
                <LineChart :x-label="terms.unit" class="mt-2" :x="structure.map((h) => h.generation)" :series="speciesSeries" :height="180" :format="(v: number) => v.toFixed(0)" />
              </div>
            </div>
          </UiPanel>

          <UiPanel :title="terms.diversityTitle">
            <p class="text-xs text-fg-subtle">{{ terms.diversityNote }}</p>
            <LineChart
              :x-label="terms.unit"
              class="mt-4"
              :x="history.map((h) => h.generation)"
              :series="[{ key: 'div', label: terms.diversity, color: palette.gold400, values: history.map((h) => h.diversity) }]"
              :height="180"
            />
          </UiPanel>
        </div>

        <!-- Champion column -->
        <UiPanel ticks title="Champion" class="h-fit xl:sticky xl:top-20">
          <template v-if="latest" #actions>
            <span class="font-mono text-[11px] text-fg-subtle">{{ (pinned !== null ? history.find((h) => h.generation === pinned) : latest)?.champion_ref }}</span>
          </template>
          <ClientOnly v-if="meta?.watchable">
            <CheckersWatch v-if="meta.game === 'checkers'" :run-id="runId" :manage-stream="false" :pinned-generation="pinned" compact @unpin="pinned = null" />
            <WatchChampion v-else :run-id="runId" :manage-stream="false" :pinned-generation="pinned" @unpin="pinned = null" />
          </ClientOnly>
          <ChampionProgram v-else-if="latest && meta?.representation === 'linear_gp'" :run-id="runId" :champion-ref="latest.champion_ref" />
          <BanditRunChampion
            v-else-if="latest && meta?.game === 'bandit'"
            :run-id="runId"
            :champion-ref="(pinned !== null ? history.find((h) => h.generation === pinned) : latest)?.champion_ref ?? latest.champion_ref"
            :scenarios="Array.isArray(run?.config?.scenarios) ? (run!.config!.scenarios as string[]) : undefined"
          />
          <p v-else-if="latest" class="text-sm text-fg-subtle">
            Champion stored as <code class="chip">{{ latest.champion_ref }}</code> -- no viewer for this representation yet.
          </p>
          <div v-else class="h-64 animate-pulse rounded-[8px] bg-raised" />
        </UiPanel>
      </div>

      <!-- Config + generations -->
      <div class="mt-6 grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
        <UiPanel title="Configuration" class="min-w-0">
          <UiKeyValues :entries="configEntries" />
          <template v-if="summaryEntries.length">
            <p class="label mt-6 mb-1">Final summary</p>
            <UiKeyValues :entries="summaryEntries" />
          </template>
        </UiPanel>
        <RunGenerationsTable class="min-w-0" :history="history" :terms="terms" :pinned="pinned" :clickable="meta?.watchable ?? false" @pin="pinned = $event" />
      </div>
    </template>
  </main>
</template>
