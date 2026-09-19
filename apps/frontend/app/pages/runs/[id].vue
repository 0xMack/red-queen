<script setup lang="ts">
import type { EvaluationRecord } from "~/types/leaderboard"
import type { RunInfo } from "~/types/telemetry"

const route = useRoute()
const runId = route.params.id as string
const config = useRuntimeConfig()

const metricsStream = useMetricsStreamStore()
const run = ref<RunInfo | null>(null)
const runError = ref<string | null>(null)

async function loadRun() {
  try {
    run.value = await $fetch<RunInfo>(`/runs/${runId}`, { baseURL: config.public.apiBase })
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
  if (typeof game !== "string") return
  try {
    const board = await $fetch<EvaluationRecord[]>(`/games/${game}/leaderboard`, { baseURL: config.public.apiBase })
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
useHead({ title: () => meta.value?.title ?? "Run" })

const history = computed(() => metricsStream.history)
const first = computed(() => history.value[0])
const latest = computed(() => history.value.at(-1))
const bestEver = computed(() =>
  history.value.reduce<(typeof history.value)[number] | null>((b, h) => (!b || h.best_fitness > b.best_fitness ? h : b), null),
)

const now = ref(Date.now() / 1000)
let clock: ReturnType<typeof setInterval> | null = null
onMounted(() => (clock = setInterval(() => (now.value = Date.now() / 1000), 5000)))
onUnmounted(() => clock && clearInterval(clock))

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
    { key: "held", label: "game score, unseen games", color: "#4ade80", values: heldOut.value.map((h) => h.held_out_score!), width: 2.5 },
    { key: "fit", label: "best training fitness", color: "#ff5c7a", values: heldOut.value.map((h) => h.best_fitness), width: 1.5, dashed: true },
  ]
  if (greedyMean.value !== null) {
    series.push({ key: "greedy", label: "greedy baseline", color: "#fbbf24", values: heldOut.value.map(() => greedyMean.value!), width: 1, dashed: true })
  }
  return series
})

const configEntries = computed(() => Object.entries(run.value?.config ?? {}))
const summaryEntries = computed(() => Object.entries(run.value?.summary ?? {}))
const recent = computed(() => history.value.slice(-25).reverse())

function displayValue(value: unknown): string {
  if (Array.isArray(value)) return value.join(" → ")
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : formatFitness(value, 4)
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
    await $fetch(`/runs/${runId}/control`, { baseURL: config.public.apiBase, method: "POST", body: { action } })
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
  <main class="mx-auto max-w-[1600px] px-4 py-8 sm:px-6 lg:px-8">
    <NuxtLink to="/runs" class="text-sm text-fg-subtle transition hover:text-fg">&larr; All runs</NuxtLink>

    <!-- Header -->
    <div class="mt-4 flex flex-wrap items-start justify-between gap-4">
      <div class="min-w-0">
        <p class="eyebrow">{{ meta?.representationLabel ?? "Training run" }}</p>
        <h1 class="mt-2 text-3xl font-semibold sm:text-4xl">{{ meta?.title ?? "Run" }}</h1>
        <div class="mt-3 flex flex-wrap items-center gap-2 text-sm">
          <StatusBadge v-if="run" :status="run.status" :stale="stale" />
          <span
            class="chip"
            :class="metricsStream.connected ? 'border-life-400/30 text-life-300' : ''"
            :title="metricsStream.connected ? 'Receiving new generations over SSE' : 'Stream not connected'"
          >
            <span class="size-1.5 rounded-full" :class="metricsStream.connected ? 'animate-live-pulse bg-life-400' : 'bg-fg-subtle'" />
            {{ metricsStream.connected ? "stream live" : "stream idle" }}
          </span>
          <button class="chip transition hover:text-fg" title="Copy run id" @click="copyId">
            {{ copied ? "copied!" : runId }}
          </button>
          <span v-if="run" class="text-fg-subtle">started {{ formatTimestamp(run.created_at) }} · ran {{ formatDuration(elapsed) }}</span>
          <span v-if="meta?.note" class="text-fg-subtle italic">· {{ meta.note }}</span>
          <span v-if="typeof run?.config?.interface === 'string'" class="chip" title="The representation this run's models were trained under (docs/design/0007)">
            {{ run.config.interface }}
          </span>
        </div>
      </div>
      <div v-if="controllable" class="flex items-center gap-2">
        <button v-if="run?.status === 'running'" class="btn-ghost btn-sm" :disabled="controlBusy" @click="control('pause')">❚❚ Pause</button>
        <template v-else>
          <button class="btn-primary btn-sm" :disabled="controlBusy" @click="control('resume')">▶ Resume</button>
          <button class="btn-ghost btn-sm" :disabled="controlBusy" @click="control('step')">Step one generation</button>
        </template>
        <p v-if="controlError" class="text-xs text-queen-300">{{ controlError }}</p>
      </div>
    </div>

    <p v-if="metricsStream.error" class="card mt-8 border-queen-500/40 p-5 text-queen-300">{{ metricsStream.error }}</p>
    <p v-else-if="runError" class="card mt-8 border-queen-500/40 p-5 text-queen-300">{{ runError }}</p>

    <template v-else>
      <!-- Stat tiles -->
      <div class="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
        <StatTile label="Generation" :value="latest ? latest.generation : '--'">
          <template #hint>
            <span v-if="meta?.targetGenerations">of {{ meta.targetGenerations }}</span>
            <span v-else>{{ history.length }} recorded</span>
          </template>
          <div v-if="progress !== null" class="mt-2 h-1 overflow-hidden rounded-full bg-raised">
            <div class="h-full rounded-full bg-queen-400 transition-all" :style="{ width: `${progress * 100}%` }" />
          </div>
        </StatTile>
        <StatTile
          label="Best fitness"
          tone="queen"
          :value="formatFitness(latest?.best_fitness, 3)"
          :hint="latest && first ? `${formatSigned(latest.best_fitness - first.best_fitness)} since gen 0` : undefined"
        />
        <StatTile label="Mean fitness" :value="formatFitness(latest?.mean_fitness, 3)" :hint="latest ? `worst ${formatFitness(latest.worst_fitness, 2)}` : undefined" />
        <StatTile
          label="Diversity"
          :value="formatFitness(latest?.diversity, 3)"
          :hint="first && latest ? `${formatSigned(latest.diversity - first.diversity, 3)} since gen 0` : undefined"
        />
        <StatTile label="Best ever" tone="gold" :value="formatFitness(bestEver?.best_fitness, 3)" :hint="bestEver ? `at generation ${bestEver.generation}` : undefined" />
        <StatTile label="Pace" :value="pace ? `${pace.toFixed(pace < 10 ? 1 : 0)}/min` : '--'" :hint="meta?.populationSize ? `population ${meta.populationSize}` : 'generations per minute'" />
      </div>

      <!-- Leaderboard standing -->
      <NuxtLink
        v-if="standing"
        :to="{ path: `/games/${standing.record.game}`, query: { watch: standing.record.entrant_id } }"
        class="card card-hover mt-3 flex flex-wrap items-center gap-x-6 gap-y-2 px-5 py-3 text-sm"
      >
        <span class="eyebrow">🏆 Leaderboard</span>
        <span class="text-fg">
          <span class="num font-semibold">#{{ standing.rank }}</span>
          <span class="text-fg-subtle"> of {{ standing.of }}</span>
        </span>
        <span class="text-fg-muted">
          held-out score
          <span class="num font-semibold text-fg">{{ standing.record.metrics.quality.mean.toFixed(2) }}</span>
          <span class="num text-fg-subtle"> ± {{ standing.record.metrics.quality.ci95.toFixed(2) }}</span>
          over {{ standing.record.metrics.quality.n }} unseen games
        </span>
        <span v-if="standing.record.metrics.quality.train_mean !== null" class="text-fg-muted">
          vs. <span class="num">{{ standing.record.metrics.quality.train_mean.toFixed(1) }}</span> on its training seeds
        </span>
        <span class="text-fg-muted"><span class="num">{{ standing.record.metrics.inference.total_us.toFixed(1) }}</span> µs / decision</span>
        <span class="ml-auto text-xs text-fg-subtle">Best fitness above is training fitness, not a game score →</span>
      </NuxtLink>

      <!-- Charts + champion -->
      <div class="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <div class="flex min-w-0 flex-col gap-6">
          <section class="card p-5">
            <div class="flex flex-wrap items-baseline justify-between gap-2">
              <h2 class="text-lg font-semibold">Fitness over generations</h2>
              <p v-if="meta?.watchable" class="text-xs text-fg-subtle">Click anywhere on the chart to watch that generation's champion.</p>
            </div>
            <FitnessChart
              class="mt-4"
              :history="history"
              :height="320"
              :marker="markerGeneration"
              :clickable="meta?.watchable ?? false"
              @select="pinned = $event"
            />
          </section>

          <section v-if="heldOut.length" class="card p-5">
            <div class="flex flex-wrap items-baseline justify-between gap-2">
              <h2 class="text-lg font-semibold">Training fitness vs. real game score</h2>
              <p v-if="heldOutPeak" class="text-xs text-fg-subtle">
                latest <span class="num text-fg">{{ heldOut.at(-1)!.held_out_score!.toFixed(2) }}</span> · peak
                <span class="num text-life-300">{{ heldOutPeak.held_out_score!.toFixed(2) }}</span> at gen {{ heldOutPeak.generation }}
              </p>
            </div>
            <p class="mt-1 text-xs text-fg-subtle">
              The champion's mean score on {{ monitorGames }} games it never trained on, checked every
              {{ run?.config?.held_out_every ?? "N" }} generations. If training fitness keeps rising while this falls, the
              population is memorizing its training games instead of learning the game.
            </p>
            <div class="mt-3 flex flex-wrap gap-4 text-xs text-fg-muted">
              <span v-for="s in heldOutSeries" :key="s.key" class="flex items-center gap-1.5">
                <span class="h-0.5 w-4 rounded" :style="{ background: s.color }" />{{ s.label }}
              </span>
            </div>
            <LineChart class="mt-2" :x="heldOut.map((h) => h.generation)" :series="heldOutSeries" :height="220" :format="(v: number) => v.toFixed(1)" />
          </section>

          <section class="card p-5">
            <h2 class="text-lg font-semibold">Population diversity</h2>
            <p class="mt-1 text-xs text-fg-subtle">Genotypic spread of the population -- a collapse toward zero is premature convergence.</p>
            <LineChart
              class="mt-4"
              :x="history.map((h) => h.generation)"
              :series="[{ key: 'div', label: 'diversity', color: '#fbbf24', values: history.map((h) => h.diversity) }]"
              :height="180"
            />
          </section>
        </div>

        <!-- Champion column -->
        <section class="card h-fit p-5 xl:sticky xl:top-20">
          <div class="flex items-baseline justify-between gap-2">
            <h2 class="text-lg font-semibold">Champion</h2>
            <span v-if="latest" class="font-mono text-[11px] text-fg-subtle">{{ (pinned !== null ? history.find((h) => h.generation === pinned) : latest)?.champion_ref }}</span>
          </div>

          <div class="mt-4">
            <ClientOnly v-if="meta?.watchable">
              <WatchChampion :run-id="runId" :manage-stream="false" :pinned-generation="pinned" @unpin="pinned = null" />
            </ClientOnly>
            <ChampionProgram v-else-if="latest && meta?.representation === 'linear_gp'" :run-id="runId" :champion-ref="latest.champion_ref" />
            <p v-else-if="latest" class="text-sm text-fg-subtle">
              Champion stored as <code class="chip">{{ latest.champion_ref }}</code> -- no viewer for this representation yet.
            </p>
            <div v-else class="h-64 animate-pulse rounded-lg bg-raised" />
          </div>
        </section>
      </div>

      <!-- Config + generations -->
      <div class="mt-6 grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
        <section class="card p-5">
          <h2 class="text-lg font-semibold">Configuration</h2>
          <dl class="mt-4 divide-y divide-line/60 text-sm">
            <div v-for="[key, value] in configEntries" :key="key" class="flex items-baseline justify-between gap-4 py-2">
              <dt class="font-mono text-xs text-fg-subtle">{{ key }}</dt>
              <dd class="text-right font-mono text-xs break-all text-fg">{{ displayValue(value) }}</dd>
            </div>
            <div v-if="meta?.parameterCount" class="flex items-baseline justify-between gap-4 py-2">
              <dt class="font-mono text-xs text-fg-subtle">parameters (derived)</dt>
              <dd class="num text-right text-xs text-fg">{{ meta.parameterCount }}</dd>
            </div>
          </dl>
          <template v-if="summaryEntries.length">
            <h3 class="mt-6 text-sm font-semibold text-fg-muted">Final summary</h3>
            <dl class="mt-2 divide-y divide-line/60 text-sm">
              <div v-for="[key, value] in summaryEntries" :key="key" class="flex items-baseline justify-between gap-4 py-2">
                <dt class="font-mono text-xs text-fg-subtle">{{ key }}</dt>
                <dd class="text-right font-mono text-xs text-fg">{{ displayValue(value) }}</dd>
              </div>
            </dl>
          </template>
        </section>

        <section class="card overflow-hidden">
          <div class="flex items-baseline justify-between px-5 pt-5">
            <h2 class="text-lg font-semibold">Recent generations</h2>
            <span class="text-xs text-fg-subtle">latest 25 of {{ history.length }}</span>
          </div>
          <div class="mt-3 max-h-[420px] overflow-auto">
            <table class="w-full text-sm">
              <thead class="sticky top-0 bg-surface">
                <tr class="border-b border-line text-left text-[11px] tracking-wide text-fg-subtle uppercase">
                  <th class="px-5 py-2 font-medium">Gen</th>
                  <th class="px-3 py-2 text-right font-medium">Best</th>
                  <th class="px-3 py-2 text-right font-medium">Δ best</th>
                  <th class="px-3 py-2 text-right font-medium">Mean</th>
                  <th class="px-3 py-2 text-right font-medium">Worst</th>
                  <th class="px-3 py-2 text-right font-medium">Diversity</th>
                  <th class="px-5 py-2 text-right font-medium">Time</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="h in recent"
                  :key="h.generation"
                  class="border-b border-line/40 last:border-0"
                  :class="[
                    meta?.watchable ? 'cursor-pointer hover:bg-raised/60' : '',
                    pinned === h.generation ? 'bg-gold-400/10' : '',
                  ]"
                  @click="meta?.watchable && (pinned = h.generation)"
                >
                  <td class="num px-5 py-1.5 text-fg-muted">{{ h.generation }}</td>
                  <td class="num px-3 py-1.5 text-right text-fg">{{ formatFitness(h.best_fitness, 3) }}</td>
                  <td class="num px-3 py-1.5 text-right text-xs">
                    <template v-for="prev in [history.find((p) => p.generation === h.generation - 1)]" :key="h.generation">
                      <span v-if="prev && h.best_fitness - prev.best_fitness > 1e-9" class="text-life-400">{{ formatSigned(h.best_fitness - prev.best_fitness, 3) }}</span>
                      <span v-else-if="prev && h.best_fitness - prev.best_fitness < -1e-9" class="text-queen-300">{{ formatSigned(h.best_fitness - prev.best_fitness, 3) }}</span>
                      <span v-else class="text-fg-subtle">·</span>
                    </template>
                  </td>
                  <td class="num px-3 py-1.5 text-right text-fg-muted">{{ formatFitness(h.mean_fitness, 3) }}</td>
                  <td class="num px-3 py-1.5 text-right text-fg-subtle">{{ formatFitness(h.worst_fitness, 3) }}</td>
                  <td class="num px-3 py-1.5 text-right text-fg-muted">{{ formatFitness(h.diversity, 3) }}</td>
                  <td class="num px-5 py-1.5 text-right text-xs text-fg-subtle">
                    {{ first ? `+${formatDuration(h.timestamp - first.timestamp)}` : "" }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </template>
  </main>
</template>
