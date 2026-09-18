<script setup lang="ts">
import type { RunInfo } from "~/types/telemetry"
import type { RunMeta } from "~/utils/runMeta"

useHead({ title: "Runs" })

const runsStore = useRunsStore()
await useAsyncData("runs", () => runsStore.fetchRuns().then(() => runsStore.runs))
onMounted(() => runsStore.fetchHistories())

const now = ref(Date.now() / 1000)
let clock: ReturnType<typeof setInterval> | null = null
onMounted(() => (clock = setInterval(() => (now.value = Date.now() / 1000), 30_000)))
onUnmounted(() => clock && clearInterval(clock))

interface Row {
  run: RunInfo
  meta: RunMeta
  generations: number
  best: number | null
  trend: number[]
  duration: number
  stale: boolean
  genome: string
}

const rows = computed<Row[]>(() =>
  runsStore.runs.map((run) => {
    const meta = describeRun(run)
    const history = runsStore.histories[run.run_id]
    const c = run.config ?? {}
    const genome = meta.network
      ? `${meta.network}${meta.parameterCount ? ` · ${meta.parameterCount}w` : ""}`
      : typeof c.num_instructions === "number"
        ? `${c.num_instructions} instr · ${c.num_registers ?? "?"} regs`
        : "--"
    const last = history?.at(-1)
    return {
      run,
      meta,
      generations: history ? history.length : 0,
      best: bestFitnessOf(run, history),
      trend: history ? downsample(history.map((h) => h.best_fitness), 60) : [],
      // For a live run the last recorded generation is fresher than the registry row.
      duration: Math.max(run.updated_at, last?.timestamp ?? 0) - run.created_at,
      stale: isStale(run, now.value),
      genome,
    }
  }),
)

function downsample(values: number[], n: number): number[] {
  if (values.length <= n) return values
  return Array.from({ length: n }, (_, i) => values[Math.round((i / (n - 1)) * (values.length - 1))]!)
}

// Filters / sorting --------------------------------------------------------------------------------
const query = ref("")
const statusFilter = ref<"all" | "running" | "completed" | "failed" | "paused">("all")
const kindFilter = ref("all")
type SortKey = "created" | "best" | "generations" | "duration"
const sortKey = ref<SortKey>("created")
const sortDesc = ref(true)

const kinds = computed(() => [...new Set(rows.value.map((r) => r.meta.representationLabel))])

const visible = computed(() => {
  const q = query.value.trim().toLowerCase()
  const filtered = rows.value.filter((r) => {
    if (statusFilter.value !== "all" && r.run.status !== statusFilter.value) return false
    if (kindFilter.value !== "all" && r.meta.representationLabel !== kindFilter.value) return false
    if (!q) return true
    return [r.run.run_id, r.meta.title, r.meta.note, r.meta.selection, r.meta.benchmark]
      .filter(Boolean)
      .some((s) => s!.toLowerCase().includes(q))
  })
  const value = (r: Row): number =>
    sortKey.value === "created"
      ? r.run.created_at
      : sortKey.value === "best"
        ? (r.best ?? -Infinity)
        : sortKey.value === "generations"
          ? r.generations
          : r.duration
  return filtered.sort((a, b) => (sortDesc.value ? value(b) - value(a) : value(a) - value(b)))
})

function sortBy(key: SortKey) {
  if (sortKey.value === key) sortDesc.value = !sortDesc.value
  else {
    sortKey.value = key
    sortDesc.value = true
  }
}

// Summary tiles --------------------------------------------------------------------------------------
const summary = computed(() => {
  const totalGens = rows.value.reduce((sum, r) => sum + r.generations, 0)
  const snake = rows.value.filter((r) => r.meta.game === "snake" && r.best !== null)
  const bestSnake = snake.length ? Math.max(...snake.map((r) => r.best!)) : null
  return {
    total: rows.value.length,
    running: rows.value.filter((r) => r.run.status === "running" && !r.stale).length,
    totalGens,
    bestSnake,
    kinds: kinds.value.length,
  }
})

const STATUSES = ["all", "running", "completed", "paused", "failed"] as const
</script>

<template>
  <main class="mx-auto max-w-[1600px] px-4 py-10 sm:px-6 lg:px-8">
    <div class="flex flex-wrap items-end justify-between gap-4">
      <div>
        <p class="eyebrow">Telemetry</p>
        <h1 class="mt-2 text-3xl font-semibold sm:text-4xl">Training runs</h1>
        <p class="mt-2 max-w-2xl text-fg-muted">
          Every run recorded by <code class="chip">libs/telemetry</code> -- genetic programming and
          neuroevolution alike. Open one for its live fitness curves and, for game runs, its champion
          playing in your browser.
        </p>
      </div>
      <button class="btn-ghost btn-sm" @click="runsStore.fetchRuns().then(() => runsStore.fetchHistories())">
        <svg viewBox="0 0 20 20" class="size-3.5" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M16 10a6 6 0 1 1-1.8-4.3M16 3.5V6h-2.5" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        Refresh
      </button>
    </div>

    <p v-if="runsStore.error" class="card mt-8 border-queen-500/40 p-5 text-queen-300">
      Couldn't reach the backend: {{ runsStore.error }}.
      <span class="text-fg-muted">Start it with <code class="chip">uv run uvicorn backend.main:app --app-dir apis/backend/src</code>.</span>
    </p>

    <template v-else>
      <div class="mt-8 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
        <StatTile label="Runs" :value="summary.total" :hint="`${summary.kinds} algorithm families`" />
        <StatTile label="Running now" :value="summary.running" :tone="summary.running ? 'life' : 'default'" hint="live over SSE" />
        <StatTile label="Generations recorded" :value="summary.totalGens.toLocaleString()" hint="across every run" />
        <StatTile label="Best Snake fitness" :value="formatFitness(summary.bestSnake, 2)" tone="queen" hint="neuroevolution" />
        <StatTile
          class="col-span-2 md:col-span-1"
          label="Latest run"
          :value="rows[0] ? formatRelative(rows[0].run.created_at, now) : '--'"
          :hint="rows[0]?.meta.title"
        />
      </div>

      <!-- Filters -->
      <div class="mt-8 flex flex-wrap items-center gap-3">
        <div class="relative min-w-56 flex-1 sm:max-w-sm">
          <svg viewBox="0 0 20 20" class="absolute top-1/2 left-3 size-4 -translate-y-1/2 text-fg-subtle" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="9" cy="9" r="6" /><path d="m14 14 4 4" stroke-linecap="round" />
          </svg>
          <input
            v-model="query"
            type="search"
            placeholder="Filter by id, game, selection…"
            class="w-full rounded-lg border border-line bg-surface py-2 pr-3 pl-9 text-sm placeholder:text-fg-subtle focus:border-queen-400/60 focus:outline-none"
          >
        </div>
        <div class="flex rounded-lg border border-line bg-surface p-0.5">
          <button
            v-for="s in STATUSES"
            :key="s"
            class="rounded-md px-3 py-1.5 text-xs font-medium capitalize transition"
            :class="statusFilter === s ? 'bg-raised text-fg shadow' : 'text-fg-subtle hover:text-fg'"
            @click="statusFilter = s"
          >
            {{ s }}
          </button>
        </div>
        <select
          v-model="kindFilter"
          class="rounded-lg border border-line bg-surface px-3 py-2 text-xs text-fg-muted focus:outline-none"
        >
          <option value="all">All algorithms</option>
          <option v-for="k in kinds" :key="k" :value="k">{{ k }}</option>
        </select>
        <span class="ml-auto text-xs text-fg-subtle">{{ visible.length }} of {{ rows.length }} runs</span>
      </div>

      <!-- Table (lg+) -->
      <div class="card mt-4 hidden overflow-x-auto lg:block">
        <table class="w-full text-sm">
          <thead>
            <tr class="border-b border-line text-left text-[11px] tracking-wide text-fg-subtle uppercase">
              <th class="px-4 py-3 font-medium">Run</th>
              <th class="px-3 py-3 font-medium">Status</th>
              <th class="px-3 py-3 font-medium">Search</th>
              <th class="px-3 py-3 text-right font-medium">Pop.</th>
              <th class="px-3 py-3 font-medium">Genome</th>
              <th class="cursor-pointer px-3 py-3 font-medium hover:text-fg" @click="sortBy('generations')">
                Generations {{ sortKey === "generations" ? (sortDesc ? "↓" : "↑") : "" }}
              </th>
              <th class="cursor-pointer px-3 py-3 text-right font-medium hover:text-fg" @click="sortBy('best')">
                Best {{ sortKey === "best" ? (sortDesc ? "↓" : "↑") : "" }}
              </th>
              <th class="px-3 py-3 font-medium">Best-fitness trend</th>
              <th class="cursor-pointer px-3 py-3 font-medium hover:text-fg" @click="sortBy('created')">
                Started {{ sortKey === "created" ? (sortDesc ? "↓" : "↑") : "" }}
              </th>
              <th class="cursor-pointer px-3 py-3 text-right font-medium hover:text-fg" @click="sortBy('duration')">
                Duration {{ sortKey === "duration" ? (sortDesc ? "↓" : "↑") : "" }}
              </th>
              <th class="px-4 py-3" />
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="r in visible"
              :key="r.run.run_id"
              class="group cursor-pointer border-b border-line/60 transition last:border-0 hover:bg-raised/60"
              @click="navigateTo(`/runs/${r.run.run_id}`)"
            >
              <td class="px-4 py-3">
                <NuxtLink :to="`/runs/${r.run.run_id}`" class="font-medium text-fg group-hover:text-queen-300" @click.stop>
                  {{ r.meta.title }}
                </NuxtLink>
                <p class="mt-0.5 flex items-center gap-2 font-mono text-[11px] text-fg-subtle">
                  {{ shortId(r.run.run_id) }}
                  <span v-if="r.meta.note" class="truncate font-sans italic">· {{ r.meta.note }}</span>
                </p>
              </td>
              <td class="px-3 py-3"><StatusBadge :status="r.run.status" :stale="r.stale" /></td>
              <td class="px-3 py-3">
                <div class="flex flex-col gap-1">
                  <span v-if="r.meta.selection" class="chip w-fit">{{ r.meta.selection }}</span>
                  <span v-if="r.meta.variation" class="w-fit max-w-48 truncate font-mono text-[11px] text-fg-subtle" :title="r.meta.variation">{{ r.meta.variation }}</span>
                  <span v-if="!r.meta.selection && !r.meta.variation" class="text-fg-subtle">--</span>
                </div>
              </td>
              <td class="num px-3 py-3 text-right text-fg-muted">{{ r.meta.populationSize ?? "--" }}</td>
              <td class="px-3 py-3 font-mono text-xs whitespace-nowrap text-fg-muted">{{ r.genome }}</td>
              <td class="px-3 py-3">
                <div class="flex items-center gap-2">
                  <span class="num w-16 text-xs text-fg-muted">
                    {{ r.generations }}<span v-if="r.meta.targetGenerations" class="text-fg-subtle">/{{ r.meta.targetGenerations }}</span>
                  </span>
                  <div v-if="r.meta.targetGenerations" class="h-1.5 w-20 overflow-hidden rounded-full bg-raised">
                    <div
                      class="h-full rounded-full"
                      :class="r.run.status === 'running' && !r.stale ? 'bg-life-400' : 'bg-fg-subtle'"
                      :style="{ width: `${Math.min(100, (r.generations / r.meta.targetGenerations) * 100)}%` }"
                    />
                  </div>
                </div>
              </td>
              <td class="num px-3 py-3 text-right font-semibold text-fg">{{ formatFitness(r.best, 3) }}</td>
              <td class="px-3 py-3"><Sparkline :values="r.trend" class="h-8 w-32" /></td>
              <td class="px-3 py-3 whitespace-nowrap text-fg-muted" :title="formatTimestamp(r.run.created_at)">
                {{ formatRelative(r.run.created_at, now) }}
              </td>
              <td class="num px-3 py-3 text-right text-fg-muted">{{ formatDuration(r.duration) }}</td>
              <td class="px-4 py-3 text-right">
                <NuxtLink v-if="r.meta.watchable" :to="`/runs/${r.run.run_id}`" class="btn-ghost btn-sm whitespace-nowrap" @click.stop>
                  ▶ Watch
                </NuxtLink>
              </td>
            </tr>
            <tr v-if="visible.length === 0">
              <td colspan="11" class="px-4 py-12 text-center text-fg-subtle">
                <template v-if="rows.length === 0">
                  No runs yet -- start one with <code class="chip">uv run python jobs/baseline_gp_run.py</code>.
                </template>
                <template v-else>No runs match these filters.</template>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Cards (< lg) -->
      <div class="mt-4 grid gap-3 sm:grid-cols-2 lg:hidden">
        <NuxtLink v-for="r in visible" :key="r.run.run_id" :to="`/runs/${r.run.run_id}`" class="card card-hover block p-4">
          <div class="flex items-start justify-between gap-3">
            <div class="min-w-0">
              <p class="truncate font-medium">{{ r.meta.title }}</p>
              <p class="font-mono text-[11px] text-fg-subtle">{{ shortId(r.run.run_id) }} · {{ formatRelative(r.run.created_at, now) }}</p>
            </div>
            <StatusBadge :status="r.run.status" :stale="r.stale" />
          </div>
          <Sparkline :values="r.trend" class="mt-3 h-10 w-full" />
          <dl class="mt-3 grid grid-cols-3 gap-2 text-xs">
            <div><dt class="text-fg-subtle">best</dt><dd class="num text-fg">{{ formatFitness(r.best, 2) }}</dd></div>
            <div>
              <dt class="text-fg-subtle">gens</dt>
              <dd class="num text-fg">{{ r.generations }}<span v-if="r.meta.targetGenerations" class="text-fg-subtle">/{{ r.meta.targetGenerations }}</span></dd>
            </div>
            <div><dt class="text-fg-subtle">duration</dt><dd class="num text-fg">{{ formatDuration(r.duration) }}</dd></div>
          </dl>
          <p v-if="r.meta.selection" class="mt-3 flex flex-wrap gap-1.5">
            <span class="chip">{{ r.meta.selection }}</span>
            <span v-if="r.meta.populationSize" class="chip">pop {{ r.meta.populationSize }}</span>
          </p>
        </NuxtLink>
      </div>
    </template>
  </main>
</template>
