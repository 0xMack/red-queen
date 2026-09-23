<script setup lang="ts">
import type { EvaluationRecord } from "~/types/leaderboard"
import type { ModelShape } from "~/utils/modelLabel"
import type { RunRow } from "~/utils/runMeta"

useHead({ title: "Runs" })

// One request for every run's list-level numbers (`GET /runs/summaries`), reused for a minute: coming back to this
// page doesn't refetch, and runs still training are polled below instead.
const runsStore = useRunsStore()
await useAsyncData("runs", () => runsStore.ensureLoaded().then(() => true))
const api = useApi()

// Where each run's champion stands on its game's leaderboard (docs/design/0007) -- the held-out score,
// which is what the leaderboard ranks by, next to the training fitness this table has always shown.
const { data: leaderboard } = await useAsyncData("runs-leaderboard-snake", () =>
  api.fetch<EvaluationRecord[]>("/games/snake/leaderboard").catch(() => []),
)
const standings = computed(() => {
  const records = leaderboard.value ?? []
  const protocol = latestProtocol(records)
  const ranked = records.filter((r) => r.protocol === protocol)
  const byRun: Record<string, { mean: number; rank: number; of: number; entrantId: string }> = {}
  ranked.forEach((r, i) => {
    // The champion itself, not a differently-behaving package variant (run:<id>@fp32).
    if (r.run_id && r.entrant_id === `run:${r.run_id}`) byRun[r.run_id] = { mean: r.metrics.quality.mean, rank: i + 1, of: ranked.length, entrantId: r.entrant_id }
  })
  return { byRun, protocol }
})

// useState, not ref: the server-rendered "3 min ago" and the first client render must agree
// (hydration), so both use the server's clock until the interval below ticks.
const now = useState("clock:now", () => Date.now() / 1000)
let clock: ReturnType<typeof setInterval> | null = null
let poll: ReturnType<typeof setInterval> | null = null
onMounted(() => {
  now.value = Date.now() / 1000 // after hydration; also refreshes a clock left over from an earlier page
  clock = setInterval(() => (now.value = Date.now() / 1000), 30_000)
  // Only a run that's still training changes: poll (one small request) while one is, and not otherwise.
  poll = setInterval(() => {
    if (runsStore.runs.some((r) => (r.status === "running" || r.status === "paused") && !isStale(r, now.value))) {
      runsStore.fetchRuns()
    }
  }, 15_000)
})
onUnmounted(() => {
  if (clock) clearInterval(clock)
  if (poll) clearInterval(poll)
})

const rows = computed<RunRow[]>(() =>
  runsStore.runs.map((run) => {
    const meta = describeRun(run)
    const summary = runsStore.summaries[run.run_id]
    const c = run.config ?? {}
    const genome = meta.network
      ? `${meta.network}${meta.parameterCount ? ` · ${meta.parameterCount}w` : ""}`
      : typeof c.num_instructions === "number"
        ? `${c.num_instructions} instr · ${c.num_registers ?? "?"} regs`
        : "--"
    const last = summary?.last ?? null
    const extras = last?.extras ?? {}
    const shape: ModelShape = {
      algorithm: meta.representationLabel,
      hidden_nodes: typeof extras.champion_hidden_nodes === "number" ? extras.champion_hidden_nodes : undefined,
      connections: typeof extras.champion_connections === "number" ? extras.champion_connections : undefined,
      layer_sizes: Array.isArray(c.layer_sizes) ? (c.layer_sizes as number[]) : undefined,
    }
    const subject = meta.title.split(" · ")[0]
    return {
      run,
      meta,
      label: meta.game ? `${subject} · ${modelLabel(shape)}` : meta.title,
      heldOut: standings.value.byRun[run.run_id] ?? null,
      generations: summary?.generations ?? 0,
      best: bestFitnessOf(run) ?? summary?.best_fitness ?? null,
      trend: summary?.trend ?? [],
      // For a live run the last recorded generation is fresher than the registry row.
      duration: Math.max(run.updated_at, last?.timestamp ?? 0) - run.created_at,
      stale: isStale(run, now.value),
      genome,
      experiment: typeof c.experiment === "string" && c.experiment ? c.experiment : null,
      arm: typeof c.arm === "string" && c.arm ? c.arm : null,
    }
  }),
)

// Filters / sorting --------------------------------------------------------------------------------
const query = ref("")
const statusFilter = ref<"all" | "running" | "completed" | "failed" | "paused">("all")
const kindFilter = ref("all")
type SortKey = "created" | "best" | "heldOut" | "generations" | "duration"
const sortKey = ref<SortKey>("created")
const sortDesc = ref(true)

const kinds = computed(() => [...new Set(rows.value.map((r) => r.meta.representationLabel))])

function sortValue(r: RunRow): number {
  switch (sortKey.value) {
    case "created":
      return r.run.created_at
    case "best":
      return r.best ?? -Infinity
    case "heldOut":
      return r.heldOut?.mean ?? -Infinity
    case "generations":
      return r.generations
    default:
      return r.duration
  }
}
const bySort = (a: number, b: number) => (sortDesc.value ? b - a : a - b)

const matching = computed(() => {
  const q = query.value.trim().toLowerCase()
  return rows.value
    .filter((r) => {
      if (statusFilter.value !== "all" && r.run.status !== statusFilter.value) return false
      if (kindFilter.value !== "all" && r.meta.representationLabel !== kindFilter.value) return false
      if (!q) return true
      return [r.run.run_id, r.label, r.meta.title, r.meta.note, r.meta.selection, r.meta.benchmark, r.experiment, r.arm]
        .filter(Boolean)
        .some((s) => s!.toLowerCase().includes(q))
    })
    .sort((a, b) => bySort(sortValue(a), sortValue(b)))
})

// Experiment groups -------------------------------------------------------------------------------------
// The runs of one comparison (jobs/snake_experiment.py and friends: arms x seeds, tagged config.experiment) are one
// collapsible row: they're read together, and there are already dozens of them per experiment.
interface RunGroup {
  experiment: string
  rows: RunRow[] // the matching runs, sorted
  total: number // every run of the experiment
  best: number | null
  generations: number
  latest: number
  status: RunRow["run"]["status"]
  statusCounts: string
  arms: string[]
  algorithms: string[]
}
type Item = { kind: "run"; row: RunRow } | { kind: "group"; group: RunGroup }

const totals = computed(() => {
  const counts: Record<string, number> = {}
  for (const r of rows.value) if (r.experiment) counts[r.experiment] = (counts[r.experiment] ?? 0) + 1
  return counts
})

function group(experiment: string, members: RunRow[]): RunGroup {
  const statuses = members.map((r) => r.run.status)
  const count = (s: string) => statuses.filter((x) => x === s).length
  const bests = members.map((r) => r.best).filter((b): b is number => b !== null)
  return {
    experiment,
    rows: members,
    total: totals.value[experiment] ?? members.length,
    best: bests.length ? Math.max(...bests) : null,
    generations: members.reduce((sum, r) => sum + r.generations, 0),
    latest: Math.max(...members.map((r) => r.run.created_at)),
    // the state that most needs attention wins
    status: count("running") ? "running" : count("paused") ? "paused" : count("failed") ? "failed" : "completed",
    statusCounts: (["running", "paused", "failed", "completed"] as const)
      .filter((s) => count(s))
      .map((s) => `${count(s)} ${s}`)
      .join(" · "),
    arms: [...new Set(members.map((r) => r.arm).filter((a): a is string => a !== null))],
    algorithms: [...new Set(members.map((r) => r.meta.representationLabel))],
  }
}

const items = computed<Item[]>(() => {
  const groups = new Map<string, RunRow[]>()
  const out: Item[] = []
  for (const r of matching.value) {
    if (!r.experiment) out.push({ kind: "run", row: r })
    else if (groups.has(r.experiment)) groups.get(r.experiment)!.push(r)
    else groups.set(r.experiment, [r])
  }
  for (const [experiment, members] of groups) out.push({ kind: "group", group: group(experiment, members) })
  // A group sorts by its most extreme member in the current direction: its first (already sorted) row.
  const key = (item: Item) => sortValue(item.kind === "run" ? item.row : item.group.rows[0]!)
  return out.sort((a, b) => bySort(key(a), key(b)))
})

const expanded = ref(new Set<string>())
function toggle(experiment: string) {
  const next = new Set(expanded.value)
  if (next.has(experiment)) next.delete(experiment)
  else next.add(experiment)
  expanded.value = next
}
// While searching, show the matching runs inside their groups rather than hiding them behind a click.
const isOpen = (experiment: string) => expanded.value.has(experiment) || query.value.trim() !== ""

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
    experiments: Object.keys(totals.value).length,
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
      <button class="btn-ghost btn-sm" @click="runsStore.fetchRuns()">
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
        <StatTile label="Runs" :value="summary.total" :hint="`${summary.kinds} algorithm families · ${summary.experiments} experiments`" />
        <StatTile label="Running now" :value="summary.running" :tone="summary.running ? 'life' : 'default'" hint="live over SSE" />
        <StatTile label="Generations recorded" :value="summary.totalGens.toLocaleString()" hint="across every run" />
        <StatTile label="Best Snake fitness" :value="formatFitness(summary.bestSnake, 2)" tone="queen" hint="training fitness, not a game score" />
        <StatTile
          class="col-span-2 md:col-span-1"
          label="Latest run"
          :value="rows[0] ? formatRelative(rows[0].run.created_at, now) : '--'"
          :hint="rows[0]?.label"
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
        <span class="ml-auto text-xs text-fg-subtle">{{ matching.length }} of {{ rows.length }} runs</span>
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
                <span title="Best training fitness: shaped reward on the games it trained on -- not a game score">
                  Best train fitness {{ sortKey === "best" ? (sortDesc ? "↓" : "↑") : "" }}
                </span>
              </th>
              <th class="cursor-pointer px-3 py-3 text-right font-medium hover:text-fg" @click="sortBy('heldOut')">
                <span :title="`Mean game score on the leaderboard's held-out games (${standings.protocol ?? 'no evaluations yet'}) -- what the leaderboard ranks by`">
                  Held-out {{ sortKey === "heldOut" ? (sortDesc ? "↓" : "↑") : "" }}
                </span>
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
            <template v-for="item in items" :key="item.kind === 'run' ? item.row.run.run_id : `exp:${item.group.experiment}`">
              <RunsTableRow v-if="item.kind === 'run'" :r="item.row" :now="now" />
              <template v-else>
                <tr
                  class="cursor-pointer border-b border-line/60 bg-surface/40 transition hover:bg-raised/60"
                  :aria-expanded="isOpen(item.group.experiment)"
                  data-experiment-group
                  @click="toggle(item.group.experiment)"
                >
                  <td class="px-4 py-3">
                    <div class="flex items-center gap-2">
                      <span class="inline-block w-3 text-fg-subtle transition" :class="isOpen(item.group.experiment) ? 'rotate-90' : ''">▶</span>
                      <div>
                        <p class="font-medium text-fg">{{ item.group.experiment }}</p>
                        <p class="mt-0.5 text-[11px] text-fg-subtle">
                          experiment · {{ item.group.rows.length }}<template v-if="item.group.rows.length !== item.group.total"> of {{ item.group.total }}</template> runs
                        </p>
                      </div>
                    </div>
                  </td>
                  <td class="px-3 py-3">
                    <StatusBadge :status="item.group.status" />
                    <p class="mt-1 text-[10px] whitespace-nowrap text-fg-subtle">{{ item.group.statusCounts }}</p>
                  </td>
                  <td class="px-3 py-3">
                    <div class="flex max-w-56 flex-wrap gap-1">
                      <span v-for="arm in item.group.arms.slice(0, 4)" :key="arm" class="chip">{{ arm }}</span>
                      <span v-if="item.group.arms.length > 4" class="text-[11px] text-fg-subtle">+{{ item.group.arms.length - 4 }} arms</span>
                    </div>
                  </td>
                  <td class="num px-3 py-3 text-right text-fg-subtle">--</td>
                  <td class="px-3 py-3 text-xs text-fg-muted">{{ item.group.algorithms.join(" · ") }}</td>
                  <td class="num px-3 py-3 text-xs text-fg-muted">{{ item.group.generations.toLocaleString() }} in all</td>
                  <td class="num px-3 py-3 text-right text-fg-muted">{{ formatFitness(item.group.best, 3) }}</td>
                  <td class="px-3 py-3 text-right">
                    <span class="cursor-help text-fg-subtle" title="Comparison runs are aggregated in their experiment, not ranked individually">--</span>
                  </td>
                  <td class="px-3 py-3" />
                  <td class="px-3 py-3 whitespace-nowrap text-fg-muted" :title="`latest run ${formatTimestamp(item.group.latest)}`">
                    {{ formatRelative(item.group.latest, now) }}
                  </td>
                  <td class="px-3 py-3" />
                  <td class="px-4 py-3 text-right">
                    <button class="btn-ghost btn-sm whitespace-nowrap" @click.stop="toggle(item.group.experiment)">
                      {{ isOpen(item.group.experiment) ? "Hide" : "Show" }} {{ item.group.rows.length }}
                    </button>
                  </td>
                </tr>
                <template v-if="isOpen(item.group.experiment)">
                  <RunsTableRow v-for="r in item.group.rows" :key="r.run.run_id" :r="r" :now="now" nested />
                </template>
              </template>
            </template>
            <tr v-if="matching.length === 0">
              <td colspan="12" class="px-4 py-12 text-center text-fg-subtle">
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
        <template v-for="item in items" :key="item.kind === 'run' ? item.row.run.run_id : `exp:${item.group.experiment}`">
          <RunsCard v-if="item.kind === 'run'" :r="item.row" :now="now" />
          <template v-else>
            <button
              class="card card-hover block p-4 text-left"
              :aria-expanded="isOpen(item.group.experiment)"
              data-experiment-group
              @click="toggle(item.group.experiment)"
            >
              <div class="flex items-start justify-between gap-3">
                <div class="min-w-0">
                  <p class="truncate font-medium">
                    <span class="inline-block w-3 text-fg-subtle transition" :class="isOpen(item.group.experiment) ? 'rotate-90' : ''">▶</span>
                    {{ item.group.experiment }}
                  </p>
                  <p class="text-[11px] text-fg-subtle">experiment · {{ item.group.total }} runs · {{ item.group.algorithms.join(" · ") }}</p>
                </div>
                <StatusBadge :status="item.group.status" />
              </div>
              <dl class="mt-3 grid grid-cols-3 gap-2 text-xs">
                <div><dt class="text-fg-subtle">best</dt><dd class="num text-fg">{{ formatFitness(item.group.best, 2) }}</dd></div>
                <div><dt class="text-fg-subtle">gens</dt><dd class="num text-fg">{{ item.group.generations.toLocaleString() }}</dd></div>
                <div><dt class="text-fg-subtle">latest</dt><dd class="num text-fg">{{ formatRelative(item.group.latest, now) }}</dd></div>
              </dl>
            </button>
            <template v-if="isOpen(item.group.experiment)">
              <RunsCard v-for="r in item.group.rows" :key="r.run.run_id" :r="r" :now="now" class="border-l-2 border-l-queen-400/40" />
            </template>
          </template>
        </template>
      </div>
    </template>
  </main>
</template>
