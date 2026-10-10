<script setup lang="ts">
import type { EvaluationRecord } from "~/types/leaderboard"
import type { Standing } from "~/composables/useRunsTable"
import { RUN_STATUSES } from "~/composables/useRunsTable"

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
  const byRun: Record<string, Standing> = {}
  ranked.forEach((r, i) => {
    // The champion itself, not a differently-behaving package variant (run:<id>@fp32).
    if (r.run_id && r.entrant_id === `run:${r.run_id}`) byRun[r.run_id] = { mean: r.metrics.quality.mean, rank: i + 1, of: ranked.length, entrantId: r.entrant_id }
  })
  return { byRun, protocol, best: ranked[0]?.metrics.quality.mean ?? null }
})

const now = useNow(30)
// Only a run that's still training changes: poll (one small request) while one is, and not otherwise.
let poll: ReturnType<typeof setInterval> | null = null
onMounted(() => {
  poll = setInterval(() => {
    if (runsStore.runs.some((r) => (r.status === "running" || r.status === "paused") && !isStale(r, now.value))) runsStore.fetchRuns()
  }, 15_000)
})
onUnmounted(() => poll && clearInterval(poll))

const { rows, kinds, query, statusFilter, kindFilter, sortKey, sortDesc, sortBy, matching, items, toggle, isOpen, summary } = useRunsTable(standings, now)
</script>

<template>
  <main class="mx-auto max-w-[1600px] px-4 pt-8 pb-10 sm:px-6 lg:px-8">
    <UiSectionHeader :level="1" eyebrow="Telemetry" title="Training runs">
      Every run recorded by <code class="chip">libs/telemetry</code> -- genetic programming, neuroevolution and reinforcement
      learning alike. Open one for its live curves and, for game runs, its champion playing in your browser.
      <template #actions>
        <button class="btn-ghost btn-sm" @click="runsStore.fetchRuns()">
          <svg viewBox="0 0 20 20" class="size-3.5" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <path d="M16 10a6 6 0 1 1-1.8-4.3M16 3.5V6h-2.5" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
          Refresh
        </button>
      </template>
    </UiSectionHeader>

    <UiPanel v-if="runsStore.error" class="mt-8 border-queen-500/40 text-queen-200">
      Couldn't reach the backend: {{ runsStore.error }}.
      <span class="text-fg-muted">Start it with <code class="chip">uv run uvicorn backend.main:app --app-dir apis/backend/src</code>.</span>
    </UiPanel>

    <template v-else>
      <div class="mt-10 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
        <StatTile label="Runs" :value="summary.total" :hint="`${summary.kinds} algorithm families · ${summary.experiments} experiments`" />
        <StatTile label="Running now" :value="summary.running" :tone="summary.running ? 'life' : 'default'" hint="live over SSE" />
        <StatTile label="Generations recorded" :value="summary.totalGens.toLocaleString()" hint="generations and iterations, every run" />
        <StatTile
          label="Best held-out Snake score"
          :value="standings.best !== null ? standings.best.toFixed(2) : '--'"
          tone="queen"
          hint="on the leaderboard's unseen games"
        />
        <StatTile
          class="col-span-2 md:col-span-1"
          label="Latest run"
          :value="rows[0] ? formatRelative(rows[0].run.created_at, now) : '--'"
          :hint="rows[0]?.label"
        />
      </div>

      <!-- Filters -->
      <div class="mt-10 flex flex-wrap items-center gap-3">
        <UiSearch v-model="query" class="min-w-56 flex-1 sm:max-w-sm" placeholder="Filter by id, game, experiment…" />
        <UiSegmented v-model="statusFilter" :options="RUN_STATUSES.map((s) => ({ value: s, label: s }))" aria-label="Status" />
        <select v-model="kindFilter" class="field py-1.5" aria-label="Algorithm">
          <option value="all">All algorithms</option>
          <option v-for="k in kinds" :key="k" :value="k">{{ k }}</option>
        </select>
        <span class="ml-auto font-mono text-xs text-fg-subtle">{{ matching.length }} of {{ rows.length }} runs</span>
      </div>

      <!-- Table (lg+) -->
      <UiPanel class="mt-4 hidden overflow-x-auto lg:block" pad="none">
        <table class="w-full text-sm">
          <thead>
            <tr class="label border-b border-line text-left">
              <th class="px-4 py-3 font-medium">Run</th>
              <th class="px-3 py-3 font-medium">Status</th>
              <th class="px-3 py-3 font-medium">Search</th>
              <th class="px-3 py-3 text-right font-medium">Pop.</th>
              <th class="px-3 py-3 font-medium">Genome</th>
              <UiSortTh column="generations" :active="sortKey" :desc="sortDesc" @sort="sortBy">Generations</UiSortTh>
              <UiSortTh column="best" :active="sortKey" :desc="sortDesc" align="right" title="Best training fitness: shaped reward on the games it trained on -- not a game score" @sort="sortBy">
                Best train fitness
              </UiSortTh>
              <UiSortTh
                column="heldOut"
                :active="sortKey"
                :desc="sortDesc"
                align="right"
                :title="`Mean game score on the leaderboard's held-out games (${standings.protocol ?? 'no evaluations yet'}) -- what the leaderboard ranks by`"
                @sort="sortBy"
              >
                Held-out
              </UiSortTh>
              <th class="px-3 py-3 font-medium">Trend</th>
              <UiSortTh column="created" :active="sortKey" :desc="sortDesc" @sort="sortBy">Started</UiSortTh>
              <UiSortTh column="duration" :active="sortKey" :desc="sortDesc" align="right" @sort="sortBy">Duration</UiSortTh>
              <th class="px-4 py-3" />
            </tr>
          </thead>
          <tbody>
            <template v-for="item in items" :key="item.kind === 'run' ? item.row.run.run_id : `exp:${item.group.experiment}`">
              <RunsTableRow v-if="item.kind === 'run'" :r="item.row" :now="now" />
              <template v-else>
                <RunsExperimentRow :group="item.group" :open="isOpen(item.group.experiment)" :now="now" @toggle="toggle(item.group.experiment)" />
                <template v-if="isOpen(item.group.experiment)">
                  <RunsTableRow v-for="r in item.group.rows" :key="r.run.run_id" :r="r" :now="now" nested />
                </template>
              </template>
            </template>
            <tr v-if="matching.length === 0">
              <td colspan="12" class="px-4 py-14 text-center text-fg-subtle">
                <template v-if="rows.length === 0">
                  No runs yet -- start one with <code class="chip">uv run python -m trainer jobs/trainer/specs/gp.yaml</code>.
                </template>
                <template v-else>No runs match these filters.</template>
              </td>
            </tr>
          </tbody>
        </table>
      </UiPanel>

      <!-- Cards (< lg) -->
      <div class="mt-4 grid gap-3 sm:grid-cols-2 lg:hidden">
        <template v-for="item in items" :key="item.kind === 'run' ? item.row.run.run_id : `exp:${item.group.experiment}`">
          <RunsCard v-if="item.kind === 'run'" :r="item.row" :now="now" />
          <template v-else>
            <RunsExperimentCard :group="item.group" :open="isOpen(item.group.experiment)" :now="now" @toggle="toggle(item.group.experiment)" />
            <template v-if="isOpen(item.group.experiment)">
              <RunsCard v-for="r in item.group.rows" :key="r.run.run_id" :r="r" :now="now" class="border-l-2 border-l-queen-400/40" />
            </template>
          </template>
        </template>
        <UiEmpty v-if="matching.length === 0" class="sm:col-span-2">No runs match these filters.</UiEmpty>
      </div>
    </template>
  </main>
</template>
