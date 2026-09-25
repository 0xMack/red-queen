<script setup lang="ts">
import type { ChartSeries } from "~/types/chart"
import { probeDevice } from "~/inference/device"
import recording from "~/data/recordings/pg-snake.json"

// Policy gradients, live on Snake: the Rust RL core (WebAssembly, in a worker) training a policy network in the
// reader's browser -- the same `PgAgent` the training jobs run (docs/design/0010 Phase 3b). Pick the rung of the
// ladder, what the snake sees and a seed; the board shows the policy's own move probabilities as it plays (the bars),
// the curve its score on 30 unseen games every 20k steps (over the previous run's), and the second chart its entropy:
// how undecided it still is. Without WebAssembly, recorded runs of each rung (jobs/export_rl_curves.py) stand in.

const GREEDY = 17.89
const NEAT = 37.95 // leaderboard's best evolved champion
const ACTIONS = ["turn left", "straight", "turn right"]
const BUDGET = 400_000
const EVAL_EVERY = 20_000
const SPEEDS = [
  { label: "watch", stepsPerSecond: 2_000 },
  { label: "fast", stepsPerSecond: 400_000 }, // as fast as this device trains (~6.5k steps/s on a desktop)
] as const
const RUNGS = [
  { id: "reinforce", label: "REINFORCE", algorithm: "reinforce", params: "" },
  { id: "baseline", label: "REINFORCE + baseline", algorithm: "reinforce", params: "baseline=1" },
  { id: "a2c", label: "A2C (actor-critic)", algorithm: "a2c", params: "" },
  { id: "ppo", label: "PPO", algorithm: "ppo", params: "" },
] as const
const OBSERVERS = [
  { id: "egocentric.v2", label: "egocentric.v2 (33: rays + reachable space)" },
  { id: "egocentric.v1", label: "egocentric.v1 (27: rays, food, tail)" },
  { id: "features.v1", label: "features.v1 (11: the Q-table's)" },
]

const config = reactive({ rung: "ppo" as (typeof RUNGS)[number]["id"], observer: "egocentric.v2", seed: 0 })
const lab = useRlLab({ speed: SPEEDS[1].stepsPerSecond })
const live = ref<boolean | null>(null)
onMounted(async () => {
  live.value = (await probeDevice()).wasm
})

const running = ref<{ label: string } | null>(null)
const previous = shallowRef<{ label: string; points: CurvePoint[] } | null>(null)
const rung = computed(() => RUNGS.find((r) => r.id === config.rung)!)

function train() {
  if (running.value && lab.points.value.length > 1) previous.value = { label: running.value.label, points: lab.points.value }
  running.value = { label: `${rung.value.label} · ${config.observer.replace(".v1", "").replace(".v2", " v2")} · seed ${config.seed}` }
  lab.start({
    algorithm: rung.value.algorithm,
    env: `snake/${config.observer}+relative3.v1`,
    params: rung.value.params,
    reward: "shaped",
    seed: config.seed,
    budget: BUDGET,
    evalEvery: EVAL_EVERY,
  })
}

const started = computed(() => lab.status.value !== "idle")
const grid = Array.from({ length: BUDGET / EVAL_EVERY + 1 }, (_, i) => i * EVAL_EVERY)

// Without WebAssembly: one recorded 2M-step run per rung, same seed.
const RECORDED = [
  { key: "reinforce", color: "#60a5fa" },
  { key: "baseline", color: "#a78bfa" },
  { key: "a2c", color: "#4ade80" },
  { key: "ppo", color: "#ff5c7a" },
  { key: "ppo-ego2", color: "#fbbf24" },
] as const

const scoreChart = computed<{ x: number[]; series: ChartSeries[] }>(() => {
  if (live.value === false) {
    const x = recording.runs.ppo.score.map((p) => p[0]!)
    return {
      x,
      series: RECORDED.map((r) => ({
        key: r.key,
        label: recording.runs[r.key].label,
        color: r.color,
        values: recording.runs[r.key].score.map((p) => p[1]!),
        width: 2,
      })),
    }
  }
  const series: ChartSeries[] = [
    { key: "now", label: running.value?.label ?? "this run", color: "#ff5c7a", values: lab.points.value.map((p) => p.score), width: 2.5 },
  ]
  if (previous.value) series.push({ key: "before", label: previous.value.label, color: "#60a5fa", values: previous.value.points.map((p) => p.score), width: 1.5 })
  series.push(
    { key: "greedy", label: `greedy (${GREEDY})`, color: "#fbbf24", values: grid.map(() => GREEDY), width: 1, dashed: true },
    { key: "neat", label: `best evolved, NEAT (${NEAT})`, color: "#a78bfa", values: grid.map(() => NEAT), width: 1, dashed: true },
  )
  return { x: grid, series }
})

// Entropy from the first evaluation on (the untrained policy has no update behind it yet); no NaN may reach LineChart.
const entropyChart = computed<{ x: number[]; series: ChartSeries[] }>(() => {
  const values = (points: CurvePoint[]) => points.slice(1).map((p) => p.entropy ?? Math.log(3))
  const series: ChartSeries[] = [{ key: "now", label: "this run", color: "#ff5c7a", values: values(lab.points.value), width: 2 }]
  if (previous.value) series.push({ key: "before", label: "run before", color: "#60a5fa", values: values(previous.value.points), width: 1.5 })
  const n = Math.max(...series.map((s) => s.values.length))
  series.push({ key: "uniform", label: "uniform (ln 3)", color: "#6b7489", values: grid.slice(1, n + 1).map(() => Math.log(3)), width: 1, dashed: true })
  return { x: grid.slice(1, n + 1), series }
})
const steps = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n))
</script>

<template>
  <div class="card not-prose my-6 p-4 sm:p-5" data-policy-gradient-lab>
    <!-- Recorded fallback: no WebAssembly here -->
    <template v-if="live === false">
      <p class="text-sm text-fg-muted">
        This browser can't run WebAssembly, so here are <strong class="text-fg">recorded runs</strong> instead: one per rung of the ladder, same seed,
        two million moves each.
      </p>
      <div class="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
        <span v-for="s in scoreChart.series" :key="s.key" class="flex items-center gap-1.5"><span class="h-0.5 w-4 rounded" :style="{ background: s.color }" />{{ s.label }}</span>
      </div>
      <LineChart class="mt-1" :x="scoreChart.x" :series="scoreChart.series" :height="220" x-label="env steps" :format="(v: number) => v.toFixed(1)" />
    </template>

    <template v-else>
      <div class="grid gap-5 lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)]">
        <div class="min-w-0">
          <div class="mx-auto w-full max-w-[320px]">
            <GridBoard v-if="lab.board.value" :state="lab.board.value" :tick-ms="90" />
            <div v-else class="flex aspect-square items-center justify-center rounded-xl border border-dashed border-line text-center text-sm text-fg-subtle">
              {{ live === null ? "Checking this device…" : "Press Train: the policy starts out choosing at random." }}
            </div>
          </div>
          <div class="mt-3 text-xs text-fg-subtle">
            <p>
              The policy's <strong class="text-fg">most likely move</strong>, played
              <template v-if="lab.board.value"> · score <span class="num text-fg">{{ lab.board.value.score }}</span></template>
            </p>
            <div v-if="lab.actionValues.value" class="mt-2 space-y-1">
              <div v-for="(p, i) in lab.actionValues.value" :key="i" class="flex items-center gap-2 font-mono">
                <span class="w-20 shrink-0" :class="i === lab.action.value ? 'text-fg' : ''">{{ ACTIONS[i] }}</span>
                <span class="h-2 rounded-sm" :class="i === lab.action.value ? 'bg-queen-400' : 'bg-raised'" :style="{ width: `${Math.max(2, p * 100)}%` }" />
                <span class="num">{{ (p * 100).toFixed(0) }}%</span>
              </div>
            </div>
          </div>
        </div>

        <div class="min-w-0">
          <div class="flex flex-wrap items-center gap-2">
            <button v-if="lab.status.value === 'running'" class="btn-primary btn-sm" @click="lab.pause">Pause</button>
            <button v-else-if="lab.status.value === 'paused'" class="btn-primary btn-sm" @click="lab.resume">Resume</button>
            <button class="btn-sm" :class="lab.status.value === 'running' || lab.status.value === 'paused' ? 'btn-ghost' : 'btn-primary'" :disabled="live !== true" @click="train">
              {{ started ? "Train again" : "Train" }}
            </button>
            <div class="ml-auto flex rounded-lg border border-line bg-surface p-0.5" role="group" aria-label="training speed">
              <button
                v-for="s in SPEEDS"
                :key="s.label"
                class="rounded-md px-2.5 py-1 text-xs font-medium transition"
                :class="lab.speed.value === s.stepsPerSecond ? 'bg-raised text-fg shadow' : 'text-fg-subtle hover:text-fg'"
                @click="lab.speed.value = s.stepsPerSecond"
              >
                {{ s.label }}
              </button>
            </div>
          </div>
          <div class="mt-3 grid gap-2 text-sm">
            <label class="flex items-center gap-2">
              <span class="w-24 shrink-0 text-fg-muted">algorithm</span>
              <select v-model="config.rung" class="min-w-0 flex-1 rounded-md border border-line bg-surface px-2 py-1 text-xs">
                <option v-for="r in RUNGS" :key="r.id" :value="r.id">{{ r.label }}</option>
              </select>
            </label>
            <div class="flex flex-wrap items-center gap-2">
              <label class="flex min-w-0 flex-1 items-center gap-2">
                <span class="w-24 shrink-0 text-fg-muted">the snake sees</span>
                <select v-model="config.observer" class="min-w-0 flex-1 rounded-md border border-line bg-surface px-2 py-1 text-xs">
                  <option v-for="o in OBSERVERS" :key="o.id" :value="o.id">{{ o.label }}</option>
                </select>
              </label>
              <label class="flex items-center gap-2">
                <span class="text-fg-muted">seed</span>
                <select v-model.number="config.seed" class="rounded-md border border-line bg-surface px-2 py-1 text-xs" aria-label="random seed">
                  <option v-for="s in 10" :key="s" :value="s - 1">{{ s - 1 }}</option>
                </select>
              </label>
            </div>
          </div>
          <p class="mt-1 text-[11px] text-fg-subtle">Changes apply when you press {{ started ? "Train again" : "Train" }}; the last run stays on the chart to compare.</p>

          <dl class="mt-3 grid grid-cols-2 gap-2 text-xs">
            <div><dt class="text-fg-subtle">env steps</dt><dd class="num text-fg">{{ steps(lab.totalSteps.value) }} / {{ steps(BUDGET) }}</dd></div>
            <div><dt class="text-fg-subtle">episodes</dt><dd class="num text-fg">{{ lab.totalEpisodes.value.toLocaleString() }}</dd></div>
          </dl>

          <div class="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
            <span v-for="s in scoreChart.series" :key="s.key" class="flex items-center gap-1.5"><span class="h-0.5 w-4 rounded" :style="{ background: s.color }" />{{ s.label }}</span>
          </div>
          <LineChart v-if="lab.points.value.length > 1 || previous" class="mt-1" :x="scoreChart.x" :series="scoreChart.series" :height="170" x-label="env steps" :format="(v: number) => v.toFixed(1)" />
          <p v-else class="mt-2 rounded-lg border border-dashed border-line p-6 text-center text-xs text-fg-subtle">
            The mean score on 30 unseen games, every 20k steps, will draw here.
          </p>
          <template v-if="lab.points.value.length > 2">
            <p class="mt-3 text-xs text-fg-subtle">the policy's entropy (nats): how undecided it still is, from ln 3 (a coin with three sides) toward 0</p>
            <LineChart class="mt-1" :x="entropyChart.x" :series="entropyChart.series" :height="100" x-label="env steps" :format="(v: number) => v.toFixed(2)" />
          </template>
          <p v-if="lab.status.value === 'done'" class="mt-2 text-xs text-life-300">Done: {{ steps(lab.totalSteps.value) }} steps. Try another rung, or what the snake sees, and train again.</p>
          <p v-if="lab.error.value" class="mt-2 text-xs text-queen-300">{{ lab.error.value }}</p>
        </div>
      </div>
    </template>
  </div>
</template>
