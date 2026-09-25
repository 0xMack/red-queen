<script setup lang="ts">
import type { ChartSeries } from "~/types/chart"
import { probeDevice } from "~/inference/device"
import recording from "~/data/recordings/dqn-snake.json"

// A DQN, live: the Rust RL core (WebAssembly, in a worker) training a Q-network on Snake in the reader's browser --
// the same `DqnAgent` the training jobs run (docs/design/0010 Phase 2b). Choose what the snake sees and which
// stabilizers are on; the curve is the greedy policy's score on 30 unseen games every 10k steps, drawn over the
// previous run's so two settings can be compared; the second chart is the network's mean Q(s, a), which is where a
// diverging run shows first. Without WebAssembly, recorded runs of the same setup (jobs/export_dqn_recording.py)
// stand in.

const GREEDY = 17.89
const TABLE = 19.51 // best Q-table on the leaderboard
const ACTIONS = ["turn left", "straight", "turn right"]
const BUDGET = 200_000
const EVAL_EVERY = 10_000
const SPEEDS = [
  { label: "watch", stepsPerSecond: 3_000 },
  { label: "fast", stepsPerSecond: 400_000 }, // as fast as this device trains (~9k steps/s on a desktop)
] as const
const OBSERVERS = [
  { id: "egocentric.v1", label: "egocentric.v1 (27: rays, food, tail)" },
  { id: "features.v1", label: "features.v1 (11: the Q-table's)" },
  { id: "grid-flat.v1", label: "grid-flat.v1 (100: every cell)" },
]

const config = reactive({ observer: "egocentric.v1", replay: true, target: true, double: false, dueling: false, nStep3: false, seed: 0 })
const lab = useRlLab({ speed: SPEEDS[1].stepsPerSecond })
const live = ref<boolean | null>(null) // null while probing
onMounted(async () => {
  live.value = (await probeDevice()).wasm
})

// what the current run was started with, and the finished run before it, for the overlay
const running = ref<{ label: string } | null>(null)
const previous = shallowRef<{ label: string; points: CurvePoint[] } | null>(null)

function describe(): string {
  const on = [config.replay && "replay", config.target && "target", config.double && "double", config.dueling && "dueling", config.nStep3 && "3-step"]
  const pieces = on.filter(Boolean).join(" + ") || "naive"
  return `${config.observer.replace(".v1", "")} · ${pieces} · seed ${config.seed}`
}

function train() {
  if (running.value && lab.points.value.length > 1) previous.value = { label: running.value.label, points: lab.points.value }
  const params: string[] = []
  if (!config.replay) params.push("replay_capacity=0")
  if (!config.target) params.push("target_update=0")
  if (config.double) params.push("double=1")
  if (config.dueling) params.push("dueling=1")
  if (config.nStep3) params.push("n_step=3")
  running.value = { label: describe() }
  lab.start({ algorithm: "dqn", observer: config.observer, params: params.join(","), reward: "shaped", seed: config.seed, budget: BUDGET, evalEvery: EVAL_EVERY })
}

const started = computed(() => lab.status.value !== "idle")
const grid = Array.from({ length: BUDGET / EVAL_EVERY + 1 }, (_, i) => i * EVAL_EVERY)

// Live: this run and the one before. Without WebAssembly: two recorded 1M-step runs, egocentric vs. features.
const scoreChart = computed<{ x: number[]; series: ChartSeries[] }>(() => {
  if (live.value === false) {
    const ego = recording.runs.egocentric.score
    const feat = recording.runs.features.score
    return {
      x: ego.map((p) => p[0]!),
      series: [
        { key: "ego", label: recording.runs.egocentric.label, color: "#ff5c7a", values: ego.map((p) => p[1]!), width: 2.5 },
        { key: "feat", label: recording.runs.features.label, color: "#60a5fa", values: feat.map((p) => p[1]!), width: 2 },
        { key: "greedy", label: `greedy (${GREEDY})`, color: "#fbbf24", values: ego.map(() => GREEDY), width: 1, dashed: true },
      ],
    }
  }
  const series: ChartSeries[] = [
    { key: "now", label: running.value?.label ?? "this run", color: "#ff5c7a", values: lab.points.value.map((p) => p.score), width: 2.5 },
  ]
  if (previous.value) series.push({ key: "before", label: previous.value.label, color: "#60a5fa", values: previous.value.points.map((p) => p.score), width: 1.5 })
  series.push(
    { key: "greedy", label: `greedy (${GREEDY})`, color: "#fbbf24", values: grid.map(() => GREEDY), width: 1, dashed: true },
    { key: "table", label: `best Q-table (${TABLE})`, color: "#4ade80", values: grid.map(() => TABLE), width: 1, dashed: true },
  )
  return { x: grid, series }
})

// Q on a symmetric log scale, so a diverging run stays on the chart; ticks are labeled in real units.
const symlog = (q: number) => Math.sign(q) * Math.log10(1 + Math.abs(q))
const unsymlog = (v: number) => Math.sign(v) * (10 ** Math.abs(v) - 1)
const formatQ = (v: number) => {
  const q = unsymlog(v)
  return Math.abs(q) >= 1000 ? q.toExponential(0) : Math.abs(q) >= 10 ? q.toFixed(0) : q.toFixed(1)
}
// From the first evaluation on (step 0 has had no updates); LineChart can't draw a gap, so no NaN may reach it.
const qChart = computed<{ x: number[]; series: ChartSeries[] }>(() => {
  const qs = (points: CurvePoint[]) => points.slice(1).map((p) => symlog(Math.max(-1e300, Math.min(p.qMean ?? 0, 1e300))))
  const series: ChartSeries[] = [{ key: "now", label: "this run", color: "#ff5c7a", values: qs(lab.points.value), width: 2 }]
  if (previous.value) series.push({ key: "before", label: "run before", color: "#60a5fa", values: qs(previous.value.points), width: 1.5 })
  const n = Math.max(...series.map((x) => x.values.length))
  return { x: grid.slice(1, n + 1), series }
})
const steps = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n))
const bestValue = computed(() => (lab.actionValues.value ? Math.max(...lab.actionValues.value) : 0))
</script>

<template>
  <div class="card not-prose my-6 p-4 sm:p-5" data-dqn-lab>
    <!-- Recorded fallback: no WebAssembly here -->
    <template v-if="live === false">
      <p class="text-sm text-fg-muted">
        This browser can't run WebAssembly, so here are two <strong class="text-fg">recorded runs</strong> instead: the same DQN for a million steps,
        once on each observation.
      </p>
      <div class="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
        <span v-for="s in scoreChart.series" :key="s.key" class="flex items-center gap-1.5"><span class="h-0.5 w-4 rounded" :style="{ background: s.color }" />{{ s.label }}</span>
      </div>
      <LineChart class="mt-1" :x="scoreChart.x" :series="scoreChart.series" :height="200" x-label="env steps" :format="(v: number) => v.toFixed(1)" />
    </template>

    <template v-else>
      <div class="grid gap-5 lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)]">
        <!-- the greedy policy playing -->
        <div class="min-w-0">
          <div class="mx-auto w-full max-w-[320px]">
            <GridBoard v-if="lab.board.value" :state="lab.board.value" :tick-ms="90" />
            <div v-else class="flex aspect-square items-center justify-center rounded-xl border border-dashed border-line text-center text-sm text-fg-subtle">
              {{ live === null ? "Checking this device…" : "Press Train: the network starts with random weights." }}
            </div>
          </div>
          <div class="mt-3 text-xs text-fg-subtle">
            <p>
              The network's <strong class="text-fg">greedy policy</strong> right now
              <template v-if="lab.board.value"> · score <span class="num text-fg">{{ lab.board.value.score }}</span></template>
            </p>
            <div v-if="lab.actionValues.value" class="mt-2 space-y-1">
              <div v-for="(q, i) in lab.actionValues.value" :key="i" class="flex items-center gap-2 font-mono">
                <span class="w-20 shrink-0" :class="i === lab.action.value ? 'text-fg' : ''">{{ ACTIONS[i] }}</span>
                <span class="h-2 rounded-sm" :class="i === lab.action.value ? 'bg-queen-400' : 'bg-raised'" :style="{ width: `${Math.max(2, Math.min(100, (Math.max(q, 0) / Math.max(bestValue, 1e-9)) * 100))}%` }" />
                <span class="num">{{ Math.abs(q) < 1e4 ? q.toFixed(2) : q.toExponential(1) }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- knobs, stats, curves -->
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

          <div class="mt-3 flex flex-wrap items-center gap-2 text-sm">
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
          <div class="mt-2 flex flex-wrap gap-x-4 gap-y-1.5 text-sm text-fg-muted">
            <label class="flex items-center gap-1.5"><input v-model="config.replay" type="checkbox" class="size-4 accent-[#ff5c7a]">replay buffer</label>
            <label class="flex items-center gap-1.5"><input v-model="config.target" type="checkbox" class="size-4 accent-[#ff5c7a]">target network</label>
            <label class="flex items-center gap-1.5"><input v-model="config.double" type="checkbox" class="size-4 accent-[#ff5c7a]">Double DQN</label>
            <label class="flex items-center gap-1.5"><input v-model="config.dueling" type="checkbox" class="size-4 accent-[#ff5c7a]">dueling heads</label>
            <label class="flex items-center gap-1.5"><input v-model="config.nStep3" type="checkbox" class="size-4 accent-[#ff5c7a]">3-step returns</label>
          </div>
          <p class="mt-1 text-[11px] text-fg-subtle">Changes apply when you press {{ started ? "Train again" : "Train" }}; the last run stays on the chart to compare.</p>

          <dl class="mt-3 grid grid-cols-3 gap-2 text-xs">
            <div><dt class="text-fg-subtle">env steps</dt><dd class="num text-fg">{{ steps(lab.totalSteps.value) }} / {{ steps(BUDGET) }}</dd></div>
            <div><dt class="text-fg-subtle">episodes</dt><dd class="num text-fg">{{ lab.totalEpisodes.value.toLocaleString() }}</dd></div>
            <div><dt class="text-fg-subtle">ε</dt><dd class="num text-fg">{{ lab.epsilon.value === null ? "--" : lab.epsilon.value.toFixed(3) }}</dd></div>
          </dl>

          <div class="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
            <span v-for="s in scoreChart.series" :key="s.key" class="flex items-center gap-1.5"><span class="h-0.5 w-4 rounded" :style="{ background: s.color }" />{{ s.label }}</span>
          </div>
          <LineChart v-if="lab.points.value.length > 1 || previous" class="mt-1" :x="scoreChart.x" :series="scoreChart.series" :height="170" x-label="env steps" :format="(v: number) => v.toFixed(1)" />
          <p v-else class="mt-2 rounded-lg border border-dashed border-line p-6 text-center text-xs text-fg-subtle">
            The mean score on 30 unseen games, every 10k steps, will draw here.
          </p>
          <template v-if="lab.points.value.length > 2">
            <p class="mt-3 text-xs text-fg-subtle">the network's mean Q(s, a) on its training batches (log scale: a diverging run leaves the chart's usual range)</p>
            <LineChart class="mt-1" :x="qChart.x" :series="qChart.series" :height="110" x-label="env steps" :format="formatQ" />
          </template>
          <p v-if="lab.status.value === 'done'" class="mt-2 text-xs text-life-300">Done: {{ steps(lab.totalSteps.value) }} steps. Change what the snake sees, or a stabilizer, and train again.</p>
          <p v-if="lab.error.value" class="mt-2 text-xs text-queen-300">{{ lab.error.value }}</p>
        </div>
      </div>
    </template>
  </div>
</template>
