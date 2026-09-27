<script setup lang="ts">
import type { ChartSeries } from "~/types/chart"
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
const live = useWasmSupport()

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

const grid = Array.from({ length: BUDGET / EVAL_EVERY + 1 }, (_, i) => i * EVAL_EVERY)

// Without WebAssembly: one recorded 2M-step run per rung, same seed.
const RECORDED = [
  { key: "reinforce", color: palette.signal400 },
  { key: "baseline", color: palette.violet400 },
  { key: "a2c", color: palette.life400 },
  { key: "ppo", color: palette.queen400 },
  { key: "ppo-ego2", color: palette.gold400 },
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
    { key: "now", label: running.value?.label ?? "this run", color: palette.queen400, values: lab.points.value.map((p) => p.score), width: 2.5 },
  ]
  if (previous.value) series.push({ key: "before", label: previous.value.label, color: palette.signal400, values: previous.value.points.map((p) => p.score), width: 1.5 })
  series.push(
    { key: "greedy", label: `greedy (${GREEDY})`, color: palette.gold400, values: grid.map(() => GREEDY), width: 1, dashed: true },
    { key: "neat", label: `best evolved, NEAT (${NEAT})`, color: palette.violet400, values: grid.map(() => NEAT), width: 1, dashed: true },
  )
  return { x: grid, series }
})

// Entropy from the first evaluation on (the untrained policy has no update behind it yet); no NaN may reach LineChart.
const entropyChart = computed<{ x: number[]; series: ChartSeries[] }>(() => {
  const values = (points: CurvePoint[]) => points.slice(1).map((p) => p.entropy ?? Math.log(3))
  const series: ChartSeries[] = [{ key: "now", label: "this run", color: palette.queen400, values: values(lab.points.value), width: 2 }]
  if (previous.value) series.push({ key: "before", label: "run before", color: palette.signal400, values: values(previous.value.points), width: 1.5 })
  const n = Math.max(...series.map((s) => s.values.length))
  series.push({ key: "uniform", label: "uniform (ln 3)", color: palette.fgSubtle, values: grid.slice(1, n + 1).map(() => Math.log(3)), width: 1, dashed: true })
  return { x: grid.slice(1, n + 1), series }
})
</script>

<template>
  <LabFrame :live="live" title="Policy gradients on Snake: REINFORCE → PPO" data-policy-gradient-lab>
    <template #recorded>
      <p class="text-sm text-fg-muted">
        This browser can't run WebAssembly, so here are <strong class="text-fg">recorded runs</strong> instead: one per rung of the ladder, same seed,
        two million moves each.
      </p>
      <LabCurve class="mt-4" :x="scoreChart.x" :series="scoreChart.series" show :height="220" />
    </template>

    <template #stage>
      <LabBoard :board="lab.board.value" :live="live" placeholder="Press Train: the policy starts out choosing at random.">
        <p>
          The policy's <strong class="text-fg">most likely move</strong>, played
          <template v-if="lab.board.value"> · score <span class="num text-fg">{{ lab.board.value.score }}</span></template>
        </p>
        <UiBars
          v-if="lab.actionValues.value"
          class="mt-2"
          :labels="ACTIONS"
          :values="lab.actionValues.value"
          :chosen="lab.action.value"
          :max="1"
          :format="(p) => `${(p * 100).toFixed(0)}%`"
        />
      </LabBoard>
    </template>

    <LabControls
      v-model:speed="lab.speed.value"
      :status="lab.status.value"
      :disabled="live !== true"
      :speeds="SPEEDS"
      :hint="`Changes apply when you press ${lab.status.value === 'idle' ? 'Train' : 'Train again'}; the last run stays on the chart to compare.`"
      @train="train"
      @pause="lab.pause"
      @resume="lab.resume"
    >
      <UiSelect v-model="config.rung" label="algorithm" :options="RUNGS.map((r) => ({ value: r.id, label: r.label }))" />
      <div class="flex flex-wrap items-end gap-x-4 gap-y-2.5">
        <UiSelect v-model="config.observer" class="flex-1" label="the snake sees" :options="OBSERVERS.map((o) => ({ value: o.id, label: o.label }))" />
        <UiSelect v-model="config.seed" class="w-24" label="seed" :options="Array.from({ length: 10 }, (_, i) => i)" />
      </div>
    </LabControls>

    <UiStats
      class="mt-4"
      :items="[
        { label: 'env steps', value: `${formatSteps(lab.totalSteps.value)} / ${formatSteps(BUDGET)}` },
        { label: 'episodes', value: lab.totalEpisodes.value.toLocaleString() },
      ]"
    />
    <LabCurve
      class="mt-4"
      :x="scoreChart.x"
      :series="scoreChart.series"
      :show="lab.points.value.length > 1 || !!previous"
      empty="The mean score on 30 unseen games, every 20k steps, will draw here."
    />
    <LabCurve
      v-if="lab.points.value.length > 2"
      class="mt-4"
      caption="Policy entropy (nats) · how undecided it still is, from ln 3 toward 0"
      :x="entropyChart.x"
      :series="entropyChart.series"
      show
      :legend="false"
      :height="100"
      :format="(v) => v.toFixed(2)"
    />
    <LabStatus :status="lab.status.value" :error="lab.error.value">
      Done: {{ formatSteps(lab.totalSteps.value) }} steps. Try another rung, or what the snake sees, and train again.
    </LabStatus>
  </LabFrame>
</template>
