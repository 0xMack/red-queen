<script setup lang="ts">
import type { ChartSeries } from "~/types/chart"
import recording from "~/data/recordings/dqn-snake.json"
import * as F from "~/data/math/dqn"
import { greedyBarTerms, greedyValues } from "~/data/math/rl-shared"

// A DQN, live: the Rust RL core (WebAssembly, in a worker) training a Q-network on Snake in the reader's browser --
// the same `DqnAgent` the training jobs run (docs/design/0010 Phase 2b). Choose what the snake sees and which
// stabilizers are on; the curve is the greedy policy's score on 30 unseen games every 10k steps, drawn over the
// previous run's so two settings can be compared; the second chart is the network's mean Q(s, a), which is where a
// diverging run shows first. Without WebAssembly, recorded runs of the same setup (jobs/export_rl_curves.py)
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
const live = useWasmSupport()

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
  lab.start({ algorithm: "dqn", env: `snake/${config.observer}+relative3.v1`, params: params.join(","), reward: "shaped", seed: config.seed, budget: BUDGET, evalEvery: EVAL_EVERY })
}

const grid = Array.from({ length: BUDGET / EVAL_EVERY + 1 }, (_, i) => i * EVAL_EVERY)

// Live: this run and the one before. Without WebAssembly: two recorded 1M-step runs, egocentric vs. features.
const scoreChart = computed<{ x: number[]; series: ChartSeries[] }>(() => {
  if (live.value === false) {
    const ego = recording.runs.egocentric.score
    const feat = recording.runs.features.score
    return {
      x: ego.map((p) => p[0]!),
      series: [
        { key: "ego", label: recording.runs.egocentric.label, color: palette.queen400, values: ego.map((p) => p[1]!), width: 2.5 },
        { key: "feat", label: recording.runs.features.label, color: palette.signal400, values: feat.map((p) => p[1]!), width: 2 },
        { key: "greedy", label: `greedy (${GREEDY})`, color: palette.gold400, values: ego.map(() => GREEDY), width: 1, dashed: true },
      ],
    }
  }
  const series: ChartSeries[] = [
    { key: "now", label: running.value?.label ?? "this run", color: palette.queen400, values: lab.points.value.map((p) => p.score), width: 2.5 },
  ]
  if (previous.value) series.push({ key: "before", label: previous.value.label, color: palette.signal400, values: previous.value.points.map((p) => p.score), width: 1.5 })
  series.push(
    { key: "greedy", label: `greedy (${GREEDY})`, color: palette.gold400, values: grid.map(() => GREEDY), width: 1, dashed: true },
    { key: "table", label: `best Q-table (${TABLE})`, color: palette.life400, values: grid.map(() => TABLE), width: 1, dashed: true },
  )
  return { x: grid, series }
})

// The rule it trains by, on top (docs/design/0012), following the switches: no target network makes θ⁻ plain θ, Double
// DQN and 3-step returns change the target's form (with both on, the 3-step form is shown). The greedy choice under the
// board is worked from the values it chose between.
const scope = useOrProvideTermScope()
const forms = computed<Record<string, string>>(() => {
  const f: Record<string, string> = {}
  if (config.nStep3) f.target = "nstep"
  else if (config.double) f.target = "double"
  if (!config.target) f["theta-minus"] = "online"
  return f
})
const greedy = computed(() => greedyValues(lab.actionValues.value, lab.action.value))

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
  const series: ChartSeries[] = [{ key: "now", label: "this run", color: palette.queen400, values: qs(lab.points.value), width: 2 }]
  if (previous.value) series.push({ key: "before", label: "run before", color: palette.signal400, values: qs(previous.value.points), width: 1.5 })
  const n = Math.max(...series.map((x) => x.values.length))
  return { x: grid.slice(1, n + 1), series }
})
</script>

<template>
  <LabFrame :live="live" title="A deep Q-network on Snake" data-dqn-lab>
    <template #recorded>
      <p class="text-sm text-fg-muted">
        This browser can't run WebAssembly, so here are two <strong class="text-fg">recorded runs</strong> instead: the same DQN for a million steps,
        once on each observation.
      </p>
      <LabCurve class="mt-4" :x="scoreChart.x" :series="scoreChart.series" show :height="200" />
    </template>

    <template #formula>
      <MathFormula :formula="F.dqnTarget" :forms="forms" bare />
      <MathFormula class="mt-1" :formula="F.dqnLoss" :forms="forms" bare />
    </template>

    <template #stage>
      <LabBoard :board="lab.board.value" :live="live" placeholder="Press Train: the network starts with random weights.">
        <p>
          The network's <strong class="text-fg">greedy policy</strong> right now
          <template v-if="lab.board.value"> · score <span class="num text-fg">{{ lab.board.value.score }}</span></template>
        </p>
        <UiBars
          v-if="lab.actionValues.value"
          class="mt-2"
          :labels="ACTIONS"
          :values="lab.actionValues.value"
          :chosen="lab.action.value"
          :format="(q) => (Math.abs(q) < 1e4 ? q.toFixed(2) : q.toExponential(1))"
          :bind="(i) => scope.target(greedyBarTerms(i))"
        />
        <MathFormula v-if="lab.actionValues.value" class="mt-2" :formula="F.greedy" :values="greedy" bare />
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
      <div class="flex flex-wrap items-end gap-x-4 gap-y-2.5">
        <UiSelect v-model="config.observer" class="flex-1" label="the snake sees" :options="OBSERVERS.map((o) => ({ value: o.id, label: o.label }))" />
        <UiSelect v-model="config.seed" class="w-24" label="seed" :options="Array.from({ length: 10 }, (_, i) => i)" />
      </div>
      <div class="flex flex-wrap gap-x-4 gap-y-1.5">
        <UiCheck v-bind="scope.target('replay')" v-model="config.replay">replay buffer</UiCheck>
        <UiCheck v-bind="scope.target('theta-minus')" v-model="config.target">target network</UiCheck>
        <UiCheck v-bind="scope.target('target')" v-model="config.double">Double DQN</UiCheck>
        <UiCheck v-model="config.dueling">dueling heads</UiCheck>
        <UiCheck v-bind="scope.target('target')" v-model="config.nStep3">3-step returns</UiCheck>
      </div>
    </LabControls>

    <UiStats
      class="mt-4"
      :items="[
        { label: 'env steps', value: `${formatSteps(lab.totalSteps.value)} / ${formatSteps(BUDGET)}` },
        { label: 'episodes', value: lab.totalEpisodes.value.toLocaleString() },
        { label: 'ε', value: lab.epsilon.value === null ? '--' : lab.epsilon.value.toFixed(3) },
      ]"
    />
    <LabCurve
      class="mt-4"
      :x="scoreChart.x"
      :series="scoreChart.series"
      :show="lab.points.value.length > 1 || !!previous"
      empty="The mean score on 30 unseen games, every 10k steps, will draw here."
    />
    <LabCurve
      v-if="lab.points.value.length > 2"
      class="mt-4"
      caption="Mean Q(s, a) on its training batches · log scale: a diverging run leaves the usual range"
      :x="qChart.x"
      :series="qChart.series"
      show
      :legend="false"
      :height="110"
      :format="formatQ"
    />
    <LabStatus :status="lab.status.value" :error="lab.error.value">
      Done: {{ formatSteps(lab.totalSteps.value) }} steps. Change what the snake sees, or a stabilizer, and train again.
    </LabStatus>
  </LabFrame>
</template>
