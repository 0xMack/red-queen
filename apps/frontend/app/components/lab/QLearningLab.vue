<script setup lang="ts">
import type { ChartSeries } from "~/types/chart"
import recording from "~/data/recordings/q-learning-snake.json"
import * as F from "~/data/math/q-learning"

// Tabular Q-learning, live: the Rust RL core (compiled to WebAssembly, in a worker) learning Snake from scratch in
// the reader's browser -- the same Trainer the training jobs run (docs/design/0010 Phase 1b). Left: the current
// greedy policy playing, move by move, with the table row it's in and the three values it's choosing between.
// Right: the knobs, and the learning curve (unseen games, against env steps) next to the greedy baseline and the
// best evolved champion. Below: the whole reachable Q-table. Without WebAssembly, a recorded run of the same
// algorithm (jobs/export_rl_recording.py) stands in.

const GREEDY = 17.89 // greedy heuristic, snake.score.v2
const NEAT = 37.95 // best evolved champion (NEAT, 409M env steps)
const ACTIONS = ["turn left", "straight", "turn right"]

// The knobs, as the reader sees them; `train()` turns them into the worker's `LabConfig`.
const config = reactive({
  algorithm: "q_learning" as "q_learning" | "sarsa",
  alpha: 0.1,
  gamma: 0.95,
  epsilonDecaySteps: 100_000,
  nStep: 1,
  optimistic: false, // initial Q 2 and epsilon 0.02: optimism does the exploring
  reward: "shaped" as "shaped" | "sparse",
})
const lab = useRlLab()
const live = useWasmSupport()

function train() {
  const params: [string, number][] = [
    ["alpha", config.alpha],
    ["gamma", config.gamma],
    ["epsilon_decay_steps", config.epsilonDecaySteps],
    ["n_step", config.nStep],
  ]
  if (config.optimistic) params.push(["initial_q", 2], ["epsilon_start", 0.02], ["epsilon_end", 0.02])
  lab.start({
    algorithm: config.algorithm,
    env: "snake/features.v1+relative3.v1",
    params: params.map(([k, v]) => `${k}=${v}`).join(","),
    reward: config.reward,
    seed: 0,
    budget: 1_000_000,
    evalEvery: 25_000,
  })
}

// The curve: live points, or the recording's
const points = computed<[number, number][]>(() => (live.value === false ? (recording.curve as [number, number][]) : lab.curve.value))
const series = computed<ChartSeries[]>(() => {
  const x = points.value
  return [
    { key: "agent", label: "Q-learning, unseen games", color: palette.queen400, values: x.map((p) => p[1]), width: 2.5 },
    { key: "greedy", label: `greedy heuristic (${GREEDY})`, color: palette.gold400, values: x.map(() => GREEDY), width: 1, dashed: true },
    { key: "neat", label: `best evolved, NEAT (${NEAT})`, color: palette.violet400, values: x.map(() => NEAT), width: 1, dashed: true },
  ]
})
// The rule it runs, on top (docs/design/0012): the knobs are linked to their symbols, and the target's form follows the
// algorithm -- Q-learning's max, SARSA's next move, or (n > 1) the n-step return. The demo board's greedy choice is
// shown worked: the row's three values and the move they pick.
const scope = useOrProvideTermScope()
const updateForms = computed<Record<string, string>>(() =>
  config.nStep > 1 ? { target: "nstep" } : config.algorithm === "sarsa" ? { target: "sarsa" } : ({} as Record<string, string>),
)
const MOVES = ["left", "straight", "right"]
const greedyValues = computed(() => {
  const q = currentQ.value
  const a = lab.action.value
  if (!q || a === null) return {}
  return { "q-left": q[0], "q-straight": q[1], "q-right": q[2], move: `\\text{${MOVES[a]}}` }
})
watch(
  () => ({ alpha: config.alpha, gamma: config.gamma, nstep: config.nStep }),
  (v) => (scope.values.value = { ...scope.values.value, ...v }),
  { immediate: true },
)
const tableValues = computed(() => (live.value === false ? recording.values : lab.values.value))
// The three values the greedy move was chosen from -- sent with the move itself, so they always agree (reading the row
// out of the table instead can lag the move by a training step, and then the bars and the choice disagree).
const currentQ = computed(() => {
  if (lab.actionValues.value?.length === 3) return lab.actionValues.value
  const v = lab.values.value
  const r = lab.row.value
  if (!v || r === null || r < 0 || v.length < (r + 1) * 3) return null
  return [v[r * 3]!, v[r * 3 + 1]!, v[r * 3 + 2]!]
})
</script>

<template>
  <LabFrame :live="live" title="Tabular Q-learning on Snake" data-q-learning-lab>
    <template #recorded>
      <p class="text-sm text-fg-muted">
        This browser can't run WebAssembly, so here is a <strong class="text-fg">recorded run</strong> of the same algorithm instead:
        {{ recording.label }} -- its learning curve and the table it ended with.
      </p>
      <LabCurve class="mt-4" :x="points.map((p) => p[0])" :series="series" show :height="200" />
      <div class="mt-5"><QTableGrid :values="tableValues" :initial="recording.initial_q" /></div>
    </template>

    <template #formula>
      <MathFormula :formula="F.update" :forms="updateForms" bare />
    </template>

    <template #stage>
      <LabBoard :board="lab.board.value" :live="live" placeholder="Press Train: the agent starts knowing nothing.">
        <p>
          The <strong class="text-fg">current greedy policy</strong> playing (no exploration)
          <template v-if="lab.board.value"> · score <span class="num text-fg">{{ lab.board.value.score }}</span></template>
        </p>
        <template v-if="currentQ">
          <p class="mt-2 font-mono">table row {{ lab.row.value }}</p>
          <UiBars
            class="mt-1"
            :labels="ACTIONS"
            :values="currentQ"
            :chosen="lab.action.value"
            :bind="(i) => scope.target([['q-left', 'q-straight', 'q-right'][i]!, 'q-row'])"
          />
          <MathFormula class="mt-2" :formula="F.greedy" :values="greedyValues" bare />
        </template>
      </LabBoard>
    </template>

    <LabControls
      v-model:speed="lab.speed.value"
      :status="lab.status.value"
      :disabled="live !== true"
      :speeds="LAB_SPEEDS"
      again-label="Start over"
      @train="train"
      @pause="lab.pause"
      @resume="lab.resume"
    >
      <div class="grid gap-x-6 gap-y-2.5 sm:grid-cols-2">
        <UiSelect v-bind="scope.target('target')" v-model="config.algorithm" label="algorithm" :options="[{ value: 'q_learning', label: 'Q-learning' }, { value: 'sarsa', label: 'SARSA' }]" />
        <UiSelect v-model="config.reward" label="reward" :options="[{ value: 'shaped', label: 'shaped (the game’s)' }, { value: 'sparse', label: 'sparse (food / death)' }]" />
        <UiRange v-bind="scope.target('alpha')" v-model="config.alpha" label="α (learn)" :min="0.02" :max="0.5" :step="0.02" :display="(v) => v.toFixed(2)" />
        <UiRange v-bind="scope.target('gamma')" v-model="config.gamma" label="γ (discount)" :min="0.5" :max="0.99" :step="0.01" :display="(v) => v.toFixed(2)" />
        <UiRange v-model="config.epsilonDecaySteps" label="ε decay" :min="10000" :max="500000" :step="10000" :display="formatSteps" :disabled="config.optimistic" />
        <UiRange v-bind="scope.target('nstep')" v-model="config.nStep" label="n-step" :min="1" :max="8" />
      </div>
      <UiCheck v-model="config.optimistic">optimistic start (every Q begins at 2, ε fixed at 0.02: optimism does the exploring)</UiCheck>
    </LabControls>

    <UiStats
      class="mt-4"
      :items="[
        { label: 'env steps', value: formatSteps(lab.totalSteps.value) },
        { label: 'episodes', value: lab.totalEpisodes.value.toLocaleString() },
        { label: 'ε', value: lab.epsilon.value === null ? '--' : lab.epsilon.value.toFixed(3) },
        { label: 'states seen', value: `${lab.statesVisited.value} / 2048` },
      ]"
    />
    <LabCurve
      class="mt-4"
      :x="points.map((p) => p[0])"
      :series="series"
      :show="points.length > 1"
      :height="190"
      empty="The mean score on 30 unseen games, every 25k steps, will draw here."
    />
    <LabStatus :status="lab.status.value" :error="lab.error.value">
      Done: {{ formatSteps(lab.totalSteps.value) }} steps. Change a knob and start over to compare.
    </LabStatus>

    <template #below>
      <p class="label mb-3">The Q-table, filling in</p>
      <QTableGrid :values="tableValues" :visits="lab.visits.value" :current="lab.row.value" />
    </template>
  </LabFrame>
</template>
