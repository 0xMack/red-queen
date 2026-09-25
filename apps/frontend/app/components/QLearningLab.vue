<script setup lang="ts">
import type { ChartSeries } from "~/types/chart"
import { probeDevice } from "~/inference/device"
import recording from "~/data/recordings/q-learning-snake.json"

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
const live = ref<boolean | null>(null) // null while probing
onMounted(async () => {
  live.value = (await probeDevice()).wasm
})

const started = computed(() => lab.status.value !== "idle")
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
    observer: "features.v1",
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
    { key: "agent", label: "Q-learning, unseen games", color: "#ff5c7a", values: x.map((p) => p[1]), width: 2.5 },
    { key: "greedy", label: `greedy heuristic (${GREEDY})`, color: "#fbbf24", values: x.map(() => GREEDY), width: 1, dashed: true },
    { key: "neat", label: `best evolved, NEAT (${NEAT})`, color: "#a78bfa", values: x.map(() => NEAT), width: 1, dashed: true },
  ]
})
const tableValues = computed(() => (live.value === false ? recording.values : lab.values.value))
const currentQ = computed(() => {
  const v = lab.values.value
  const r = lab.row.value
  if (!v || r === null || r < 0 || v.length < (r + 1) * 3) return null
  return [v[r * 3]!, v[r * 3 + 1]!, v[r * 3 + 2]!]
})
const steps = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n))
</script>

<template>
  <div class="card not-prose my-6 p-4 sm:p-5" data-q-learning-lab>
    <!-- Recorded fallback: no WebAssembly here -->
    <template v-if="live === false">
      <p class="text-sm text-fg-muted">
        This browser can't run WebAssembly, so here is a <strong class="text-fg">recorded run</strong> of the same algorithm instead:
        {{ recording.label }} -- its learning curve and the table it ended with.
      </p>
      <LineChart class="mt-3" :x="points.map((p) => p[0])" :series="series" :height="200" x-label="env steps" :format="(v: number) => v.toFixed(1)" />
      <div class="mt-4"><QTableGrid :values="tableValues" :initial="recording.initial_q" /></div>
    </template>

    <template v-else>
      <div class="grid gap-5 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <!-- the greedy policy playing -->
        <div class="min-w-0">
          <div class="mx-auto w-full max-w-[340px]">
            <GridBoard v-if="lab.board.value" :state="lab.board.value" :tick-ms="90" />
            <div v-else class="flex aspect-square items-center justify-center rounded-xl border border-dashed border-line text-center text-sm text-fg-subtle">
              {{ live === null ? "Checking this device…" : "Press Train: the agent starts knowing nothing." }}
            </div>
          </div>
          <div class="mt-3 text-xs text-fg-subtle">
            <p>
              The <strong class="text-fg">current greedy policy</strong> playing (no exploration)
              <template v-if="lab.board.value"> · score <span class="num text-fg">{{ lab.board.value.score }}</span></template>
            </p>
            <p v-if="currentQ" class="mt-1 font-mono">
              row {{ lab.row.value }}:
              <span v-for="(q, i) in currentQ" :key="i" :class="i === lab.action.value ? 'text-fg' : ''">
                {{ ACTIONS[i] }} {{ q.toFixed(2) }}<template v-if="i < 2"> · </template>
              </span>
            </p>
          </div>
        </div>

        <!-- knobs, stats, curve -->
        <div class="min-w-0">
          <div class="flex flex-wrap items-center gap-2">
            <button v-if="lab.status.value === 'running'" class="btn-primary btn-sm" @click="lab.pause">Pause</button>
            <button v-else-if="lab.status.value === 'paused'" class="btn-primary btn-sm" @click="lab.resume">Resume</button>
            <button class="btn-sm" :class="lab.status.value === 'running' || lab.status.value === 'paused' ? 'btn-ghost' : 'btn-primary'" :disabled="live !== true" @click="train">
              {{ started ? "Start over" : "Train" }}
            </button>
            <div class="ml-auto flex rounded-lg border border-line bg-surface p-0.5" role="group" aria-label="training speed">
              <button
                v-for="s in LAB_SPEEDS"
                :key="s.label"
                class="rounded-md px-2.5 py-1 text-xs font-medium transition"
                :class="lab.speed.value === s.stepsPerSecond ? 'bg-raised text-fg shadow' : 'text-fg-subtle hover:text-fg'"
                @click="lab.speed.value = s.stepsPerSecond"
              >
                {{ s.label }}
              </button>
            </div>
          </div>

          <div class="mt-3 grid gap-x-5 gap-y-2 text-sm sm:grid-cols-2">
            <label class="flex items-center gap-2">
              <span class="w-20 shrink-0 text-fg-muted">algorithm</span>
              <select v-model="config.algorithm" class="min-w-0 flex-1 rounded-md border border-line bg-surface px-2 py-1 text-xs">
                <option value="q_learning">Q-learning</option>
                <option value="sarsa">SARSA</option>
              </select>
            </label>
            <label class="flex items-center gap-2">
              <span class="w-20 shrink-0 text-fg-muted">reward</span>
              <select v-model="config.reward" class="min-w-0 flex-1 rounded-md border border-line bg-surface px-2 py-1 text-xs">
                <option value="shaped">shaped (game's)</option>
                <option value="sparse">sparse (food / death)</option>
              </select>
            </label>
            <label class="flex items-center gap-2">
              <span class="w-20 shrink-0 text-fg-muted">α (learn)</span>
              <input v-model.number="config.alpha" type="range" min="0.02" max="0.5" step="0.02" class="min-w-0 flex-1 accent-[#ff5c7a]" aria-label="learning rate">
              <span class="num w-9 text-right">{{ config.alpha.toFixed(2) }}</span>
            </label>
            <label class="flex items-center gap-2">
              <span class="w-20 shrink-0 text-fg-muted">γ (discount)</span>
              <input v-model.number="config.gamma" type="range" min="0.5" max="0.99" step="0.01" class="min-w-0 flex-1 accent-[#ff5c7a]" aria-label="discount">
              <span class="num w-9 text-right">{{ config.gamma.toFixed(2) }}</span>
            </label>
            <label class="flex items-center gap-2" :class="config.optimistic ? 'opacity-40' : ''">
              <span class="w-20 shrink-0 text-fg-muted">ε decay</span>
              <input v-model.number="config.epsilonDecaySteps" type="range" min="10000" max="500000" step="10000" :disabled="config.optimistic" class="min-w-0 flex-1 accent-[#ff5c7a]" aria-label="epsilon decay steps">
              <span class="num w-9 text-right">{{ steps(config.epsilonDecaySteps) }}</span>
            </label>
            <label class="flex items-center gap-2">
              <span class="w-20 shrink-0 text-fg-muted">n-step</span>
              <input v-model.number="config.nStep" type="range" min="1" max="8" step="1" class="min-w-0 flex-1 accent-[#ff5c7a]" aria-label="n-step returns">
              <span class="num w-9 text-right">{{ config.nStep }}</span>
            </label>
            <label class="flex items-center gap-2 sm:col-span-2">
              <input v-model="config.optimistic" type="checkbox" class="size-4 accent-[#ff5c7a]">
              <span class="text-fg-muted">optimistic start (every Q begins at 2, ε fixed at 0.02: optimism does the exploring)</span>
            </label>
          </div>
          <p class="mt-1 text-[11px] text-fg-subtle">Changes apply when you press {{ started ? "Start over" : "Train" }}.</p>

          <dl class="mt-3 grid grid-cols-4 gap-2 text-xs">
            <div><dt class="text-fg-subtle">env steps</dt><dd class="num text-fg">{{ steps(lab.totalSteps.value) }}</dd></div>
            <div><dt class="text-fg-subtle">episodes</dt><dd class="num text-fg">{{ lab.totalEpisodes.value.toLocaleString() }}</dd></div>
            <div><dt class="text-fg-subtle">ε</dt><dd class="num text-fg">{{ lab.epsilon.value === null ? "--" : lab.epsilon.value.toFixed(3) }}</dd></div>
            <div><dt class="text-fg-subtle">states seen</dt><dd class="num text-fg">{{ lab.statesVisited.value }} / 2048</dd></div>
          </dl>

          <div class="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
            <span v-for="s in series" :key="s.key" class="flex items-center gap-1.5"><span class="h-0.5 w-4 rounded" :style="{ background: s.color }" />{{ s.label }}</span>
          </div>
          <LineChart v-if="points.length > 1" class="mt-1" :x="points.map((p) => p[0])" :series="series" :height="190" x-label="env steps" :format="(v: number) => v.toFixed(1)" />
          <p v-else class="mt-2 rounded-lg border border-dashed border-line p-6 text-center text-xs text-fg-subtle">
            The mean score on 30 unseen games, every 25k steps, will draw here.
          </p>
          <p v-if="lab.status.value === 'done'" class="mt-2 text-xs text-life-300">Done: {{ steps(lab.totalSteps.value) }} steps. Change a knob and start over to compare.</p>
          <p v-if="lab.error.value" class="mt-2 text-xs text-queen-300">{{ lab.error.value }}</p>
        </div>
      </div>

      <div class="mt-6">
        <p class="mb-2 text-sm font-medium text-fg">The Q-table, filling in</p>
        <QTableGrid :values="tableValues" :visits="lab.visits.value" :current="lab.row.value" />
      </div>
    </template>
  </div>
</template>
