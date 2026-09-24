<script setup lang="ts">
// The numbers from `jobs/rl_experiment.py report --name rl-tabular-v1` (docs/design/0010 Phase 1a): twelve variants of
// tabular Q-learning on Snake (features.v1), five rng seeds each, 1M environment steps each (q-long: 5M), every run's
// *final* table playing the same 200 held-out games. Hard-coded on purpose, like the other chapters' cited results:
// the run data is gitignored, and a chapter's figures shouldn't change under a reader because someone re-ran a job.
//
// One dot per training run; the bar is the arm's mean; p is an exact paired permutation test against `q-learning`
// (runs with the same rng seed paired).
interface Arm {
  key: string
  label: string
  sub: string
  scores: number[]
  p: number | null
}

const GREEDY = 17.89 // the greedy heuristic on the same 200 games (snake.score.v2)

const ARMS: Arm[] = [
  { key: "q-learning", label: "Q-learning", sub: "α 0.1 · γ 0.95 · ε 1 → 0.05 over 100k", scores: [16.73, 17.3, 19.09, 18.16, 16.89], p: null },
  { key: "q-optimistic", label: "optimistic start", sub: "every Q starts at 2 · ε 0.02", scores: [19.51, 20.3, 17.27, 18.69, 21.14], p: 0.1875 },
  { key: "q-gamma-0.9", label: "γ 0.9", sub: "shorter horizon", scores: [17.33, 20.2, 19.32, 19.11, 18.52], p: 0.0625 },
  { key: "q-eps-fast", label: "ε fast", sub: "decays over 20k steps", scores: [18.77, 18.16, 18.57, 18.05, 19.89], p: 0.25 },
  { key: "sarsa-n3", label: "SARSA, 3-step", sub: "on-policy, n = 3", scores: [18.68, 18.83, 19.87, 18.25, 17.3], p: 0.0625 },
  { key: "q-sparse", label: "sparse reward", sub: "food +1 · death −1 · nothing else", scores: [18.38, 17.98, 18.3, 18.8, 17.73], p: 0.25 },
  { key: "q-n3", label: "Q-learning, 3-step", sub: "n = 3", scores: [16.95, 17.4, 19.03, 18.11, 18.31], p: 0.3125 },
  { key: "q-long", label: "5× longer", sub: "5M steps", scores: [17.76, 17.97, 18.32, 16.89, 18.54], p: 0.6875 },
  { key: "q-alpha-0.3", label: "α 0.3", sub: "bigger steps", scores: [17.64, 18.32, 15.14, 18.79, 18.11], p: 1 },
  { key: "q-eps-slow", label: "ε slow", sub: "decays over 500k steps", scores: [17.39, 17.44, 17.48, 16.65, 18.93], p: 1 },
  { key: "sarsa", label: "SARSA", sub: "on-policy, n = 1", scores: [16.49, 17.95, 17.05, 18.67, 17.25], p: 0.9375 },
  { key: "q-gamma-0.99", label: "γ 0.99", sub: "longer horizon", scores: [13.99, 18.49, 17.32, 18.0, 18.95], p: 0.8125 },
]

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
const sd = (xs: number[]) => Math.sqrt(xs.reduce((s, x) => s + (x - mean(xs)) ** 2, 0) / (xs.length - 1))

const W = 460
const rowH = 26
const left = 150
const right = W - 12
const top = 16
const axisMin = 12
const axisMax = 24
const x = (v: number) => left + ((v - axisMin) / (axisMax - axisMin)) * (right - left)
const H = top + ARMS.length * rowH + 20
const ticks = [12, 15, 18, 21, 24]
</script>

<template>
  <figure class="card my-8 p-5" data-q-learning-results>
    <p class="eyebrow">Held-out Snake score · twelve Q-learning variants · five seeds each</p>

    <svg :viewBox="`0 0 ${W} ${H}`" class="mt-3 block h-auto w-full" role="img" aria-label="Strip plot of each run's held-out Snake score, per Q-learning variant">
      <g v-for="t in ticks" :key="t">
        <line :x1="x(t)" :x2="x(t)" :y1="top - 6" :y2="top + ARMS.length * rowH" stroke="#222837" />
        <text :x="x(t)" :y="H - 4" text-anchor="middle" class="fill-fg-subtle font-mono text-[9px]">{{ t }}</text>
      </g>
      <line :x1="x(GREEDY)" :x2="x(GREEDY)" :y1="top - 6" :y2="top + ARMS.length * rowH" stroke="#fbbf24" stroke-dasharray="4 3" stroke-opacity="0.8" />
      <text :x="x(GREEDY) + 4" :y="top - 5" class="fill-gold-300 font-mono text-[9px]">greedy {{ GREEDY }}</text>
      <g v-for="(arm, i) in ARMS" :key="arm.key">
        <text :x="left - 8" :y="top + i * rowH + 16" text-anchor="end" class="font-mono text-[9px]" :class="i === 0 ? 'fill-fg' : 'fill-fg-muted'">{{ arm.label }}</text>
        <line :x1="x(mean(arm.scores))" :x2="x(mean(arm.scores))" :y1="top + i * rowH + 5" :y2="top + i * rowH + 21" :stroke="i === 0 ? '#ff5c7a' : '#60a5fa'" stroke-width="2.5" />
        <circle v-for="(s, k) in arm.scores" :key="k" :cx="x(s)" :cy="top + i * rowH + 13" r="3.5" :fill="i === 0 ? '#ff5c7a' : '#60a5fa'" fill-opacity="0.5" :stroke="i === 0 ? '#ff5c7a' : '#60a5fa'" />
      </g>
    </svg>

    <div class="mt-4 overflow-x-auto">
      <table class="w-full min-w-[440px] text-xs">
        <thead class="text-left text-[10px] tracking-wide text-fg-subtle uppercase">
          <tr>
            <th class="py-1 font-medium">variant</th>
            <th class="font-medium">score (mean ± sd)</th>
            <th class="font-medium">vs Q-learning</th>
            <th class="font-medium">p</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="arm in ARMS" :key="arm.key" class="border-t border-line">
            <td class="py-1.5 text-fg">{{ arm.label }}<span class="text-fg-subtle"> · {{ arm.sub }}</span></td>
            <td class="num text-fg">{{ mean(arm.scores).toFixed(2) }} ± {{ sd(arm.scores).toFixed(2) }}</td>
            <td class="num text-fg-muted">{{ arm.p === null ? "--" : `${mean(arm.scores) - mean(ARMS[0]!.scores) >= 0 ? "+" : ""}${(mean(arm.scores) - mean(ARMS[0]!.scores)).toFixed(2)}` }}</td>
            <td class="num text-fg-muted">{{ arm.p === null ? "--" : arm.p.toFixed(2) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <figcaption class="mt-3 text-xs text-fg-subtle">
      Each dot is a separate training run (a different rng seed) whose final table played the same 200 games none of them trained on; the bar is the
      mean of five. Every run visited the same 256 states. Each trained in about half a second (the 5M-step arm in 2.6 s) in the Rust core. From
      <code>jobs/rl_experiment.py report --name rl-tabular-v1</code>; the runs are on the <NuxtLink to="/runs">runs page</NuxtLink>, grouped under
      the experiment's name.
    </figcaption>
  </figure>
</template>
