<script setup lang="ts">
// The numbers from `jobs/snake_experiment.py report --name neat-vs-neuro-v1` (docs/design/0008): four arms, five rng seeds
// each, identical budget (population 100, 250 generations, 5 fresh training games per generation, 125,000 training
// episodes), every run's *final* champion scored on the same 200 held-out Snake games. Hard-coded on purpose, like the
// other chapters' cited results: the run data itself is gitignored, and a chapter's figures shouldn't change under a
// reader because someone re-ran a job. Regenerate with the command above if the algorithms change.
//
// One dot per training run (i.e. per rng seed); the tick is the arm's mean.
interface Arm {
  key: string
  label: string
  sub: string
  scores: number[]
  params: number
  hidden: string
  seconds: number
  color: string
}

const GREEDY = 18.47 // the 3-line greedy heuristic on the same 200 games (the leaderboard's reference)

const ARMS: Arm[] = [
  { key: "lex", label: "Neuroevolution", sub: "fixed 11-16-3 · lexicase", scores: [15.95, 14.04, 17.61, 17.62, 16.59], params: 243, hidden: "16 (fixed)", seconds: 364, color: "#94a3b8" },
  { key: "tour", label: "Neuroevolution", sub: "fixed 11-16-3 · tournament", scores: [17.93, 16.19, 19.45, 20.53, 16.68], params: 243, hidden: "16 (fixed)", seconds: 385, color: "#60a5fa" },
  { key: "nospec", label: "NEAT, no speciation", sub: "ablation: one species", scores: [1.5, 20.22, 20.54, 21.38, 14.23], params: 41, hidden: "5.6 (1–13)", seconds: 79, color: "#fbbf24" },
  { key: "neat", label: "NEAT", sub: "with speciation", scores: [20.58, 20.55, 19.5, 19.18, 21.05], params: 60, hidden: "7.8 (6–11)", seconds: 94, color: "#ff5c7a" },
]

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
const sd = (xs: number[]) => Math.sqrt(xs.reduce((s, x) => s + (x - mean(xs)) ** 2, 0) / (xs.length - 1))

const W = 460
const rowH = 44
const left = 6
const right = W - 12
const top = 14
const axisMax = 24
const x = (v: number) => left + (v / axisMax) * (right - left)
const H = top + ARMS.length * rowH + 22
const ticks = [0, 5, 10, 15, 20]
</script>

<template>
  <figure class="card my-8 p-5">
    <p class="eyebrow">Held-out Snake score · four algorithms · five seeds each</p>

    <svg :viewBox="`0 0 ${W} ${H}`" class="mt-3 block h-auto w-full" role="img" aria-label="Strip plot of each run's held-out Snake score, per algorithm">
      <g v-for="t in ticks" :key="t">
        <line :x1="x(t)" :x2="x(t)" :y1="top - 6" :y2="top + ARMS.length * rowH" stroke="#222837" />
        <text :x="x(t)" :y="H - 6" text-anchor="middle" class="fill-fg-subtle font-mono text-[9px]">{{ t }}</text>
      </g>
      <line :x1="x(GREEDY)" :x2="x(GREEDY)" :y1="top - 6" :y2="top + ARMS.length * rowH" stroke="#fbbf24" stroke-dasharray="4 3" stroke-opacity="0.8" />
      <text :x="x(GREEDY) - 4" :y="top - 4" text-anchor="end" class="fill-gold-300 font-mono text-[9px]">greedy heuristic {{ GREEDY }}</text>
      <g v-for="(arm, i) in ARMS" :key="arm.key">
        <text :x="left" :y="top + i * rowH + 12" class="fill-fg-muted font-mono text-[9px]">{{ arm.label }} · {{ arm.sub }}</text>
        <line :x1="x(mean(arm.scores))" :x2="x(mean(arm.scores))" :y1="top + i * rowH + 18" :y2="top + i * rowH + 38" :stroke="arm.color" stroke-width="2.5" />
        <circle v-for="(s, k) in arm.scores" :key="k" :cx="x(s)" :cy="top + i * rowH + 28" r="4" :fill="arm.color" fill-opacity="0.55" :stroke="arm.color" />
      </g>
    </svg>

    <div class="mt-4 overflow-x-auto">
      <table class="w-full min-w-[440px] text-xs">
        <thead class="text-left text-[10px] tracking-wide text-fg-subtle uppercase">
          <tr>
            <th class="py-1 font-medium">algorithm</th>
            <th class="font-medium">score (mean ± sd)</th>
            <th class="font-medium">range</th>
            <th class="font-medium">parameters</th>
            <th class="font-medium">hidden nodes</th>
            <th class="font-medium">training</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="arm in ARMS" :key="arm.key" class="border-t border-line">
            <td class="py-1.5 text-fg"><span class="mr-2 inline-block size-2 rounded-full" :style="{ background: arm.color }" />{{ arm.label }}<span class="text-fg-subtle"> · {{ arm.sub }}</span></td>
            <td class="num text-fg">{{ mean(arm.scores).toFixed(2) }} ± {{ sd(arm.scores).toFixed(2) }}</td>
            <td class="num text-fg-muted">{{ Math.min(...arm.scores).toFixed(1) }}–{{ Math.max(...arm.scores).toFixed(1) }}</td>
            <td class="num text-fg-muted">{{ arm.params }}</td>
            <td class="num text-fg-muted">{{ arm.hidden }}</td>
            <td class="num text-fg-muted">{{ arm.seconds }} s</td>
          </tr>
        </tbody>
      </table>
    </div>
    <figcaption class="mt-3 text-xs text-fg-subtle">
      Each dot is a separate training run (a different rng seed) whose final champion played the same 200 games none of them trained on; the bar is the mean of
      the five. Every arm gets 125,000 training episodes. Training time is this project's pure-Python implementation on one machine -- a reference for
      relative cost, not a benchmark. From <code>jobs/snake_experiment.py report --name neat-vs-neuro-v1</code>; the individual runs are on the
      <NuxtLink to="/runs">runs page</NuxtLink>.
    </figcaption>
  </figure>
</template>
