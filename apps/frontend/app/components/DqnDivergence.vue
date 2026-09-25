<script setup lang="ts">
import recording from "~/data/recordings/dqn-snake.json"

// The two runs of `rl-dqn-v1` that diverged, next to a healthy one (jobs/export_rl_curves.py): the network's mean
// Q(s, a) over each 50k steps of training, and its score on unseen games. Divergence is rare -- 1 run in 20 without a
// target network -- so a reader's live demo would almost never show it; these are the real ones. Q is drawn on a
// symmetric log scale, sign(q) * log10(1 + |q|), so -5,000, 2.6 and 8e9 fit on one axis.

interface Run {
  label: string
  score: number[][]
  q_mean: number[][]
}
const RUNS: { key: keyof typeof recording.runs; color: string }[] = [
  { key: "naive-diverged", color: "#ff5c7a" },
  { key: "replay-diverged", color: "#fbbf24" },
  { key: "healthy", color: "#4ade80" },
]
const runs = RUNS.map((r) => ({ ...r, run: recording.runs[r.key] as Run }))

const W = 460
const left = 46
const right = W - 10
const MAX_STEPS = 1_000_000
const x = (steps: number) => left + (steps / MAX_STEPS) * (right - left)

const symlog = (q: number) => Math.sign(q) * Math.log10(1 + Math.abs(q))
const Q_TOP = 20
const Q_H = 190
const Q_MIN = symlog(-1e4)
const Q_MAX = symlog(1e10)
const qy = (q: number) => Q_TOP + ((Q_MAX - symlog(Math.max(q, -1e4))) / (Q_MAX - Q_MIN)) * Q_H
const Q_TICKS: [number, string][] = [
  [-1e4, "−10⁴"],
  [-100, "−100"],
  [0, "0"],
  [100, "100"],
  [1e4, "10⁴"],
  [1e6, "10⁶"],
  [1e8, "10⁸"],
  [1e10, "10¹⁰"],
]

const S_TOP = Q_TOP + Q_H + 34
const S_H = 70
const sy = (score: number) => S_TOP + (1 - score / 32) * S_H
const H = S_TOP + S_H + 24

const path = (points: number[][], y: (v: number) => number) =>
  points.map(([s, v], i) => `${i ? "L" : "M"}${x(s!).toFixed(1)},${y(v!).toFixed(1)}`).join("")
</script>

<template>
  <figure class="card my-8 p-5" data-dqn-divergence>
    <p class="eyebrow">Three real runs · the network's value estimates, and its score</p>
    <div class="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
      <span v-for="r in runs" :key="r.key" class="flex items-center gap-1.5">
        <span class="h-0.5 w-4 rounded" :style="{ background: r.color }" />{{ r.run.label }}
      </span>
    </div>
    <svg :viewBox="`0 0 ${W} ${H}`" class="mt-2 block h-auto w-full" role="img" aria-label="Mean Q-value and held-out score over training, for two diverged runs and a healthy one">
      <!-- Q panel -->
      <text :x="left" :y="Q_TOP - 8" class="fill-fg-subtle font-mono text-[9px]">mean Q(s, a) (symmetric log scale)</text>
      <rect :x="left" :y="qy(5)" :width="right - left" :height="qy(0) - qy(5)" fill="#4ade80" fill-opacity="0.08" />
      <text :x="right - 4" :y="qy(5) - 3" text-anchor="end" class="fill-life-300 font-mono text-[8px]">what returns can actually be: about 0-5</text>
      <g v-for="[q, label] in Q_TICKS" :key="label">
        <line :x1="left" :x2="right" :y1="qy(q)" :y2="qy(q)" stroke="#222837" />
        <text :x="left - 5" :y="qy(q) + 3" text-anchor="end" class="fill-fg-subtle font-mono text-[8px]">{{ label }}</text>
      </g>
      <path v-for="r in runs" :key="`q${r.key}`" :d="path(r.run.q_mean, qy)" fill="none" :stroke="r.color" stroke-width="2" />

      <!-- score panel -->
      <text :x="left" :y="S_TOP - 8" class="fill-fg-subtle font-mono text-[9px]">score on unseen games</text>
      <g v-for="s in [0, 16, 32]" :key="s">
        <line :x1="left" :x2="right" :y1="sy(s)" :y2="sy(s)" stroke="#222837" />
        <text :x="left - 5" :y="sy(s) + 3" text-anchor="end" class="fill-fg-subtle font-mono text-[8px]">{{ s }}</text>
      </g>
      <path v-for="r in runs" :key="`s${r.key}`" :d="path(r.run.score, sy)" fill="none" :stroke="r.color" stroke-width="2" />
      <g v-for="t in [0, 250_000, 500_000, 750_000, 1_000_000]" :key="t">
        <text :x="x(t)" :y="H - 6" text-anchor="middle" class="fill-fg-subtle font-mono text-[8px]">{{ t === 0 ? "0" : `${t / 1000}k` }}</text>
      </g>
    </svg>
    <figcaption class="mt-2 text-xs text-fg-subtle">
      Rewards are about ±1 a step and γ is 0.95, so no honest estimate can leave the green band by much. The run without replay spiked to 3.7 × 10⁷
      and then swung negative; the one with replay but no target network climbed steadily past 10⁹ -- each update chasing a target that its own
      last update had just raised. Neither ever learned to play. The healthy run has the same replay buffer plus a target network. From
      <code>jobs/export_rl_curves.py</code> (experiment <code>rl-dqn-v1</code>).
    </figcaption>
  </figure>
</template>
