<script setup lang="ts">
import type { ArmResult } from "~/components/ArmResults.vue"

// `jobs/checkers_selfplay_experiment.py report --name rl-selfplay-v1` (docs/design/0010 Phase 4): each run's final
// network, searching 3 plies, against a fixed field -- material search to 2, 3 and 4 plies and the strongest evolved
// evaluator on the versus leaderboard (32 -> 16 -> 1, 3 plies) -- 20 games each, seats alternating. Points per game
// (win 1, draw 1/2); 200k self-play games per run, 5 seeds per arm, compared with pure self-play seed for seed.

const SP = [0.644, 0.45, 0.694, 0.662, 0.556]
const ARMS: ArmResult[] = [
  { key: "sp", label: "self-play", sub: "λ 0.7", scores: SP, highlight: true },
  {
    key: "pool",
    label: "+ opponent pool",
    sub: "half the games vs past selves",
    scores: [0.625, 0.581, 0.613, 0.581, 0.625],
    versus: { label: "self-play", difference: 0.004, p: 1 },
  },
  {
    key: "lambda0",
    label: "λ 0",
    sub: "one-step TD",
    scores: [0.581, 0.6, 0.594, 0.569, 0.513],
    versus: { label: "self-play", difference: -0.03, p: 0.5625 },
  },
  {
    key: "lambda1",
    label: "λ 1",
    sub: "Monte-Carlo: the result only",
    scores: [0.312, 0.356, 0.344, 0.331, 0.4],
    versus: { label: "self-play", difference: -0.253, p: 0.0625 },
  },
]

// mean points per game against each field member, per arm
const OPPONENTS = ["material 2-ply", "material 3-ply", "material 4-ply", "evolved, 3-ply"]
const BY_OPPONENT: { label: string; points: number[] }[] = [
  { label: "self-play", points: [0.75, 0.595, 0.42, 0.64] },
  { label: "+ opponent pool", points: [0.785, 0.57, 0.36, 0.705] },
  { label: "λ 0", points: [0.775, 0.56, 0.395, 0.555] },
  { label: "λ 1", points: [0.49, 0.35, 0.26, 0.295] },
]
</script>

<template>
  <div>
    <ArmResults
      eyebrow="Points per game against the field · 3-ply search · 200k self-play games"
      :arms="ARMS"
      :references="[{ value: 0.5, label: 'even', color: '#6b7489' }]"
      :axis="[0.2, 0.8]"
      :ticks="[0.2, 0.35, 0.5, 0.65, 0.8]"
    >
      Each dot is one training run's final network (a different rng seed); its score is points per game (win 1, draw ½) over 80 games against the
      field below. From <code>jobs/checkers_selfplay_experiment.py report --name rl-selfplay-v1</code>.
    </ArmResults>
    <div class="card not-prose -mt-4 mb-8 overflow-x-auto p-5">
      <p class="eyebrow">Mean points per game, by opponent</p>
      <table class="mt-3 w-full min-w-[440px] text-xs">
        <thead class="text-left text-[10px] tracking-wide text-fg-subtle uppercase">
          <tr>
            <th class="py-1 font-medium">variant</th>
            <th v-for="o in OPPONENTS" :key="o" class="font-medium">{{ o }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in BY_OPPONENT" :key="row.label" class="border-t border-line">
            <td class="py-1.5 text-fg">{{ row.label }}</td>
            <td v-for="(p, i) in row.points" :key="i" class="num" :class="p > 0.5 ? 'text-life-300' : p < 0.5 ? 'text-queen-300' : 'text-fg-muted'">{{ p.toFixed(2) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
