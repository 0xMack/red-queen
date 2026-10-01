<script setup lang="ts">
import { generations, initial, PBT_INTERVAL, PBT_RUN, type PbtSettings } from "~/data/pbtHistory"

// One population-based training run, generation by generation (docs/design/0014; data from jobs/export_pbt_history.py).
// Eight lanes, one per member. A dot is the member at a round robin: its size is its score there, its colour the lineage
// its weights descend from -- when a bottom-quarter member copies a top-quarter one (the curves), it takes on that
// member's colour. Below, the learning rate each member trained with: copied, then nudged x0.8 or x1.25.
const members = initial.length
const G = generations.length

// lineage[g][m]: which original member's weights member m carries in round g
const lineage = (() => {
  const out: number[][] = [Array.from({ length: members }, (_, m) => m)]
  for (let g = 0; g < G - 1; g++) {
    const next = [...out[g]!]
    for (const [loser, winner] of generations[g]!.copied) next[loser] = out[g]![winner]!
    out.push(next)
  }
  return out
})()
// the settings each member trained with during interval g (before round g's exploit)
const trainedWith = (g: number): PbtSettings[] => (g === 0 ? initial : generations[g - 1]!.settings)

const COLORS = [palette.queen400, palette.signal400, palette.life400, palette.gold400, palette.violet400, palette.teal400, palette.orange400, palette.pink400]
const W = 600
const LANE = 22
const TOP = 16
const H = TOP + members * LANE + 8
const LEFT = 30
const x = (g: number) => LEFT + (g / Math.max(1, G - 1)) * (W - LEFT - 14)
const y = (m: number) => TOP + m * LANE + LANE / 2

// learning-rate panel: log scale 1e-4 ... 1e-2
const LH = 120
const ly = (lr: number) => 8 + ((Math.log10(1e-2) - Math.log10(lr)) / 2) * (LH - 20)
const lrPath = (m: number) =>
  Array.from({ length: G }, (_, g) => `${g ? "L" : "M"}${x(g).toFixed(1)},${ly(trainedWith(g)[m]!.learning_rate).toFixed(1)}`).join("")

const hovered = ref<{ g: number; m: number } | null>(null)
const info = computed(() => {
  if (!hovered.value) return null
  const { g, m } = hovered.value
  const s = trainedWith(g)[m]!
  const copied = generations[g]!.copied.find(([loser]) => loser === m)
  return {
    head: `Round ${g + 1} (${((g + 1) * PBT_INTERVAL).toLocaleString()} games each) · member ${m}`,
    body: `scored ${generations[g]!.scores[m]!.toFixed(2)} · learning rate ${s.learning_rate.toExponential(1)} · λ ${s.lambda.toFixed(2)} · pool ${Math.round(s.pool_fraction * 100)}% · ${s.pfsp ? "prioritized" : "uniform"} pool`,
    copied: copied ? `then copied member ${copied[1]} and nudged its settings` : null,
  }
})
const survivors = computed(() => new Set(lineage[G - 1]).size)
</script>

<template>
  <UiFigure :title="`A population of eight, ${G} rounds`" kind="Result">
    <svg :viewBox="`0 0 ${W} ${H}`" class="block h-auto w-full" role="img" aria-label="Population lineages over the rounds" font-family="Geist Mono, monospace">
      <text v-for="m in members" :key="`l${m}`" x="4" :y="y(m - 1) + 3" font-size="9" :fill="palette.fgSubtle">#{{ m - 1 }}</text>
      <g v-for="(gen, g) in generations" :key="`c${g}`">
        <path
          v-for="[loser, winner] in gen.copied"
          :key="`${loser}-${winner}`"
          :d="`M${x(g)},${y(winner)} C${x(g) + 10},${y(winner)} ${x(g + 1) - 10},${y(loser)} ${x(g + 1)},${y(loser)}`"
          fill="none"
          :stroke="COLORS[lineage[g]![winner]! % COLORS.length]"
          stroke-opacity="0.6"
          stroke-width="1.25"
        />
      </g>
      <g v-for="(gen, g) in generations" :key="`d${g}`">
        <circle
          v-for="m in members"
          :key="m"
          :cx="x(g)"
          :cy="y(m - 1)"
          :r="(2 + Math.max(0, gen.scores[m - 1]! - 0.2) * 9).toFixed(2)"
          :fill="COLORS[lineage[g]![m - 1]! % COLORS.length]"
          :fill-opacity="hovered && (hovered.g !== g || hovered.m !== m - 1) ? 0.45 : 0.9"
          class="cursor-default"
          @mouseenter="hovered = { g, m: m - 1 }"
          @mouseleave="hovered = null"
        />
      </g>
    </svg>
    <p class="mt-1 min-h-[2.5rem] text-xs text-fg-muted">
      <template v-if="info">
        <span class="text-fg">{{ info.head }}</span> -- {{ info.body }}<template v-if="info.copied">; {{ info.copied }}</template>.
      </template>
      <template v-else>Hover a dot. By the last round, {{ survivors }} of the 8 original lineages survive.</template>
    </p>
    <p class="label mt-3">learning rate each member trained with</p>
    <svg :viewBox="`0 0 ${W} ${LH}`" class="mt-1 block h-auto w-full" role="img" aria-label="Learning rates over the rounds" font-family="Geist Mono, monospace">
      <g font-size="9" :fill="palette.fgSubtle">
        <line v-for="lr in [1e-4, 1e-3, 1e-2]" :key="lr" :x1="LEFT" :x2="W - 14" :y1="ly(lr)" :y2="ly(lr)" :stroke="palette.line" stroke-dasharray="2 3" />
        <text v-for="lr in [1e-4, 1e-3, 1e-2]" :key="`t${lr}`" x="2" :y="ly(lr) + 3">{{ lr.toExponential(0) }}</text>
      </g>
      <path v-for="m in members" :key="m" :d="lrPath(m - 1)" fill="none" :stroke="COLORS[lineage[G - 1]![m - 1]! % COLORS.length]" stroke-opacity="0.75" stroke-width="1.5" />
    </svg>
    <template #caption>
      Run <code>{{ PBT_RUN }}</code> of <code>jobs/checkers_pbt_run.py</code> (<code>selfplay-v4</code>, arm <code>g-pbt</code>, seed 0): eight
      2 × 64 learners, {{ PBT_INTERVAL.toLocaleString() }} self-play games each between round robins. Dot size is the round-robin score (2
      ballot openings per pair, both seats -- a noisy ranking, which is part of why lineages can win by luck). Colour is lineage: whose weights
      a member carries. Learning-rate lines are coloured by the lineage each member ended in.
    </template>
  </UiFigure>
</template>
