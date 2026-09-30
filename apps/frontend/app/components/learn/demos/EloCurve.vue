<script setup lang="ts">
import * as F from "~/data/math/measuring-strength"

// Elo, as a curve (docs/design/0013): two ratings in, A's expected points per game out. The sliders are the formula's
// R_A and R_B; the curve is the expected score against every rating difference, with this pair marked on it. Presets
// are real entrants' ratings from the Checkers leaderboard (checkers.versus.v2).
const rA = ref(927)
const rB = ref(663)
const PRESETS = [
  { label: "TD-Leaf 4-ply vs Material 4-ply", a: 927, b: 663 },
  { label: "TD-Leaf 4-ply vs its 3-ply self", a: 927, b: 828 },
  { label: "Material 2-ply vs Random", a: 426, b: 0 },
  { label: "Even", a: 500, b: 500 },
]

const scope = useOrProvideTermScope()
watch(
  [rA, rB],
  ([a, b]) => {
    scope.values.value = { "r-a": a, "r-b": b }
    scope.expected.value = { "e-a": expectedScore(a - b) }
  },
  { immediate: true },
)

const W = 520
const H = 220
const PAD = { l: 40, r: 16, t: 14, b: 34 }
const RANGE = 800
const x = (d: number) => PAD.l + ((d + RANGE) / (2 * RANGE)) * (W - PAD.l - PAD.r)
const y = (s: number) => PAD.t + (1 - s) * (H - PAD.t - PAD.b)
// Rounded: the server and the browser must write identical path strings.
const curve = Array.from({ length: 81 }, (_, i) => {
  const d = -RANGE + i * 20
  return `${i ? "L" : "M"}${x(d).toFixed(1)},${y(expectedScore(d)).toFixed(2)}`
}).join("")
const diff = computed(() => Math.max(-RANGE, Math.min(RANGE, rA.value - rB.value)))
const e = computed(() => expectedScore(rA.value - rB.value))
</script>

<template>
  <UiFigure title="Elo: a rating difference is a prediction">
    <div class="flex flex-wrap gap-2">
      <button v-for="p in PRESETS" :key="p.label" class="btn-ghost btn-sm" @click="(rA = p.a), (rB = p.b)">{{ p.label }}</button>
    </div>
    <div class="mt-4 grid gap-x-6 gap-y-2 sm:grid-cols-2">
      <UiRange v-bind="scope.target('r-a')" v-model="rA" label="A's rating" :min="-100" :max="1100" :step="1" />
      <UiRange v-bind="scope.target('r-b')" v-model="rB" label="B's rating" :min="-100" :max="1100" :step="1" />
    </div>
    <div class="math-panel mt-4">
      <MathFormula :formula="F.expected" bare />
    </div>
    <svg :viewBox="`0 0 ${W} ${H}`" class="mt-4 block h-auto w-full" role="img" aria-label="Expected score against rating difference" font-family="Geist Mono, monospace">
      <g font-size="10" :fill="palette.fgSubtle">
        <line :x1="PAD.l" :x2="W - PAD.r" :y1="y(0)" :y2="y(0)" :stroke="palette.lineStrong" />
        <line v-for="s in [0.25, 0.5, 0.75, 1]" :key="s" :x1="PAD.l" :x2="W - PAD.r" :y1="y(s)" :y2="y(s)" :stroke="palette.line" stroke-dasharray="2 3" />
        <text v-for="s in [0, 0.25, 0.5, 0.75, 1]" :key="`t${s}`" :x="PAD.l - 6" :y="y(s) + 3" text-anchor="end">{{ s.toFixed(2) }}</text>
        <text v-for="d in [-800, -400, 0, 400, 800]" :key="`d${d}`" :x="x(d)" :y="H - PAD.b + 16" text-anchor="middle">{{ d > 0 ? `+${d}` : d }}</text>
        <text :x="W - PAD.r" :y="H - 4" text-anchor="end">A's rating − B's rating</text>
      </g>
      <path :d="curve" fill="none" :stroke="palette.signal400" stroke-width="2" />
      <g v-bind="scope.target('e-a')">
        <line :x1="x(diff)" :x2="x(diff)" :y1="y(0)" :y2="y(e)" :stroke="palette.queen400" stroke-dasharray="3 3" />
        <line :x1="PAD.l" :x2="x(diff)" :y1="y(e)" :y2="y(e)" :stroke="palette.queen400" stroke-dasharray="3 3" />
        <circle :cx="x(diff)" :cy="y(e)" r="5" :fill="palette.queen400" :stroke="palette.surface" stroke-width="2" />
      </g>
    </svg>
    <p class="mt-2 text-sm text-fg-muted">
      A is <span class="num text-fg">{{ rA - rB >= 0 ? "+" : "" }}{{ rA - rB }}</span> and expects
      <span class="num text-fg">{{ e.toFixed(2) }}</span> points per game: over 100 games, about
      <span class="num text-fg">{{ Math.round(e * 100) }}</span> points to B's <span class="num text-fg">{{ 100 - Math.round(e * 100) }}</span>.
    </p>
    <template #caption>
      Only differences matter: add 1,000 to both ratings and nothing changes. Every 400 points multiplies the odds by ten, so each 200 points
      buys less score than the last -- 0.50, 0.76, 0.91, 0.97 -- and near the top, a big difference in strength is a small difference in points.
    </template>
  </UiFigure>
</template>
