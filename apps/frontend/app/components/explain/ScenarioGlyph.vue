<script setup lang="ts">
// What a bandit scenario's machines really do -- the reveal a player only sees after the game -- as one still picture
// (the explainer card's visual). Values follow the scenario's setup in libs/games/rust/core/src/bandit.rs, drawn for
// one representative game: every real game shuffles the machines and jitters the rates. Gold: the best machine.
const props = defineProps<{ id: string }>()

const W = 280
const H = 140
const BASE = 108
const TOP = 18

interface Group {
  name?: string
  color?: string
  /** true means (payout per pull). */
  means: number[]
  /** a ± band per machine (noisy payouts). */
  spread?: number[]
  labels: string[]
  /** machine drawn as a door (pays 0, leads on). */
  door?: number
}

const pct = (p: number) => `${Math.round(p * 100)}%`

const spec = computed<{ groups: Group[]; max: number; note: string } | null>(() => {
  switch (props.id) {
    case "classic": {
      const p = [0.3, 0.75, 0.15, 0.6, 0.45]
      return { groups: [{ means: p, labels: p.map(pct) }], max: 1, note: "win rates, 15% to 75%" }
    }
    case "close-call": {
      const p = [0.47, 0.44, 0.55, 0.49, 0.46]
      return { groups: [{ means: p, labels: p.map(pct) }], max: 1, note: "one at 55%, the rest 42–50%" }
    }
    case "lucky-start": {
      const m = [5.5, 4, 7, 4.75, 6.25]
      return { groups: [{ means: m, spread: m.map(() => 3), labels: m.map((v) => v.toFixed(1)) }], max: 11, note: "average payout ± 3: the noise hides the gaps" }
    }
    case "jackpot": {
      const m = [0.45, 1.0, 0.8, 0.35, 0.55]
      return { groups: [{ means: m, labels: ["45%", "2% × 50", "0.8 always", "35%", "55%"] }], max: 1.2, note: "worth per pull: the rare jackpot is best" }
    }
    case "too-many-arms": {
      const p = [0.42, 0.11, 0.66, 0.28, 0.73, 0.19, 0.55, 0.08, 0.37, 0.61, 0.24, 0.79, 0.47, 0.15, 0.33, 0.52]
      return { groups: [{ means: p, labels: p.map(() => "") }], max: 1, note: "16 machines, 5–80%: about six pulls each" }
    }
    case "two-lamps": {
      const red = [0.3, 0.75, 0.15, 0.6, 0.45]
      return {
        groups: [
          { name: "red lamp", color: palette.queen400, means: red, labels: red.map(pct) },
          { name: "blue lamp", color: palette.signal400, means: red.map((p) => 0.9 - p), labels: red.map((p) => pct(0.9 - p)) },
        ],
        max: 1,
        note: "blue mirrors red: 0.9 − p",
      }
    }
    case "detour": {
      const red = [0.3, 0.75, 0, 0.6, 0.45]
      const gold = [1.5, 2.4, 1.2, 1.8, 2.1]
      return {
        groups: [
          { name: "red room", color: palette.queen400, means: red, labels: red.map((p, i) => (i === 2 ? "door" : pct(p))), door: 2 },
          { name: "gold room", color: palette.gold400, means: gold, labels: gold.map((v) => v.toFixed(1)) },
        ],
        max: 2.6,
        note: "worth per pull; the door pays 0 and leads to gold",
      }
    }
    default:
      return null
  }
})

// Drifting is over time, not a bar chart.
const drift = computed(() => {
  if (props.id !== "drifting") return null
  const p = [0.3, 0.75, 0.15, 0.6, 0.45]
  const x = (t: number) => 20 + (t / 200) * (W - 40)
  const y = (v: number) => BASE - (BASE - TOP) * v
  return {
    lines: p.map((v, arm) => ({
      arm,
      d: arm === 1 ? `M${x(0)},${y(v)} L${x(60)},${y(v)} L${x(60)},${y(0.15)} L${x(200)},${y(0.15)}` : `M${x(0)},${y(v)} L${x(200)},${y(v)}`,
      // B ends on C's level: label it where it starts instead.
      labelX: arm === 1 ? x(0) + 2 : x(200) + 3,
      labelY: arm === 1 ? y(v) - 4 : y(v) + 3,
    })),
    x,
    y,
  }
})

const layout = computed(() => {
  const s = spec.value
  if (!s) return []
  const n = s.groups.reduce((a, g) => a + g.means.length, 0)
  const gap = s.groups.length > 1 ? 16 : 0
  const slot = (W - 24 - gap * (s.groups.length - 1)) / n
  let x0 = 12
  return s.groups.map((g) => {
    const best = g.means.indexOf(Math.max(...g.means))
    const bars = g.means.map((m, i) => {
      const cx = x0 + slot * (i + 0.5)
      return { i, cx, w: Math.min(22, slot * 0.62), m, h: ((BASE - TOP) * Math.min(m, s.max)) / s.max, best: i === best, label: g.labels[i]!, spread: g.spread?.[i] }
    })
    const start = x0
    x0 += slot * g.means.length + gap
    return { ...g, bars, start, end: x0 - gap }
  })
})
const yOf = (v: number) => BASE - ((BASE - TOP) * v) / (spec.value?.max ?? 1)
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="block h-auto w-full" role="img" :aria-label="`How the machines pay in ${id}`" font-family="Geist Mono, monospace">
    <template v-if="drift">
      <text x="12" y="11" font-size="8.5" :fill="palette.fgSubtle">win rate of each machine, over 200 pulls</text>
      <line :x1="drift.x(0)" :x2="drift.x(200)" :y1="BASE" :y2="BASE" :stroke="palette.lineStrong" />
      <rect :x="drift.x(50)" :y="TOP" :width="drift.x(70) - drift.x(50)" :height="BASE - TOP" :fill="alpha('queen400', 0.08)" />
      <path v-for="l in drift.lines" :key="l.arm" :d="l.d" fill="none" :stroke="armColor(l.arm)" :stroke-width="l.arm === 1 ? 2.2 : 1.3" :opacity="l.arm === 1 ? 1 : 0.6" />
      <text v-for="l in drift.lines" :key="`t${l.arm}`" :x="l.labelX" :y="l.labelY" font-size="9" font-family="Instrument Serif, serif" :fill="armColor(l.arm)">{{ armName(l.arm) }}</text>
      <text v-for="t in [0, 50, 100, 150, 200]" :key="t" :x="drift.x(t)" :y="BASE + 11" text-anchor="middle" font-size="7.5" :fill="palette.fgSubtle">{{ t }}</text>
      <text :x="drift.x(70) + 3" :y="TOP + 8" font-size="7.5" :fill="palette.queen300">B breaks</text>
    </template>

    <template v-else-if="spec">
      <text x="12" y="11" font-size="8.5" :fill="palette.fgSubtle">{{ spec.note }}</text>
      <line x1="10" :x2="W - 10" :y1="BASE" :y2="BASE" :stroke="palette.lineStrong" />
      <g v-for="g in layout" :key="g.name ?? 'all'">
        <text v-if="g.name" :x="(g.start + g.end) / 2" :y="BASE + 30" text-anchor="middle" font-size="8" :fill="g.color">{{ g.name }}</text>
        <g v-for="b in g.bars" :key="b.i">
          <line
            v-if="b.spread"
            :x1="b.cx"
            :x2="b.cx"
            :y1="yOf(Math.min(spec.max, b.m + b.spread))"
            :y2="yOf(Math.max(0, b.m - b.spread))"
            :stroke="armColor(b.i)"
            stroke-width="6"
            opacity="0.22"
            stroke-linecap="round"
          />
          <rect
            v-if="b.m > 0"
            :x="b.cx - b.w / 2"
            :y="BASE - b.h"
            :width="b.w"
            :height="b.h"
            rx="2"
            :fill="armColor(b.i)"
            :opacity="b.best ? 1 : 0.55"
            :stroke="b.best ? palette.gold300 : 'none'"
            stroke-width="1.5"
          />
          <!-- the jackpot: a thin spike off the chart -->
          <line v-if="id === 'jackpot' && b.i === 1" :x1="b.cx" :x2="b.cx" :y1="TOP - 4" :y2="BASE - b.h" :stroke="armColor(b.i)" stroke-dasharray="1.5 2" />
          <g v-if="g.door === b.i">
            <rect :x="b.cx - b.w / 2" :y="BASE - 20" :width="b.w" height="20" rx="2" fill="none" :stroke="palette.gold400" stroke-dasharray="2 2" />
            <text :x="b.cx" :y="BASE - 6" text-anchor="middle" font-size="9" :fill="palette.gold400">→</text>
          </g>
          <text v-if="spec.groups[0]!.means.length <= 5" :x="b.cx" :y="BASE + 10" text-anchor="middle" font-size="9.5" font-family="Instrument Serif, serif" :fill="palette.fg">{{ armName(b.i) }}</text>
          <text v-if="b.label" :x="b.cx" :y="BASE + 20" text-anchor="middle" font-size="7" :fill="b.best ? palette.gold300 : palette.fgSubtle">{{ b.label }}</text>
        </g>
      </g>
    </template>
  </svg>
</template>
