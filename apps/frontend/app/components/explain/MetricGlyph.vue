<script setup lang="ts">
import type { MetricMode } from "~/types/explain"

// What a leaderboard number means, as one small picture (the metric explainers' visual). Illustrative values; the
// cited bandit skills are the chapter's (data/banditResults.ts).
defineProps<{ mode: MetricMode }>()

const W = 280
const H = 140
const x100 = (v: number) => 24 + (v / 100) * (W - 48)

// Regret: cumulative expected payout, the best machine's against a strategy's, over 100 pulls.
const regret = (() => {
  const best: string[] = []
  const mine: string[] = []
  let m = 0
  for (let t = 0; t <= 100; t += 4) {
    const x = 24 + (t / 100) * (W - 48)
    best.push(`${x},${110 - t * 0.75 * 0.95}`)
    m += t < 30 ? 4 * 0.45 : 4 * 0.7
    mine.push(`${x},${110 - m * 0.95}`)
  }
  return { best: best.join(" "), mine: mine.join(" "), area: `${best.join(" ")} ${[...mine].reverse().join(" ")}` }
})()

// Best machine: which pulls went to it -- rarely at first, then mostly.
const pulls = Array.from({ length: 100 }, (_, t) => ((t * 37) % 100) / 100 < (t < 25 ? 0.25 : 0.85))
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="block h-auto w-full" role="img" :aria-label="`What ${mode} means`" font-family="Geist Mono, monospace">
    <g v-if="mode === 'skill'">
      <line :x1="x100(0)" :x2="x100(100)" y1="62" y2="62" :stroke="palette.lineStrong" stroke-width="2" />
      <g v-for="m in [{ v: 0, t: 'random' }, { v: 100, t: 'best machine, every pull' }]" :key="m.v">
        <line :x1="x100(m.v)" :x2="x100(m.v)" y1="54" y2="70" :stroke="palette.fgMuted" />
        <text :x="x100(m.v)" y="84" :text-anchor="m.v ? 'end' : 'start'" font-size="8" :fill="palette.fgMuted">{{ m.v }} · {{ m.t }}</text>
      </g>
      <g v-for="(p, i) in [{ v: 38.8, t: 'greedy' }, { v: 57.1, t: 'ε 0.1' }, { v: 77.9, t: 'optimistic' }]" :key="p.t">
        <circle :cx="x100(p.v)" cy="62" r="4" :fill="seriesColors[i]" />
        <text :x="x100(p.v)" :y="i % 2 ? 32 : 47" text-anchor="middle" font-size="8" :fill="palette.fg">{{ p.t }} {{ p.v }}</text>
      </g>
      <text x="24" y="112" font-size="8.5" :fill="palette.fgMuted">skill = (expected − random) / (best − random)</text>
      <text x="24" y="125" font-size="7.5" :fill="palette.fgSubtle">scores the choices, not the luck of the payouts</text>
    </g>

    <g v-else-if="mode === 'held-out'">
      <text x="24" y="20" font-size="8" :fill="palette.fgMuted">training seeds</text>
      <rect v-for="i in 12" :key="`t${i}`" :x="24 + (i - 1) * 9" y="28" width="7" height="7" rx="1.5" :fill="palette.signal400" opacity="0.7" />
      <text x="150" y="20" font-size="8" :fill="palette.fgMuted">held-out seeds</text>
      <rect v-for="i in 12" :key="`h${i}`" :x="150 + (i - 1) * 9" y="28" width="7" height="7" rx="1.5" :fill="palette.life400" />
      <line x1="140" x2="140" y1="14" y2="44" :stroke="palette.queen300" stroke-dasharray="2 2" />
      <text x="24" y="66" font-size="8.5" :fill="palette.fg">trained on these →</text>
      <text x="150" y="66" font-size="8.5" :fill="palette.fg">ranked on these</text>
      <text x="24" y="100" font-size="7.5" :fill="palette.fgSubtle">same seed = same food sequence, for every entrant</text>
      <text x="24" y="112" font-size="7.5" :fill="palette.fgSubtle">score = mean food eaten per game</text>
    </g>

    <g v-else-if="mode === 'points'">
      <text v-for="(n, i) in ['A', 'B', 'C', 'D']" :key="`c${n}`" :x="86 + i * 26" y="16" text-anchor="middle" font-size="8" :fill="palette.fgMuted">{{ n }}</text>
      <g v-for="(row, r) in [['', '1', '½', '1'], ['0', '', '1', '½'], ['½', '0', '', '1'], ['0', '½', '0', '']]" :key="`r${r}`">
        <text x="66" :y="34 + r * 22" text-anchor="end" font-size="8" :fill="palette.fgMuted">{{ "ABCD"[r] }}</text>
        <g v-for="(v, c) in row" :key="`v${c}`">
          <rect :x="74 + c * 26" :y="22 + r * 22" width="24" height="18" rx="2" :fill="v === '1' ? alpha('life400', 0.35) : v === '½' ? alpha('gold400', 0.25) : v === '0' ? alpha('queen400', 0.18) : palette.raised" />
          <text :x="86 + c * 26" :y="34 + r * 22" text-anchor="middle" font-size="8.5" :fill="palette.fg">{{ v }}</text>
        </g>
        <text x="190" :y="34 + r * 22" font-size="8.5" :fill="palette.fg">{{ [0.83, 0.5, 0.5, 0.17][r]!.toFixed(2) }}</text>
      </g>
      <text x="190" y="16" font-size="8" :fill="palette.fgMuted">per game</text>
      <text x="24" y="124" font-size="7.5" :fill="palette.fgSubtle">win 1 · draw ½ · loss 0, against everyone, both colours</text>
    </g>

    <g v-else-if="mode === 'interval'">
      <g v-for="(e, i) in [{ m: 70, ci: 6, y: 34, t: '#1' }, { m: 63, ci: 5, y: 62, t: '#2 ≈' }, { m: 40, ci: 5, y: 90, t: '#3' }]" :key="i">
        <line :x1="x100(e.m - e.ci)" :x2="x100(e.m + e.ci)" :y1="e.y" :y2="e.y" :stroke="seriesColors[i]" stroke-width="5" opacity="0.35" stroke-linecap="round" />
        <circle :cx="x100(e.m)" :cy="e.y" r="3.5" :fill="seriesColors[i]" />
        <text x="12" :y="e.y + 3" font-size="8.5" :fill="palette.fg">{{ e.t }}</text>
      </g>
      <rect :x="x100(56)" y="24" :width="x100(77) - x100(56)" height="46" fill="none" :stroke="palette.gold400" stroke-dasharray="2 2" />
      <text x="24" y="120" font-size="7.5" :fill="palette.fgSubtle">#1 and #2 overlap: their order is not clear</text>
    </g>

    <g v-else-if="mode === 'level'">
      <g v-for="l in [0, 1, 2, 3]" :key="l">
        <rect x="24" :y="100 - l * 26" :width="60 + l * 50" height="20" rx="3" :fill="alpha('fg', 0.04)" :stroke="LEVEL_COLORS[l]" />
        <text x="32" :y="113 - l * 26" font-size="8.5" :fill="LEVEL_COLORS[l]">L{{ l }} {{ LEVEL_NAMES[l] }}</text>
      </g>
      <text x="90" y="113" font-size="7.5" :fill="palette.fgSubtle">pixels, as a person sees it</text>
      <text x="140" y="87" font-size="7.5" :fill="palette.fgSubtle">grid-flat, board32</text>
      <text x="24" y="14" font-size="7.5" :fill="palette.fgSubtle">↑ more of the thinking done for the model</text>
      <text x="190" y="61" font-size="7.5" :fill="palette.fgSubtle">egocentric</text>
      <text x="238" y="35" font-size="7.5" :fill="palette.fgSubtle">features</text>
    </g>

    <g v-else-if="mode === 'gap'">
      <g v-for="(b, i) in [{ t: 'training seeds', v: 42 }, { t: 'held-out', v: 29 }]" :key="b.t">
        <rect :x="60 + i * 70" :y="110 - b.v * 2" width="40" :height="b.v * 2" rx="2" :fill="i ? palette.life400 : palette.signal400" />
        <text :x="80 + i * 70" y="124" text-anchor="middle" font-size="8" :fill="palette.fgMuted">{{ b.t }}</text>
        <text :x="80 + i * 70" :y="104 - b.v * 2" text-anchor="middle" font-size="8.5" :fill="palette.fg">{{ b.v }}</text>
      </g>
      <line x1="200" x2="200" :y1="110 - 84" :y2="110 - 58" :stroke="palette.queen300" stroke-width="1.5" />
      <text x="207" y="44" font-size="8.5" :fill="palette.queen300">gap +13</text>
      <text x="207" y="56" font-size="7.5" :fill="palette.fgSubtle">memorised its</text>
      <text x="207" y="66" font-size="7.5" :fill="palette.fgSubtle">training boards</text>
    </g>

    <g v-else-if="mode === 'regret'">
      <polygon :points="regret.area" :fill="alpha('queen400', 0.18)" />
      <polyline :points="regret.best" fill="none" :stroke="palette.gold400" stroke-width="1.5" />
      <polyline :points="regret.mine" fill="none" :stroke="palette.signal400" stroke-width="1.5" />
      <text x="190" y="36" font-size="8" :fill="palette.gold400">best machine</text>
      <text x="200" y="66" font-size="8" :fill="palette.signal400">a strategy</text>
      <text x="110" y="72" font-size="8" :fill="palette.queen300">regret</text>
      <text x="24" y="126" font-size="7.5" :fill="palette.fgSubtle">expected payout over 100 pulls; the shaded gap is given up</text>
    </g>

    <g v-else-if="mode === 'best-rate'">
      <rect v-for="(b, t) in pulls" :key="t" :x="24 + (t % 25) * 9.4" :y="24 + Math.floor(t / 25) * 16" width="7.5" height="12" rx="1.5" :fill="b ? palette.gold400 : palette.raised" />
      <text x="24" y="16" font-size="8" :fill="palette.fgMuted">100 pulls, in order</text>
      <text x="24" y="104" font-size="8.5" :fill="palette.fg">{{ pulls.filter(Boolean).length }}% went to the best machine</text>
      <text x="24" y="118" font-size="7.5" :fill="palette.fgSubtle">few early, while exploring; most once it's found</text>
    </g>

    <g v-else-if="mode === 'cost'">
      <g v-for="(c, i) in [{ t: 'inference', v: 'µs per decision', w: 0.2 }, { t: 'params', v: 'numbers in the model', w: 0.55 }, { t: 'training', v: 'episodes · time', w: 0.9 }]" :key="c.t">
        <text x="24" :y="30 + i * 32" font-size="8.5" :fill="palette.fg">{{ c.t }}</text>
        <rect x="90" :y="22 + i * 32" :width="c.w * 160" height="10" rx="2" :fill="seriesColors[i]" opacity="0.7" />
        <text x="90" :y="44 + i * 32" font-size="7.5" :fill="palette.fgSubtle">{{ c.v }}</text>
      </g>
    </g>
  </svg>
</template>
