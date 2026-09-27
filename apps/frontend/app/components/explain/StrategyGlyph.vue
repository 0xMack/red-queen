<script setup lang="ts">
import { useId } from "vue"
import type { StrategyMode } from "~/types/explain"

// How a strategy decides, as one still picture (the explainer card's visual): the same five machines and what this
// strategy sees in them -- estimates, an uncertainty bonus, a belief, a probability, a table -- and which one it
// pulls next (▼). The numbers are illustrative, chosen so each mode's rule picks a different machine for its own
// reason; the panel's live demo plays the real thing.
const props = defineProps<{ mode: StrategyMode }>()
const uid = useId()

const W = 280
const H = 140
const BASE = 112
const TOP = 26
const scale = (v: number) => (BASE - TOP) * Math.max(0, Math.min(1, v))
const x = (i: number) => 34 + i * 48

interface Bar {
  value: number
  /** Drawn hatched: a value the strategy assumed, not measured. */
  assumed?: boolean
  /** A band above/around the bar: [low, high]. */
  band?: [number, number]
  /** A point (a sample). */
  dot?: number
  label?: string
}

const view = computed<{ bars: Bar[]; pick: number | null; note: string; any?: boolean; split?: boolean }>(() => {
  switch (props.mode) {
    case "random":
      return { bars: [0, 1, 2, 3, 4].map(() => ({ value: 0.2, label: "20%" })), pick: null, any: true, note: "every machine equally likely" }
    case "greedy":
      return { bars: [0.34, 0.62, 0.47, 0.21, 0].map((value, i) => ({ value, label: i === 4 ? "never tried" : value.toFixed(2) })), pick: 1, note: "always the highest average" }
    case "epsilon":
      return { bars: [0.34, 0.62, 0.47, 0.21, 0.4].map((value) => ({ value, label: value.toFixed(2) })), pick: 1, split: true, note: "" }
    case "optimistic":
      return {
        bars: [0.34, 0.62, 0.47, 0.21, 1].map((value, i) => ({ value, assumed: i === 4, label: i === 4 ? "assumed max" : value.toFixed(2) })),
        pick: 4,
        note: "untried machines look best until tried",
      }
    case "ucb": {
      const est = [0.34, 0.62, 0.47, 0.21, 0.4]
      const bonus = [0.17, 0.13, 0.22, 0.32, 0.45]
      return { bars: est.map((value, i) => ({ value, band: [value, value + bonus[i]!], label: `+${bonus[i]!.toFixed(2)}` })), pick: 4, note: "estimate + bonus for being unsure" }
    }
    case "thompson": {
      const est = [0.34, 0.62, 0.47, 0.21, 0.5]
      const sd = [0.1, 0.07, 0.17, 0.2, 0.42]
      const draw = [0.38, 0.58, 0.71, 0.3, 0.52]
      return { bars: est.map((value, i) => ({ value: 0, band: [value - sd[i]!, value + sd[i]!], dot: draw[i], label: draw[i]!.toFixed(2) })), pick: 2, note: "one draw from each belief; highest draw wins" }
    }
    case "gradient":
      return { bars: [0.12, 0.41, 0.2, 0.07, 0.2].map((value) => ({ value, label: `${Math.round(value * 100)}%` })), pick: null, any: true, note: "softmax of preferences: chance of each pull" }
    default:
      return { bars: [], pick: null, note: "" }
  }
})

const barModes = ["random", "greedy", "epsilon", "optimistic", "ucb", "thompson", "gradient"]

// Q-table modes: a row per situation.
const TABLE = {
  "q-table": { rows: [{ name: "every pull", color: palette.fgMuted, values: [0.31, 0.58, 0.44, 0.19, 0.36] }], pick: [0, 1] as const },
  "q-lookahead": {
    rows: [
      { name: "red room", color: palette.queen400, values: [0.3, 0.6, 2.2, 0.15, 0.45] },
      { name: "gold room", color: palette.gold400, values: [1.2, 2.1, 1.5, 2.4, 1.8] },
    ],
    pick: [0, 2] as const,
  },
}
const table = computed(() => (props.mode === "q-table" || props.mode === "q-lookahead" ? TABLE[props.mode] : null))
const cellX = (i: number) => 76 + i * 40
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="block h-auto w-full" role="img" :aria-label="`How ${mode} chooses`">
    <!-- Bar modes: five machines -->
    <g v-if="barModes.includes(mode)" font-family="Geist Mono, monospace">
      <line x1="10" :x2="W - 10" :y1="BASE" :y2="BASE" :stroke="palette.lineStrong" />
      <!-- ε-greedy's coin: 1 - ε to the best, ε to any -->
      <g v-if="view.split">
        <rect x="14" y="4" :width="(W - 28) * 0.9" height="11" rx="2" :fill="alpha('fg', 0.14)" />
        <rect :x="14 + (W - 28) * 0.9 + 2" y="4" :width="(W - 28) * 0.1 - 2" height="11" rx="2" :fill="alpha('queen400', 0.45)" />
        <text x="19" y="12.5" font-size="8" :fill="palette.fg">1 − ε: pull the best estimate</text>
        <text :x="W - 16" y="12.5" font-size="8" text-anchor="end" :fill="palette.fg">ε: any</text>
      </g>
      <text v-else-if="view.note" x="12" y="12" font-size="8.5" :fill="palette.fgSubtle">{{ view.note }}</text>

      <g v-for="(b, i) in view.bars" :key="i">
        <!-- band: UCB's bonus or Thompson's belief -->
        <rect
          v-if="b.band"
          :x="x(i) - 11"
          :y="BASE - scale(b.band[1])"
          width="22"
          :height="Math.max(1, scale(b.band[1]) - scale(b.band[0]))"
          rx="2"
          :fill="alpha(mode === 'thompson' ? 'fg' : 'signal400', mode === 'thompson' ? 0.1 : 0.22)"
          :stroke="mode === 'thompson' ? armColor(i) : palette.signal400"
          stroke-dasharray="2 2"
          stroke-width="0.8"
        />
        <rect
          v-if="b.value > 0"
          :x="x(i) - 11"
          :y="BASE - scale(b.value)"
          width="22"
          :height="scale(b.value)"
          rx="2"
          :fill="b.assumed ? `url(#${uid}-hatch)` : armColor(i)"
          :stroke="b.assumed ? armColor(i) : 'none'"
          :opacity="view.pick === null || view.pick === i || b.assumed ? 0.95 : 0.4"
        />
        <rect v-else-if="!b.band" :x="x(i) - 11" :y="BASE - 18" width="22" height="18" rx="2" fill="none" :stroke="palette.lineStrong" stroke-dasharray="2 2" />
        <circle v-if="b.dot !== undefined" :cx="x(i)" :cy="BASE - scale(b.dot)" r="3.2" :fill="armColor(i)" :stroke="palette.bg" />
        <text :x="x(i)" :y="BASE + 11" text-anchor="middle" font-size="10" font-family="Instrument Serif, serif" :fill="palette.fg">{{ armName(i) }}</text>
        <text :x="x(i)" :y="BASE + 22" text-anchor="middle" font-size="7.5" :fill="palette.fgSubtle">{{ b.label }}</text>
        <!-- the pick -->
        <text
          v-if="view.pick === i"
          :x="x(i)"
          :y="BASE - Math.max(scale(b.value), b.band ? scale(b.band[1]) : 0, b.dot !== undefined ? scale(b.dot) : 0, b.value > 0 ? 0 : 18) - 5"
          text-anchor="middle"
          font-size="10"
          :fill="palette.queen300"
        >▼</text>
        <text v-if="view.any && view.pick === null" :x="x(i)" :y="BASE - scale(b.value) - 5" text-anchor="middle" font-size="7" :fill="palette.fgSubtle">⚄</text>
      </g>
    </g>

    <!-- Table modes: a row per situation -->
    <g v-else-if="table" font-family="Geist Mono, monospace">
      <text v-for="i in 5" :key="`h${i}`" :x="cellX(i - 1) + 18" y="18" text-anchor="middle" font-size="10" font-family="Instrument Serif, serif" :fill="armColor(i - 1)">{{ armName(i - 1) }}</text>
      <g v-for="(row, r) in table.rows" :key="row.name">
        <text x="70" :y="38 + r * 30" text-anchor="end" font-size="8" :fill="row.color">{{ row.name }}</text>
        <g v-for="(v, i) in row.values" :key="i">
          <rect
            :x="cellX(i)"
            :y="25 + r * 30"
            width="36"
            height="22"
            rx="3"
            :fill="alpha('life400', 0.06 + 0.4 * (v / Math.max(...row.values)))"
            :stroke="table.pick[0] === r && table.pick[1] === i ? palette.queen300 : palette.line"
            :stroke-width="table.pick[0] === r && table.pick[1] === i ? 1.5 : 0.8"
          />
          <text :x="cellX(i) + 18" :y="39 + r * 30" text-anchor="middle" font-size="8.5" :fill="palette.fg">{{ v.toFixed(2) }}</text>
        </g>
      </g>
      <template v-if="mode === 'q-lookahead'">
        <path :d="`M ${cellX(2) + 18} 47 C ${cellX(2) + 18} 52, ${cellX(3) + 18} 50, ${cellX(3) + 18} 55`" fill="none" :stroke="palette.gold400" stroke-dasharray="2 2" :marker-end="`url(#${uid}-arrow)`" />
        <text x="12" y="110" font-size="8" :fill="palette.fgMuted">door C pays 0, but its value is</text>
        <text x="12" y="122" font-size="8.5" :fill="palette.fg">0 + γ · max(gold room) = 0.9 × 2.4</text>
      </template>
      <template v-else>
        <text x="12" y="84" font-size="8" :fill="palette.fgMuted">one row: nothing tells pulls apart</text>
        <text x="12" y="104" font-size="8.5" :fill="palette.fg">Q[s][a] += α · (r + γ · max Q[s′] − Q[s][a])</text>
        <text x="12" y="118" font-size="8" :fill="palette.fgSubtle">γ 0 here: a pull has no future</text>
      </template>
    </g>

    <!-- Snake's greedy rule: toward the food, unless it's deadly -->
    <g v-else-if="mode === 'heuristic'" font-family="Geist Mono, monospace">
      <g v-for="c in 49" :key="c">
        <rect :x="70 + ((c - 1) % 7) * 16" :y="6 + Math.floor((c - 1) / 7) * 16" width="15" height="15" rx="2" :fill="palette.sunken" :stroke="palette.line" stroke-width="0.6" />
      </g>
      <rect v-for="(s, i) in [[3, 4], [3, 5], [2, 5], [2, 4], [2, 3]]" :key="i" :x="70 + s[0]! * 16" :y="6 + s[1]! * 16" width="15" height="15" rx="3" :fill="alpha('life400', 0.55)" />
      <rect :x="70 + 3 * 16" :y="6 + 3 * 16" width="15" height="15" rx="3" :fill="palette.life400" />
      <circle :cx="70 + 5 * 16 + 7.5" :cy="6 + 1 * 16 + 7.5" r="4.5" :fill="palette.queen400" />
      <!-- heading up from (3,3): straight = (3,2) ok and toward food; left = (2,3) body -->
      <text :x="70 + 3 * 16 + 7.5" :y="6 + 2 * 16 + 11" text-anchor="middle" font-size="10" :fill="palette.life300">↑</text>
      <text :x="70 + 2 * 16 + 7.5" :y="6 + 3 * 16 + 11" text-anchor="middle" font-size="10" :fill="palette.queen300">✕</text>
      <text :x="70 + 4 * 16 + 7.5" :y="6 + 3 * 16 + 11" text-anchor="middle" font-size="10" :fill="palette.fgSubtle">→</text>
      <text x="12" y="132" font-size="8" :fill="palette.fgSubtle">safe and toward the food? take it. Otherwise any safe move.</text>
    </g>

    <!-- Alpha-beta search: a small game tree -->
    <g v-else-if="mode === 'search'" font-family="Geist Mono, monospace">
      <g :stroke="palette.lineStrong">
        <line x1="140" y1="18" x2="70" y2="58" />
        <line x1="140" y1="18" x2="210" y2="58" :stroke="palette.queen300" />
        <line v-for="(x2, i) in [40, 70, 100]" :key="`l${i}`" x1="70" y1="58" :x2="x2" y2="100" />
        <line x1="210" y1="58" x2="180" y2="100" :stroke="palette.queen300" />
        <line x1="210" y1="58" x2="210" y2="100" />
        <line x1="210" y1="58" x2="240" y2="100" stroke-dasharray="2 3" />
      </g>
      <circle cx="140" cy="18" r="8" :fill="palette.raised" :stroke="palette.queen300" />
      <text x="140" y="21" text-anchor="middle" font-size="8" :fill="palette.fg">+1</text>
      <text x="156" y="14" font-size="7.5" :fill="palette.fgSubtle">me: max</text>
      <g v-for="(n, i) in [{ x: 70, v: '−2' }, { x: 210, v: '+1' }]" :key="`n${i}`">
        <rect :x="n.x - 9" y="50" width="18" height="16" rx="3" :fill="palette.raised" :stroke="i ? palette.queen300 : palette.lineStrong" />
        <text :x="n.x" y="61" text-anchor="middle" font-size="8" :fill="palette.fg">{{ n.v }}</text>
      </g>
      <text x="232" y="56" font-size="7.5" :fill="palette.fgSubtle">them: min</text>
      <g v-for="(n, i) in [{ x: 40, v: '0' }, { x: 70, v: '−2' }, { x: 100, v: '+3' }, { x: 180, v: '+1' }, { x: 210, v: '+4' }, { x: 240, v: '✂' }]" :key="`m${i}`">
        <text :x="n.x" y="112" text-anchor="middle" font-size="8.5" :fill="n.v === '✂' ? palette.fgSubtle : palette.fgMuted">{{ n.v }}</text>
      </g>
      <text x="12" y="132" font-size="8" :fill="palette.fgSubtle">leaves: material count · ✂ pruned: can't change the answer</text>
    </g>

    <!-- First legal: the move list, top one taken -->
    <g v-else-if="mode === 'first-legal'" font-family="Geist Mono, monospace">
      <g v-for="(m, i) in ['c3 → d4', 'c3 → b4', 'e3 → f4', 'g3 → h4']" :key="m">
        <rect x="80" :y="14 + i * 24" width="120" height="19" rx="3" :fill="i === 0 ? alpha('queen400', 0.18) : palette.sunken" :stroke="i === 0 ? palette.queen300 : palette.line" />
        <text x="92" :y="27 + i * 24" font-size="9" :fill="i === 0 ? palette.fg : palette.fgSubtle">{{ i + 1 }}. {{ m }}</text>
      </g>
      <text x="12" y="132" font-size="8" :fill="palette.fgSubtle">always the first move the engine lists</text>
    </g>

    <defs>
      <pattern :id="`${uid}-hatch`" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="5" height="5" :fill="alpha('fg', 0.06)" />
        <line x1="0" y1="0" x2="0" y2="5" :stroke="palette.fgMuted" stroke-width="1.4" />
      </pattern>
      <marker :id="`${uid}-arrow`" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto">
        <path d="M0,0 L6,3 L0,6 z" :fill="palette.gold400" />
      </marker>
    </defs>
  </svg>
</template>
