<script setup lang="ts">
import { checkersSnapshot } from "~/data/checkersSnapshot"
import { DIRS, FEATURE_NAMES, cellValue, features, rays, sketchStart, space, turn, type SnakeSketch } from "~/data/explainers/snakeSketch"

// What an observer sees (the representation explainers' visual): for Snake, one board with the observer's view
// drawn over it and its readout beside it; for Checkers, the 32 squares as signed numbers; for the bandit, the table
// the observer gives a strategy. `state` animates it (the panel's live view); without one it's a fixed position.
const props = defineProps<{ id: string; state?: SnakeSketch | null }>()

const W = 280
const H = 140
const C = 12 // Snake cell size
const OX = 8
const OY = 12

const s = computed(() => props.state ?? sketchStart())
const game = computed(() => props.id.split("/")[0])
const observer = computed(() => props.id.split("/")[1] ?? "")
const cx = (x: number) => OX + x * C
const cy = (y: number) => OY + y * C
const cells = computed(() => {
  const out: { x: number; y: number; v: number }[] = []
  for (let y = 0; y < s.value.h; y++) for (let x = 0; x < s.value.w; x++) out.push({ x, y, v: cellValue(s.value, [x, y]) })
  return out
})
const CELL_FILL = [palette.sunken, alpha("life400", 0.5), palette.life400, palette.queen400]

const feats = computed(() => features(s.value))
// features.v1's danger bits are about these three cells (off-board ones aren't drawn).
const dangerCells = computed(() =>
  [0, -1, 1]
    .map((move, k) => {
      const d = DIRS[turn(s.value.heading, move)]!
      return { move, x: s.value.body[0]![0] + d[0], y: s.value.body[0]![1] + d[1], deadly: feats.value[k] === 1 }
    })
    .filter((c) => c.x >= 0 && c.y >= 0 && c.x < s.value.w && c.y < s.value.h),
)
const rayList = computed(() => (observer.value.startsWith("egocentric") ? rays(s.value) : []))
const spaces = computed(() => (observer.value === "egocentric.v2" ? space(s.value) : []))
// v2's shading: the region the best move keeps open.
const roomiest = computed(() => [...spaces.value].sort((a, b) => b.share - a.share)[0] ?? null)
const nearness = (d: number | null) => (d === null ? 0 : 1 / d)
const MOVE_NAME: Record<number, string> = { [-1]: "left", 0: "straight", 1: "right" }

// grid-onehot's three channels.
const CHANNELS = [
  { name: "body", v: 1, color: "life400" as const },
  { name: "head", v: 2, color: "life300" as const },
  { name: "food", v: 3, color: "queen400" as const },
]

// Checkers: dark squares, signed from the mover's (black's) point of view.
const checkers = computed(() => {
  const at = new Map(checkersSnapshot.cells.map((c) => [`${c.x},${c.y}`, c.label]))
  const out: { x: number; y: number; v: number; label: string | undefined }[] = []
  for (let y = 0; y < 8; y++)
    for (let x = 0; x < 8; x++) {
      if ((x + y) % 2 === 0) continue
      const label = at.get(`${x},${y}`)
      const sign = label?.startsWith("black") ? 1 : -1
      out.push({ x, y, label, v: label ? sign * (label.endsWith("king") ? 2 : 1) : 0 })
    }
  return out
})
const signed = (v: number) => (v > 0 ? `+${v}` : String(v))
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="block h-auto w-full" role="img" :aria-label="`What ${id} sees`" font-family="Geist Mono, monospace">
    <!-- ================= Snake ================= -->
    <g v-if="game === 'snake'">
      <rect :x="OX - 1" :y="OY - 1" :width="s.w * C + 2" :height="s.h * C + 2" rx="3" fill="none" :stroke="palette.lineStrong" />
      <!-- v2: the room the roomiest move keeps -->
      <rect v-for="(c, i) in roomiest?.cells ?? []" :key="`f${i}`" :x="cx(c[0])" :y="cy(c[1])" :width="C" :height="C" :fill="alpha('signal400', 0.13)" />
      <g v-for="c in cells" :key="`${c.x},${c.y}`">
        <rect
          v-if="c.v || observer === 'grid-flat.v1'"
          :x="cx(c.x) + 0.5"
          :y="cy(c.y) + 0.5"
          :width="C - 1"
          :height="C - 1"
          :rx="c.v === 3 ? 5.5 : 2"
          :fill="c.v ? CELL_FILL[c.v] : 'transparent'"
          :opacity="observer === 'grid-flat.v1' && !c.v ? 0 : 1"
        />
        <text v-if="observer === 'grid-flat.v1'" :x="cx(c.x) + C / 2" :y="cy(c.y) + 8.5" text-anchor="middle" font-size="6.5" :fill="c.v ? palette.bg : palette.fgSubtle">{{ c.v }}</text>
      </g>
      <!-- features.v1: the three cells it checks for danger -->
      <template v-if="observer === 'features.v1'">
        <rect
          v-for="d in dangerCells"
          :key="`d${d.move}`"
          :x="cx(d.x)"
          :y="cy(d.y)"
          :width="C"
          :height="C"
          rx="2"
          fill="none"
          :stroke="d.deadly ? palette.queen300 : palette.life300"
          stroke-width="1.3"
          stroke-dasharray="2 1.5"
        />
      </template>
      <!-- egocentric: the rays from the head -->
      <g v-for="r in rayList" :key="r.name">
        <line
          :x1="cx(s.body[0]![0]) + C / 2"
          :y1="cy(s.body[0]![1]) + C / 2"
          :x2="cx(r.end[0]) + C / 2 + (r.dir[0] * C) / 2"
          :y2="cy(r.end[1]) + C / 2 + (r.dir[1] * C) / 2"
          :stroke="r.food !== null ? palette.queen300 : r.body !== null ? palette.life300 : palette.signal400"
          stroke-width="0.9"
          :opacity="0.85"
        />
      </g>

      <!-- Readouts -->
      <g v-if="observer === 'features.v1'">
        <g v-for="(name, i) in FEATURE_NAMES" :key="name">
          <rect x="140" :y="10 + i * 11.3" width="8" height="8" rx="1.5" :fill="feats[i] ? (i < 3 ? palette.queen400 : palette.life400) : palette.sunken" :stroke="palette.lineStrong" stroke-width="0.6" />
          <text x="153" :y="17 + i * 11.3" font-size="7.5" :fill="feats[i] ? palette.fg : palette.fgSubtle">{{ name }}</text>
          <text x="270" :y="17 + i * 11.3" font-size="7.5" text-anchor="end" :fill="feats[i] ? palette.fg : palette.fgSubtle">{{ feats[i] }}</text>
        </g>
      </g>
      <g v-else-if="observer === 'grid-flat.v1'">
        <text x="140" y="16" font-size="8" :fill="palette.fgMuted">100 inputs, row by row</text>
        <rect
          v-for="(c, i) in cells"
          :key="`r${i}`"
          :x="140 + (i % 25) * 5.2"
          :y="24 + Math.floor(i / 25) * 9"
          width="4.4"
          height="7"
          rx="1"
          :fill="c.v ? CELL_FILL[c.v] : palette.raised"
        />
        <text x="140" y="75" font-size="6.5" :fill="palette.fgSubtle">0 empty · 1 body · 2 head · 3 food</text>
        <text x="140" y="92" font-size="7.5" :fill="palette.fgSubtle">no neighbours, no heading:</text>
        <text x="140" y="103" font-size="7.5" :fill="palette.fgSubtle">the network must find the</text>
        <text x="140" y="114" font-size="7.5" :fill="palette.fgSubtle">board's shape by itself</text>
      </g>
      <g v-else-if="observer === 'grid-onehot.v1'">
        <g v-for="(ch, k) in CHANNELS" :key="ch.name">
          <text :x="140 + k * 44" y="16" font-size="7.5" :fill="palette.fgMuted">{{ ch.name }}</text>
          <rect v-for="c in cells" :key="`${ch.name}${c.x},${c.y}`" :x="140 + k * 44 + c.x * 4" :y="22 + c.y * 4" width="3.5" height="3.5" :fill="c.v === ch.v ? palette[ch.color] : palette.raised" />
        </g>
        <text x="140" y="78" font-size="7.5" :fill="palette.fgMuted">heading</text>
        <g v-for="d in 4" :key="`h${d}`">
          <rect :x="180 + (d - 1) * 14" y="71" width="11" height="10" rx="1.5" :fill="d - 1 === s.heading ? palette.life400 : palette.raised" />
          <text :x="185.5 + (d - 1) * 14" y="79" font-size="7" text-anchor="middle" :fill="d - 1 === s.heading ? palette.bg : palette.fgSubtle">{{ "→↓←↑"[d - 1] }}</text>
        </g>
        <text x="140" y="100" font-size="7.5" :fill="palette.fgSubtle">3 × 100 + 4 = 304 yes/no inputs</text>
      </g>
      <g v-else-if="rayList.length">
        <g v-for="(r, i) in rayList" :key="`t${r.name}`">
          <text x="140" :y="15 + i * 10" font-size="6.5" :fill="palette.fgSubtle">{{ r.name }}</text>
          <rect x="190" :y="9 + i * 10" :width="nearness(r.wall) * 60" height="2" :fill="palette.signal400" />
          <rect x="190" :y="11.5 + i * 10" :width="nearness(r.body) * 60" height="2" :fill="palette.life400" />
          <rect x="190" :y="14 + i * 10" :width="nearness(r.food) * 60" height="2" :fill="palette.queen400" />
        </g>
        <text x="140" y="87" font-size="6.5" :fill="palette.fgSubtle">nearness: <tspan :fill="palette.signal400">wall</tspan> · <tspan :fill="palette.life400">body</tspan> · <tspan :fill="palette.queen400">food</tspan></text>
        <template v-if="observer === 'egocentric.v2'">
          <text x="140" y="101" font-size="6.5" :fill="palette.signal300">room after each move · tail?</text>
          <g v-for="(sp, k) in spaces" :key="`s${sp.move}`">
            <text x="140" :y="111 + k * 10" font-size="6.5" :fill="palette.fgSubtle">{{ MOVE_NAME[sp.move] }}</text>
            <rect x="190" :y="106 + k * 10" :width="Math.max(0.5, sp.share * 60)" height="5" rx="1" :fill="palette.signal300" />
            <circle cx="262" :cy="108.5 + k * 10" r="2.2" :fill="sp.tail ? palette.life400 : palette.raised" :stroke="palette.lineStrong" stroke-width="0.5" />
          </g>
        </template>
        <template v-else>
          <text x="140" y="101" font-size="6.5" :fill="palette.fgSubtle">+ food and tail as ahead/right</text>
          <text x="140" y="111" font-size="6.5" :fill="palette.fgSubtle">offsets, apples eaten, hunger</text>
        </template>
      </g>
    </g>

    <!-- ================= Checkers ================= -->
    <g v-else-if="game === 'checkers'">
      <g v-for="y in 8" :key="`y${y}`">
        <rect v-for="x in 8" :key="`x${x}`" :x="8 + (x - 1) * 15" :y="10 + (y - 1) * 15" width="15" height="15" :fill="(x + y) % 2 === 1 ? palette.raised : palette.sunken" />
      </g>
      <g v-for="c in checkers" :key="`${c.x},${c.y}`">
        <circle v-if="c.label" :cx="15.5 + c.x * 15" :cy="17.5 + c.y * 15" r="5.5" :fill="c.label.startsWith('black') ? palette.fg : palette.queen500" :stroke="c.label.endsWith('king') ? palette.gold400 : 'none'" stroke-width="1.5" />
      </g>
      <text x="142" y="16" font-size="8" :fill="palette.fgMuted">32 inputs (black to move)</text>
      <g v-for="(c, i) in checkers" :key="`v${i}`">
        <rect :x="142 + (i % 4) * 32" :y="23 + Math.floor(i / 4) * 12.5" width="29" height="10.5" rx="2" :fill="c.v > 0 ? alpha('fg', 0.12 * c.v) : c.v < 0 ? alpha('queen400', -0.14 * c.v) : palette.sunken" />
        <text :x="156.5 + (i % 4) * 32" :y="31 + Math.floor(i / 4) * 12.5" text-anchor="middle" font-size="7.5" :fill="c.v ? palette.fg : palette.fgSubtle">{{ signed(c.v) }}</text>
      </g>
    </g>

    <!-- ================= Bandit ================= -->
    <g v-else-if="game === 'bandit'">
      <text v-for="i in 5" :key="`a${i}`" :x="92 + (i - 1) * 38" y="22" text-anchor="middle" font-size="11" font-family="Instrument Serif, serif" :fill="armColor(i - 1)">{{ armName(i - 1) }}</text>
      <g v-for="(row, r) in observer === 'lamp.v1' ? [{ name: 'red lamp', color: palette.queen400 }, { name: 'blue lamp', color: palette.signal400 }] : [{ name: 'every pull', color: palette.fgMuted }]" :key="row.name">
        <circle v-if="observer === 'lamp.v1'" cx="16" :cy="41 + r * 30" r="5" :fill="row.color" />
        <text x="66" :y="44 + r * 30" text-anchor="end" font-size="8" :fill="row.color">{{ row.name }}</text>
        <g v-for="i in 5" :key="`c${i}`">
          <rect :x="75 + (i - 1) * 38" :y="30 + r * 30" width="34" height="22" rx="3" :fill="palette.sunken" :stroke="palette.line" />
          <text :x="92 + (i - 1) * 38" :y="44 + r * 30" text-anchor="middle" font-size="8.5" :fill="palette.fgSubtle">?</text>
        </g>
      </g>
      <text x="12" y="118" font-size="8" :fill="palette.fgSubtle">{{ observer === "lamp.v1" ? "the lamp picks the row: a Q-table with two states" : "nothing to observe: one row, one value per machine" }}</text>
    </g>
  </svg>
</template>
