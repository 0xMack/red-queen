<script setup lang="ts">
import type { ChapterArtKind } from "~/data/learnChapters"
import { checkersSnapshot } from "~/data/checkersSnapshot"

// Cover diagrams for Learn chapters -- each one sketches the chapter's actual mechanism (the GA loop,
// a Pareto front, a linear program vs. its expression tree, ...), not decorative clip art. Drawn at a
// fixed 400x250 viewBox and scaled to fit the card/banner.
// fit: "cover" crops to fill (cards with a fixed aspect ratio); "contain" letterboxes the whole
// diagram (e.g. the chapter header, whose height depends on the title length).
withDefaults(defineProps<{ kind: ChapterArtKind; fit?: "cover" | "contain" }>(), { fit: "cover" })

// Deterministic pseudo-random points so SSR and client render identically.
function rand(seed: number) {
  let s = seed
  return () => {
    s = (s * 16807) % 2147483647
    return s / 2147483647
  }
}

const population = (() => {
  const r = rand(11)
  return Array.from({ length: 22 }, () => ({ x: 40 + r() * 90, y: 70 + r() * 110, f: r() }))
})()

const paretoPoints = (() => {
  const r = rand(5)
  return Array.from({ length: 34 }, () => {
    const a = r()
    const b = r()
    return { x: 40 + a * 320, y: 30 + b * 180 }
  })
})()
// Non-dominated set, minimizing both complexity (x) and error (up the page, so *larger* SVG y is
// lower error) -- computed, not hand-picked.
const front = paretoPoints
  .filter((p) => !paretoPoints.some((q) => q !== p && q.x <= p.x && q.y >= p.y && (q.x < p.x || q.y > p.y)))
  .sort((a, b) => a.x - b.x)

// A slice of a Q-table: rows of three action values, the chosen (largest) one highlighted.
const qRows = (() => {
  const r = rand(17)
  return Array.from({ length: 7 }, () => {
    const q = [r() * 2 - 0.5, r() * 2 - 0.5, r() * 2 - 0.5]
    return { q, best: q.indexOf(Math.max(...q)) }
  })
})()

const attention = (() => {
  const r = rand(3)
  const tokens = "the_queen".split("")
  const n = tokens.length
  return {
    tokens,
    cells: Array.from({ length: n * n }, (_, i) => {
      const row = Math.floor(i / n)
      const col = i % n
      return { row, col, v: col > row ? 0 : Math.pow(r(), 2) * (col === row ? 1 : 0.9) }
    }),
  }
})()
</script>

<template>
  <svg
    viewBox="0 0 400 250"
    class="block h-full w-full bg-[#0b0e14]"
    :preserveAspectRatio="fit === 'cover' ? 'xMidYMid slice' : 'xMidYMid meet'"
  >
    <defs>
      <radialGradient id="art-glow" cx="50%" cy="45%" r="60%">
        <stop offset="0" stop-color="#ef3b5d" stop-opacity="0.18" />
        <stop offset="1" stop-color="#ef3b5d" stop-opacity="0" />
      </radialGradient>
      <marker id="art-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#6b7489" />
      </marker>
    </defs>
    <rect width="400" height="250" fill="#0b0e14" />
    <rect width="400" height="250" fill="url(#art-glow)" />

    <!-- Population -> evaluate -> select -> vary, around a loop -->
    <g v-if="kind === 'ga-loop'" font-family="JetBrains Mono, monospace" font-size="10">
      <circle v-for="(p, i) in population" :key="i" :cx="p.x" :cy="p.y" :r="3 + p.f * 3" :fill="p.f > 0.7 ? '#ff5c7a' : '#4ade80'" :fill-opacity="0.35 + p.f * 0.6" />
      <text x="85" y="200" text-anchor="middle" fill="#a0a8ba">population</text>
      <g v-for="(step, i) in ['evaluate', 'select', 'vary']" :key="step">
        <rect :x="190 + (i % 2) * 90" :y="45 + i * 55" width="80" height="30" rx="8" fill="#151924" stroke="#323a4d" />
        <text :x="230 + (i % 2) * 90" :y="64 + i * 55" text-anchor="middle" fill="#e9ebf1">{{ step }}</text>
      </g>
      <path d="M140 110 C 160 70, 170 60, 188 60" fill="none" stroke="#6b7489" stroke-width="1.5" marker-end="url(#art-arrow)" />
      <path d="M270 62 C 300 70, 320 85, 320 98" fill="none" stroke="#6b7489" stroke-width="1.5" marker-end="url(#art-arrow)" />
      <path d="M300 130 C 290 150, 280 158, 272 162" fill="none" stroke="#6b7489" stroke-width="1.5" marker-end="url(#art-arrow)" />
      <path d="M190 172 C 170 180, 150 170, 136 158" fill="none" stroke="#ff5c7a" stroke-width="1.5" marker-end="url(#art-arrow)" />
      <text x="162" y="200" fill="#ff8fa3">next generation</text>
    </g>

    <!-- Pareto front: error vs. complexity -->
    <g v-else-if="kind === 'selection'" font-family="JetBrains Mono, monospace" font-size="10">
      <line x1="36" y1="220" x2="370" y2="220" stroke="#323a4d" />
      <line x1="36" y1="220" x2="36" y2="22" stroke="#323a4d" />
      <text x="370" y="238" text-anchor="end" fill="#6b7489">complexity →</text>
      <text x="44" y="18" fill="#6b7489">↑ error</text>
      <circle v-for="(p, i) in paretoPoints" :key="i" :cx="p.x" :cy="p.y" r="3.5" fill="#60a5fa" fill-opacity="0.35" />
      <polyline :points="front.map((p) => `${p.x},${p.y}`).join(' ')" fill="none" stroke="#ff5c7a" stroke-width="1.5" stroke-dasharray="4 3" />
      <circle v-for="(p, i) in front" :key="`f${i}`" :cx="p.x" :cy="p.y" r="5" fill="#ff5c7a" stroke="#0b0e14" stroke-width="1.5" />
      <text :x="(front[0]?.x ?? 60) + 12" :y="(front[0]?.y ?? 60) + 4" fill="#ff8fa3">Pareto front</text>
    </g>

    <!-- A linear program next to the expression tree it computes -->
    <g v-else-if="kind === 'genomes'" font-family="JetBrains Mono, monospace" font-size="11">
      <rect x="24" y="40" width="150" height="170" rx="10" fill="#151924" stroke="#323a4d" />
      <text v-for="(line, i) in ['r1 = x0 × x0', 'r2 = r1 - x0', 'r3 = r0 + r2', 'r0 = r2 × r1', 'r1 = r3 ÷ x0']" :key="i" x="38" :y="70 + i * 30" :fill="i === 2 || i === 4 ? '#6b7489' : '#e9ebf1'" :text-decoration="i === 2 || i === 4 ? 'line-through' : undefined">{{ line }}</text>
      <path d="M185 125 L215 125" stroke="#6b7489" stroke-width="1.5" marker-end="url(#art-arrow)" />
      <g>
        <line x1="300" y1="55" x2="255" y2="105" stroke="#323a4d" stroke-width="1.5" />
        <line x1="300" y1="55" x2="345" y2="105" stroke="#323a4d" stroke-width="1.5" />
        <line x1="255" y1="105" x2="232" y2="160" stroke="#323a4d" stroke-width="1.5" />
        <line x1="255" y1="105" x2="278" y2="160" stroke="#323a4d" stroke-width="1.5" />
        <line x1="345" y1="105" x2="322" y2="160" stroke="#323a4d" stroke-width="1.5" />
        <line x1="345" y1="105" x2="368" y2="160" stroke="#323a4d" stroke-width="1.5" />
        <line x1="232" y1="160" x2="215" y2="205" stroke="#323a4d" stroke-width="1.5" />
        <line x1="232" y1="160" x2="249" y2="205" stroke="#323a4d" stroke-width="1.5" />
        <g v-for="n in [
          { x: 300, y: 55, t: '×', op: true },
          { x: 255, y: 105, t: '-', op: true },
          { x: 345, y: 105, t: '×', op: true },
          { x: 232, y: 160, t: '×', op: true },
          { x: 278, y: 160, t: 'x', op: false },
          { x: 322, y: 160, t: 'x', op: false },
          { x: 368, y: 160, t: 'x', op: false },
          { x: 215, y: 205, t: 'x', op: false },
          { x: 249, y: 205, t: 'x', op: false },
        ]" :key="`${n.x},${n.y}`">
          <circle :cx="n.x" :cy="n.y" r="13" :fill="n.op ? '#2a1119' : '#0f1f16'" :stroke="n.op ? '#ff5c7a' : '#4ade80'" />
          <text :x="n.x" :y="n.y + 4" text-anchor="middle" :fill="n.op ? '#ff8fa3' : '#86efac'">{{ n.t }}</text>
        </g>
      </g>
    </g>

    <!-- A computation graph with forward values and backward gradients -->
    <g v-else-if="kind === 'autodiff'" font-family="JetBrains Mono, monospace" font-size="11">
      <g v-for="e in [[70, 70, 170, 100], [70, 160, 170, 130], [210, 115, 280, 115], [70, 215, 280, 140], [320, 125, 360, 125]]" :key="e.join()">
        <line :x1="e[0]" :y1="e[1]" :x2="e[2]" :y2="e[3]" stroke="#323a4d" stroke-width="1.5" marker-end="url(#art-arrow)" />
      </g>
      <g v-for="n in [
        { x: 50, y: 70, t: 'x', v: '2.0' },
        { x: 50, y: 160, t: 'w', v: '-0.5' },
        { x: 190, y: 115, t: '×', v: '-1.0' },
        { x: 50, y: 215, t: 'b', v: '0.3' },
        { x: 300, y: 125, t: '+', v: '-0.7' },
        { x: 375, y: 125, t: 'tanh', v: '' },
      ]" :key="n.t">
        <circle :cx="n.x" :cy="n.y" r="18" fill="#151924" stroke="#60a5fa" />
        <text :x="n.x" :y="n.y + 4" text-anchor="middle" fill="#e9ebf1">{{ n.t }}</text>
        <text v-if="n.v" :x="n.x" :y="n.y - 24" text-anchor="middle" fill="#93c5fd" font-size="9">{{ n.v }}</text>
      </g>
      <path d="M355 150 C 300 200, 150 200, 70 180" fill="none" stroke="#ff5c7a" stroke-dasharray="4 3" stroke-width="1.5" marker-end="url(#art-arrow)" />
      <text x="210" y="215" text-anchor="middle" fill="#ff8fa3">∂L/∂w flows backward</text>
    </g>

    <!-- A causal attention matrix over characters -->
    <g v-else-if="kind === 'attention'" font-family="JetBrains Mono, monospace" font-size="10">
      <g transform="translate(120, 26)">
        <rect
          v-for="c in attention.cells"
          :key="`${c.row},${c.col}`"
          :x="c.col * 20"
          :y="c.row * 20"
          width="18"
          height="18"
          rx="3"
          :fill="c.v === 0 ? '#12151e' : '#ff5c7a'"
          :fill-opacity="c.v === 0 ? 1 : 0.12 + c.v * 0.88"
        />
        <text v-for="(t, i) in attention.tokens" :key="`c${i}`" :x="i * 20 + 9" y="-6" text-anchor="middle" fill="#a0a8ba">{{ t }}</text>
        <text v-for="(t, i) in attention.tokens" :key="`r${i}`" x="-8" :y="i * 20 + 13" text-anchor="end" fill="#a0a8ba">{{ t }}</text>
      </g>
    </g>

    <!-- A real Checkers position -->
    <g v-else-if="kind === 'checkers'">
      <foreignObject x="95" y="15" width="220" height="220">
        <CheckersBoard :state="checkersSnapshot" />
      </foreignObject>
    </g>

    <!-- A fixed network read off a flat list of weights, one of them mid-mutation -->
    <g v-else-if="kind === 'neuroevolution'" font-family="JetBrains Mono, monospace" font-size="10">
      <g v-for="(from, fi) in [0, 1, 2]" :key="`e${fi}`">
        <line v-for="(to, ti) in [0, 1, 2, 3]" :key="`e${fi}${ti}`" :x1="90" :y1="55 + fi * 45" :x2="200" :y2="40 + ti * 40" :stroke="(fi + ti) % 3 === 0 ? '#ff5c7a' : '#4ade80'" :stroke-opacity="0.25 + ((fi * 3 + ti * 5) % 7) / 12" stroke-width="1.5" />
      </g>
      <line v-for="(from, fi) in [0, 1, 2, 3]" :key="`o${fi}`" :x1="200" :y1="40 + fi * 40" x2="310" y2="100" :stroke="fi % 2 ? '#ff5c7a' : '#4ade80'" :stroke-opacity="0.3 + fi * 0.15" stroke-width="1.5" />
      <circle v-for="fi in [0, 1, 2]" :key="`i${fi}`" cx="90" :cy="55 + fi * 45" r="8" fill="#151924" stroke="#323a4d" />
      <circle v-for="ti in [0, 1, 2, 3]" :key="`h${ti}`" cx="200" :cy="40 + ti * 40" r="8" fill="#151924" stroke="#323a4d" />
      <circle cx="310" cy="100" r="9" fill="#0f1f16" stroke="#4ade80" />
      <g>
        <rect v-for="i in 14" :key="`g${i}`" :x="28 + (i - 1) * 25" y="188" width="22" height="26" rx="4" :fill="i === 6 ? '#3a2a0d' : '#151924'" :stroke="i === 6 ? '#fbbf24' : '#323a4d'" />
        <text v-for="(w, i) in ['.4', '-1', '.9', '.2', '-.7', '1.3', '.1', '-.3', '.8', '-.5', '.6', '-1', '.3', '.7']" :key="`t${i}`" :x="39 + i * 25" y="205" text-anchor="middle" :fill="i === 5 ? '#fcd34d' : '#a0a8ba'">{{ w }}</text>
      </g>
      <text x="200" y="232" text-anchor="middle" fill="#6b7489">the genome: one flat list, mutated in place</text>
    </g>

    <!-- A NEAT graph that has grown a hidden node, genes tagged with innovation numbers -->
    <g v-else-if="kind === 'neat'" font-family="JetBrains Mono, monospace" font-size="10">
      <g fill="none" stroke-width="1.6">
        <path d="M70 60 C 130 60, 150 60, 210 60" stroke="#4ade80" stroke-opacity="0.5" stroke-dasharray="4 3" />
        <path d="M70 60 C 110 60, 120 110, 160 130" stroke="#fbbf24" stroke-width="2.4" />
        <path d="M160 130 C 190 130, 200 70, 210 62" stroke="#fbbf24" stroke-width="2.4" />
        <path d="M70 125 C 130 125, 150 130, 160 130" stroke="#4ade80" stroke-opacity="0.6" />
        <path d="M70 125 C 140 125, 170 190, 210 190" stroke="#ff5c7a" stroke-opacity="0.55" />
        <path d="M70 190 C 130 190, 160 190, 210 190" stroke="#4ade80" stroke-opacity="0.5" />
        <path d="M70 190 C 130 190, 170 65, 210 60" stroke="#ff5c7a" stroke-opacity="0.4" />
        <path d="M210 60 L330 90" stroke="#4ade80" stroke-opacity="0.6" />
        <path d="M210 190 L330 100" stroke="#4ade80" stroke-opacity="0.6" />
      </g>
      <circle v-for="n in [{ x: 70, y: 60 }, { x: 70, y: 125 }, { x: 70, y: 190 }, { x: 210, y: 60 }, { x: 210, y: 190 }, { x: 330, y: 95 }]" :key="`${n.x}${n.y}`" :cx="n.x" :cy="n.y" r="8" fill="#151924" stroke="#323a4d" />
      <circle cx="160" cy="130" r="9" fill="#3a2a0d" stroke="#fbbf24" stroke-width="2" />
      <text x="160" y="152" text-anchor="middle" fill="#fcd34d">new node</text>
      <text x="118" y="88" fill="#fcd34d">#7</text>
      <text x="196" y="104" fill="#fcd34d">#8</text>
      <text x="128" y="55" fill="#6b7489">#1 off</text>
      <rect x="28" y="214" width="344" height="24" rx="6" fill="#151924" stroke="#323a4d" />
      <text x="200" y="230" text-anchor="middle" fill="#a0a8ba">innovation numbers line up different structures</text>
    </g>

    <!-- A board state becomes a row of the Q-table; the update rule underneath -->
    <g v-else-if="kind === 'q-table'" font-family="JetBrains Mono, monospace" font-size="10">
      <g>
        <rect v-for="i in 36" :key="`c${i}`" :x="30 + ((i - 1) % 6) * 18" :y="40 + Math.floor((i - 1) / 6) * 18" width="16" height="16" rx="2" fill="#151924" />
        <rect v-for="s in [[2, 3], [3, 3], [4, 3]]" :key="`s${s[0]}`" :x="30 + s[0] * 18" :y="40 + s[1] * 18" width="16" height="16" rx="3" fill="#4ade80" :fill-opacity="s[0] === 4 ? 1 : 0.6" />
        <circle :cx="30 + 4 * 18 + 8" :cy="40 + 1 * 18 + 8" r="5" fill="#ff5c7a" />
        <text x="84" y="164" text-anchor="middle" fill="#6b7489">state s</text>
      </g>
      <path d="M150 95 L205 95" stroke="#6b7489" stroke-width="1.5" marker-end="url(#art-arrow)" />
      <g>
        <text v-for="(h, i) in ['↰', '↑', '↱']" :key="`h${i}`" :x="245 + i * 44" y="34" text-anchor="middle" fill="#a0a8ba" font-size="12">{{ h }}</text>
        <g v-for="(row, ri) in qRows" :key="`r${ri}`">
          <rect x="220" :y="42 + ri * 20" width="134" height="18" rx="3" :fill="ri === 2 ? '#2a1520' : '#151924'" :stroke="ri === 2 ? '#ff5c7a' : 'none'" />
          <text v-for="(v, ci) in row.q" :key="`v${ci}`" :x="245 + ci * 44" :y="55 + ri * 20" text-anchor="middle" :fill="ci === row.best ? '#4ade80' : '#6b7489'">{{ v.toFixed(2) }}</text>
        </g>
      </g>
      <rect x="28" y="196" width="344" height="36" rx="6" fill="#151924" stroke="#323a4d" />
      <text x="200" y="218" text-anchor="middle" fill="#a0a8ba">Q(s,a) ← Q(s,a) + α [ r + γ·max Q(s′,·) − Q(s,a) ]</text>
    </g>

    <!-- Training job -> telemetry -> API/SSE -> browser -->
    <g v-else-if="kind === 'pipeline'" font-family="JetBrains Mono, monospace" font-size="10">
      <g v-for="(s, i) in [
        { t: 'jobs/*.py', s: 'evolve()' },
        { t: 'telemetry', s: 'jsonl + sqlite' },
        { t: 'FastAPI', s: 'SSE stream' },
        { t: 'browser', s: 'live chart' },
      ]" :key="s.t">
        <rect :x="18 + i * 96" y="95" width="80" height="56" rx="10" fill="#151924" :stroke="i === 3 ? '#ff5c7a' : '#323a4d'" />
        <text :x="58 + i * 96" y="120" text-anchor="middle" fill="#e9ebf1">{{ s.t }}</text>
        <text :x="58 + i * 96" y="137" text-anchor="middle" fill="#6b7489" font-size="9">{{ s.s }}</text>
        <line v-if="i < 3" :x1="100 + i * 96" y1="123" :x2="112 + i * 96" y2="123" stroke="#6b7489" stroke-width="1.5" marker-end="url(#art-arrow)" />
      </g>
      <circle v-for="i in 5" :key="i" :cx="120 + i * 40" cy="190" r="3" fill="#4ade80" :fill-opacity="0.2 + i * 0.16" />
      <text x="200" y="215" text-anchor="middle" fill="#86efac">one event per generation</text>
    </g>
  </svg>
</template>
