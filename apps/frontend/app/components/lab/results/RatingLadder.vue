<script setup lang="ts">
// The Checkers leaderboard two ways (docs/design/0013): points per game against the whole field on the left, the Elo
// rating fitted to the same games on the right, one line per entrant. `jobs/evaluate_versus.py`, protocol
// checkers.versus.v2 (2026-09-30): 20 entrants, 12 ballot openings (24 games) per pairing. Hard-coded from that run.
interface Row {
  label: string
  kind: "trained" | "baseline"
  points: number
  elo: number
  lo: number
  hi: number
}
const ROWS: Row[] = [
  { label: "TD-Leaf 2×64 · 4-ply", kind: "trained", points: 0.923, elo: 927, lo: 884, hi: 982 },
  { label: "TD-Leaf 2×64 · 3-ply", kind: "trained", points: 0.864, elo: 828, lo: 787, hi: 877 },
  { label: "TD 2×64 · 4-ply", kind: "trained", points: 0.842, elo: 797, lo: 752, hi: 851 },
  { label: "TD 2×64 · 3-ply", kind: "trained", points: 0.788, elo: 727, lo: 685, hi: 779 },
  { label: "Material 4-ply", kind: "baseline", points: 0.734, elo: 663, lo: 624, hi: 714 },
  { label: "TD 16 · 3-ply", kind: "trained", points: 0.658, elo: 581, lo: 541, hi: 632 },
  { label: "Material 3-ply", kind: "baseline", points: 0.638, elo: 560, lo: 515, hi: 610 },
  { label: "Evolved 16 · 3-ply", kind: "trained", points: 0.635, elo: 557, lo: 516, hi: 602 },
  { label: "Evolved 16 · 3-ply (2)", kind: "trained", points: 0.613, elo: 534, lo: 496, hi: 583 },
  { label: "NEAT · 3-ply", kind: "trained", points: 0.527, elo: 447, lo: 404, hi: 496 },
  { label: "Material 2-ply", kind: "baseline", points: 0.507, elo: 426, lo: 388, hi: 475 },
  { label: "NEAT · 3-ply (2)", kind: "trained", points: 0.481, elo: 400, lo: 360, hi: 448 },
  { label: "NEAT 3 hidden · 3-ply", kind: "trained", points: 0.455, elo: 373, lo: 331, hi: 420 },
  { label: "Evolved 16 · 3-ply (3)", kind: "trained", points: 0.308, elo: 213, lo: 168, hi: 264 },
  { label: "Evolved 12 · 1-ply", kind: "trained", points: 0.243, elo: 137, lo: 94, hi: 177 },
  { label: "Evolved 16 · 1-ply", kind: "trained", points: 0.181, elo: 57, lo: 12, hi: 103 },
  { label: "Evolved 12 · 1-ply (2)", kind: "trained", points: 0.172, elo: 45, lo: -6, hi: 95 },
  { label: "First legal", kind: "baseline", points: 0.165, elo: 35, lo: -13, hi: 91 },
  { label: "Random", kind: "baseline", points: 0.14, elo: 0, lo: 0, hi: 0 },
  { label: "Material 1-ply", kind: "baseline", points: 0.125, elo: -23, lo: -57, hi: 31 },
]

const W = 560
const H = 380
const TOP = 24
const BOTTOM = H - 24
const LEFT = 175
const RIGHT = W - 140
const yPoints = (p: number) => BOTTOM - p * (BOTTOM - TOP)
const yElo = (e: number) => BOTTOM - ((e + 100) / 1100) * (BOTTOM - TOP)
const hovered = ref<number | null>(null)
const colour = (r: Row, i: number) => (hovered.value === i ? palette.queen300 : r.kind === "baseline" ? palette.fgSubtle : palette.signal400)
// Label the leader, the best baseline and the anchor -- or, while one is hovered, only that one; the rest would collide.
const labelled = (i: number) => (hovered.value === null ? [0, 4, 18].includes(i) : hovered.value === i)

// The field test: the leader's numbers fitted to the whole round robin, and again with only the 11 entrants rated 400
// or more (the same games, the others' dropped; arena.versus_stats.bradley_terry on the records' per-opponent results).
const FIELDS = [
  { label: "All 20 entrants", points: 0.923, aboveMaterial4: 265 },
  { label: "Only the 11 rated 400+", points: 0.858, aboveMaterial4: 264 },
]
</script>

<template>
  <UiFigure title="The same games, two yardsticks" kind="Result">
    <svg :viewBox="`0 0 ${W} ${H}`" class="block h-auto w-full" role="img" aria-label="Points per game against Elo rating for every entrant" font-family="Geist Mono, monospace">
      <g font-size="10" :fill="palette.fgMuted">
        <text :x="LEFT" y="12" text-anchor="middle">points per game</text>
        <text :x="RIGHT" y="12" text-anchor="middle">Elo rating</text>
        <line :x1="LEFT" :x2="LEFT" :y1="TOP" :y2="BOTTOM" :stroke="palette.lineStrong" />
        <line :x1="RIGHT" :x2="RIGHT" :y1="TOP" :y2="BOTTOM" :stroke="palette.lineStrong" />
        <text v-for="p in [0, 0.25, 0.5, 0.75, 1]" :key="`p${p}`" :x="LEFT - 6" :y="yPoints(p) + 3" text-anchor="end" :fill="palette.fgSubtle">{{ p.toFixed(2) }}</text>
        <text v-for="e in [0, 250, 500, 750, 1000]" :key="`e${e}`" :x="RIGHT + 6" :y="yElo(e) + 3" :fill="palette.fgSubtle">{{ e }}</text>
      </g>
      <g v-for="(r, i) in ROWS" :key="r.label" class="cursor-default" @mouseenter="hovered = i" @mouseleave="hovered = null">
        <line :x1="LEFT" :x2="RIGHT" :y1="yPoints(r.points)" :y2="yElo(r.elo)" :stroke="colour(r, i)" :stroke-width="hovered === i ? 2.5 : 1.25" :stroke-opacity="hovered === null || hovered === i ? 0.85 : 0.2" />
        <line :x1="LEFT" :x2="RIGHT" :y1="yPoints(r.points)" :y2="yElo(r.elo)" stroke="transparent" stroke-width="8" />
        <line :x1="RIGHT" :x2="RIGHT" :y1="yElo(r.lo)" :y2="yElo(r.hi)" :stroke="colour(r, i)" stroke-width="5" stroke-opacity="0.25" stroke-linecap="round" />
        <circle :cx="LEFT" :cy="yPoints(r.points)" r="3" :fill="colour(r, i)" />
        <circle :cx="RIGHT" :cy="yElo(r.elo)" r="3" :fill="colour(r, i)" />
        <template v-if="labelled(i)">
          <text :x="LEFT - 42" :y="yPoints(r.points) + 3" text-anchor="end" font-size="9.5" :fill="hovered === i ? palette.fg : palette.fgMuted">{{ r.label }}</text>
          <text :x="RIGHT + 38" :y="yElo(r.elo) + 3" font-size="9.5" :fill="hovered === i ? palette.fg : palette.fgMuted">{{ r.elo }} [{{ r.lo }}, {{ r.hi }}]</text>
        </template>
      </g>
    </svg>
    <div class="mt-4 well p-4">
      <p class="label">Change the field, keep the games: the leader (TD-Leaf, 4 plies)</p>
      <table class="mt-2 w-full text-xs">
        <thead class="label text-left">
          <tr>
            <th class="py-1 font-medium">fitted to</th>
            <th class="font-medium">points per game</th>
            <th class="font-medium">rating above Material 4-ply</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="f in FIELDS" :key="f.label" class="border-t border-line">
            <td class="py-1.5 text-fg">{{ f.label }}</td>
            <td class="num text-fg">{{ f.points.toFixed(3) }}</td>
            <td class="num text-fg">+{{ f.aboveMaterial4 }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <template #caption>
      Every entrant on the Checkers leaderboard (blue: trained, grey: fixed baselines; hover a line for its name and interval). On this field
      the two yardsticks agree on the order, because they come from the same games. What differs is what they depend on. Drop the nine weakest
      entrants and the leader's points per game fall by 0.065, though it played exactly the same games against everyone left. Its rating
      relative to Material 4-ply doesn't move. <code>jobs/evaluate_versus.py</code>, protocol <code>checkers.versus.v2</code>.
    </template>
  </UiFigure>
</template>
