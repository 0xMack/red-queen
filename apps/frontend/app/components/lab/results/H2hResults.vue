<script setup lang="ts">
// Every Checkers self-play comparison, measured again head to head (docs/design/0013):
// `jobs/checkers_selfplay_experiment.py h2h --name <experiment> [--full]`. Each variant's final network plays its
// baseline's from the same training seed, both searching 3 plies, one game pair per ballot opening; five seeds.
// "Field" is what the old report said (points per game against material-2/3/4 and an evolved network, the variant's
// mean minus its baseline's, exact paired p over seeds). "SPRT" is the sequential test (elo0 0, elo1 50, α = β = 0.05,
// pairs pooled over the seeds). "Head to head" plays the whole ballot: the Elo difference over 870 pairs, and each seed's.
interface Row {
  arm: string
  vs: string
  what: string
  field: { diff: number; p: number }
  sprt: { verdict: "H1" | "H0"; pairs: number }
  elo: number
  lo: number
  hi: number
  seeds: number[]
}
const ROWS: Row[] = [
  { arm: "TD-Leaf, 3 plies", vs: "plain TD", what: "train through a 3-ply search", field: { diff: 0.104, p: 0.125 }, sprt: { verdict: "H1", pairs: 25 }, elo: 120, lo: 108, hi: 132, seeds: [103, 124, 127, 120, 126] },
  { arm: "TD-Leaf, 2 plies", vs: "plain TD", what: "train through a 2-ply search", field: { diff: 0.067, p: 0.125 }, sprt: { verdict: "H1", pairs: 40 }, elo: 81, lo: 70, hi: 93, seeds: [82, 71, 81, 54, 119] },
  { arm: "2 × 64", vs: "1 × 64", what: "a second hidden layer, 1M games", field: { diff: 0.162, p: 0.0625 }, sprt: { verdict: "H1", pairs: 20 }, elo: 143, lo: 131, hi: 156, seeds: [152, 135, 114, 173, 146] },
  { arm: "1 × 192", vs: "2 × 64", what: "the same weights in one wide layer", field: { diff: -0.149, p: 0.0625 }, sprt: { verdict: "H0", pairs: 20 }, elo: -151, lo: -163, hi: -139, seeds: [-152, -117, -186, -171, -132] },
  { arm: "2 × 32", vs: "1 × 64", what: "two small layers, same weights", field: { diff: 0.019, p: 0.6875 }, sprt: { verdict: "H1", pairs: 35 }, elo: 36, lo: 24, hi: 47, seeds: [28, 12, 3, 77, 60] },
  { arm: "1 × 64, 1M", vs: "1 × 16, 1M", what: "a 4× wider layer, 1M games", field: { diff: 0.032, p: 0.5 }, sprt: { verdict: "H1", pairs: 20 }, elo: 55, lo: 45, hi: 66, seeds: [48, 18, 66, 45, 102] },
  { arm: "1 × 64, 200k", vs: "1 × 16, 200k", what: "a 4× wider layer", field: { diff: -0.027, p: 0.375 }, sprt: { verdict: "H0", pairs: 40 }, elo: 16, lo: 5, hi: 28, seeds: [27, -15, 2, 52, 16] },
  { arm: "1M games", vs: "200k games", what: "5× the training, 1 × 16", field: { diff: -0.04, p: 0.5 }, sprt: { verdict: "H0", pairs: 25 }, elo: -14, lo: -25, hi: -3, seeds: [-6, 11, -23, -4, -49] },
  { arm: "opponent pool", vs: "self-play", what: "half the games vs past selves", field: { diff: 0.004, p: 1 }, sprt: { verdict: "H0", pairs: 20 }, elo: 5, lo: -6, hi: 16, seeds: [24, 21, -7, 22, -33] },
  { arm: "λ 0", vs: "λ 0.7", what: "one-step TD", field: { diff: -0.03, p: 0.5625 }, sprt: { verdict: "H0", pairs: 120 }, elo: -13, lo: -22, hi: -3, seeds: [-15, 12, -16, -8, -37] },
  { arm: "λ 1", vs: "λ 0.7", what: "Monte-Carlo: the result only", field: { diff: -0.253, p: 0.0625 }, sprt: { verdict: "H0", pairs: 20 }, elo: -148, lo: -161, hi: -136, seeds: [-186, -150, -152, -117, -140] },
]

// The strip's axis: -200 ... +200 Elo, as a CSS position (HTML, so the dots stay round at any width).
const R = 200
const pos = (e: number) => `${(((Math.max(-R, Math.min(R, e)) + R) / (2 * R)) * 100).toFixed(2)}%`
const signed = (v: number, d = 0) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}`
</script>

<template>
  <UiFigure title="Head to head, against the old yardstick" kind="Result">
    <div class="hidden grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] gap-4 pb-2 sm:grid">
      <span class="label">variant vs baseline · head to head · SPRT · old field report</span>
      <span class="label flex justify-between"><span>−200</span><span>each seed's Elo difference</span><span>+200</span></span>
    </div>
    <ul class="divide-y divide-line">
      <li v-for="r in ROWS" :key="r.arm + r.vs" class="grid gap-x-4 gap-y-1 py-2.5 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] sm:items-center">
        <div class="min-w-0 text-xs">
          <p><span class="text-fg">{{ r.arm }}</span> <span class="text-fg-subtle">vs {{ r.vs }}</span> <span class="text-[11px] text-fg-subtle">· {{ r.what }}</span></p>
          <p class="num mt-0.5 flex flex-wrap gap-x-3">
            <span class="text-fg" title="Elo difference over the whole ballot, five seeds pooled (870 pairs), 95% interval">{{ signed(r.elo) }} Elo <span class="text-fg-subtle">[{{ signed(r.lo) }}, {{ signed(r.hi) }}]</span></span>
            <span :class="r.sprt.verdict === 'H1' ? 'text-life-300' : 'text-queen-300'" title="Sequential test (elo0 0, elo1 50): verdict, and the pairs it took">{{ r.sprt.verdict === 'H1' ? 'stronger' : 'no stronger' }} · {{ r.sprt.pairs }} pairs</span>
            <span class="text-fg-subtle" title="The old report: points per game against the field, mean difference and exact paired p">field {{ signed(r.field.diff, 3) }}, p {{ r.field.p < 0.1 ? r.field.p.toFixed(4) : r.field.p.toFixed(2) }}</span>
          </p>
        </div>
        <div class="relative h-[26px]" role="img" :aria-label="`Per-seed Elo differences: ${r.seeds.join(', ')}`">
          <span class="absolute inset-y-0.5 w-px bg-line-strong" :style="{ left: pos(0) }" />
          <span
            class="absolute top-1/2 h-2 -translate-y-1/2 rounded-sm"
            :class="r.elo > 0 ? 'bg-life-400/25' : 'bg-queen-400/25'"
            :style="{ left: pos(r.lo), width: `max(2px, calc(${pos(r.hi)} - ${pos(r.lo)}))` }"
          />
          <span
            v-for="(s, i) in r.seeds"
            :key="i"
            class="absolute top-1/2 size-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full"
            :class="s > 0 ? 'bg-life-400' : 'bg-queen-400'"
            :style="{ left: pos(s) }"
          />
        </div>
      </li>
    </ul>
    <template #caption>
      Every self-play variant from the previous chapter against the arm it changes one thing from, same training seed on both sides, both
      searching 3 plies. <em>Elo</em>: all 174 ballot openings × 5 seeds, the pooled difference with its interval (the shaded bar), and each
      seed's (the dots). <em>Stronger / no stronger</em>: the sequential test's verdict and the game pairs it needed, seeds pooled.
      <em>Field</em>: what the old report said (points per game against material search and an evolved network, and its p over five seeds).
      <code>jobs/checkers_selfplay_experiment.py h2h</code>.
    </template>
  </UiFigure>
</template>
