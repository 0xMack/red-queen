<script setup lang="ts">
import * as F from "~/data/math/measuring-strength"

// The sequential probability ratio test, run by hand (docs/design/0013). Two simulated players `truth` Elo apart play
// game pairs (utils/versusStats.ts `simulatePair`: draws most likely between equals, the expected score what Elo says),
// and after every pair the LLR moves up or down; crossing a bound decides. One test animates pair by pair, and its
// numbers work the LLR formula; a thousand tests at once show what the test promises -- how often it is wrong, and how
// many pairs it needs. The budget is the ballot: 174 pairs, then "inconclusive".
const BUDGET = 174
const ALPHA = 0.05
const BETA = 0.05
const { lower, upper } = sprtBounds(ALPHA, BETA)

const truth = ref(60)
const elo1 = ref(50)
const ELO1 = [
  { value: 20, label: "20" },
  { value: 50, label: "50" },
  { value: 100, label: "100" },
]

type Verdict = "H1" | "H0" | "inconclusive"
interface Run {
  path: number[]
  pairs: number[]
  verdict: Verdict | null
}

let seed = 1
function runTest(rand: () => number, onPair?: (run: Run) => void): Run {
  const run: Run = { path: [0], pairs: [], verdict: null }
  while (run.pairs.length < BUDGET && !run.verdict) {
    run.pairs.push(simulatePair(rand, truth.value))
    const value = run.pairs.length >= 20 ? llr(run.pairs, 0, elo1.value) : 0
    run.path.push(value)
    if (value >= upper) run.verdict = "H1"
    else if (value <= lower) run.verdict = "H0"
    onPair?.(run)
  }
  run.verdict ??= "inconclusive"
  return run
}

// --- One test, animated -----------------------------------------------------------------------------------------------
const current = shallowRef<Run | null>(null)
const history = shallowRef<Run[]>([])
const playing = ref(false)
let timer: ReturnType<typeof setInterval> | null = null

function playOne() {
  stop()
  const rand = seededRandom(seed++)
  const all = runTest(rand)
  let shown = 0
  playing.value = true
  timer = setInterval(() => {
    shown += 2
    const done = shown >= all.pairs.length
    current.value = { path: all.path.slice(0, shown + 1), pairs: all.pairs.slice(0, shown), verdict: done ? all.verdict : null }
    if (done) {
      stop()
      history.value = [...history.value.slice(-11), all]
    }
  }, 30)
}
function stop() {
  if (timer) clearInterval(timer)
  timer = null
  playing.value = false
}
onBeforeUnmount(stop)

// --- A thousand tests ---------------------------------------------------------------------------------------------------
const batch = shallowRef<{ H1: number; H0: number; inconclusive: number; meanPairs: number; n: number; truth: number; elo1: number } | null>(null)
function runMany(n = 1000) {
  const rand = seededRandom(1000 + seed++)
  const counts = { H1: 0, H0: 0, inconclusive: 0 }
  let pairs = 0
  for (let i = 0; i < n; i++) {
    const run = runTest(rand)
    counts[run.verdict!]++
    pairs += run.pairs.length
  }
  batch.value = { ...counts, meanPairs: pairs / n, n, truth: truth.value, elo1: elo1.value }
}
watch([truth, elo1], () => {
  batch.value = null
  history.value = []
})

// --- The formula, worked on the current run --------------------------------------------------------------------------
const scope = useOrProvideTermScope()
const stats = computed(() => {
  const pairs = current.value?.pairs ?? []
  if (pairs.length < 2) return null
  const scores = pairs.map((p) => p / 2)
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length
  const variance = Math.max(scores.reduce((a, s) => a + (s - mean) ** 2, 0) / scores.length, 1e-3)
  return { n: pairs.length, mean, variance, llr: llr(pairs, 0, elo1.value) }
})
watch(
  [stats, elo1],
  ([s, e1]) => {
    const s0 = expectedScore(0)
    const s1 = expectedScore(e1)
    scope.values.value = s ? { n: s.n, s0, s1, sbar: (s0 + s1) / 2, xbar: s.mean, sigma2: s.variance } : {}
    scope.expected.value = s ? { llr: s.llr } : {}
  },
  { immediate: true },
)

// --- The plot -------------------------------------------------------------------------------------------------------------
const W = 520
const H = 240
const PAD = { l: 34, r: 12, t: 12, b: 26 }
const Y = 6
const x = (i: number) => PAD.l + (i / BUDGET) * (W - PAD.l - PAD.r)
const y = (v: number) => PAD.t + ((Y - Math.max(-Y, Math.min(Y, v))) / (2 * Y)) * (H - PAD.t - PAD.b)
const pathOf = (run: Run) => run.path.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("")
const TONE: Record<Verdict, string> = { H1: palette.life400, H0: palette.queen400, inconclusive: palette.fgMuted }
const verdictText = (v: Verdict | null) =>
  v === "H1" ? "H1: A is stronger" : v === "H0" ? "H0: A is no stronger" : v === "inconclusive" ? "inconclusive: the ballot ran out" : "playing…"
const pct = (k: number) => (batch.value ? `${((k / batch.value.n) * 100).toFixed(1)}%` : "")
</script>

<template>
  <LabFrame :live="true" title="A sequential test, simulated" runtime="TypeScript" split="none">
    <template #formula>
      <MathFormula :formula="F.llr" bare />
      <MathFormula class="mt-1" :formula="F.stop" bare />
    </template>

    <div class="space-y-5">
      <div class="min-w-0">
        <svg :viewBox="`0 0 ${W} ${H}`" class="block h-auto w-full" role="img" aria-label="The LLR after each game pair" font-family="Geist Mono, monospace">
          <g font-size="10" :fill="palette.fgSubtle">
            <line :x1="PAD.l" :x2="W - PAD.r" :y1="y(0)" :y2="y(0)" :stroke="palette.line" />
            <line :x1="PAD.l" :x2="W - PAD.r" :y1="y(upper)" :y2="y(upper)" :stroke="palette.life400" stroke-dasharray="4 3" v-bind="scope.target('upper', { interactive: false })" />
            <line :x1="PAD.l" :x2="W - PAD.r" :y1="y(lower)" :y2="y(lower)" :stroke="palette.queen400" stroke-dasharray="4 3" v-bind="scope.target('lower', { interactive: false })" />
            <text :x="W - PAD.r" :y="y(upper) - 4" text-anchor="end" :fill="palette.life300">H1: stronger</text>
            <text :x="W - PAD.r" :y="y(lower) + 12" text-anchor="end" :fill="palette.queen300">H0: no stronger</text>
            <line :x1="x(20)" :x2="x(20)" :y1="PAD.t" :y2="H - PAD.b" :stroke="palette.line" stroke-dasharray="1 3" />
            <text :x="x(20) + 3" :y="PAD.t + 8">first look</text>
            <text v-for="t in [0, 50, 100, 150]" :key="t" :x="x(t)" :y="H - 8" text-anchor="middle">{{ t }}</text>
            <text :x="W - PAD.r" :y="H - 8" text-anchor="end">{{ BUDGET }} pairs</text>
            <text v-for="v in [-4, 0, 4]" :key="`v${v}`" :x="PAD.l - 6" :y="y(v) + 3" text-anchor="end">{{ v }}</text>
          </g>
          <path v-for="(run, i) in history" :key="i" :d="pathOf(run)" fill="none" :stroke="TONE[run.verdict!]" stroke-opacity="0.22" stroke-width="1.25" />
          <path v-if="current" :d="pathOf(current)" fill="none" :stroke="current.verdict ? TONE[current.verdict] : palette.fg" stroke-width="2" v-bind="scope.target('llr', { interactive: false })" />
        </svg>
        <p class="mt-2 text-xs" :style="{ color: current?.verdict ? TONE[current.verdict] : undefined }">
          {{ current ? `${verdictText(current.verdict)} · ${current.pairs.length} pairs` : "Press “Play one test”." }}
        </p>
      </div>

      <div class="min-w-0 space-y-4">
        <UiRange v-model="truth" label="How much stronger A really is" :min="-100" :max="200" :step="5" :display="(v) => `${v > 0 ? '+' : ''}${v} Elo`" />
        <div class="flex flex-wrap items-center justify-between gap-2 text-sm">
          <span class="text-fg-muted">elo1: the gain worth detecting</span>
          <UiSegmented v-model="elo1" :options="ELO1" aria-label="elo1" />
        </div>
        <div class="flex flex-wrap gap-2">
          <button class="btn-accent btn-sm" :disabled="playing" @click="playOne">Play one test</button>
          <button class="btn-ghost btn-sm" @click="runMany()">Run 1,000 tests</button>
        </div>
        <UiStats
          v-if="batch"
          :cols="2"
          :items="[
            { label: 'Says stronger (H1)', value: pct(batch.H1), tone: 'life' },
            { label: 'Says no stronger (H0)', value: pct(batch.H0), tone: 'queen' },
            { label: 'Ran out of openings', value: pct(batch.inconclusive) },
            { label: 'Pairs, on average', value: batch.meanPairs.toFixed(0) },
          ]"
        />
        <p class="text-[11px] text-fg-subtle">
          H0 is "no stronger than 0 Elo", H1 "at least elo1 stronger"; α = β = 0.05. Between the two, either answer is acceptable. The first look
          comes after 20 pairs: before that the variance is too rough to trust, and the test is wrong more often than it promises.
        </p>
      </div>
    </div>
  </LabFrame>
</template>
