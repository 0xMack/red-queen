<script setup lang="ts">
// NEAT running live on XOR: 150 genomes that start with no hidden nodes, and whatever structure the run ends up
// with, it grew itself. Watch three things -- the fitness curve (jumps when a useful hidden node is found), the
// species bars (each colour is one species; height = its share of the population), and the champion's graph.
// Then flip speciation off and run the batch experiment: same algorithm, same budget, only the one mechanism
// removed. This is the utils/neat.ts port of evolve.neat.evolve_neat, so what you see here is the real algorithm.
import type { ChartSeries } from "~/types/chart"
import { complexity, DEFAULT_CONFIG, forward, NeatRun, XOR_CASES, xorGenomeFitness, type GenerationReport } from "~/utils/neat"

const POP = 150
const speciation = ref(true)
const threshold = ref(3)
const seed = ref(2)

function makeRun() {
  return new NeatRun(2, 1, POP, { ...DEFAULT_CONFIG, speciation: speciation.value, compatibilityThreshold: threshold.value }, xorGenomeFitness, seed.value)
}
const run = shallowRef(makeRun())
const reports = ref<GenerationReport[]>([])
const playing = ref(false)
let timer: ReturnType<typeof setInterval> | null = null

function stop() {
  playing.value = false
  if (timer) clearInterval(timer)
  timer = null
}
function reset() {
  stop()
  run.value = makeRun()
  reports.value = []
}
function step(n = 1) {
  const fresh: GenerationReport[] = []
  for (let i = 0; i < n; i++) fresh.push(run.value.step())
  reports.value = [...reports.value, ...fresh]
}
function togglePlay() {
  if (playing.value) return stop()
  playing.value = true
  timer = setInterval(() => step(), 120)
}
watch([speciation, threshold], reset)
onUnmounted(stop)

const latest = computed(() => reports.value.at(-1) ?? null)
const size = computed(() => (latest.value ? complexity(latest.value.champion) : null))
const solvedAt = computed(() => reports.value.find((r) => r.best > 0.98)?.generation ?? null)
const outputs = computed(() => (latest.value ? XOR_CASES.map((c) => forward(latest.value!.champion, c.input)[0]!) : XOR_CASES.map(() => 0)))
const series = computed<ChartSeries[]>(() => [
  { key: "best", label: "best", color: "#4ade80", values: reports.value.map((r) => r.best), width: 2.5 },
  { key: "mean", label: "mean", color: "#ff5c7a", values: reports.value.map((r) => r.mean), width: 1.5 },
])

// Species bars over the last WINDOW generations.
const PALETTE = ["#ff5c7a", "#60a5fa", "#4ade80", "#fbbf24", "#c084fc", "#2dd4bf", "#fb923c", "#94a3b8", "#f472b6", "#a3e635"]
const WINDOW = 70
const BW = 460
const BH = 96
const bars = computed(() => {
  const recent = reports.value.slice(-WINDOW)
  const w = BW / Math.max(recent.length, 1)
  return recent.map((r, i) => {
    let y = BH
    return {
      key: r.generation,
      x: i * w,
      w: Math.max(w - 0.5, 0.5),
      segments: [...r.speciesSizes]
        .sort((a, b) => a.id - b.id)
        .map((s) => {
          const h = (s.size / POP) * BH
          y -= h
          return { id: s.id, y, h }
        }),
    }
  })
})

// The batch experiment: 8 runs each with speciation on and off, same budget.
interface BatchRow { speciation: boolean; solved: number; runs: number; medianGen: number | null; meanHidden: number }
const batch = ref<BatchRow[] | null>(null)
const batchProgress = ref(0)
const batching = ref(false)
const BATCH_RUNS = 8
const BATCH_GENERATIONS = 80

async function runBatch() {
  stop()
  batching.value = true
  batch.value = null
  const rows: BatchRow[] = []
  let done = 0
  for (const on of [true, false]) {
    const solvedGens: number[] = []
    let hidden = 0
    for (let s = 0; s < BATCH_RUNS; s++) {
      const r = new NeatRun(2, 1, POP, { ...DEFAULT_CONFIG, speciation: on }, xorGenomeFitness, 1000 + s)
      let last: GenerationReport | null = null
      let solved: number | null = null
      for (let g = 0; g < BATCH_GENERATIONS; g++) {
        last = r.step()
        if (solved === null && last.best > 0.98) solved = g
      }
      if (solved !== null) solvedGens.push(solved)
      hidden += complexity(last!.champion).hidden
      batchProgress.value = ++done / (BATCH_RUNS * 2)
      await new Promise((resolve) => setTimeout(resolve)) // let the page paint between runs
    }
    solvedGens.sort((a, b) => a - b)
    rows.push({ speciation: on, solved: solvedGens.length, runs: BATCH_RUNS, medianGen: solvedGens.length ? solvedGens[Math.floor(solvedGens.length / 2)]! : null, meanHidden: hidden / BATCH_RUNS })
  }
  batch.value = rows
  batching.value = false
}
</script>

<template>
  <figure class="card my-8 p-5">
    <div class="flex flex-wrap items-center gap-2">
      <p class="eyebrow mr-auto">Live · NEAT evolving XOR from zero hidden nodes</p>
      <button class="btn-primary btn-sm" @click="togglePlay">{{ playing ? "Pause" : "Play" }}</button>
      <button class="btn-ghost btn-sm" @click="step()">Step</button>
      <button class="btn-ghost btn-sm" @click="step(10)">+10</button>
      <button class="btn-ghost btn-sm" @click="seed++; reset()">New start</button>
    </div>

    <div class="mt-4 grid gap-x-6 gap-y-3 sm:grid-cols-2">
      <label class="flex items-center gap-3 text-sm">
        <input v-model="speciation" type="checkbox" class="size-4 accent-[#ff5c7a]">
        <span class="text-fg-muted">speciation on</span>
      </label>
      <label class="flex items-center gap-3 text-sm" :class="speciation ? '' : 'opacity-40'">
        <span class="w-24 shrink-0 text-fg-muted">threshold δₜ</span>
        <input v-model.number="threshold" type="range" min="0.5" max="6" step="0.1" :disabled="!speciation" class="min-w-0 flex-1 accent-[#ff5c7a]" aria-label="compatibility threshold">
        <span class="num w-8 text-right text-fg">{{ threshold.toFixed(1) }}</span>
      </label>
    </div>

    <div class="mt-5 grid gap-5 md:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
      <div class="min-w-0">
        <div class="flex flex-wrap gap-4 text-xs text-fg-muted">
          <span v-for="s in series" :key="s.key" class="flex items-center gap-1.5"><span class="h-0.5 w-4 rounded" :style="{ background: s.color }" />{{ s.label }} fitness</span>
        </div>
        <LineChart v-if="reports.length > 1" class="mt-1" :x="reports.map((r) => r.generation)" :series="series" :height="160" />
        <p v-else class="mt-2 rounded-lg border border-dashed border-line p-8 text-center text-xs text-fg-subtle">Press Play or Step to start evolving.</p>

        <p class="mt-4 text-[11px] text-fg-subtle">
          species · <span class="num text-fg">{{ latest?.speciesSizes.length ?? "-" }}</span> now (each colour is one species; height = share of the population)
        </p>
        <svg :viewBox="`0 0 ${BW} ${BH}`" class="mt-1 block h-auto w-full rounded-lg border border-line bg-sunken" role="img" aria-label="species share of the population by generation">
          <g v-for="b in bars" :key="b.key">
            <rect v-for="s in b.segments" :key="s.id" :x="b.x" :y="s.y" :width="b.w" :height="s.h" :fill="PALETTE[s.id % PALETTE.length]" fill-opacity="0.85" />
          </g>
        </svg>
      </div>

      <div class="min-w-0">
        <p class="text-[11px] text-fg-subtle">
          champion · generation {{ latest?.generation ?? "-" }} · fitness <span class="num text-fg">{{ latest ? latest.best.toFixed(3) : "-" }}</span>
          <template v-if="size"> · <span class="num text-fg">{{ size.hidden }}</span> hidden, <span class="num text-fg">{{ size.connections }}</span> connections</template>
          <template v-if="solvedAt !== null"> · <span class="text-life-300">solved at generation {{ solvedAt }}</span></template>
        </p>
        <div class="mt-1 rounded-lg border border-line bg-sunken p-3">
          <NeatDiagram v-if="latest" :genome="latest.champion" :input-labels="['x₀', 'x₁']" :output-labels="['XOR']" />
          <div v-else class="flex h-32 items-center justify-center text-xs text-fg-subtle">no champion yet</div>
        </div>
        <div class="mt-3"><XorTable :outputs="outputs" label="champion says" /></div>
      </div>
    </div>

    <div class="mt-6 rounded-lg border border-line bg-sunken p-4">
      <div class="flex flex-wrap items-center justify-between gap-2">
        <p class="text-sm font-medium text-fg">What does speciation buy? Run it {{ BATCH_RUNS }} times each way.</p>
        <button class="btn-ghost btn-sm" :disabled="batching" @click="runBatch">{{ batching ? `Running… ${Math.round(batchProgress * 100)}%` : batch ? "Run again" : "Run the experiment" }}</button>
      </div>
      <table v-if="batch" class="mt-3 w-full text-xs">
        <thead class="text-left text-[10px] tracking-wide text-fg-subtle uppercase">
          <tr><th class="py-1 font-medium">setting</th><th class="font-medium">solved XOR</th><th class="font-medium">median generation</th><th class="font-medium">hidden nodes (mean)</th></tr>
        </thead>
        <tbody>
          <tr v-for="row in batch" :key="String(row.speciation)" class="border-t border-line">
            <td class="py-1.5 text-fg">{{ row.speciation ? "speciation on" : "speciation off" }}</td>
            <td class="num" :class="row.solved === row.runs ? 'text-life-300' : 'text-queen-300'">{{ row.solved }} / {{ row.runs }}</td>
            <td class="num text-fg-muted">{{ row.medianGen ?? "-" }}</td>
            <td class="num text-fg-muted">{{ row.meanHidden.toFixed(1) }}</td>
          </tr>
        </tbody>
      </table>
      <p v-else class="mt-2 text-xs text-fg-subtle">
        {{ BATCH_GENERATIONS }} generations, population {{ POP }}, seeds 1000–{{ 1000 + BATCH_RUNS - 1 }}, identical budgets. "Solved" = best fitness above 0.98.
      </p>
    </div>
  </figure>
</template>
