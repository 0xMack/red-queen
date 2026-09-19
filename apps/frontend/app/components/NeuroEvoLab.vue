<script setup lang="ts">
// The whole neuroevolution loop, running live on XOR: a population of flat weight vectors, score every one, keep
// the best 20%, refill the population with Gaussian-mutated copies of them (the champion always survives
// unchanged). It's Evolution Strategies -- the same loop `evolve()` runs on Snake, at a size that solves in a
// second. Change the step size or population mid-run and watch what it does to the dots: each column is one
// generation, each dot one individual's fitness.
import type { ChartSeries } from "~/types/chart"
import { XOR_CASES } from "~/utils/neat"
import { EsRun, output, XOR_SHAPE, xorWeightsFitness, type EsReport } from "~/utils/neuro"

const popSize = ref(50)
const logSigma = ref(-0.5)
const sigma = computed(() => 10 ** logSigma.value)
const seed = ref(4)

const run = shallowRef(new EsRun(XOR_SHAPE, popSize.value, sigma.value, xorWeightsFitness, seed.value))
const reports = ref<EsReport[]>([])
const playing = ref(false)
let timer: ReturnType<typeof setInterval> | null = null

function reset() {
  stop()
  run.value = new EsRun(XOR_SHAPE, popSize.value, sigma.value, xorWeightsFitness, seed.value)
  reports.value = []
}
function step(n = 1) {
  for (let i = 0; i < n; i++) {
    run.value.sigma = sigma.value
    reports.value = [...reports.value, run.value.step()]
  }
}
function stop() {
  playing.value = false
  if (timer) clearInterval(timer)
  timer = null
}
function togglePlay() {
  if (playing.value) return stop()
  playing.value = true
  timer = setInterval(() => step(), 110)
}
function newSeed() {
  seed.value++
  reset()
}
watch(popSize, reset)
onUnmounted(stop)

const latest = computed(() => reports.value.at(-1) ?? null)
const solvedAt = computed(() => reports.value.find((r) => r.best > 0.98)?.generation ?? null)
const outputs = computed(() => (latest.value ? XOR_CASES.map((c) => output(latest.value!.champion, XOR_SHAPE, c.input)) : XOR_CASES.map(() => 0)))
const series = computed<ChartSeries[]>(() => [
  { key: "best", label: "best", color: "#4ade80", values: reports.value.map((r) => r.best), width: 2.5 },
  { key: "mean", label: "mean", color: "#ff5c7a", values: reports.value.map((r) => r.mean), width: 1.5 },
])

// Dot cloud: last 60 generations.
const WINDOW = 60
const cloud = computed(() => reports.value.slice(-WINDOW))
const CW = 460
const CH = 130
const lowY = 0.3
const dotX = (i: number) => 8 + (i / Math.max(WINDOW - 1, 1)) * (CW - 16)
const dotY = (f: number) => CH - 6 - ((Math.max(lowY, f) - lowY) / (1 - lowY)) * (CH - 14)
</script>

<template>
  <figure class="card my-8 p-5">
    <div class="flex flex-wrap items-center gap-2">
      <p class="eyebrow mr-auto">Live · evolving a 2-3-1 network to compute XOR</p>
      <button class="btn-primary btn-sm" @click="togglePlay">{{ playing ? "Pause" : "Play" }}</button>
      <button class="btn-ghost btn-sm" @click="step()">Step</button>
      <button class="btn-ghost btn-sm" @click="step(10)">+10</button>
      <button class="btn-ghost btn-sm" @click="newSeed">New start</button>
    </div>

    <div class="mt-4 grid gap-x-6 gap-y-2 sm:grid-cols-2">
      <label class="flex items-center gap-3 text-sm">
        <span class="w-24 shrink-0 text-fg-muted">step size σ</span>
        <input v-model.number="logSigma" type="range" min="-2" max="0.5" step="0.02" class="min-w-0 flex-1 accent-[#ff5c7a]" aria-label="mutation step size sigma">
        <span class="num w-12 text-right text-fg">{{ sigma.toFixed(sigma < 0.1 ? 3 : 2) }}</span>
      </label>
      <label class="flex items-center gap-3 text-sm">
        <span class="w-24 shrink-0 text-fg-muted">population</span>
        <input v-model.number="popSize" type="range" min="10" max="200" step="10" class="min-w-0 flex-1 accent-[#ff5c7a]" aria-label="population size">
        <span class="num w-12 text-right text-fg">{{ popSize }}</span>
      </label>
    </div>

    <div class="mt-5 grid gap-5 md:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
      <div class="min-w-0">
        <p class="text-[11px] text-fg-subtle">every individual's fitness, by generation</p>
        <svg :viewBox="`0 0 ${CW} ${CH}`" class="mt-1 block h-auto w-full rounded-lg border border-line bg-sunken" role="img" aria-label="population fitness by generation">
          <line x1="0" x2="460" :y1="dotY(0.75)" :y2="dotY(0.75)" stroke="#fbbf24" stroke-opacity="0.35" stroke-dasharray="3 3" />
          <text x="6" :y="dotY(0.75) - 3" class="fill-gold-300 font-mono text-[8px]">0.75: the best a network with no hidden layer could do</text>
          <g v-for="(r, gi) in cloud" :key="r.generation">
            <circle v-for="(f, k) in r.fitnesses" :key="k" :cx="dotX(gi)" :cy="dotY(f)" r="1.4" fill="#93c5fd" fill-opacity="0.35" />
            <circle :cx="dotX(gi)" :cy="dotY(r.best)" r="2.2" fill="#4ade80" />
          </g>
        </svg>
        <div class="mt-4 flex flex-wrap gap-4 text-xs text-fg-muted">
          <span v-for="s in series" :key="s.key" class="flex items-center gap-1.5"><span class="h-0.5 w-4 rounded" :style="{ background: s.color }" />{{ s.label }} fitness</span>
        </div>
        <LineChart v-if="reports.length > 1" class="mt-1" :x="reports.map((r) => r.generation)" :series="series" :height="150" />
        <p v-else class="mt-2 rounded-lg border border-dashed border-line p-6 text-center text-xs text-fg-subtle">Press Play or Step to start evolving.</p>
      </div>

      <div class="min-w-0">
        <p class="text-[11px] text-fg-subtle">
          champion · generation {{ latest?.generation ?? "-" }} · fitness <span class="num text-fg">{{ latest ? latest.best.toFixed(3) : "-" }}</span>
          <template v-if="solvedAt !== null"> · <span class="text-life-300">solved at generation {{ solvedAt }}</span></template>
        </p>
        <div class="mt-1 rounded-lg border border-line bg-sunken p-3">
          <NetworkDiagram v-if="latest" :weights="latest.champion" :layer-sizes="XOR_SHAPE" :input-labels="['x₀', 'x₁']" :output-labels="['XOR']" />
          <div v-else class="flex h-32 items-center justify-center text-xs text-fg-subtle">no champion yet</div>
        </div>
        <div class="mt-3"><XorTable :outputs="outputs" label="champion says" /></div>
      </div>
    </div>
  </figure>
</template>
