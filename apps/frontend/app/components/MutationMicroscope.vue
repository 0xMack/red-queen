<script setup lang="ts">
// What Gaussian mutation actually does to fitness. One parent (a nearly-solved XOR network, fitness 0.92) is
// mutated 400 times -- every weight gets independent N(0, σ) noise, exactly evolve.neuro.GaussianMutation --
// and the histogram shows how each child's fitness compares with its parent's. Tiny σ: almost every child is a
// near-copy (safe, but nothing to select on). Large σ: children are essentially random networks, nearly all of
// them worse. The useful step size lives in between -- which is why σ is the one number worth tuning.
import { Rng } from "~/utils/neat"
import { xorWeightsFitness } from "~/utils/neuro"

const PARENT = [4.09, -1.46, 4.71, 2.11, 2.05, 1.22, -0.23, -0.97, -1.55, -1.35, 2.9, -0.97, -0.87]
const parentFitness = xorWeightsFitness(PARENT)

const logSigma = ref(-0.7) // sigma = 10^logSigma
const sigma = computed(() => 10 ** logSigma.value)
const seed = ref(1)
const CHILDREN = 400

const deltas = computed(() => {
  const rng = new Rng(seed.value)
  return Array.from({ length: CHILDREN }, () => xorWeightsFitness(PARENT.map((w) => w + rng.gauss(0, sigma.value))) - parentFitness)
})

const LOW = -0.6
const HIGH = 0.1
const BINS = 35
const bins = computed(() => {
  const counts = Array<number>(BINS).fill(0)
  for (const d of deltas.value) counts[Math.min(BINS - 1, Math.max(0, Math.floor(((d - LOW) / (HIGH - LOW)) * BINS)))]!++
  return counts
})
const tallest = computed(() => Math.max(1, ...bins.value))
const better = computed(() => deltas.value.filter((d) => d > 1e-9).length / CHILDREN)
const worse = computed(() => deltas.value.filter((d) => d < -1e-9).length / CHILDREN)
const best = computed(() => Math.max(...deltas.value))

const W = 460
const H = 150
const zeroX = ((0 - LOW) / (HIGH - LOW)) * W
</script>

<template>
  <figure class="card my-8 p-5">
    <div class="flex flex-wrap items-baseline justify-between gap-2">
      <p class="eyebrow">Mutation microscope · one parent, {{ CHILDREN }} children</p>
      <button class="btn-ghost btn-sm" @click="seed++">Resample</button>
    </div>

    <label class="mt-4 flex items-center gap-3 text-sm">
      <span class="w-24 shrink-0 text-fg-muted">step size σ</span>
      <input v-model.number="logSigma" type="range" min="-2" max="0.7" step="0.02" class="min-w-0 flex-1 accent-[#ff5c7a]" aria-label="mutation step size sigma">
      <span class="num w-14 text-right text-fg">{{ sigma.toFixed(sigma < 0.1 ? 3 : 2) }}</span>
    </label>

    <svg :viewBox="`0 0 ${W} ${H + 22}`" class="mt-4 block h-auto w-full" role="img" aria-label="histogram of child fitness minus parent fitness">
      <g v-for="(count, i) in bins" :key="i">
        <rect
          :x="(i / BINS) * W + 1"
          :y="H - (count / tallest) * (H - 8)"
          :width="W / BINS - 2"
          :height="(count / tallest) * (H - 8)"
          :fill="LOW + ((i + 0.5) / BINS) * (HIGH - LOW) > 0 ? '#4ade80' : '#ff5c7a'"
          fill-opacity="0.75"
          rx="1.5"
        />
      </g>
      <line :x1="zeroX" :x2="zeroX" y1="0" :y2="H" stroke="#e9ebf1" stroke-dasharray="3 3" stroke-opacity="0.6" />
      <text :x="zeroX + 4" y="10" class="fill-fg-muted font-mono text-[9px]">same as parent</text>
      <text x="0" :y="H + 14" class="fill-fg-subtle font-mono text-[9px]">much worse</text>
      <text :x="W" :y="H + 14" text-anchor="end" class="fill-fg-subtle font-mono text-[9px]">better →</text>
    </svg>

    <div class="mt-3 grid grid-cols-3 gap-2 text-center">
      <div class="rounded-lg border border-line bg-sunken px-3 py-2">
        <p class="text-[10px] tracking-wide text-fg-subtle uppercase">better</p>
        <p class="num text-lg text-life-300">{{ Math.round(better * 100) }}%</p>
      </div>
      <div class="rounded-lg border border-line bg-sunken px-3 py-2">
        <p class="text-[10px] tracking-wide text-fg-subtle uppercase">worse</p>
        <p class="num text-lg text-queen-300">{{ Math.round(worse * 100) }}%</p>
      </div>
      <div class="rounded-lg border border-line bg-sunken px-3 py-2">
        <p class="text-[10px] tracking-wide text-fg-subtle uppercase">best child</p>
        <p class="num text-lg text-fg">{{ best >= 0 ? "+" : "" }}{{ best.toFixed(3) }}</p>
      </div>
    </div>
    <figcaption class="mt-3 text-xs text-fg-subtle">
      Parent fitness {{ parentFitness.toFixed(3) }}. Selection can only work with what mutation offers: at σ = 0.01 every
      child lands within a hair of its parent's fitness (almost nothing to choose between); at σ = 5 nearly every child is far worse.
    </figcaption>
  </figure>
</template>
