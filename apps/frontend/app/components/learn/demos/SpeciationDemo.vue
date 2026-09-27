<script setup lang="ts">
// Speciation, in two steps. Top: the compatibility distance δ = c₁·E/N + c₂·D/N + c₃·W̄ between the two parents from
// the crossover demo, broken into its three terms (E excess genes, D disjoint genes, W̄ mean weight difference over
// matching genes), with the coefficients and the threshold δₜ as sliders. Bottom: a dozen genomes drawn from the same
// pool of ten possible genes, each shown as a barcode (which innovations it has), assigned to species exactly the way
// evolve.neat does it -- compare to each species' representative in turn, join the first one within δₜ, else found a
// new species. Slide δₜ and watch the population split and merge.
import { compatibility, Rng, type Genome, type Gene } from "~/utils/neat"
import { INNOVATION_EDGES, PARENT_A, PARENT_B } from "~/utils/neatExamples"

const c1 = ref(1)
const c2 = ref(1)
const c3 = ref(0.4)
const threshold = ref(6)
const seed = ref(7)

// Built here, not inline in the template: the template auto-unwraps top-level refs, so an inline list would hand
// each slider a plain number instead of the ref it needs to write to.
const sliders = [
  { name: "c₁ excess", model: c1, max: 3 },
  { name: "c₂ disjoint", model: c2, max: 3 },
  { name: "c₃ weights", model: c3, max: 3 },
  { name: "δₜ threshold", model: threshold, max: 10 },
]

const coefficients = computed(() => ({ excessCoefficient: c1.value, disjointCoefficient: c2.value, weightCoefficient: c3.value }))
const parts = computed(() => compatibility(PARENT_A, PARENT_B, coefficients.value))
const terms = computed(() => [
  { label: `c₁·E/N = ${c1.value.toFixed(1)}·${parts.value.excess}/${parts.value.n}`, value: (c1.value * parts.value.excess) / parts.value.n, color: "bg-signal-400" },
  { label: `c₂·D/N = ${c2.value.toFixed(1)}·${parts.value.disjoint}/${parts.value.n}`, value: (c2.value * parts.value.disjoint) / parts.value.n, color: "bg-gold-400" },
  { label: `c₃·W̄ = ${c3.value.toFixed(1)}·${parts.value.meanWeightDiff.toFixed(2)}`, value: c3.value * parts.value.meanWeightDiff, color: "bg-life-400" },
])
const scaleMax = computed(() => Math.max(parts.value.distance, threshold.value) * 1.15 || 1)

// A population of possible genomes: A and B, then random subsets of the ten genes with random weights.
const INNOVATIONS = Object.keys(INNOVATION_EDGES).map(Number)
const population = computed<Genome[]>(() => {
  const rng = new Rng(seed.value)
  const others = Array.from({ length: 10 }, () => {
    const genes: Gene[] = INNOVATIONS.filter(() => rng.random() < 0.55).map((innovation) => {
      const [source, target] = INNOVATION_EDGES[innovation]!
      return { innovation, source, target, weight: rng.uniform(-1, 1), enabled: true }
    })
    return { numInputs: 3, numOutputs: 1, genes: genes.length ? genes : [{ innovation: 1, source: 0, target: 4, weight: 0.5, enabled: true }] }
  })
  return [PARENT_A, PARENT_B, ...others]
})

const PALETTE = [palette.queen400, palette.signal400, palette.life400, palette.gold400, palette.violet400, palette.teal400, palette.orange400, palette.fgMuted, palette.pink400, palette.life300, palette.signal300, palette.pink400]
const assignment = computed(() => {
  const reps: { genome: Genome }[] = []
  return population.value.map((g) => {
    let index = reps.findIndex((r) => compatibility(g, r.genome, coefficients.value).distance < threshold.value)
    let distance = index >= 0 ? compatibility(g, reps[index]!.genome, coefficients.value).distance : 0
    if (index < 0) {
      reps.push({ genome: g })
      index = reps.length - 1
      distance = 0
    }
    return { genome: g, species: index, distance }
  })
})
const speciesCount = computed(() => new Set(assignment.value.map((a) => a.species)).size)
const grouped = computed(() => [...assignment.value].sort((x, y) => x.species - y.species || x.distance - y.distance))
const label = (i: number) => (i === 0 ? "A" : i === 1 ? "B" : `g${i - 1}`)
</script>

<template>
  <UiFigure title="Compatibility distance · parents A and B from the crossover demo">

    <div class="mt-4 grid gap-x-6 gap-y-2 sm:grid-cols-2">
      <UiRange v-for="s in sliders" :key="s.name" v-model="s.model.value" :label="s.name" :min="0" :max="s.max" :step="0.1" :display="(v) => v.toFixed(1)" />
    </div>

    <div class="mt-4 well p-4">
      <div class="relative h-6 overflow-hidden rounded-md bg-line/60">
        <div class="flex h-full">
          <div v-for="t in terms" :key="t.label" class="h-full" :class="t.color" :style="{ width: `${(t.value / scaleMax) * 100}%` }" :title="t.label" />
        </div>
        <div class="absolute inset-y-0 w-0.5 bg-fg" :style="{ left: `${(threshold / scaleMax) * 100}%` }" title="threshold δₜ" />
      </div>
      <div class="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-fg-muted">
        <span v-for="t in terms" :key="t.label" class="flex items-center gap-1.5"><span class="size-2 rounded-sm" :class="t.color" />{{ t.label }} = <span class="num text-fg">{{ t.value.toFixed(2) }}</span></span>
      </div>
      <p class="mt-3 text-sm">
        δ = <span class="num text-fg">{{ parts.distance.toFixed(2) }}</span>
        <span class="text-fg-subtle"> vs threshold </span><span class="num text-fg">{{ threshold.toFixed(1) }}</span>
        → <span :class="parts.distance < threshold ? 'text-life-300' : 'text-queen-300'">{{ parts.distance < threshold ? "same species" : "different species" }}</span>
        <span class="ml-2 text-[11px] text-fg-subtle">(N = {{ parts.n }}: genomes under 20 genes aren't size-normalized)</span>
      </p>
    </div>

    <div class="mt-6 flex flex-wrap items-baseline justify-between gap-2">
      <p class="label">A population, sorted into <span class="num text-fg">{{ speciesCount }}</span> species</p>
      <button class="btn-ghost btn-sm" @click="seed++">New population</button>
    </div>
    <div class="mt-3 space-y-1">
      <div v-for="a in grouped" :key="population.indexOf(a.genome)" class="flex items-center gap-3 well py-1 pr-2 pl-1">
        <span class="h-6 w-1.5 rounded-full" :style="{ background: PALETTE[a.species % PALETTE.length] }" />
        <span class="num w-6 text-[11px] text-fg-muted">{{ label(population.indexOf(a.genome)) }}</span>
        <div class="flex flex-1 gap-0.5">
          <span
            v-for="i in INNOVATIONS"
            :key="i"
            class="num flex h-5 min-w-0 flex-1 items-center justify-center rounded-sm text-[9px]"
            :class="a.genome.genes.some((g) => g.innovation === i) ? 'text-bg' : 'bg-line/40 text-fg-subtle/50'"
            :style="a.genome.genes.some((g) => g.innovation === i) ? { background: PALETTE[a.species % PALETTE.length] } : undefined"
          >{{ i }}</span>
        </div>
        <span class="num w-16 text-right text-[10px] text-fg-subtle">{{ a.distance === 0 ? "founder" : `δ ${a.distance.toFixed(1)}` }}</span>
      </div>
    </div>
    <template #caption>
      Each barcode lists which of the ten possible genes a genome has. Genomes that share most genes and have similar weights land in the same
      colour. Fitness sharing then divides each member's fitness by its species' size -- so a species can't take over the population just by being
      big, and a small new one isn't wiped out before its weights have been tuned.
    </template>
  </UiFigure>
</template>
