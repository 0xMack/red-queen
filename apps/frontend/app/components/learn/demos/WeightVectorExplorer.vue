<script setup lang="ts">
// The first idea of neuroevolution, made touchable: the whole "genome" is one flat list of numbers, and a
// network is just a reading of that list. Click a cell to see which connection it is, drag it, and watch the
// XOR outputs (and the fitness) move. This is evolve.neuro.WeightVector's real layout -- per layer, every
// weight (out×in) and then every bias -- on a 2-3-1 network small enough to see all 13 numbers at once.
import { XOR_CASES } from "~/utils/neat"
import { describeIndex, output, segments, XOR_SHAPE, xorWeightsFitness } from "~/utils/neuro"
import * as F from "~/data/math/neuroevolution"

// A hand-checked XOR solution (found by the ES in the lab below, then rounded).
const SOLVED = [3.96, -1.73, 4.86, 2.42, 1.87, 0.92, -0.8, -1.33, -2.09, -0.8, 3.96, -3.45, -2.24]

const weights = ref([...SOLVED])
const selected = ref(6)
const row = ref(1)

const layers = XOR_SHAPE
const segs = segments(layers)
const fitness = computed(() => xorWeightsFitness(weights.value))
const outputs = computed(() => XOR_CASES.map((c) => output(weights.value, layers, c.input)))
const live = computed(() => forwardActivations(weights.value, layers, XOR_CASES[row.value]!.input))

const HIDDEN = ["hidden 0", "hidden 1", "hidden 2"]
const unit = (layer: number, index: number) => (layer === 0 ? `x${index === 0 ? "₀" : "₁"}` : layer === layers.length - 2 ? "the output" : HIDDEN[index]!)
const meaning = computed(() => {
  const d = describeIndex(layers, selected.value)
  const target = d.layer === layers.length - 2 ? "the output" : HIDDEN[d.to]!
  return d.kind === "weight"
    ? `the weight on the connection from ${d.layer === 0 ? unit(0, d.from!) : HIDDEN[d.from!]!} into ${target}`
    : `the bias of ${target}`
})

function cellColor(w: number): string {
  const t = Math.min(1, Math.abs(w) / 5)
  return w >= 0 ? alpha("life400", 0.1 + 0.55 * t) : alpha("queen400", 0.1 + 0.55 * t)
}

// The output neuron as a formula, worked for the selected XOR row from the genome and the hidden activations
// (docs/design/0012); genome cells 9-12 are its weights and bias. In development the formula's output is checked
// against this explorer's own forward pass.
const scope = useOrProvideTermScope()
const OUT_TERMS: Record<number, string> = { 9: "w-out-0", 10: "w-out-1", 11: "w-out-2", 12: "b-out" }
watch(
  [weights, live],
  ([wts, act]) => {
    const hidden = act[1]!
    scope.values.value = { "w-out-0": wts[9]!, "w-out-1": wts[10]!, "w-out-2": wts[11]!, "b-out": wts[12]!, "h-0": hidden[0]!, "h-1": hidden[1]!, "h-2": hidden[2]! }
    scope.expected.value = { y: act[2]![0]! }
  },
  { immediate: true, deep: true },
)

function randomize() {
  weights.value = weights.value.map(() => (Math.random() * 2 - 1) * 2)
}
</script>

<template>
  <UiFigure>
    <div class="grid gap-5 md:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
      <div class="min-w-0">
        <p class="label">The genome, as a flat list of {{ weights.length }} numbers</p>
        <div class="mt-3 flex flex-wrap gap-x-3 gap-y-3">
          <div v-for="seg in segs" :key="`${seg.layer}${seg.kind}`">
            <p class="mb-1 text-[10px] text-fg-subtle">layer {{ seg.layer + 1 }} {{ seg.kind }}</p>
            <div class="flex gap-1">
              <button
                v-for="i in seg.end - seg.start"
                :key="seg.start + i - 1"
                class="num h-9 w-11 rounded-md border text-[11px] transition"
                :class="selected === seg.start + i - 1 ? 'border-gold-400 text-fg' : 'border-line text-fg-muted hover:border-line-strong'"
                :style="{ background: cellColor(weights[seg.start + i - 1]!) }"
                :aria-label="`genome index ${seg.start + i - 1}`"
                v-bind="OUT_TERMS[seg.start + i - 1] ? scope.target(OUT_TERMS[seg.start + i - 1]!) : {}"
                @click="selected = seg.start + i - 1"
              >
                {{ weights[seg.start + i - 1]!.toFixed(1) }}
              </button>
            </div>
          </div>
        </div>

        <div class="mt-4 well p-3 text-sm">
          <p class="text-fg-muted">
            <span class="font-mono text-xs text-gold-300">genome[{{ selected }}]</span> is {{ meaning }}.
          </p>
          <label class="mt-2 flex items-center gap-3 text-xs text-fg-subtle">
            <input v-model.number="weights[selected]" type="range" min="-6" max="6" step="0.05" class="min-w-0 flex-1" :aria-label="`value of genome index ${selected}`">
            <span class="num w-12 text-right text-fg">{{ weights[selected]!.toFixed(2) }}</span>
          </label>
        </div>

        <div class="mt-4 flex items-center justify-between gap-3">
          <p class="text-xs text-fg-subtle">
            fitness <span class="num text-base text-fg" v-bind="scope.target('fitness')">{{ fitness.toFixed(3) }}</span>
            <span class="ml-1">(1.0 = XOR solved)</span>
          </p>
          <span class="flex gap-2">
            <button class="btn-ghost btn-sm" @click="weights = [...SOLVED]">Solved</button>
            <button class="btn-ghost btn-sm" @click="randomize">Randomize</button>
          </span>
        </div>
      </div>

      <div class="min-w-0">
        <p class="label">…read as a network</p>
        <div class="mt-3 well p-3">
          <NetworkDiagram :weights="weights" :layer-sizes="layers" :activations="live" :input-labels="['x₀', 'x₁']" :output-labels="['XOR']" />
        </div>
        <div class="mt-3">
          <XorTable :outputs="outputs" :active="row" selectable @select="row = $event" />
        </div>
      </div>
    </div>
    <div class="math-panel mt-5">
      <MathFormula :formula="F.xorOutput" bare />
      <MathFormula class="mt-1" :formula="F.xorFitness" bare />
    </div>
    <template #caption>
      Nothing here is special about the numbers being a list: it's just the simplest thing an evolutionary algorithm can copy,
      mutate, and store. Click an XOR row to light up the network for that input.
    </template>
  </UiFigure>
</template>
