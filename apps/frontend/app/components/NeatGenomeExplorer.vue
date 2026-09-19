<script setup lang="ts">
// A NEAT genome you can grow by hand. Start from the minimal structure (inputs wired straight to the output, no
// hidden nodes) and apply the same mutations `evolve.neat` uses: perturb weights, add a connection, add a node
// (which *splits* a connection), toggle a gene. Every structural change gets an innovation number from a shared
// tracker -- that number is the gene's identity, and it's what lets two genomes with different structures be
// compared later. New genes are highlighted; the log says exactly what each button did.
import {
  activations,
  addConnection,
  addNode,
  complexity,
  DEFAULT_CONFIG,
  forward,
  InnovationTracker,
  initialGenome,
  legalNewConnections,
  mutateWeights,
  Rng,
  toggleConnection,
  XOR_CASES,
  xorGenomeFitness,
  type Genome,
} from "~/utils/neat"

const NODE_NAMES = ["x₀", "x₁", "bias", "out"]
const name = (id: number) => (id < 4 ? NODE_NAMES[id]! : `h${id}`)
const START = "Start: every input and the bias wired to the output. No hidden nodes."

function fresh(seed: number) {
  const rng = new Rng(seed)
  const tracker = new InnovationTracker(2 + 1 + 1)
  return { rng, tracker, genome: initialGenome(2, 1, tracker, rng) }
}

const seed = ref(11)
const state = shallowRef(fresh(seed.value))
const genome = computed(() => state.value.genome)
const highlight = ref<number[]>([])
const log = ref<string[]>([START])
const row = ref(1)

const size = computed(() => complexity(genome.value))
const fitness = computed(() => xorGenomeFitness(genome.value))
const outputs = computed(() => XOR_CASES.map((c) => forward(genome.value, c.input)[0]!))
const live = computed(() => activations(genome.value, XOR_CASES[row.value]!.input))
const canConnect = computed(() => legalNewConnections(genome.value).length > 0)

function apply(kind: "weights" | "connection" | "node" | "toggle") {
  const { rng, tracker } = state.value
  const before: Genome = genome.value
  let next: Genome
  if (kind === "weights") next = mutateWeights(before, { ...DEFAULT_CONFIG, weightReplaceRate: 0.15 }, rng)
  else if (kind === "connection") next = addConnection(before, tracker, DEFAULT_CONFIG, rng)
  else if (kind === "node") next = addNode(before, tracker, DEFAULT_CONFIG, rng)
  else next = toggleConnection(before, rng)

  const known = new Set(before.genes.map((c) => c.innovation))
  const added = next.genes.filter((c) => !known.has(c.innovation))
  const flipped = next.genes.filter((c) => known.has(c.innovation) && before.genes.find((b) => b.innovation === c.innovation)!.enabled !== c.enabled)
  highlight.value = [...added, ...flipped].map((c) => c.innovation)

  let line: string
  if (kind === "weights") {
    line = "Perturbed every weight (a few replaced outright). Structure unchanged."
  } else if (kind === "toggle") {
    const c = flipped[0]!
    line = `Toggled innovation ${c.innovation} (${name(c.source)} → ${name(c.target)}) ${c.enabled ? "on" : "off"}.`
  } else if (next === before) {
    line = kind === "node" ? "No connection could be split." : "No legal new connection (every allowed pair already exists)."
  } else if (kind === "connection") {
    const c = added[0]!
    line = `Added connection ${name(c.source)} → ${name(c.target)} as innovation ${c.innovation}, weight ${c.weight.toFixed(2)}.`
  } else {
    const off = flipped[0]!
    const [into, out] = added
    line =
      `Split ${name(off.source)} → ${name(off.target)} (innovation ${off.innovation}, now disabled) with new node ${name(into!.target)}: ` +
      `${name(into!.source)} → ${name(into!.target)} weight 1.00, ${name(out!.source)} → ${name(out!.target)} keeps the old weight ${out!.weight.toFixed(2)}.`
  }
  log.value = [line, ...log.value].slice(0, 6)
  state.value = { ...state.value, genome: next }
}

function reset() {
  seed.value++
  state.value = fresh(seed.value)
  highlight.value = []
  log.value = [START]
}
</script>

<template>
  <figure class="card my-8 p-5">
    <div class="flex flex-wrap items-center gap-2">
      <p class="eyebrow mr-auto">Grow a genome · {{ size.hidden }} hidden nodes · {{ size.connections }} enabled connections</p>
      <button class="btn-ghost btn-sm" @click="apply('weights')">Perturb weights</button>
      <button class="btn-ghost btn-sm" :disabled="!canConnect" @click="apply('connection')">Add connection</button>
      <button class="btn-primary btn-sm" @click="apply('node')">Add node</button>
      <button class="btn-ghost btn-sm" @click="apply('toggle')">Toggle a gene</button>
      <button class="btn-ghost btn-sm" @click="reset">Start over</button>
    </div>

    <div class="mt-4 grid gap-5 md:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
      <div class="min-w-0">
        <div class="rounded-lg border border-line bg-sunken p-3">
          <NeatDiagram :genome="genome" :activations="live" :highlight="highlight" :input-labels="['x₀', 'x₁']" :output-labels="['XOR']" show-innovations />
        </div>
        <p class="mt-2 text-[11px] text-fg-subtle">
          numbers on the edges are innovation numbers · <span class="text-gold-300">gold</span> = just changed · dashed grey = disabled (still in the genome)
        </p>
        <div class="mt-3">
          <XorTable :outputs="outputs" :active="row" selectable @select="row = $event" />
          <p class="mt-1 text-xs text-fg-subtle">fitness <span class="num text-fg">{{ fitness.toFixed(3) }}</span> · a network with no hidden node can score at most 0.75</p>
        </div>
      </div>

      <div class="min-w-0">
        <p class="text-[11px] tracking-wide text-fg-subtle uppercase">The genome: one row per connection gene</p>
        <div class="mt-1 max-h-60 overflow-y-auto rounded-lg border border-line">
          <table class="w-full text-xs">
            <thead class="sticky top-0 bg-raised text-left text-[10px] tracking-wide text-fg-subtle uppercase">
              <tr>
                <th class="px-2 py-1.5 font-medium">innov</th>
                <th class="font-medium">connection</th>
                <th class="font-medium">weight</th>
                <th class="pr-2 font-medium">on</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="c in genome.genes" :key="c.innovation" class="border-t border-line" :class="highlight.includes(c.innovation) ? 'bg-gold-400/10' : ''">
                <td class="num px-2 py-1 text-fg">{{ c.innovation }}</td>
                <td class="num py-1" :class="c.enabled ? 'text-fg-muted' : 'text-fg-subtle line-through'">{{ name(c.source) }} → {{ name(c.target) }}</td>
                <td class="num py-1" :class="c.weight >= 0 ? 'text-life-300' : 'text-queen-300'">{{ c.weight.toFixed(2) }}</td>
                <td class="py-1 pr-2" :class="c.enabled ? 'text-life-300' : 'text-fg-subtle'">{{ c.enabled ? "✓" : "–" }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <ul class="mt-3 space-y-1.5 text-xs text-fg-muted">
          <li v-for="(line, i) in log" :key="line + i" :class="i === 0 ? 'text-fg' : 'text-fg-subtle'">{{ i === 0 ? "▸" : "·" }} {{ line }}</li>
        </ul>
      </div>
    </div>
  </figure>
</template>
