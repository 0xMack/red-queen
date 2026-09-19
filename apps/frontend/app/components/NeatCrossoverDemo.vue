<script setup lang="ts">
// Crossover between two genomes with *different structures*, done by lining genes up on innovation number. Each row
// is one innovation. Matching genes (in both parents) are inherited from a random parent; disjoint and excess genes
// (in one parent only) come only from the fitter parent; a gene disabled in either parent is usually disabled in the
// child. Swap which parent is fitter, or re-roll the coin flips, and watch the child change. This is
// evolve.neat.crossover -- the same code, ported.
import { align, crossover, DEFAULT_CONFIG, Rng, type Genome } from "~/utils/neat"
import { INNOVATION_EDGES, PARENT_A, PARENT_B } from "~/utils/neatExamples"

const fitter = ref<"A" | "B">("B")
const seed = ref(3)

const NODE = ["x₀", "x₁", "x₂", "bias", "out"]
const nodeName = (id: number) => NODE[id] ?? `h${id}`

const alignment = computed(() => align(PARENT_A, PARENT_B))
const child = computed<Genome>(() => {
  const [first, second] = fitter.value === "A" ? [PARENT_A, PARENT_B] : [PARENT_B, PARENT_A]
  return crossover(first, second, DEFAULT_CONFIG, new Rng(seed.value))
})

type Kind = "matching" | "disjoint" | "excess"
const rows = computed(() => {
  const a = new Map(PARENT_A.genes.map((g) => [g.innovation, g]))
  const b = new Map(PARENT_B.genes.map((g) => [g.innovation, g]))
  const c = new Map(child.value.genes.map((g) => [g.innovation, g]))
  const excess = new Set([...alignment.value.excessA, ...alignment.value.excessB].map((g) => g.innovation))
  return Object.keys(INNOVATION_EDGES)
    .map(Number)
    .map((i) => {
      const kind: Kind | null = a.has(i) && b.has(i) ? "matching" : excess.has(i) ? "excess" : a.has(i) || b.has(i) ? "disjoint" : null
      const [source, target] = INNOVATION_EDGES[i]!
      return { i, a: a.get(i), b: b.get(i), c: c.get(i), kind, edge: `${nodeName(source)} → ${nodeName(target)}` }
    })
})

const KIND_STYLE: Record<Kind, string> = {
  matching: "bg-life-400/10 text-life-300",
  disjoint: "bg-gold-400/10 text-gold-300",
  excess: "bg-signal-400/10 text-signal-300",
}
const fmt = (g?: { weight: number; enabled: boolean }) => (g ? `${g.weight.toFixed(1)}${g.enabled ? "" : " ✗"}` : "")
</script>

<template>
  <figure class="card my-8 p-5">
    <div class="flex flex-wrap items-center gap-2">
      <p class="eyebrow mr-auto">Crossover by innovation number</p>
      <span class="text-xs text-fg-muted">fitter parent</span>
      <div class="flex rounded-lg border border-line bg-sunken p-0.5">
        <button v-for="p in ['A', 'B'] as const" :key="p" class="num rounded-md px-3 py-1 text-xs transition" :class="fitter === p ? 'bg-raised text-fg shadow' : 'text-fg-subtle hover:text-fg'" @click="fitter = p">{{ p }}</button>
      </div>
      <button class="btn-ghost btn-sm" @click="seed++">Re-roll coin flips</button>
    </div>

    <div class="mt-4 overflow-x-auto rounded-lg border border-line">
      <table class="w-full min-w-[480px] text-xs">
        <thead class="bg-raised text-left text-[10px] tracking-wide text-fg-subtle uppercase">
          <tr>
            <th class="px-3 py-1.5 font-medium">innov</th>
            <th class="font-medium">connection</th>
            <th class="font-medium">parent A</th>
            <th class="font-medium">parent B</th>
            <th class="font-medium">kind</th>
            <th class="pr-3 font-medium">child</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in rows" :key="r.i" class="border-t border-line">
            <td class="num px-3 py-1.5 text-fg">{{ r.i }}</td>
            <td class="num text-fg-muted">{{ r.edge }}</td>
            <td class="num" :class="r.a?.enabled === false ? 'text-fg-subtle' : 'text-fg'">{{ fmt(r.a) || "·" }}</td>
            <td class="num" :class="r.b?.enabled === false ? 'text-fg-subtle' : 'text-fg'">{{ fmt(r.b) || "·" }}</td>
            <td>
              <span v-if="r.kind" class="rounded px-1.5 py-0.5 text-[10px]" :class="KIND_STYLE[r.kind]">{{ r.kind }}</span>
            </td>
            <td class="num pr-3" :class="r.c ? (r.c.enabled ? 'text-fg' : 'text-fg-subtle') : 'text-fg-subtle'">
              {{ fmt(r.c) || "·" }}
              <span v-if="r.c && r.kind === 'matching'" class="ml-1 text-[10px] text-fg-subtle">({{ r.c.weight === r.a?.weight ? "A" : "B" }})</span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="mt-2 text-[11px] text-fg-subtle">✗ = gene disabled in that genome · · = the parent doesn't have this gene</p>

    <div class="mt-4 grid gap-3 sm:grid-cols-3">
      <div v-for="(g, label) in { 'parent A': PARENT_A, 'parent B': PARENT_B, child: child }" :key="label" class="rounded-lg border border-line bg-sunken p-2">
        <p class="mb-1 text-center text-[11px] text-fg-subtle">{{ label }}</p>
        <NeatDiagram :genome="g" :input-labels="['x₀', 'x₁', 'x₂']" :output-labels="['out']" :height="150" />
      </div>
    </div>

    <figcaption class="mt-3 text-xs text-fg-subtle">
      Without the innovation column there would be no way to know that A's connection <span class="num">x₁ → h5</span> and B's are the
      <em>same gene</em> -- or that B's <span class="num">h5 → h6</span> is one A has never seen. The child keeps only genes the fitter parent has:
      matching genes are a coin flip, the rest are inherited, never invented.
    </figcaption>
  </figure>
</template>
