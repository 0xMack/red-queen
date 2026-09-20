<script setup lang="ts">
// A NEAT genome drawn as the graph it is: inputs (and the bias) on the left, outputs on the right, and every
// hidden node in a column by its depth in between -- so as structure grows you can watch layers appear, which
// a fixed-shape NetworkDiagram can't show. Green = excitatory, red = inhibitory, thickness = |weight|;
// disabled genes stay visible (dashed) because they are still part of the genome and still count in crossover
// and speciation. With `activations` (utils/neat.ts activations()) nodes light up and edges show their live
// contribution; `highlight` marks genes by innovation number (e.g. the ones a mutation just added).
import { biasId, hiddenIds, outputIds, type Genome } from "~/utils/neat"

const props = withDefaults(
  defineProps<{
    genome: Genome
    activations?: Map<number, number> | null
    highlight?: number[]
    inputLabels?: string[]
    outputLabels?: string[]
    showDisabled?: boolean
    showInnovations?: boolean
    // Fixed drawing height in viewBox units; default fits the tallest column.
    height?: number
    /** Fill the parent's height (aspect ratio kept) instead of sizing to the parent's width. */
    fit?: boolean
  }>(),
  { activations: null, highlight: () => [], inputLabels: () => [], outputLabels: () => [], showDisabled: true, showInnovations: false, height: 0, fit: false },
)

const W = 460
const padY = 22
const labelW = computed(() => (props.inputLabels.length || props.outputLabels.length ? 84 : 26))

// Depth of every node = longest path from an input, over all genes (enabled or not) so toggling a gene
// never rearranges the drawing.
const columns = computed(() => {
  const g = props.genome
  const incoming = new Map<number, number[]>()
  for (const c of g.genes) (incoming.get(c.target) ?? incoming.set(c.target, []).get(c.target)!).push(c.source)
  const first = g.numInputs + 1 + g.numOutputs
  const memo = new Map<number, number>()
  const depth = (n: number): number => {
    if (n < g.numInputs + 1) return 0
    if (memo.has(n)) return memo.get(n)!
    memo.set(n, 0) // cycle guard; genomes are acyclic, but never hang a render on a bad artifact
    const d = 1 + Math.max(0, ...(incoming.get(n) ?? []).map(depth))
    memo.set(n, d)
    return d
  }
  const hidden = hiddenIds(g)
  const maxHidden = Math.max(0, ...hidden.filter((n) => n >= first).map(depth))
  const cols: number[][] = Array.from({ length: maxHidden + 2 }, () => [])
  for (let i = 0; i <= g.numInputs; i++) cols[0]!.push(i)
  for (const n of hidden) cols[Math.min(depth(n), maxHidden)]!.push(n)
  cols[maxHidden + 1] = outputIds(g)
  return cols
})

const H = computed(() => props.height || Math.max(150, Math.max(...columns.value.map((c) => c.length)) * 24 + padY * 2))

const positions = computed(() => {
  const map = new Map<number, { x: number; y: number }>()
  const cols = columns.value
  const left = labelW.value
  const right = W - labelW.value
  cols.forEach((col, ci) => {
    const x = cols.length === 1 ? W / 2 : left + (ci / (cols.length - 1)) * (right - left)
    col.forEach((id, i) => map.set(id, { x, y: col.length === 1 ? H.value / 2 : padY + (i / (col.length - 1)) * (H.value - padY * 2) }))
  })
  return map
})

const maxAbs = computed(() => Math.max(1e-9, ...props.genome.genes.filter((c) => c.enabled).map((c) => Math.abs(c.weight))))
const highlighted = computed(() => new Set(props.highlight))

// Attribute numbers are rounded: activations come from Math.tanh, which Node and the browser can disagree on in the
// last digit -- an unrounded opacity is a hydration mismatch waiting to happen.
const r3 = (n: number) => Math.round(n * 1000) / 1000

const edges = computed(() => {
  const list: { key: number; d: string; mx: number; my: number; color: string; opacity: number; width: number; dashed: boolean; hot: boolean; innovation: number }[] = []
  for (const c of props.genome.genes) {
    if (!c.enabled && !props.showDisabled) continue
    const a = positions.value.get(c.source)
    const b = positions.value.get(c.target)
    if (!a || !b) continue
    const mid = (a.x + b.x) / 2
    const strength = Math.abs(c.weight) / maxAbs.value
    const live = props.activations?.get(c.source)
    const contribution = live === undefined ? null : Math.min(1, Math.abs(c.weight * live) / (maxAbs.value * 0.7))
    const hot = highlighted.value.has(c.innovation)
    list.push({
      key: c.innovation,
      d: `M${a.x},${a.y}C${mid},${a.y} ${mid},${b.y} ${b.x},${b.y}`,
      mx: mid,
      my: (a.y + b.y) / 2,
      color: hot ? "#fbbf24" : !c.enabled ? "#6b7489" : c.weight >= 0 ? "#4ade80" : "#ff5c7a",
      opacity: r3(hot ? 0.95 : !c.enabled ? 0.45 : contribution === null ? 0.25 + 0.6 * strength : 0.08 + 0.85 * contribution),
      width: r3(hot ? 2.6 : !c.enabled ? 1 : 0.6 + 2 * strength),
      dashed: !c.enabled,
      hot,
      innovation: c.innovation,
    })
  }
  // Weakest first so strong edges land on top.
  return list.sort((x, y) => Number(x.hot) - Number(y.hot) || x.width - y.width)
})

const nodes = computed(() => {
  const g = props.genome
  const outs = new Set(outputIds(g))
  const chosen = props.activations && outs.size
    ? [...outs].reduce((best, id) => ((props.activations!.get(id) ?? -Infinity) > (props.activations!.get(best) ?? -Infinity) ? id : best))
    : null
  return [...positions.value.entries()].map(([id, p]) => {
    const kind = id < g.numInputs ? "input" : id === biasId(g) ? "bias" : outs.has(id) ? "output" : "hidden"
    const a = props.activations?.get(id)
    const t = a === undefined ? 0 : Math.min(1, Math.abs(a))
    const fill = a === undefined ? (kind === "hidden" ? "#232a3d" : "#1c2130") : a >= 0 ? `rgb(74 222 128 / ${r3(0.15 + 0.85 * t)})` : `rgb(255 92 122 / ${r3(0.15 + 0.85 * t)})`
    const label =
      kind === "input" ? (props.inputLabels[id] ?? `x${id}`) : kind === "bias" ? "bias" : kind === "output" ? (props.outputLabels[id - g.numInputs - 1] ?? `y${id - g.numInputs - 1}`) : `h${id}`
    return { id, kind, ...p, fill, label, chosen: chosen === id }
  })
})
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="block w-full" :class="fit ? 'h-full' : 'h-auto'" role="img" :aria-label="`NEAT network: ${genome.numInputs} inputs, ${hiddenIds(genome).length} hidden nodes, ${genome.numOutputs} outputs`">
    <path
      v-for="e in edges"
      :key="e.key"
      :d="e.d"
      :stroke="e.color"
      :stroke-opacity="e.opacity"
      :stroke-width="e.width"
      :stroke-dasharray="e.dashed ? '3 3' : undefined"
      fill="none"
    />
    <template v-if="showInnovations">
      <text
        v-for="e in edges"
        :key="`t${e.key}`"
        :x="e.mx"
        :y="e.my - 3"
        text-anchor="middle"
        class="font-mono text-[8px]"
        :class="e.hot ? 'fill-gold-300' : 'fill-fg-subtle'"
      >
        {{ e.innovation }}
      </text>
    </template>
    <g v-for="n in nodes" :key="n.id">
      <circle
        :cx="n.x"
        :cy="n.y"
        :r="n.kind === 'hidden' ? 6 : 5"
        :fill="n.fill"
        :stroke="n.chosen ? '#fbbf24' : n.kind === 'bias' ? '#93c5fd' : '#323a4d'"
        :stroke-width="n.chosen ? 2 : 1"
      />
      <text v-if="n.kind === 'hidden'" :x="n.x" :y="n.y - 10" text-anchor="middle" class="fill-fg-subtle font-mono text-[8px]">{{ n.label }}</text>
      <text
        v-else-if="n.kind !== 'output'"
        :x="n.x - 9"
        :y="n.y"
        text-anchor="end"
        dominant-baseline="middle"
        class="font-mono text-[9px]"
        :class="n.kind === 'bias' ? 'fill-signal-300' : (activations?.get(n.id) ?? 0) > 0.5 ? 'fill-fg' : 'fill-fg-subtle'"
      >
        {{ n.label }}
      </text>
      <text v-else :x="n.x + 9" :y="n.y" dominant-baseline="middle" class="font-mono text-[9px]" :class="n.chosen ? 'fill-gold-300' : 'fill-fg-subtle'">{{ n.label }}</text>
    </g>
  </svg>
</template>
