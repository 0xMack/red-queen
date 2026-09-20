<script setup lang="ts">
// A champion's actual network: every weight drawn as an edge (green positive, red negative,
// opacity by magnitude). With `activations` (forwardActivations() on the live observation), nodes
// light up by activation and edges by their live contribution |w * a| -- so you can watch which
// sensor is driving the current move. Purely a visualization; the worker's Python copy decides.
const props = withDefaults(
  defineProps<{
    weights: readonly number[]
    layerSizes: readonly number[]
    activations?: number[][] | null
    inputLabels?: readonly string[]
    outputLabels?: readonly string[]
    /** Fill the parent's height (aspect ratio kept) instead of sizing to the parent's width. */
    fit?: boolean
  }>(),
  { activations: null, inputLabels: () => [], outputLabels: () => [], fit: false },
)

const W = 460
const labelW = 88
const top = 14
// Wide layers (e.g. the 100-input grid representation) are compressed to fit ~320px of height
// instead of growing the diagram, and edges are capped to the strongest few hundred -- drawing
// every one of ~2.5k edges each tick is both unreadable and slow.
const MAX_HEIGHT = 320
const MAX_EDGES = 450

const maxLayer = computed(() => Math.max(...props.layerSizes))
const nodeGap = computed(() => Math.min(17, MAX_HEIGHT / Math.max(1, maxLayer.value - 1)))
const nodeR = computed(() => Math.max(1.6, Math.min(5, nodeGap.value * 0.3)))
const H = computed(() => top * 2 + (maxLayer.value - 1) * nodeGap.value)

const nodes = computed(() => {
  const cols = props.layerSizes.length
  const left = labelW
  const right = W - labelW
  return props.layerSizes.map((size, li) => {
    const x = cols === 1 ? (left + right) / 2 : left + (li / (cols - 1)) * (right - left)
    const offset = ((maxLayer.value - size) * nodeGap.value) / 2
    return Array.from({ length: size }, (_, ni) => ({ x, y: top + offset + ni * nodeGap.value }))
  })
})

// The chosen action -- argmax of the output layer, same rule as the worker's snake_policy_action.
const chosenOutput = computed(() => {
  const out = props.activations?.at(-1)
  if (!out) return null
  return out.indexOf(Math.max(...out))
})

const edges = computed(() => {
  const list: { d: string; color: string; opacity: number; width: number; strength: number }[] = []
  const maxAbs = Math.max(1e-9, ...props.weights.map(Math.abs))
  for (let li = 0; li < props.layerSizes.length - 1; li++) {
    const from = nodes.value[li]!
    const to = nodes.value[li + 1]!
    const input = props.activations?.[li]
    for (let o = 0; o < to.length; o++) {
      for (let k = 0; k < from.length; k++) {
        const w = weightAt(props.weights, props.layerSizes, li, k, o)
        const strength = input ? Math.min(1, Math.abs(w * input[k]!) / (maxAbs * 0.6)) : Math.abs(w) / maxAbs
        if (strength < 0.04) continue
        const a = from[k]!
        const b = to[o]!
        list.push({
          d: `M${a.x},${a.y}L${b.x},${b.y}`,
          color: w >= 0 ? "#4ade80" : "#ff5c7a",
          // rounded: server (Node) and browser can differ in Math.tanh's last digit -> hydration mismatch
          opacity: Math.round((input ? 0.08 + 0.85 * strength : 0.06 + 0.5 * strength) * 1000) / 1000,
          width: Math.round((0.5 + 1.6 * strength) * 1000) / 1000,
          strength,
        })
      }
    }
  }
  // Strongest MAX_EDGES only, drawn weakest-first so strong edges land on top.
  return list
    .sort((a, b) => b.strength - a.strength)
    .slice(0, MAX_EDGES)
    .reverse()
})

function nodeFill(li: number, ni: number): string {
  const a = props.activations?.[li]?.[ni]
  if (a === undefined) return "#1c2130"
  const t = Math.min(1, Math.abs(a))
  const alpha = Math.round((0.15 + 0.85 * t) * 1000) / 1000
  return a >= 0 ? `rgb(74 222 128 / ${alpha})` : `rgb(255 92 122 / ${alpha})`
}
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="block w-full" :class="fit ? 'h-full' : 'h-auto'">
    <path
      v-for="(e, i) in edges"
      :key="i"
      :d="e.d"
      :stroke="e.color"
      :stroke-opacity="e.opacity"
      :stroke-width="e.width"
      fill="none"
    />
    <g v-for="(layer, li) in nodes" :key="li">
      <circle
        v-for="(n, ni) in layer"
        :key="ni"
        :cx="n.x"
        :cy="n.y"
        :r="nodeR"
        :fill="nodeFill(li, ni)"
        :stroke="li === layerSizes.length - 1 && chosenOutput === ni ? '#fbbf24' : '#323a4d'"
        :stroke-width="li === layerSizes.length - 1 && chosenOutput === ni ? 2 : 1"
      />
    </g>
    <text
      v-for="(label, i) in inputLabels"
      :key="`in${i}`"
      :x="labelW - 10"
      :y="nodes[0]?.[i]?.y"
      text-anchor="end"
      dominant-baseline="middle"
      class="font-mono text-[9px]"
      :class="(activations?.[0]?.[i] ?? 0) > 0.5 ? 'fill-fg' : 'fill-fg-subtle'"
    >
      {{ label }}
    </text>
    <text
      v-for="(label, i) in outputLabels"
      :key="`out${i}`"
      :x="W - labelW + 10"
      :y="nodes.at(-1)?.[i]?.y"
      dominant-baseline="middle"
      class="font-mono text-[9px]"
      :class="chosenOutput === i ? 'fill-gold-300' : 'fill-fg-subtle'"
    >
      {{ label }}
    </text>
  </svg>
</template>
