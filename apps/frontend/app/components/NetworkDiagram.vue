<script setup lang="ts">
// A champion's actual network: every weight drawn as an edge (green positive, red negative,
// opacity by magnitude). With `activations` (forwardActivations() on the live observation), nodes
// light up by activation and edges by their live contribution |w * a| -- so you can watch which
// sensor is driving the current move. Purely a visualization; the worker's Python copy decides.
const props = withDefaults(
  defineProps<{
    weights: number[]
    layerSizes: number[]
    activations?: number[][] | null
    inputLabels?: string[]
    outputLabels?: string[]
  }>(),
  { activations: null, inputLabels: () => [], outputLabels: () => [] },
)

const W = 460
const nodeGap = 17
const labelW = 88
const top = 14

const maxLayer = computed(() => Math.max(...props.layerSizes))
const H = computed(() => top * 2 + (maxLayer.value - 1) * nodeGap)

const nodes = computed(() => {
  const cols = props.layerSizes.length
  const left = labelW
  const right = W - labelW
  return props.layerSizes.map((size, li) => {
    const x = cols === 1 ? (left + right) / 2 : left + (li / (cols - 1)) * (right - left)
    const offset = ((maxLayer.value - size) * nodeGap) / 2
    return Array.from({ length: size }, (_, ni) => ({ x, y: top + offset + ni * nodeGap }))
  })
})

// The chosen action -- argmax of the output layer, same rule as the worker's snake_policy_action.
const chosenOutput = computed(() => {
  const out = props.activations?.at(-1)
  if (!out) return null
  return out.indexOf(Math.max(...out))
})

const edges = computed(() => {
  const list: { d: string; color: string; opacity: number; width: number }[] = []
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
          opacity: input ? 0.08 + 0.85 * strength : 0.06 + 0.5 * strength,
          width: 0.5 + 1.6 * strength,
        })
      }
    }
  }
  // Strong edges last so they draw on top.
  return list.sort((a, b) => a.opacity - b.opacity)
})

function nodeFill(li: number, ni: number): string {
  const a = props.activations?.[li]?.[ni]
  if (a === undefined) return "#1c2130"
  const t = Math.min(1, Math.abs(a))
  return a >= 0 ? `rgb(74 222 128 / ${0.15 + 0.85 * t})` : `rgb(255 92 122 / ${0.15 + 0.85 * t})`
}
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="block h-auto w-full">
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
        r="5"
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
