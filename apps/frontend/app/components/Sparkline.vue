<script setup lang="ts">
// A tiny inline trend line for tables/cards -- no axes, just shape. Fixed viewBox, stretched to fit.
const props = withDefaults(defineProps<{ values: number[]; color?: string; fill?: boolean }>(), {
  color: "#ff5c7a",
  fill: true,
})

const W = 120
const H = 32
const uid = useId()

const paths = computed(() => {
  const v = props.values
  if (v.length < 2) return null
  const min = Math.min(...v)
  const max = Math.max(...v)
  const span = max - min || 1
  const pts = v.map((y, i) => [(i / (v.length - 1)) * W, H - 2 - ((y - min) / span) * (H - 4)] as const)
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("")
  return { line, area: `${line}L${W},${H}L0,${H}Z`, last: pts.at(-1)! }
})
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none" class="block overflow-visible">
    <template v-if="paths">
      <defs>
        <linearGradient :id="`spark-${uid}`" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" :stop-color="color" stop-opacity="0.3" />
          <stop offset="1" :stop-color="color" stop-opacity="0" />
        </linearGradient>
      </defs>
      <path v-if="fill" :d="paths.area" :fill="`url(#spark-${uid})`" />
      <path :d="paths.line" fill="none" :stroke="color" stroke-width="1.5" vector-effect="non-scaling-stroke" stroke-linejoin="round" />
    </template>
    <line v-else x1="0" :x2="W" :y1="H / 2" :y2="H / 2" stroke="#323a4d" stroke-dasharray="3 3" vector-effect="non-scaling-stroke" />
  </svg>
</template>
