<script setup lang="ts">
import type { ChartSeries } from "~/types/chart"

// A lab's learning curve: an optional caption, the legend, and the chart -- or, until there's enough to draw, a
// placeholder saying what will appear.
withDefaults(
  defineProps<{
    x: number[]
    series: ChartSeries[]
    show: boolean
    caption?: string
    empty?: string
    height?: number
    xLabel?: string
    format?: (v: number) => string
    legend?: boolean
  }>(),
  { caption: undefined, empty: undefined, height: 170, xLabel: "env steps", format: (v: number) => v.toFixed(1), legend: true },
)
</script>

<template>
  <div>
    <p v-if="caption" class="label mb-2">{{ caption }}</p>
    <template v-if="show">
      <UiLegend v-if="legend" :series="series" />
      <LineChart class="mt-1" :x="x" :series="series" :height="height" :x-label="xLabel" :format="format" />
    </template>
    <UiEmpty v-else-if="empty" compact>{{ empty }}</UiEmpty>
  </div>
</template>
