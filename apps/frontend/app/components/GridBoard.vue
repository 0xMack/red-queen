<script setup lang="ts">
import type { RenderState } from "~/types/games"

const props = defineProps<{ state: RenderState }>()

const cellSize = 24

const colorFor: Record<string, string> = {
  head: "#0f172a",
  body: "#475569",
  food: "#ef4444",
}
</script>

<template>
  <svg
    :viewBox="`0 0 ${props.state.width * cellSize} ${props.state.height * cellSize}`"
    class="rounded-lg border border-slate-200 bg-slate-50"
    :style="{ width: `${props.state.width * cellSize}px` }"
  >
    <rect
      v-for="cell in props.state.cells"
      :key="`${cell.x},${cell.y}`"
      :x="cell.x * cellSize"
      :y="cell.y * cellSize"
      :width="cellSize - 1"
      :height="cellSize - 1"
      :fill="colorFor[cell.label] ?? '#cbd5e1'"
      rx="2"
    />
  </svg>
</template>
