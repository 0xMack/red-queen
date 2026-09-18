<script setup lang="ts">
import type { GridCell } from "~/types/games"

// Renders games.checkers' render_state() shape ({width, height, cells} with
// "black_man"/"red_king"/... labels -- the same shape Snake uses, just a wider label vocabulary).
const props = defineProps<{ state: { width: number; height: number; cells: GridCell[] } }>()

const S = 40
const pieces = computed(() =>
  props.state.cells.map((c) => ({
    ...c,
    red: c.label.startsWith("red"),
    king: c.label.endsWith("king"),
  })),
)
</script>

<template>
  <svg :viewBox="`0 0 ${state.width * S} ${state.height * S}`" class="block h-auto w-full rounded-xl border border-line">
    <template v-for="y in state.height" :key="y">
      <rect
        v-for="x in state.width"
        :key="x"
        :x="(x - 1) * S"
        :y="(y - 1) * S"
        :width="S"
        :height="S"
        :fill="(x + y) % 2 === 0 ? '#1d2230' : '#12151e'"
      />
    </template>
    <g v-for="p in pieces" :key="`${p.x},${p.y}`" :transform="`translate(${p.x * S + S / 2}, ${p.y * S + S / 2})`">
      <circle r="15" :fill="p.red ? '#ef3b5d' : '#e9ebf1'" />
      <circle r="11" fill="none" :stroke="p.red ? '#9b1535' : '#a0a8ba'" stroke-width="1.5" />
      <path v-if="p.king" d="M-7 4 L-8 -4 L-3.5 0 L0 -6 L3.5 0 L8 -4 L7 4 Z" :fill="p.red ? '#fff' : '#c81e45'" />
    </g>
  </svg>
</template>
