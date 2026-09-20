<script setup lang="ts">
import type { Ply } from "~/types/versus"

// Moves in pairs, first seat then second, tucked into a disclosure: it is a record to copy from, not
// something to watch, so it stays out of the way (and out of the layout) until asked for. Game-agnostic:
// the engine wrote each ply's notation.
const props = defineProps<{ history: Ply[]; players: readonly string[] }>()
const rows = computed(() => {
  const out: { n: number; a?: string; b?: string }[] = []
  props.history.forEach((ply, i) => {
    const row = (out[Math.floor(i / 2)] ??= { n: Math.floor(i / 2) + 1 })
    row[ply.player === 0 ? "a" : "b"] = ply.text
  })
  return out
})
</script>

<template>
  <details class="card group text-sm">
    <summary class="flex cursor-pointer list-none items-center justify-between px-3 py-2">
      <span class="eyebrow">Move history</span>
      <span class="num text-xs text-fg-subtle">{{ history.length }} plies <span class="ml-1 inline-block transition group-open:rotate-90">›</span></span>
    </summary>
    <div class="h-48 overflow-y-auto border-t border-line px-3 py-2">
      <p v-if="!rows.length" class="text-fg-subtle">No moves yet. {{ players[0] }} moves first.</p>
      <ol v-else class="grid grid-cols-[2rem_1fr_1fr] gap-x-3 gap-y-0.5 font-mono text-xs">
        <li v-for="row in rows" :key="row.n" class="contents">
          <span class="text-fg-subtle">{{ row.n }}.</span>
          <span>{{ row.a }}</span>
          <span>{{ row.b }}</span>
        </li>
      </ol>
    </div>
  </details>
</template>
