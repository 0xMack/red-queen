<script setup lang="ts">
import type { Seat } from "~/types/versus"

// Who plays each seat: a human or any strategy. Game-agnostic (docs/design/0006).
defineProps<{
  players: readonly string[]
  seats: readonly Seat[]
  strategies: { id: string; label: string }[]
}>()
const emit = defineEmits<{ seat: [player: 0 | 1, id: Seat] }>()
</script>

<template>
  <div class="grid gap-4 sm:grid-cols-2">
    <label v-for="p in ([0, 1] as const)" :key="p" class="block text-sm">
      <span class="eyebrow">{{ players[p] }} plays</span>
      <select
        class="mt-1.5 w-full rounded-lg border border-line bg-surface px-3 py-2 text-fg"
        :value="seats[p]"
        @change="emit('seat', p, ($event.target as HTMLSelectElement).value)"
      >
        <option value="human">Human</option>
        <option v-for="s in strategies" :key="s.id" :value="s.id">{{ s.label }}</option>
      </select>
    </label>
  </div>
</template>
