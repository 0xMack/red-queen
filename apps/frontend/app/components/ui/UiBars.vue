<script setup lang="ts">
// Labelled horizontal bars for a handful of values -- what a policy is choosing between (Q-values, move
// probabilities). `chosen` highlights one; `max` sets the full-width value (default: the largest). `bind` adds attributes
// to each row -- a formula's term scope uses it to link a bar to the term it is (docs/design/0012).
const props = defineProps<{
  labels: readonly string[]
  values: readonly number[]
  chosen?: number | null
  max?: number
  format?: (v: number) => string
  bind?: (i: number) => Record<string, unknown>
}>()
const top = computed(() => props.max ?? Math.max(1e-9, ...props.values))
</script>

<template>
  <div class="space-y-1 font-mono text-xs text-fg-subtle">
    <div v-for="(v, i) in values" :key="i" class="grid grid-cols-[5.5rem_minmax(0,1fr)_3.5rem] items-center gap-2" v-bind="bind?.(i) ?? {}">
      <span class="truncate" :class="i === chosen ? 'text-fg' : ''">{{ labels[i] }}</span>
      <span class="h-2 overflow-hidden rounded-[2px] bg-raised">
        <span
          class="block h-full rounded-[2px] transition-[width] duration-150"
          :class="i === chosen ? 'bg-queen-400' : 'bg-line-strong'"
          :style="{ width: `${Math.max(2, Math.min(100, (Math.max(v, 0) / top) * 100))}%` }"
        />
      </span>
      <span class="num text-right" :class="i === chosen ? 'text-fg' : ''">{{ format ? format(v) : v.toFixed(2) }}</span>
    </div>
  </div>
</template>
