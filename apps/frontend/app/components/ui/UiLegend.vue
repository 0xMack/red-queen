<script setup lang="ts">
// A chart's key: one swatch per series (dashed series get a dashed swatch). Takes `ChartSeries` as-is.
defineProps<{ series: readonly { key: string; label: string; color: string; dashed?: boolean }[]; suffix?: string }>()
</script>

<template>
  <ul class="flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
    <li v-for="s in series" :key="s.key" class="flex items-center gap-1.5">
      <svg width="16" height="4" aria-hidden="true" class="shrink-0">
        <line x1="1" x2="15" y1="2" y2="2" :stroke="s.color" stroke-width="2" stroke-linecap="round" :stroke-dasharray="s.dashed ? '3 3' : undefined" />
      </svg>
      {{ s.label }}<template v-if="suffix"> {{ suffix }}</template>
    </li>
  </ul>
</template>
