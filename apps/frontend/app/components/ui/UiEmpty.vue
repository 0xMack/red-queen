<script setup lang="ts">
// Where something will appear: a chart before its first point, a board before training starts, an empty list.
// `square` keeps a board's footprint so the layout doesn't jump when it arrives.
defineProps<{ square?: boolean; compact?: boolean }>()
</script>

<template>
  <div
    class="flex flex-col items-center justify-center gap-2 rounded-[8px] border border-dashed border-line-strong/70 bg-[repeating-linear-gradient(135deg,transparent_0_8px,rgb(255_255_255/0.012)_8px_16px)] text-center text-fg-subtle"
    :class="[square ? 'aspect-square' : '', compact ? 'p-4 text-xs' : 'p-6 text-sm']"
  >
    <slot />
  </div>
</template>
