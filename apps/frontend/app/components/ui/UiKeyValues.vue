<script setup lang="ts">
// A list of key/value pairs, keys on the left in mono, values on the right -- a config, a summary. Long values wrap
// rather than push the page wider.
defineProps<{ entries: readonly (readonly [string, string | number])[] }>()
</script>

<template>
  <dl class="divide-y divide-line/60 text-sm">
    <div v-for="[key, value] in entries" :key="key" class="flex items-baseline justify-between gap-4 py-2">
      <dt class="shrink-0 font-mono text-xs text-fg-subtle">{{ key }}</dt>
      <dd class="min-w-0 text-right font-mono text-xs break-all text-fg">{{ value }}</dd>
    </div>
  </dl>
</template>
