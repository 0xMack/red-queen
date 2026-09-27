<script setup lang="ts">
import type { Fact } from "~/types/explain"

// An explainer's facts: a label and a value, the value's meaning on hover (and, with `meanings`, written under it).
defineProps<{ facts: Fact[]; meanings?: boolean }>()
</script>

<template>
  <dl class="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-xs">
    <template v-for="f in facts" :key="f.label">
      <dt class="font-mono text-[11px] text-fg-subtle">{{ f.label }}</dt>
      <dd class="min-w-0 text-fg" :title="f.hint">
        <span class="font-mono break-words">{{ f.value }}</span>
        <span v-if="meanings && f.hint" class="block text-[11px] text-fg-subtle">{{ f.hint }}</span>
      </dd>
    </template>
  </dl>
</template>
