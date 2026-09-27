<script setup lang="ts" generic="T extends string | number">
// A labelled dropdown. `options` may be plain values (shown as themselves) or {value, label}. Stacked (label above)
// by default, so it fits a narrow column; `inline` puts the label beside it.
const props = defineProps<{ label?: string; options: readonly (T | { value: T; label: string })[]; inline?: boolean; labelWidth?: string }>()
const model = defineModel<T>({ required: true })
const items = computed(() => props.options.map((o) => (typeof o === "object" ? o : { value: o, label: String(o) })))
</script>

<template>
  <label class="min-w-0 text-sm" :class="inline ? 'flex items-center gap-2' : 'block'">
    <span v-if="label" class="block shrink-0 truncate text-fg-muted" :class="[inline ? '' : 'mb-1.5', labelWidth]">{{ label }}</span>
    <select v-model="model" class="field w-full min-w-0 py-1.5" :aria-label="label">
      <option v-for="o in items" :key="String(o.value)" :value="o.value">{{ o.label }}</option>
    </select>
  </label>
</template>
