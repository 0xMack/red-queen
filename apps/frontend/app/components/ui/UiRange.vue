<script setup lang="ts">
// A labelled slider: the label and its current value (`display` formats it) on one line, the slider under them --
// an instrument's dial, readable at any column width.
const props = withDefaults(defineProps<{ label: string; min: number; max: number; step?: number; display?: (v: number) => string; disabled?: boolean }>(), {
  step: 1,
  display: undefined,
})
const model = defineModel<number>({ required: true })
const shown = computed(() => (props.display ? props.display(model.value) : String(model.value)))
const fill = computed(() => `${((model.value - props.min) / (props.max - props.min || 1)) * 100}%`)
</script>

<template>
  <label class="block min-w-0 text-sm" :class="disabled ? 'pointer-events-none opacity-40' : ''">
    <span class="flex items-baseline justify-between gap-3">
      <span class="truncate text-fg-muted">{{ label }}</span>
      <span class="num shrink-0 text-xs text-fg">{{ shown }}</span>
    </span>
    <input
      v-model.number="model"
      type="range"
      :min="min"
      :max="max"
      :step="step"
      :disabled="disabled"
      class="ui-range mt-1.5 w-full"
      :style="{ '--fill': fill }"
      :aria-label="label"
    >
  </label>
</template>
