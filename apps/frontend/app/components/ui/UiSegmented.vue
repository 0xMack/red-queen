<script setup lang="ts" generic="T extends string | number | null">
// One choice out of a few, shown all at once: Watch/Play, a speed, a chart's axis, a filter. `accent` marks the
// option that should read as the call to action when selected (Play, on a game page).
const props = withDefaults(
  defineProps<{
    options: readonly { value: T; label: string; title?: string; disabled?: boolean }[]
    size?: "sm" | "md"
    accent?: T
    ariaLabel?: string
  }>(),
  { size: "sm" },
)
const model = defineModel<T>({ required: true })
</script>

<template>
  <div class="inline-flex max-w-full flex-wrap rounded-[8px] border border-line bg-sunken p-0.5" role="radiogroup" :aria-label="ariaLabel">
    <button
      v-for="o in props.options"
      :key="String(o.value)"
      type="button"
      role="radio"
      :aria-checked="model === o.value"
      :title="o.title"
      :disabled="o.disabled"
      class="rounded-[6px] font-medium whitespace-nowrap transition disabled:cursor-not-allowed disabled:opacity-40"
      :class="[
        size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-4 py-1.5 text-sm',
        model === o.value
          ? accent !== undefined && o.value === accent
            ? 'bg-queen-500 text-white'
            : 'bg-raised text-fg shadow-[inset_0_0_0_1px_var(--color-line-strong)]'
          : 'text-fg-subtle hover:text-fg',
      ]"
      @click="model = o.value"
    >
      {{ o.label }}
    </button>
  </div>
</template>
