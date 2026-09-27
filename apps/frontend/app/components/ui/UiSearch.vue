<script setup lang="ts">
// A search field with its icon. `size: "lg"` for a page's main search (the Learn index).
withDefaults(defineProps<{ placeholder?: string; size?: "md" | "lg" }>(), { placeholder: "Search…", size: "md" })
const model = defineModel<string>({ required: true })
const input = ref<HTMLInputElement | null>(null)
defineExpose({ focus: () => input.value?.focus(), blur: () => input.value?.blur() })
</script>

<template>
  <div class="relative">
    <svg
      viewBox="0 0 20 20"
      class="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-fg-subtle"
      :class="size === 'lg' ? 'size-4.5' : 'size-4'"
      fill="none"
      stroke="currentColor"
      stroke-width="1.8"
      aria-hidden="true"
    >
      <circle cx="9" cy="9" r="6" /><path d="m14 14 4 4" stroke-linecap="round" />
    </svg>
    <input
      ref="input"
      v-model="model"
      type="search"
      :placeholder="placeholder"
      class="w-full rounded-[8px] border border-line-strong bg-sunken pr-3 text-fg transition placeholder:text-fg-subtle hover:border-fg-subtle focus:border-queen-400 focus:outline-none"
      :class="size === 'lg' ? 'py-3 pl-10 text-base' : 'py-2 pl-9 text-sm'"
    >
    <slot />
  </div>
</template>
