<script setup lang="ts">
// A labelled checkbox; the label is the default slot (or `label`).
defineProps<{ label?: string; disabled?: boolean }>()
const model = defineModel<boolean>({ required: true })
</script>

<template>
  <label class="inline-flex cursor-pointer items-center gap-2 text-sm text-fg-muted transition hover:text-fg" :class="disabled ? 'pointer-events-none opacity-40' : ''">
    <input v-model="model" type="checkbox" class="size-3.5" :disabled="disabled">
    <slot>{{ label }}</slot>
  </label>
</template>
