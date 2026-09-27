<script setup lang="ts">
// A figure in a chapter: a plate captioned "Fig. N · <kind>" and a title, controls on the right (`actions`), the
// figure itself, and an optional `caption` underneath. N comes from a CSS counter (main.css `.ui-figure`), so a
// chapter's figures number themselves in reading order with no bookkeeping -- and it's identical on server and client.
withDefaults(defineProps<{ title?: string; kind?: string; pad?: "none" | "sm" | "md" | "lg" }>(), { title: undefined, kind: "Interactive", pad: "md" })
</script>

<template>
  <UiPanel as="figure" class="ui-figure not-prose my-8" :pad="pad">
    <template #header>
      <span class="fig-num label">{{ kind }}</span>
      <span v-if="title || $slots.title" class="text-sm font-semibold text-fg"><slot name="title">{{ title }}</slot></span>
    </template>
    <template v-if="$slots.actions" #actions><slot name="actions" /></template>
    <slot />
    <template v-if="$slots.caption" #footer>
      <figcaption><slot name="caption" /></figcaption>
    </template>
  </UiPanel>
</template>
