<script setup lang="ts">
import type { ExplainLink } from "~/types/explain"

// Where to read more: a chapter, a section of one, a training run. A chapter link is a book mark (§ for a section).
defineProps<{ links: ExplainLink[] }>()
defineEmits<{ navigate: [] }>()
const MARK: Record<ExplainLink["kind"], string> = { chapter: "❡", section: "§", run: "↗", page: "→", doc: "¶" }
</script>

<template>
  <ul class="space-y-1">
    <li v-for="l in links" :key="l.to">
      <NuxtLink :to="l.to" class="group flex items-baseline gap-2 text-xs text-fg-muted transition hover:text-fg" @click="$emit('navigate')">
        <span class="w-3 shrink-0 text-center font-mono text-queen-300">{{ MARK[l.kind] }}</span>
        <span class="underline decoration-line-strong underline-offset-4 group-hover:decoration-queen-300">{{ l.label }}</span>
      </NuxtLink>
    </li>
  </ul>
</template>
