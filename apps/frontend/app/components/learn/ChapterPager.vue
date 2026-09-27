<script setup lang="ts">
import { learnChapters, type LearnChapter } from "~/data/learnChapters"

// Previous / next chapter, among the written ones.
const props = defineProps<{ current: LearnChapter }>()
const available = learnChapters.filter((c) => c.status === "available")
const i = computed(() => available.findIndex((c) => c.slug === props.current.slug))
const prev = computed(() => (i.value > 0 ? available[i.value - 1] : null))
const next = computed(() => (i.value >= 0 && i.value < available.length - 1 ? available[i.value + 1] : null))
</script>

<template>
  <nav class="grid gap-3 sm:grid-cols-2" aria-label="Chapter navigation">
    <NuxtLink v-if="prev" :to="prev.path" class="card card-hover group block p-5">
      <p class="label">← Previous</p>
      <p class="mt-2 font-display text-xl leading-tight transition group-hover:text-queen-200">{{ prev.title }}</p>
    </NuxtLink>
    <span v-else />
    <NuxtLink v-if="next" :to="next.path" class="card card-hover group block p-5 text-right">
      <p class="label text-queen-300">Next chapter →</p>
      <p class="mt-2 font-display text-xl leading-tight transition group-hover:text-queen-200">{{ next.title }}</p>
    </NuxtLink>
  </nav>
</template>
