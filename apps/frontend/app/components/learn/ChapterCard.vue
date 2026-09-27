<script setup lang="ts">
import type { LearnChapter } from "~/data/learnChapters"

// A chapter on an index. Unwritten chapters render as a plain (unlinked) card rather than being omitted.
defineProps<{ chapter: LearnChapter; number: number }>()
const NuxtLink = resolveComponent("NuxtLink")
</script>

<template>
  <component
    :is="chapter.status === 'available' ? NuxtLink : 'div'"
    :to="chapter.status === 'available' ? chapter.path : undefined"
    class="card group flex flex-col overflow-hidden"
    :class="chapter.status === 'available' ? 'card-hover' : 'opacity-60'"
  >
    <div class="relative aspect-[16/10] overflow-hidden border-b border-line bg-sunken">
      <img
        v-if="chapter.image"
        :src="chapter.image"
        :alt="`${chapter.title} cover`"
        class="size-full object-cover object-top transition duration-700 group-hover:scale-[1.03]"
        loading="lazy"
      >
      <ChapterArt v-else :kind="chapter.art" class="transition duration-700 group-hover:scale-[1.03]" />
      <UiBadge v-if="chapter.status === 'coming-soon'" class="absolute top-3 right-3 bg-bg/70 backdrop-blur">soon</UiBadge>
    </div>
    <div class="flex flex-1 flex-col p-5">
      <p class="flex items-baseline gap-2">
        <span class="font-display text-xl text-queen-400 italic">{{ number }}.</span>
        <span class="label">{{ chapter.part }}</span>
      </p>
      <h3 class="mt-1 font-display text-[1.6rem] leading-[1.1] font-normal transition group-hover:text-queen-200">{{ chapter.title }}</h3>
      <p class="mt-2.5 text-sm leading-relaxed text-fg-muted">{{ chapter.summary }}</p>
      <div class="mt-auto flex flex-wrap items-center gap-1.5 pt-4">
        <span v-if="chapter.readMinutes" class="chip text-fg">{{ chapter.readMinutes }} min</span>
        <span v-for="tag in chapter.tags.slice(0, 2)" :key="tag" class="chip">{{ tag }}</span>
      </div>
    </div>
  </component>
</template>
