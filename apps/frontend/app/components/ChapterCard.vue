<script setup lang="ts">
import type { LearnChapter } from "~/data/learnChapters"

defineProps<{ chapter: LearnChapter; number: number }>()
// Unwritten chapters render as a plain (unlinked) card rather than being omitted.
const NuxtLink = resolveComponent("NuxtLink")
</script>

<template>
  <component
    :is="chapter.status === 'available' ? NuxtLink : 'div'"
    :to="chapter.status === 'available' ? chapter.path : undefined"
    class="card group flex flex-col overflow-hidden"
    :class="chapter.status === 'available' ? 'card-hover' : 'opacity-70'"
  >
    <div class="relative aspect-[16/10] overflow-hidden border-b border-line bg-sunken">
      <img
        v-if="chapter.image"
        :src="chapter.image"
        :alt="`${chapter.title} cover`"
        class="size-full object-cover object-top transition duration-500 group-hover:scale-[1.03]"
        loading="lazy"
      >
      <ChapterArt v-else :kind="chapter.art" class="transition duration-500 group-hover:scale-[1.03]" />
      <span class="absolute top-3 left-3 rounded-md border border-line-strong bg-bg/70 px-2 py-0.5 font-mono text-[11px] text-fg-muted backdrop-blur">
        {{ String(number).padStart(2, "0") }}
      </span>
      <span
        v-if="chapter.status === 'coming-soon'"
        class="absolute top-3 right-3 rounded-full border border-line-strong bg-bg/70 px-2 py-0.5 text-[11px] text-fg-muted backdrop-blur"
      >
        Coming soon
      </span>
    </div>
    <div class="flex flex-1 flex-col p-5">
      <p class="eyebrow">{{ chapter.part }}</p>
      <h3 class="mt-2 text-lg leading-snug font-semibold transition group-hover:text-queen-200">{{ chapter.title }}</h3>
      <p class="mt-2 text-sm leading-relaxed text-fg-muted">{{ chapter.summary }}</p>
      <div class="mt-auto flex flex-wrap items-center gap-1.5 pt-4">
        <span v-if="chapter.readMinutes" class="chip text-fg">{{ chapter.readMinutes }} min read</span>
        <span v-for="tag in chapter.tags.slice(0, 3)" :key="tag" class="chip">{{ tag }}</span>
      </div>
    </div>
  </component>
</template>
