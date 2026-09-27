<script setup lang="ts">
import { chapterNumber, learnChapters, type LearnChapter } from "~/data/learnChapters"

// A chapter's title page, generated from data/learnChapters.ts: number and part, the title, the summary, what it
// covers, what it builds on -- and its cover.
const props = defineProps<{ chapter: LearnChapter }>()
const prerequisites = computed(() =>
  (props.chapter.prerequisites ?? []).map((slug) => learnChapters.find((c) => c.slug === slug)).filter((c) => !!c),
)
</script>

<template>
  <header class="card ticks relative overflow-hidden">
    <div class="grid md:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
      <div class="relative z-10 p-6 sm:p-9">
        <p class="flex items-baseline gap-3">
          <span class="font-display text-4xl leading-none text-queen-400 italic">{{ chapterNumber(chapter.slug) }}.</span>
          <span class="label">{{ chapter.part }}</span>
        </p>
        <h1 class="mt-4 text-[2.4rem] leading-[1.02] sm:text-[3.1rem]">{{ chapter.title }}</h1>
        <p class="mt-4 text-[15.5px] leading-relaxed text-fg-muted">{{ chapter.summary }}</p>
        <div class="mt-6 flex flex-wrap items-center gap-1.5">
          <span v-if="chapter.readMinutes" class="chip text-fg">{{ chapter.readMinutes }} min read</span>
          <span v-for="tag in chapter.tags" :key="tag" class="chip">{{ tag }}</span>
        </div>
        <p v-if="prerequisites.length" class="mt-5 text-sm text-fg-subtle">
          Builds on
          <template v-for="(p, i) in prerequisites" :key="p.slug">
            <NuxtLink :to="p.path" class="link">{{ p.title }}</NuxtLink><span v-if="i < prerequisites.length - 1">, </span>
          </template>.
        </p>
      </div>
      <div class="relative hidden min-h-60 border-l border-line bg-sunken md:block">
        <img v-if="chapter.image" :src="chapter.image" :alt="`${chapter.title} cover`" class="absolute inset-0 size-full object-cover object-top">
        <ChapterArt v-else :kind="chapter.art" fit="contain" class="absolute inset-0" />
        <div class="absolute inset-0 bg-gradient-to-r from-surface via-surface/5 to-transparent" />
      </div>
    </div>
  </header>
</template>
