<script setup lang="ts">
import { chapterBySlug, chapterHref, type LearnChapter } from "~/data/learnChapters"
import { palette } from "~/utils/palette"

// A chapter's title page, generated from data/learnChapters.ts: where it sits on the path being read, the title, the
// summary, what it covers, what it builds on (and pairs well with) -- and its cover.
const props = defineProps<{ chapter: LearnChapter }>()
const position = useLearnPath(toRef(props, "chapter"))
const prerequisites = computed(() => (props.chapter.prerequisites ?? []).map(chapterBySlug).filter((c) => !!c))
const related = computed(() => (props.chapter.related ?? []).map(chapterBySlug).filter((c) => !!c))
</script>

<template>
  <header class="card ticks relative overflow-hidden">
    <div class="grid md:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
      <div class="relative z-10 p-6 sm:p-9">
        <p class="flex items-center gap-2.5">
          <span class="size-2.5 rounded-full" :style="{ background: palette[position.path.color] }" />
          <span class="label">{{ position.path.title }} · {{ position.index + 1 }} of {{ position.stops.length }}</span>
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
            <NuxtLink :to="chapterHref(p, position.path)" class="link">{{ p.title }}</NuxtLink><span v-if="i < prerequisites.length - 1">, </span>
          </template>.
        </p>
        <p v-if="related.length" class="mt-1.5 text-sm text-fg-subtle">
          Pairs well with
          <template v-for="(p, i) in related" :key="p.slug">
            <NuxtLink :to="p.path" class="link">{{ p.title }}</NuxtLink><span v-if="i < related.length - 1">, </span>
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
