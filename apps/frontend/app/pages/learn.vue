<script setup lang="ts">
import { learnChapters } from "~/data/learnChapters"

// Parent route for every /learn/* page. The contents page (/learn) renders full-width on its own; a chapter gets the
// textbook frame: the path it's being read on down the left (?path=, else the chapter's home path), a generated
// title page, prev/next along that path and where the graph forks from here, and an "on this page" outline on the
// right built from the chapter's own <h2>s at runtime.
const route = useRoute()
const chapter = computed(() => learnChapters.find((c) => c.path === route.path.replace(/\/$/, "")) ?? null)
useHead({ title: () => chapter.value?.title ?? "Learn" })

const body = ref<HTMLElement | null>(null)
const { outline, activeId } = useChapterOutline(body)
const progress = useReadingProgress()

// Reaching the end of a chapter marks it read (the map and sidebar show it).
const { markRead } = useReadChapters()
watch(progress, (p) => p > 0.9 && chapter.value && markRead(chapter.value.slug))
</script>

<template>
  <div>
    <NuxtPage v-if="!chapter" />

    <div v-else class="mx-auto max-w-[1600px] px-4 sm:px-6 lg:px-8">
      <div class="fixed inset-x-0 top-14 z-40 h-px">
        <div class="h-full bg-queen-400 transition-[width] duration-150" :style="{ width: `${progress * 100}%` }" />
      </div>

      <div class="grid gap-10 py-8 lg:grid-cols-[250px_minmax(0,1fr)] xl:grid-cols-[250px_minmax(0,1fr)_210px]">
        <ChapterNav :current="chapter" />

        <div class="min-w-0">
          <ChapterHeader :chapter="chapter" />
          <div ref="body" class="mx-auto mt-12 max-w-3xl">
            <NuxtPage />
          </div>
          <ChapterPager class="mx-auto mt-20 max-w-3xl" :current="chapter" />
        </div>

        <aside class="hidden xl:block">
          <ChapterOutline :outline="outline" :active-id="activeId" />
        </aside>
      </div>
    </div>
  </div>
</template>
