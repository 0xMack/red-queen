<script setup lang="ts">
import { chapterParts, learnChapters, type LearnChapter } from "~/data/learnChapters"

// The textbook's sidebar: search, then every chapter grouped by part, the current one marked. Collapses to a
// "Chapter N of M" toggle on narrow screens.
const props = defineProps<{ current: LearnChapter }>()
const parts = chapterParts()
const route = useRoute()
const open = ref(false)
watch(() => route.fullPath, () => (open.value = false))
const number = computed(() => learnChapters.indexOf(props.current) + 1)
</script>

<template>
  <aside class="lg:sticky lg:top-20 lg:h-[calc(100vh-6rem)] lg:overflow-y-auto lg:pr-2 lg:pb-8">
    <LearnSearch />
    <button class="btn-ghost mt-3 w-full justify-between lg:hidden" :aria-expanded="open" @click="open = !open">
      <span>Chapter {{ number }} of {{ learnChapters.length }}</span>
      <span>{{ open ? "▴" : "▾" }}</span>
    </button>
    <nav class="mt-6 space-y-7" :class="open ? 'block' : 'hidden lg:block'" aria-label="Chapters">
      <NuxtLink to="/learn" class="flex items-center gap-2 text-sm text-fg-subtle transition hover:text-fg">
        <span>←</span> Contents
      </NuxtLink>
      <div v-for="group in parts" :key="group.part">
        <p class="label mb-2.5"><span class="text-queen-300/80">{{ group.numeral }}.</span> {{ group.part }}</p>
        <ul class="space-y-px border-l border-line">
          <li v-for="{ chapter: c, number: n } in group.chapters" :key="c.slug">
            <NuxtLink
              v-if="c.status === 'available'"
              :to="c.path"
              class="-ml-px flex gap-3 border-l py-1.5 pr-2 pl-3 text-[13.5px] leading-snug transition"
              :class="c.slug === current.slug ? 'border-queen-400 text-fg' : 'border-transparent text-fg-muted hover:border-fg-subtle hover:text-fg'"
            >
              <span class="w-4 shrink-0 text-right font-display text-[15px] leading-5 italic" :class="c.slug === current.slug ? 'text-queen-400' : 'text-fg-subtle'">{{ n }}</span>
              <span>{{ c.title }}</span>
            </NuxtLink>
            <span v-else class="-ml-px flex gap-3 border-l border-transparent py-1.5 pr-2 pl-3 text-[13.5px] text-fg-subtle/70">
              <span class="w-4 shrink-0 text-right font-display text-[15px] leading-5 italic">{{ n }}</span>
              <span>{{ c.title }} <span class="text-[10px] tracking-wide uppercase">· soon</span></span>
            </span>
          </li>
        </ul>
      </div>
    </nav>
  </aside>
</template>
