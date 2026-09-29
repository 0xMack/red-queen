<script setup lang="ts">
import { chapterBySlug, chapterHref, learnPaths, pathsThrough, type LearnChapter } from "~/data/learnChapters"
import { palette } from "~/utils/palette"

// The textbook's sidebar: search, the path this chapter is being read on drawn as a line of stops (read ones filled,
// the current one ringed), a switch to any other path through this chapter, and the rest of the paths below.
// Collapses to a "Stop N of M" toggle on narrow screens.
const props = defineProps<{ current: LearnChapter }>()
const current = toRef(props, "current")
const position = useLearnPath(current)
const alternatives = computed(() => pathsThrough(props.current).filter((p) => p.id !== position.value.path.id))
const otherPaths = computed(() => learnPaths.filter((p) => !p.chapters.includes(props.current.slug)))
const assumed = computed(() => (position.value.path.assumes ?? []).map((slug) => chapterBySlug(slug)!))
const { isRead } = useReadChapters()

const route = useRoute()
const open = ref(false)
watch(() => route.fullPath, () => (open.value = false))
</script>

<template>
  <aside class="lg:sticky lg:top-20 lg:h-[calc(100vh-6rem)] lg:overflow-y-auto lg:pr-2 lg:pb-8">
    <LearnSearch />
    <button class="btn-ghost mt-3 w-full justify-between lg:hidden" :aria-expanded="open" @click="open = !open">
      <span class="flex items-center gap-2">
        <span class="size-2 rounded-full" :style="{ background: palette[position.path.color] }" />
        {{ position.path.title }} · {{ position.index + 1 }} of {{ position.stops.length }}
      </span>
      <span>{{ open ? "▴" : "▾" }}</span>
    </button>

    <nav class="mt-6 space-y-8" :class="open ? 'block' : 'hidden lg:block'" aria-label="Learning path">
      <NuxtLink to="/learn" class="flex items-center gap-2 text-sm text-fg-subtle transition hover:text-fg">
        <span>←</span> All paths
      </NuxtLink>

      <div>
        <p class="label mb-1">Path</p>
        <p class="flex items-center gap-2 font-display text-xl leading-tight">
          <span class="size-2.5 shrink-0 rounded-full" :style="{ background: palette[position.path.color] }" />
          {{ position.path.title }}
        </p>
        <p v-if="assumed.length" class="mt-2 text-xs leading-relaxed text-fg-subtle">
          Assumes
          <template v-for="(c, i) in assumed" :key="c.slug">
            <NuxtLink :to="c.path" class="link">{{ c.short }}</NuxtLink><span v-if="i < assumed.length - 1">, </span>
          </template>.
        </p>

        <ol class="relative mt-4">
          <li v-for="(c, i) in position.stops" :key="c.slug" class="relative flex gap-3 pb-1">
            <!-- the line between this stop and the next -->
            <span
              v-if="i < position.stops.length - 1"
              class="absolute top-4 left-[6px] h-[calc(100%-4px)] w-0.5"
              :style="{ background: palette[position.path.color], opacity: i < position.index ? 0.9 : 0.3 }"
            />
            <span
              class="relative z-10 mt-[7px] size-3.5 shrink-0 rounded-full border-2"
              :class="c.slug === current.slug ? 'ring-4 ring-bg' : ''"
              :style="{
                borderColor: palette[position.path.color],
                background: c.slug === current.slug || isRead(c.slug) ? palette[position.path.color] : palette.bg,
              }"
            />
            <NuxtLink
              v-if="c.status === 'available'"
              :to="chapterHref(c, position.path)"
              class="min-w-0 py-1 text-[13.5px] leading-snug transition"
              :class="c.slug === current.slug ? 'text-fg' : 'text-fg-muted hover:text-fg'"
              :aria-current="c.slug === current.slug ? 'page' : undefined"
            >
              {{ c.title }}
              <span v-if="c.readMinutes" class="ml-1 font-mono text-[10.5px] text-fg-subtle">{{ c.readMinutes }}m</span>
            </NuxtLink>
            <span v-else class="py-1 text-[13.5px] text-fg-subtle/70">{{ c.title }} <span class="text-[10px] tracking-wide uppercase">· soon</span></span>
          </li>
        </ol>
      </div>

      <div v-if="alternatives.length">
        <p class="label mb-2">Also on</p>
        <ul class="space-y-1">
          <li v-for="p in alternatives" :key="p.id">
            <NuxtLink :to="chapterHref(current, p)" class="flex items-center gap-2 text-[13.5px] text-fg-muted transition hover:text-fg">
              <span class="size-2 rounded-full" :style="{ background: palette[p.color] }" />
              {{ p.title }} <span class="text-fg-subtle">· switch</span>
            </NuxtLink>
          </li>
        </ul>
      </div>

      <div v-if="otherPaths.length">
        <p class="label mb-2">Other paths</p>
        <ul class="space-y-1">
          <li v-for="p in otherPaths" :key="p.id">
            <NuxtLink :to="chapterHref(chapterBySlug(p.chapters[0]!)!, p)" class="flex items-center gap-2 text-[13.5px] text-fg-subtle transition hover:text-fg">
              <span class="size-2 rounded-full opacity-70" :style="{ background: palette[p.color] }" />
              {{ p.title }}
            </NuxtLink>
          </li>
        </ul>
      </div>
    </nav>
  </aside>
</template>
