<script setup lang="ts">
import { chapterHref, chaptersBuildingOn, learnPaths, pathsThrough, type LearnChapter } from "~/data/learnChapters"
import { palette } from "~/utils/palette"

// The end of a chapter: previous / next along the path being read, then where the graph forks from here -- every
// chapter that builds on this one, each shown on its own home path, so a reader can change direction deliberately.
const props = defineProps<{ current: LearnChapter }>()
const position = useLearnPath(toRef(props, "current"))
const forks = computed(() =>
  chaptersBuildingOn(props.current.slug)
    .filter((c) => c.status === "available" && c.slug !== position.value.next?.slug)
    .map((c) => ({ chapter: c, path: pathsThrough(c)[0]! })),
)
const nextPaths = computed(() => learnPaths.filter((p) => p.id !== position.value.path.id && !p.chapters.includes(props.current.slug)))
</script>

<template>
  <nav aria-label="Chapter navigation">
    <div class="grid gap-3 sm:grid-cols-2">
      <NuxtLink v-if="position.prev" :to="chapterHref(position.prev, position.path)" class="card card-hover group block p-5">
        <p class="label">← Previous</p>
        <p class="mt-2 font-display text-xl leading-tight transition group-hover:text-queen-200">{{ position.prev.title }}</p>
      </NuxtLink>
      <span v-else />
      <NuxtLink v-if="position.next" :to="chapterHref(position.next, position.path)" class="card card-hover group block p-5 text-right">
        <p class="label text-queen-300">Next on {{ position.path.title }} →</p>
        <p class="mt-2 font-display text-xl leading-tight transition group-hover:text-queen-200">{{ position.next.title }}</p>
      </NuxtLink>
      <div v-else class="card p-5 text-right">
        <p class="label" :style="{ color: palette[position.path.color] }">End of the {{ position.path.title }} path</p>
        <p class="mt-2 text-sm text-fg-muted">Pick another path below, or see them all on the <NuxtLink to="/learn" class="link">map</NuxtLink>.</p>
      </div>
    </div>

    <div v-if="forks.length" class="mt-10">
      <p class="label">This chapter also leads to</p>
      <ul class="mt-3 grid gap-3 sm:grid-cols-2">
        <li v-for="{ chapter: c, path } in forks" :key="c.slug">
          <NuxtLink :to="chapterHref(c, path)" class="card card-hover group flex h-full flex-col p-4">
            <p class="flex items-center gap-2 text-xs text-fg-subtle">
              <span class="size-2 rounded-full" :style="{ background: palette[path.color] }" />{{ path.title }}
            </p>
            <p class="mt-1.5 font-display text-lg leading-tight transition group-hover:text-queen-200">{{ c.title }}</p>
          </NuxtLink>
        </li>
      </ul>
    </div>

    <div v-if="!position.next && nextPaths.length" class="mt-10">
      <p class="label">Other paths</p>
      <ul class="mt-3 grid gap-3 sm:grid-cols-2">
        <li v-for="p in nextPaths" :key="p.id">
          <LearnPathCard :path="p" compact />
        </li>
      </ul>
    </div>
  </nav>
</template>
