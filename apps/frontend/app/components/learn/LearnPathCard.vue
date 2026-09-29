<script setup lang="ts">
import { chapterBySlug, chapterHref, pathMinutes, type LearnPath } from "~/data/learnChapters"
import { palette } from "~/utils/palette"

// One learning path: its goal, its stops as a strip of stations (read ones filled), and a link to the first unread
// stop -- "Start" on a fresh path, "Continue" part-way. `compact` drops the summary and the strip.
const props = defineProps<{ path: LearnPath; compact?: boolean; highlighted?: boolean }>()
const { isRead } = useReadChapters()
const stops = computed(() => props.path.chapters.map((slug) => chapterBySlug(slug)!))
const readCount = computed(() => stops.value.filter((c) => isRead(c.slug)).length)
const resume = computed(() => stops.value.find((c) => c.status === "available" && !isRead(c.slug)) ?? stops.value[0]!)
const color = computed(() => palette[props.path.color])
</script>

<template>
  <NuxtLink
    :to="chapterHref(resume, path)"
    class="card card-hover group flex h-full flex-col overflow-hidden"
    :class="highlighted ? 'border-line-strong' : ''"
  >
    <div class="h-1" :style="{ background: color }" />
    <div class="flex flex-1 flex-col p-5">
      <p class="flex items-center justify-between gap-3 font-mono text-[11px] text-fg-subtle">
        <span>{{ stops.length }} {{ stops.length === 1 ? "chapter" : "chapters" }} · ~{{ pathMinutes(path) }} min</span>
        <span v-if="readCount">{{ readCount }}/{{ stops.length }} read</span>
      </p>
      <h3 class="mt-2 font-display text-[1.7rem] leading-tight font-normal transition group-hover:text-queen-200">{{ path.title }}</h3>
      <p class="mt-1 text-sm text-fg">{{ path.goal }}</p>
      <p v-if="!compact" class="mt-2.5 text-sm leading-relaxed text-fg-muted">{{ path.summary }}</p>

      <ol v-if="!compact" class="mt-5 flex items-center" :aria-label="`${path.title} stops`">
        <li v-for="(c, i) in stops" :key="c.slug" class="flex items-center" :class="i < stops.length - 1 ? 'flex-1' : ''">
          <span
            class="size-3 shrink-0 rounded-full border-2"
            :title="c.title"
            :style="{ borderColor: color, background: isRead(c.slug) ? color : palette.surface }"
          />
          <span v-if="i < stops.length - 1" class="h-0.5 flex-1" :style="{ background: color, opacity: 0.45 }" />
        </li>
      </ol>

      <p class="mt-auto flex items-center justify-between gap-3 pt-5 text-sm">
        <span class="truncate text-fg-subtle">{{ readCount ? "Next up" : "Starts with" }}: <span class="text-fg-muted">{{ resume.short }}</span></span>
        <span class="shrink-0 transition group-hover:translate-x-0.5" :style="{ color }">{{ readCount ? "Continue" : "Start" }} →</span>
      </p>
    </div>
  </NuxtLink>
</template>
