<script setup lang="ts">
import { chapterParts, learnChapters } from "~/data/learnChapters"

useHead({ title: "Learn" })

// The textbook's contents page, laid out like one: parts in order, each chapter a numbered line with its summary,
// what it covers and how long it takes -- and its cover, to recognize it by.
const available = learnChapters.filter((c) => c.status === "available")
const totalMinutes = available.reduce((sum, c) => sum + (c.readMinutes ?? 0), 0)
const parts = chapterParts()
const NuxtLink = resolveComponent("NuxtLink")
</script>

<template>
  <main class="mx-auto max-w-[1600px] px-4 pt-8 pb-10 sm:px-6 lg:px-8">
    <div class="grid gap-10 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)] lg:items-end">
      <div>
        <UiSectionHeader :level="1" eyebrow="The interactive textbook" title="Learn how it actually works">
          Foundations first -- each chapter builds on the last, cites the real code, and shows real results from real runs
          (including the parts that didn't work the first time). The demos run live, in your browser.
        </UiSectionHeader>
        <LearnSearch class="mt-8 max-w-xl" size="lg" placeholder="Search e.g. “lexicase”, “reward hacking”, “Pareto”…" />
        <p class="mt-4 flex flex-wrap gap-x-5 gap-y-1 font-mono text-xs text-fg-subtle">
          <span><span class="text-fg">{{ available.length }}</span> chapters</span>
          <span><span class="text-fg">~{{ totalMinutes }}</span> min of reading</span>
          <span><span class="text-fg">{{ parts.length }}</span> parts</span>
        </p>
      </div>

      <NuxtLink :to="available[0]!.path" class="card card-hover ticks group block overflow-hidden">
        <div class="aspect-[16/8] border-b border-line">
          <ChapterArt :kind="available[0]!.art" />
        </div>
        <div class="flex items-end justify-between gap-4 p-5">
          <div>
            <p class="label text-queen-300">Start here · chapter 1</p>
            <p class="mt-1.5 font-display text-[1.7rem] leading-tight transition group-hover:text-queen-200">{{ available[0]!.title }}</p>
          </div>
          <span class="btn-primary btn-sm shrink-0">Begin →</span>
        </div>
      </NuxtLink>
    </div>

    <!-- Contents -->
    <section class="mt-24">
      <p class="label">Contents</p>
      <div class="mt-4 border-t border-line-strong">
        <div v-for="group in parts" :key="group.part" class="grid gap-x-10 border-b border-line py-8 lg:grid-cols-[220px_minmax(0,1fr)]">
          <div class="mb-4 lg:mb-0">
            <p class="font-display text-lg text-queen-400 italic">Part {{ group.numeral }}</p>
            <h2 class="text-[1.9rem] leading-tight">{{ group.part }}</h2>
          </div>
          <ol class="divide-y divide-line/70">
            <li v-for="{ chapter: c, number } in group.chapters" :key="c.slug">
              <component
                :is="c.status === 'available' ? NuxtLink : 'div'"
                :to="c.status === 'available' ? c.path : undefined"
                class="group grid grid-cols-[2.75rem_minmax(0,1fr)] items-start gap-x-4 py-5 first:pt-0 md:grid-cols-[2.75rem_minmax(0,1fr)_168px] md:gap-x-6"
                :class="c.status === 'available' ? '' : 'opacity-50'"
              >
                <span class="font-display text-[2rem] leading-none text-fg-subtle italic transition group-hover:text-queen-400">{{ number }}</span>
                <div class="min-w-0">
                  <h3 class="font-display text-[1.65rem] leading-tight font-normal transition group-hover:text-queen-200">
                    {{ c.title }}
                    <span v-if="c.status !== 'available'" class="ml-2 align-middle font-mono text-[10px] tracking-widest text-fg-subtle uppercase">soon</span>
                  </h3>
                  <p class="mt-1.5 max-w-2xl text-[15px] leading-relaxed text-fg-muted">{{ c.summary }}</p>
                  <p class="mt-3 flex flex-wrap items-center gap-1.5">
                    <span v-if="c.readMinutes" class="chip text-fg">{{ c.readMinutes }} min</span>
                    <span v-for="tag in c.tags.slice(0, 4)" :key="tag" class="chip">{{ tag }}</span>
                  </p>
                </div>
                <div class="relative hidden aspect-[16/10] overflow-hidden rounded-[6px] border border-line bg-sunken md:block">
                  <img v-if="c.image" :src="c.image" :alt="`${c.title} cover`" class="size-full object-cover object-top opacity-80 transition duration-500 group-hover:scale-[1.04] group-hover:opacity-100" loading="lazy">
                  <ChapterArt v-else :kind="c.art" class="opacity-80 transition duration-500 group-hover:scale-[1.04] group-hover:opacity-100" />
                </div>
              </component>
            </li>
          </ol>
        </div>
      </div>
    </section>
  </main>
</template>
