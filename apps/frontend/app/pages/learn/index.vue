<script setup lang="ts">
import { chapterBySlug, chapterHref, learnChapters, learnPath, learnPaths, type LearnPath, type LearnPathId } from "~/data/learnChapters"
import { palette } from "~/utils/palette"

useHead({ title: "Learn" })

// The textbook's contents, drawn as what it is: not one line but a few paths through a graph of chapters. The map
// first (hover a path card to light its line), the paths as cards, then every chapter under its home path.
const available = learnChapters.filter((c) => c.status === "available")
const totalMinutes = available.reduce((sum, c) => sum + (c.readMinutes ?? 0), 0)
const highlight = ref<LearnPathId | null>(null)
const { isRead } = useReadChapters()
const stopsOf = (p: LearnPath) => p.chapters.map((slug) => chapterBySlug(slug)!)
const NuxtLink = resolveComponent("NuxtLink")
</script>

<template>
  <main class="mx-auto max-w-[1600px] px-4 pt-8 pb-10 sm:px-6 lg:px-8">
    <div class="grid gap-8 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)] lg:items-end">
      <UiSectionHeader :level="1" eyebrow="The interactive textbook" title="Learn how it actually works">
        Pick a path. The chapters branch rather than run in one line -- evolution, reinforcement learning and neural
        networks meet in a few places and part ways in others. Every chapter cites the real code, shows real results
        (including what didn't work the first time), and runs its demos live in your browser.
      </UiSectionHeader>
      <div>
        <LearnSearch size="lg" placeholder="Search e.g. “lexicase”, “reward hacking”, “Pareto”…" />
        <p class="mt-4 flex flex-wrap gap-x-5 gap-y-1 font-mono text-xs text-fg-subtle">
          <span><span class="text-fg">{{ learnPaths.length }}</span> paths</span>
          <span><span class="text-fg">{{ available.length }}</span> chapters</span>
          <span><span class="text-fg">~{{ totalMinutes }}</span> min of reading</span>
        </p>
      </div>
    </div>

    <section class="card ticks mt-12 px-2 py-4 sm:px-6" aria-label="Map">
      <LearnMap :highlight="highlight" @hover-path="(id) => (highlight = id)" />
      <p class="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1 px-2 text-xs text-fg-subtle">
        <span class="flex items-center gap-1.5"><span class="size-3 rounded-full border-[3px] border-fg bg-bg" /> interchange: on two paths</span>
        <span class="flex items-center gap-1.5"><span class="w-5 border-t-2 border-dotted border-fg-subtle" /> assumed from another path</span>
        <span class="flex items-center gap-1.5"><span class="size-2.5 rounded-full bg-fg-muted" /> read</span>
      </p>
    </section>

    <section class="mt-10 grid gap-5 sm:grid-cols-2 xl:grid-cols-4" aria-label="Paths">
      <div v-for="p in learnPaths" :key="p.id" @mouseenter="highlight = p.id" @mouseleave="highlight = null">
        <LearnPathCard :path="p" :highlighted="highlight === p.id" />
      </div>
    </section>

    <!-- Every chapter, under its path -->
    <section class="mt-24">
      <p class="label">Every chapter</p>
      <div class="mt-4 border-t border-line-strong">
        <div v-for="p in learnPaths" :key="p.id" class="grid gap-x-10 border-b border-line py-8 lg:grid-cols-[220px_minmax(0,1fr)]">
          <div class="mb-4 lg:mb-0">
            <p class="flex items-center gap-2 text-sm text-fg-subtle">
              <span class="size-2.5 rounded-full" :style="{ background: palette[p.color] }" /> Path
            </p>
            <h2 class="text-[1.9rem] leading-tight">{{ p.title }}</h2>
            <p class="mt-2 text-sm text-fg-muted">{{ p.goal }}</p>
            <p v-if="p.assumes?.length" class="mt-2 text-xs text-fg-subtle">
              Assumes
              <template v-for="(slug, i) in p.assumes" :key="slug">
                <NuxtLink :to="chapterBySlug(slug)!.path" class="link">{{ chapterBySlug(slug)!.short }}</NuxtLink><span v-if="i < p.assumes.length - 1">, </span>
              </template>.
            </p>
          </div>
          <ol class="divide-y divide-line/70">
            <li v-for="(c, i) in stopsOf(p)" :key="c.slug">
              <component
                :is="c.status === 'available' ? NuxtLink : 'div'"
                :to="c.status === 'available' ? chapterHref(c, p) : undefined"
                class="group grid grid-cols-[2.75rem_minmax(0,1fr)] items-start gap-x-4 py-5 first:pt-0 md:grid-cols-[2.75rem_minmax(0,1fr)_168px] md:gap-x-6"
                :class="c.status === 'available' ? '' : 'opacity-50'"
              >
                <span class="relative flex h-8 items-center">
                  <span
                    class="size-4 rounded-full border-[3px]"
                    :style="{ borderColor: palette[p.color], background: isRead(c.slug) ? palette[p.color] : 'transparent' }"
                  />
                  <span class="ml-1.5 font-mono text-[11px] text-fg-subtle">{{ i + 1 }}</span>
                </span>
                <div class="min-w-0">
                  <h3 class="font-display text-[1.65rem] leading-tight font-normal transition group-hover:text-queen-200">
                    {{ c.title }}
                    <span v-if="c.home !== p.id" class="ml-2 align-middle font-mono text-[10px] tracking-widest text-fg-subtle uppercase">
                      also on {{ learnPath(c.home)!.title }}
                    </span>
                  </h3>
                  <p class="mt-1.5 max-w-2xl text-[15px] leading-relaxed text-fg-muted">{{ c.summary }}</p>
                  <p class="mt-3 flex flex-wrap items-center gap-1.5">
                    <span v-if="c.readMinutes" class="chip text-fg">{{ c.readMinutes }} min</span>
                    <span v-for="tag in c.tags.slice(0, 4)" :key="tag" class="chip">{{ tag }}</span>
                  </p>
                </div>
                <div class="relative hidden aspect-[16/10] overflow-hidden rounded-[6px] border border-line bg-sunken md:block">
                  <img
                    v-if="c.image"
                    :src="c.image"
                    :alt="`${c.title} cover`"
                    class="size-full object-cover object-top opacity-80 transition duration-500 group-hover:scale-[1.04] group-hover:opacity-100"
                    loading="lazy"
                  >
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
