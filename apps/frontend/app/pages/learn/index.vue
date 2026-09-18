<script setup lang="ts">
import { learnChapters } from "~/data/learnChapters"

useHead({ title: "Learn" })

const available = learnChapters.filter((c) => c.status === "available")
const totalMinutes = available.reduce((sum, c) => sum + (c.readMinutes ?? 0), 0)
const parts = [...new Set(learnChapters.map((c) => c.part))]
const activePart = ref<string | null>(null)
const shown = computed(() =>
  learnChapters
    .map((chapter, i) => ({ chapter, number: i + 1 }))
    .filter(({ chapter }) => !activePart.value || chapter.part === activePart.value),
)
</script>

<template>
  <main class="mx-auto max-w-[1600px] px-4 py-10 sm:px-6 lg:px-8">
    <section class="card relative overflow-hidden p-6 sm:p-10">
      <div class="pointer-events-none absolute -top-40 -right-40 size-[480px] rounded-full bg-queen-500/10 blur-3xl" />
      <div class="relative grid gap-8 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)] lg:items-center">
        <div>
          <p class="eyebrow">The interactive textbook</p>
          <h1 class="mt-3 text-4xl font-semibold sm:text-5xl">Learn how it actually works</h1>
          <p class="mt-4 max-w-2xl text-lg text-fg-muted">
            Foundations first -- each chapter builds on the last, cites the real code, and shows real
            results from real runs (including the parts that didn't work the first time).
          </p>
          <div class="mt-6 max-w-xl">
            <LearnSearch placeholder="Search e.g. “lexicase”, “reward hacking”, “Pareto”…" />
          </div>
          <div class="mt-4 flex flex-wrap gap-2 text-xs">
            <span class="chip text-fg">{{ available.length }} of {{ learnChapters.length }} chapters written</span>
            <span class="chip">~{{ totalMinutes }} min of reading</span>
            <span class="chip">live demos embedded</span>
          </div>
        </div>

        <NuxtLink :to="available[0]!.path" class="card card-hover group block overflow-hidden">
          <div class="aspect-[16/9] border-b border-line">
            <ChapterArt :kind="available[0]!.art" />
          </div>
          <div class="p-5">
            <p class="eyebrow">Start here · Chapter 1</p>
            <p class="mt-2 text-lg font-semibold group-hover:text-queen-200">{{ available[0]!.title }}</p>
            <p class="mt-1 text-sm text-fg-muted">{{ available[0]!.summary }}</p>
          </div>
        </NuxtLink>
      </div>
    </section>

    <div class="mt-12 flex flex-wrap items-center gap-2">
      <h2 class="mr-4 text-2xl font-semibold">All chapters</h2>
      <button
        class="rounded-full border px-3 py-1 text-xs transition"
        :class="activePart === null ? 'border-queen-400/50 bg-queen-500/10 text-queen-200' : 'border-line text-fg-muted hover:text-fg'"
        @click="activePart = null"
      >
        All
      </button>
      <button
        v-for="part in parts"
        :key="part"
        class="rounded-full border px-3 py-1 text-xs transition"
        :class="activePart === part ? 'border-queen-400/50 bg-queen-500/10 text-queen-200' : 'border-line text-fg-muted hover:text-fg'"
        @click="activePart = part"
      >
        {{ part }}
      </button>
    </div>

    <div class="mt-6 grid gap-5 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
      <ChapterCard v-for="{ chapter, number } in shown" :key="chapter.slug" :chapter="chapter" :number="number" />
    </div>
  </main>
</template>
