<script setup lang="ts">
import { learnChapters } from "~/data/learnChapters"
</script>

<template>
  <main class="mx-auto max-w-3xl p-6">
    <h1 class="text-2xl font-semibold text-slate-900">Learn</h1>
    <p class="mt-1 text-sm text-slate-500">
      How this project's algorithms actually work, in order -- each chapter builds on the ones
      before it, and links out to the real code and real results, not just the theory.
    </p>

    <div class="mt-6 flex flex-col gap-3">
      <ChapterCard
        v-for="chapter in learnChapters"
        :key="chapter.path"
        :title="chapter.title"
        :summary="chapter.summary"
        :path="chapter.path"
        :status="chapter.status"
      />
    </div>
  </main>
</template>
