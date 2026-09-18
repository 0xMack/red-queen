<script setup lang="ts">
import { games } from "~/data/games"
</script>

<template>
  <main class="mx-auto max-w-3xl p-6">
    <h1 class="text-2xl font-semibold text-slate-900">Games</h1>
    <p class="mt-1 text-sm text-slate-500">
      Every game here runs entirely in your browser, via Pyodide -- the exact same
      <code class="rounded bg-slate-100 px-1 py-0.5 text-xs">libs/games</code> Python code the
      <NuxtLink to="/runs" class="text-blue-600 hover:underline">training runs</NuxtLink> and its own
      tests exercise, not a separate copy.
    </p>

    <div class="mt-6 grid gap-4 sm:grid-cols-2">
      <GameCard
        v-for="game in games"
        :key="game.title"
        :title="game.title"
        :summary="game.summary"
        :status="game.status"
        :play-href="game.playHref"
        :runs-href="game.runsHref"
      />
    </div>
  </main>
</template>
