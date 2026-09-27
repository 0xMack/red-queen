<script setup lang="ts">
import { games } from "~/data/games"

useHead({ title: "Games" })
const playable = games.filter((g) => g.status === "available").length
</script>

<template>
  <main class="mx-auto max-w-[1600px] px-4 pt-8 pb-10 sm:px-6 lg:px-8">
    <UiSectionHeader :level="1" eyebrow="The arena" title="Games">
      Purpose-built environments to test every algorithm against. Each runs entirely in your browser as WebAssembly -- the
      same <code class="chip">libs/games</code> Rust core the <NuxtLink to="/runs" class="link">training runs</NuxtLink> play
      through Python, not a JavaScript re-implementation.
      <template #actions>
        <UiStats
          class="w-full sm:w-auto"
          :items="[
            { label: 'playable', value: playable, tone: 'life' },
            { label: 'server calls / move', value: 0, tone: 'queen' },
          ]"
        />
      </template>
    </UiSectionHeader>

    <div class="mt-10 grid gap-6 md:grid-cols-2 2xl:grid-cols-3">
      <GameCard v-for="game in games" :key="game.slug" :game="game" />

      <UiEmpty class="min-h-72">
        <p class="font-display text-2xl text-fg-muted">Your game here</p>
        <p class="max-w-xs text-sm">
          Implement <code class="chip">Environment</code> (or <code class="chip">MultiAgentEnvironment</code>) in the
          <code class="chip">libs/games</code> Rust core, add a <code class="chip">GameModule</code> and a stage, and it plugs into
          training, the leaderboard and this page.
        </p>
      </UiEmpty>
    </div>
  </main>
</template>
