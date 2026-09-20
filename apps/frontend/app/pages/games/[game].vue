<script setup lang="ts">
import { games } from "~/data/games"
import { gameModules } from "~/games/registry"

// One page per game, one layout for all of them (docs/design/0007): this route only looks the game up and
// hands it to GamePage, which renders the shared skeleton -- stage, ranked leaderboard, measurements,
// representations -- around whatever the game's module (app/games/) supplies. A game in data/games.ts with
// no module yet is a "coming soon" stub.
const slug = useRoute().params.game as string
const game = games.find((g) => g.slug === slug)
const gameModule = gameModules[slug]
useHead({ title: game?.title ?? slug })

const board = gameModule ? await useGameBoard(slug) : null
</script>

<template>
  <GamePage v-if="game && gameModule && board" :game="game" :module="gameModule" :board="board" />
  <main v-else class="mx-auto max-w-[1600px] px-4 py-8 sm:px-6 lg:px-8">
    <NuxtLink to="/games" class="text-sm text-fg-subtle transition hover:text-fg">&larr; Games</NuxtLink>
    <div v-if="!game" class="card mt-8 p-8 text-center text-fg-muted">
      No game called “{{ slug }}”. <NuxtLink to="/games" class="link">See all games</NuxtLink>.
    </div>
    <template v-else>
      <p class="eyebrow mt-4">{{ game.tagline }}</p>
      <h1 class="mt-2 text-3xl font-semibold sm:text-4xl">{{ game.title }}</h1>
      <p class="mt-2 max-w-3xl text-fg-muted">{{ game.summary }}</p>
      <p class="card mt-8 p-6 text-fg-muted">Coming soon.</p>
    </template>
  </main>
</template>
