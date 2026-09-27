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
  <main v-else class="mx-auto max-w-[1600px] px-4 pt-8 pb-10 sm:px-6 lg:px-8">
    <UiSectionHeader :level="1" :eyebrow="game?.tagline ?? 'Games'" :title="game?.title ?? slug" :back="{ to: '/games', label: 'Games' }">
      {{ game ? game.summary : `No game called “${slug}”.` }}
    </UiSectionHeader>
    <UiEmpty class="mt-10 min-h-60">
      <template v-if="game">Coming soon.</template>
      <NuxtLink v-else to="/games" class="link">See all games</NuxtLink>
    </UiEmpty>
  </main>
</template>
