<script setup lang="ts">
import type { GameEntry } from "~/data/games"
import { checkersSnapshot } from "~/data/checkersSnapshot"

// A game on an index: its board as the cover, what kind of game it is, and the three ways in -- watch, play,
// the leaderboard. `compact` for a grid of several (the landing page).
defineProps<{ game: GameEntry; compact?: boolean }>()
</script>

<template>
  <article class="card group flex flex-col overflow-hidden" :class="game.status === 'available' ? 'card-hover' : 'opacity-70'">
    <NuxtLink
      :to="game.href"
      class="relative block overflow-hidden border-b border-line bg-sunken"
      :class="compact ? 'aspect-[16/10]' : 'aspect-[4/3]'"
    >
      <img
        v-if="game.image"
        :src="game.image"
        :alt="`${game.title} screenshot`"
        class="size-full object-cover opacity-90 transition duration-700 group-hover:scale-[1.03] group-hover:opacity-100"
        loading="lazy"
      >
      <div v-else-if="game.art === 'checkers'" class="flex size-full items-center justify-center p-6">
        <CheckersBoard :state="checkersSnapshot" class="max-h-full max-w-[72%] transition duration-700 group-hover:scale-[1.03]" />
      </div>
      <BanditArt v-else-if="game.art === 'bandit'" class="transition duration-700 group-hover:scale-[1.03]" />
      <div class="absolute inset-0 bg-gradient-to-t from-surface via-transparent to-transparent" />
      <UiBadge class="absolute top-3 left-3 bg-bg/70 backdrop-blur" :tone="game.status === 'available' ? 'life' : 'neutral'" dot>
        {{ game.status === "available" ? "playable" : "coming soon" }}
      </UiBadge>
    </NuxtLink>

    <div class="flex flex-1 flex-col p-5">
      <p class="label">{{ game.tagline }}</p>
      <h3 class="mt-1.5 font-display text-[1.9rem] leading-none font-normal">{{ game.title }}</h3>
      <p class="mt-3 text-sm leading-relaxed text-fg-muted" :class="compact ? 'line-clamp-3' : ''">{{ game.summary }}</p>
      <div v-if="!compact" class="mt-4 flex flex-wrap gap-1.5">
        <span v-for="fact in game.facts" :key="fact" class="chip">{{ fact }}</span>
      </div>
      <div v-if="game.status === 'available'" class="mt-auto flex flex-wrap gap-2 pt-5">
        <NuxtLink :to="game.href" class="btn-primary btn-sm">Watch the models</NuxtLink>
        <NuxtLink :to="{ path: game.href, query: { mode: 'play' } }" class="btn-ghost btn-sm">Play</NuxtLink>
        <NuxtLink :to="{ path: game.href, hash: '#leaderboard' }" class="btn-quiet btn-sm">Leaderboard</NuxtLink>
      </div>
      <p v-else class="mt-auto pt-5 text-xs text-fg-subtle">Framework built · UI in progress</p>
    </div>
  </article>
</template>
