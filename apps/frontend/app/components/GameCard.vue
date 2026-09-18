<script setup lang="ts">
import type { GameEntry } from "~/data/games"
import { checkersSnapshot } from "~/data/checkersSnapshot"

defineProps<{ game: GameEntry; compact?: boolean }>()
</script>

<template>
  <article
    class="card group flex flex-col overflow-hidden"
    :class="game.status === 'available' ? 'card-hover' : ''"
  >
    <NuxtLink
      :to="game.playHref ?? '/games'"
      class="relative block overflow-hidden border-b border-line bg-sunken"
      :class="compact ? 'aspect-[16/10]' : 'aspect-[4/3]'"
    >
      <img
        v-if="game.image"
        :src="game.image"
        :alt="`${game.title} screenshot`"
        class="size-full object-cover transition duration-500 group-hover:scale-[1.03]"
        loading="lazy"
      >
      <div v-else-if="game.art === 'checkers'" class="flex size-full items-center justify-center p-6">
        <CheckersBoard :state="checkersSnapshot" class="max-h-full max-w-[75%] opacity-80" />
      </div>
      <div class="absolute inset-0 bg-gradient-to-t from-bg/80 via-transparent to-transparent" />
      <span
        class="absolute top-3 left-3 rounded-full border px-2 py-0.5 text-[11px] font-medium backdrop-blur"
        :class="game.status === 'available' ? 'border-life-400/30 bg-bg/60 text-life-300' : 'border-line-strong bg-bg/60 text-fg-muted'"
      >
        {{ game.status === "available" ? "Playable now" : "Coming soon" }}
      </span>
    </NuxtLink>

    <div class="flex flex-1 flex-col p-5">
      <p class="eyebrow">{{ game.tagline }}</p>
      <h3 class="mt-2 text-xl font-semibold">{{ game.title }}</h3>
      <p class="mt-2 text-sm leading-relaxed text-fg-muted" :class="compact ? 'line-clamp-3' : ''">{{ game.summary }}</p>
      <div v-if="!compact" class="mt-4 flex flex-wrap gap-1.5">
        <span v-for="fact in game.facts" :key="fact" class="chip">{{ fact }}</span>
      </div>
      <div class="mt-auto flex flex-wrap gap-2 pt-5">
        <NuxtLink v-if="game.playHref" :to="game.playHref" class="btn-primary btn-sm">Play</NuxtLink>
        <NuxtLink v-if="game.runsHref" :to="game.runsHref" class="btn-ghost btn-sm">Watch trained runs</NuxtLink>
        <span v-if="game.status === 'coming-soon'" class="text-xs text-fg-subtle">Framework built · UI in progress</span>
      </div>
    </div>
  </article>
</template>
