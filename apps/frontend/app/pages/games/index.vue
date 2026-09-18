<script setup lang="ts">
import { games } from "~/data/games"

useHead({ title: "Games" })
</script>

<template>
  <main class="mx-auto max-w-[1600px] px-4 py-10 sm:px-6 lg:px-8">
    <div class="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:items-end">
      <div>
        <p class="eyebrow">Arena</p>
        <h1 class="mt-2 text-3xl font-semibold sm:text-4xl">Games</h1>
        <p class="mt-3 max-w-2xl text-fg-muted">
          Purpose-built environments to test every algorithm against. Each one runs entirely in your
          browser via Pyodide -- the exact same <code class="chip">libs/games</code> Python code the
          <NuxtLink to="/runs" class="link">training runs</NuxtLink> and its own tests exercise, not a
          JavaScript re-implementation.
        </p>
      </div>
      <div class="grid grid-cols-3 gap-3">
        <StatTile label="Playable" :value="games.filter((g) => g.status === 'available').length" tone="life" />
        <StatTile label="In progress" :value="games.filter((g) => g.status === 'coming-soon').length" />
        <StatTile label="Server calls / move" value="0" tone="queen" />
      </div>
    </div>

    <div class="mt-10 grid gap-6 md:grid-cols-2 2xl:grid-cols-3">
      <GameCard v-for="game in games" :key="game.slug" :game="game" />

      <div class="card flex flex-col items-center justify-center gap-3 border-dashed p-8 text-center">
        <p class="font-display text-lg font-semibold text-fg-muted">Your game here</p>
        <p class="max-w-xs text-sm text-fg-subtle">
          Implement <code class="chip">evolve.Environment</code> (or <code class="chip">MultiAgentEnvironment</code>)
          plus <code class="chip">render_state()</code> in <code class="chip">libs/games</code> and it plugs into
          training, telemetry, and this UI.
        </p>
      </div>
    </div>
  </main>
</template>
