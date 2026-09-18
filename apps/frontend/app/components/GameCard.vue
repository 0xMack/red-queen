<script setup lang="ts">
defineProps<{
  title: string
  summary: string
  status: "available" | "coming-soon"
  playHref?: string
  // Not a specific run (ids rot/expire) -- links to /runs so a visitor can pick any recorded run
  // of this game to watch, e.g. filtered by config.game on that page.
  runsHref?: string
}>()
</script>

<template>
  <div
    class="rounded-lg border p-5"
    :class="status === 'available' ? 'border-slate-200 bg-white' : 'border-dashed border-slate-200 opacity-60'"
  >
    <div class="flex items-center gap-2">
      <h3 class="font-semibold" :class="status === 'available' ? 'text-slate-900' : 'text-slate-700'">
        {{ title }}
      </h3>
      <span
        v-if="status === 'coming-soon'"
        class="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-500"
      >
        Coming soon
      </span>
    </div>
    <p class="mt-1.5 text-sm" :class="status === 'available' ? 'text-slate-600' : 'text-slate-500'">
      {{ summary }}
    </p>
    <div v-if="status === 'available'" class="mt-3 flex gap-4 text-sm font-medium">
      <NuxtLink v-if="playHref" :to="playHref" class="text-blue-600 hover:underline">Play &rarr;</NuxtLink>
      <NuxtLink v-if="runsHref" :to="runsHref" class="text-blue-600 hover:underline">Watch a trained run &rarr;</NuxtLink>
    </div>
  </div>
</template>
