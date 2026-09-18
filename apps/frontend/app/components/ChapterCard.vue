<script setup lang="ts">
defineProps<{
  title: string
  summary: string
  path: string
  status: "available" | "coming-soon"
}>()
</script>

<template>
  <NuxtLink
    v-if="status === 'available'"
    :to="path"
    class="block rounded-lg border border-slate-200 bg-white p-5 transition hover:border-blue-300 hover:shadow-sm"
  >
    <h3 class="font-semibold text-slate-900">{{ title }}</h3>
    <p class="mt-1.5 text-sm text-slate-600">{{ summary }}</p>
  </NuxtLink>
  <div v-else class="rounded-lg border border-dashed border-slate-200 p-5 opacity-60">
    <div class="flex items-center gap-2">
      <h3 class="font-semibold text-slate-700">{{ title }}</h3>
      <span class="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-500">
        Coming soon
      </span>
    </div>
    <p class="mt-1.5 text-sm text-slate-500">{{ summary }}</p>
  </div>
</template>
