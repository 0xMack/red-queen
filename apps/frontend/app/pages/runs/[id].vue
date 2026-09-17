<script setup lang="ts">
const route = useRoute()
const runId = route.params.id as string

const metricsStream = useMetricsStreamStore()

onMounted(() => {
  metricsStream.start(runId)
})
onUnmounted(() => {
  metricsStream.stop()
})

const latest = computed(() => metricsStream.history.at(-1))
</script>

<template>
  <main class="mx-auto max-w-4xl p-6">
    <NuxtLink to="/" class="text-sm text-slate-500 hover:underline">&larr; all runs</NuxtLink>

    <div class="mt-2 flex items-center gap-3">
      <h1 class="font-mono text-lg text-slate-900">{{ runId }}</h1>
      <span
        class="rounded-full px-2 py-0.5 text-xs font-medium"
        :class="metricsStream.connected ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600'"
      >
        {{ metricsStream.connected ? "live" : "connecting..." }}
      </span>
    </div>

    <p v-if="metricsStream.error" class="mt-4 text-red-600">{{ metricsStream.error }}</p>

    <template v-else>
      <FitnessChart :history="metricsStream.history" class="mt-6" />

      <dl v-if="latest" class="mt-4 grid grid-cols-3 gap-4 text-sm">
        <div>
          <dt class="text-slate-500">generation</dt>
          <dd class="font-mono text-slate-900">{{ latest.generation }}</dd>
        </div>
        <div>
          <dt class="text-slate-500">best fitness</dt>
          <dd class="font-mono text-slate-900">{{ latest.best_fitness.toFixed(4) }}</dd>
        </div>
        <div>
          <dt class="text-slate-500">diversity</dt>
          <dd class="font-mono text-slate-900">{{ latest.diversity.toFixed(4) }}</dd>
        </div>
      </dl>

      <p class="mt-4 text-sm text-slate-500">{{ metricsStream.history.length }} generations recorded</p>
    </template>
  </main>
</template>
