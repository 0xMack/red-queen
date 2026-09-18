<script setup lang="ts">
const runsStore = useRunsStore()
await useAsyncData("runs", () => runsStore.fetchRuns())

function statusClasses(status: string): string {
  return (
    {
      running: "bg-blue-100 text-blue-700",
      completed: "bg-green-100 text-green-700",
      failed: "bg-red-100 text-red-700",
      paused: "bg-amber-100 text-amber-700",
    }[status] ?? "bg-slate-100 text-slate-700"
  )
}
</script>

<template>
  <main class="mx-auto max-w-3xl p-6">
    <h1 class="text-2xl font-semibold text-slate-900">Runs</h1>
    <p class="mt-1 text-sm text-slate-500">Training runs recorded in libs/telemetry.</p>

    <p v-if="runsStore.error" class="mt-4 text-red-600">{{ runsStore.error }}</p>

    <ul v-else class="mt-4 divide-y divide-slate-200 rounded-lg border border-slate-200 bg-white">
      <li v-for="run in runsStore.runs" :key="run.run_id">
        <NuxtLink
          :to="`/runs/${run.run_id}`"
          class="flex items-center justify-between px-4 py-3 hover:bg-slate-50"
        >
          <span class="font-mono text-sm text-slate-700">{{ run.run_id }}</span>
          <span class="rounded-full px-2 py-0.5 text-xs font-medium" :class="statusClasses(run.status)">
            {{ run.status }}
          </span>
        </NuxtLink>
      </li>
      <li v-if="runsStore.runs.length === 0" class="px-4 py-6 text-center text-slate-500">
        No runs yet -- start one with <code class="font-mono">uv run python jobs/baseline_gp_run.py</code>.
      </li>
    </ul>
  </main>
</template>
