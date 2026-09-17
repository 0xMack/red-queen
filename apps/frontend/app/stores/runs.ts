import { defineStore } from "pinia"
import type { RunInfo } from "~/types/telemetry"

export const useRunsStore = defineStore("runs", () => {
  const runs = ref<RunInfo[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function fetchRuns() {
    loading.value = true
    error.value = null
    try {
      const config = useRuntimeConfig()
      runs.value = await $fetch<RunInfo[]>("/runs", { baseURL: config.public.apiBase })
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  return { runs, loading, error, fetchRuns }
})
