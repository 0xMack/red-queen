import { defineStore } from "pinia"
import type { GenerationStats, RunInfo } from "~/types/telemetry"

export const useRunsStore = defineStore("runs", () => {
  const runs = ref<RunInfo[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  // run_id -> full metrics history. Fetched per run (client-side, in parallel) for the runs table's
  // sparklines/generation counts -- there's no aggregate endpoint, and at this project's scale
  // (a handful of runs, a few hundred generations each) N small requests is simpler than adding one.
  const histories = ref<Record<string, GenerationStats[]>>({})

  async function fetchRuns() {
    loading.value = true
    error.value = null
    try {
      const api = useApi()
      const fetched = await api.fetch<RunInfo[]>("/runs")
      runs.value = [...fetched].sort((a, b) => b.created_at - a.created_at)
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  async function fetchHistories(ids = runs.value.map((r) => r.run_id)) {
    const api = useApi()
    await Promise.all(
      ids.map(async (id) => {
        try {
          const history = await api.fetch<GenerationStats[]>(`/runs/${id}/metrics/history`)
          histories.value = { ...histories.value, [id]: history }
        } catch {
          // A run with no metrics yet (or an unreachable file) just shows no sparkline.
        }
      }),
    )
  }

  return { runs, loading, error, histories, fetchRuns, fetchHistories }
})
