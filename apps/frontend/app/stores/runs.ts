import { defineStore } from "pinia"
import type { RunInfo, RunSummary } from "~/types/telemetry"

// How long a loaded run list stays good enough to show without asking the backend again: revisiting a page within
// it costs nothing. Live runs are refreshed by the runs page's own poll instead.
const FRESH_MS = 60_000

export const useRunsStore = defineStore("runs", () => {
  const runs = ref<RunInfo[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  // run_id -> what a runs list shows (counts, best, last stats, a 60-point trend): one request for every run
  // (`GET /runs/summaries`), not each run's full history. Held raw and replaced in one assignment: a deep-reactive
  // map filled one run at a time made the runs page recompute and re-render every row once per run.
  const summaries = shallowRef<Record<string, RunSummary>>({})
  const loadedAt = ref(0) // Date.now() of the last successful load, 0 = never

  async function fetchRuns() {
    loading.value = true
    error.value = null
    try {
      const api = useApi()
      const [fetched, summaryList] = await Promise.all([
        api.fetch<RunInfo[]>("/runs"),
        api.fetch<RunSummary[]>("/runs/summaries").catch(() => [] as RunSummary[]),
      ])
      runs.value = [...fetched].sort((a, b) => b.created_at - a.created_at)
      summaries.value = markRaw(Object.fromEntries(summaryList.map((s) => [s.run_id, markRaw(s)])))
      loadedAt.value = Date.now()
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  /** Load unless what's already here is fresh (or `force`). */
  async function ensureLoaded({ force = false, maxAgeMs = FRESH_MS } = {}) {
    if (!force && loadedAt.value && Date.now() - loadedAt.value < maxAgeMs && !error.value) return
    await fetchRuns()
  }

  return { runs, loading, error, summaries, loadedAt, fetchRuns, ensureLoaded }
})
