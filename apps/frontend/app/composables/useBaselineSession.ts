import type { LoadedPolicy } from "~/composables/useWatchSession"
import type { RunInfo, GenerationStats } from "~/types/telemetry"

// Watch a games.baselines policy (e.g. the greedy heuristic) play -- the same code
// jobs/evaluate.py scores (the Rust game core's baselines), run in the worker as WebAssembly. Returns the same shape as useWatchSession so
// WatchChampion can render either kind of leaderboard entrant; the run/generation parts are simply
// empty, since a baseline was never trained.
export function useBaselineSession(baseline: string, interfaceId: string) {
  const session = useSnakeSession()

  function start() {
    session.resetStats()
    session.start({ baseline, interfaceId })
  }

  onUnmounted(() => session.stop())

  return {
    ...session,
    run: ref<RunInfo | null>(null),
    lastLoadedRef: ref<string | null>(null),
    policy: ref<LoadedPolicy | null>(null),
    pinnedGeneration: ref<number | null>(null),
    targetStats: computed<GenerationStats | null>(() => null),
    unsupported: ref(null),
    awaitingConfirmation: ref<number | null>(null),
    source: ref<"published" | "on-demand" | null>(null),
    currentPackage: ref(null),
    start,
    retry: start,
    pin: (_generation: number | null) => {},
    confirmDownload: () => {},
  }
}
