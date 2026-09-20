import type { EvaluationRecord } from "~/types/leaderboard"
import type { VersusStrategy } from "~/types/versus"
import { brainFromApi, STATIC_STRATEGIES, wasmStrategy, type CheckersBrain, type CheckersPosition } from "~/utils/checkersEngine"

// The Checkers leaderboard's entrants as players (docs/design/0007): each `EvaluationRecord` becomes a
// `VersusStrategy` keyed by its entrant id, so the stage's seats, the arena and the leaderboard all speak
// about the same entrants. A baseline is a Rust player named in its id (`baseline:material-2`). A champion is
// a trained evaluator: its network comes from the backend as plain numbers (`/runs/{id}/artifacts/{ref}/brain`
// -- layered weights, or a NEAT genome compiled to its evaluation plan, so no client re-implements either) and
// is searched to the depth it was evaluated at (`metrics.model.search_depth`). Each is a few KB, fetched once,
// and an entrant appears in the list as soon as its brain lands. With no leaderboard (backend unreachable) it
// falls back to the static players, so the page still plays.
export function useCheckersEntrants(entries: MaybeRefOrGetter<EvaluationRecord[]>) {
  const baseURL = useRuntimeConfig().public.apiBase
  const brains = shallowRef<Record<string, CheckersBrain>>({})
  const failed = ref<string[]>([])

  async function fetchBrain(record: EvaluationRecord) {
    try {
      const payload = await $fetch(`/runs/${record.run_id}/artifacts/${record.champion_ref}/brain`, { baseURL })
      brains.value = { ...brains.value, [record.entrant_id]: brainFromApi(payload as Parameters<typeof brainFromApi>[0]) }
    } catch (e) {
      failed.value = [...failed.value, `${entrantLabel(record)}: ${e instanceof Error ? e.message : String(e)}`]
    }
  }

  const requested = new Set<string>()
  watch(
    () => toValue(entries),
    (list) => {
      for (const r of list) {
        if (r.entrant_kind !== "champion" || !r.run_id || !r.champion_ref || requested.has(r.entrant_id)) continue
        requested.add(r.entrant_id)
        fetchBrain(r)
      }
    },
    { immediate: true },
  )

  const strategies = computed<VersusStrategy<CheckersPosition>[]>(() => {
    const list = toValue(entries)
    if (!list.length) return STATIC_STRATEGIES
    return list.flatMap((r) => {
      const depth = r.metrics.model.search_depth ?? 1
      const description = `${r.metrics.model.description}${depth > 1 ? `, searching ${depth} plies` : ""}. ${r.metrics.quality.mean.toFixed(2)} points per game.`
      if (r.entrant_kind === "baseline") {
        return [wasmStrategy({ id: r.entrant_id, kind: r.entrant_id.replace(/^baseline:/, ""), label: r.label, description })]
      }
      const brain = brains.value[r.entrant_id]
      return brain ? [wasmStrategy({ id: r.entrant_id, kind: "evaluator", label: entrantLabel(r), description, brain, depth })] : []
    })
  })

  return { strategies, failed }
}
