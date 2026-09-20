import type { EvaluationRecord, InterfaceInfo } from "~/types/leaderboard"

// A game's leaderboard data (docs/design/0007), the same fetch for every game: the evaluations
// (`/games/{game}/leaderboard`, produced by jobs/evaluate.py or jobs/evaluate_versus.py -- never
// training fitness) and the representations (`/games/{game}/interfaces`). Records from different
// protocol versions aren't comparable, so only the newest protocol's are ranked. An unreachable backend
// leaves `entries` empty and sets `error`; the page then still lets you play.
export async function useGameBoard(slug: string) {
  const config = useRuntimeConfig()
  const baseURL = config.public.apiBase

  // Both requests start before either is awaited: after an `await`, a plain composable (unlike a
  // <script setup>, whose compiler preserves it) has lost Nuxt's context, so a second useAsyncData
  // after the first await would throw.
  const leaderboard = useAsyncData(`leaderboard-${slug}`, () => $fetch<EvaluationRecord[]>(`/games/${slug}/leaderboard`, { baseURL }))
  const representations = useAsyncData(`interfaces-${slug}`, () =>
    $fetch<InterfaceInfo[]>(`/games/${slug}/interfaces`, { baseURL }).catch(() => []),
  )
  const [{ data: records, error }, { data: interfaceList }] = await Promise.all([leaderboard, representations])

  const protocol = computed(() => [...new Set((records.value ?? []).map((r) => r.protocol))].sort().at(-1) ?? null)
  const entries = computed(() => (records.value ?? []).filter((r) => r.protocol === protocol.value))
  const interfaces = computed(() => interfaceList.value ?? [])
  const interfacesById = computed(() => Object.fromEntries(interfaces.value.map((i) => [i.id, i])))
  const protocolInfo = computed(() => entries.value[0]?.metrics.protocol)

  return { error, protocol, entries, interfaces, interfacesById, protocolInfo }
}

export type GameBoard = Awaited<ReturnType<typeof useGameBoard>>
