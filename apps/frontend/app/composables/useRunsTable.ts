import type { ModelShape } from "~/utils/modelLabel"
import type { RunRow } from "~/utils/runMeta"

// The runs table's model (pages/runs/index.vue), apart from how it's drawn: every run as a `RunRow`, the filters
// (search, status, algorithm), sorting, and experiment groups -- the runs of one comparison (jobs/snake_experiment.py
// and friends: arms x seeds, tagged config.experiment) collapse into one row, since they're read together and there
// are dozens per experiment. `standings` maps a run to its champion's place on its game's leaderboard.

export type Standing = { mean: number; rank: number; of: number; entrantId: string }

export interface RunGroup {
  experiment: string
  rows: RunRow[] // the matching runs, sorted
  total: number // every run of the experiment
  best: number | null
  generations: number
  latest: number
  status: RunRow["run"]["status"]
  statusCounts: string
  arms: string[]
  algorithms: string[]
}
export type RunItem = { kind: "run"; row: RunRow } | { kind: "group"; group: RunGroup }
export type RunSortKey = "created" | "best" | "heldOut" | "generations" | "duration"
export const RUN_STATUSES = ["all", "running", "completed", "paused", "failed"] as const

export function useRunsTable(standings: Ref<{ byRun: Record<string, Standing> }>, now: Ref<number>) {
  const runsStore = useRunsStore()

  const rows = computed<RunRow[]>(() =>
    runsStore.runs.map((run) => {
      const meta = describeRun(run)
      const summary = runsStore.summaries[run.run_id]
      const c = run.config ?? {}
      const genome = meta.network
        ? `${meta.network}${meta.parameterCount ? ` · ${meta.parameterCount}w` : ""}`
        : typeof c.num_instructions === "number"
          ? `${c.num_instructions} instr · ${c.num_registers ?? "?"} regs`
          : "--"
      const last = summary?.last ?? null
      const extras = last?.extras ?? {}
      const shape: ModelShape = {
        algorithm: meta.representationLabel,
        hidden_nodes: typeof extras.champion_hidden_nodes === "number" ? extras.champion_hidden_nodes : undefined,
        connections: typeof extras.champion_connections === "number" ? extras.champion_connections : undefined,
        layer_sizes: Array.isArray(c.layer_sizes) ? (c.layer_sizes as number[]) : undefined,
      }
      const subject = meta.title.split(" · ")[0]
      return {
        run,
        meta,
        label: meta.game ? `${subject} · ${modelLabel(shape)}` : meta.title,
        heldOut: standings.value.byRun[run.run_id] ?? null,
        generations: summary?.generations ?? 0,
        best: bestFitnessOf(run) ?? summary?.best_fitness ?? null,
        trend: summary?.trend ?? [],
        // For a live run the last recorded generation is fresher than the registry row.
        duration: Math.max(run.updated_at, last?.timestamp ?? 0) - run.created_at,
        stale: isStale(run, now.value),
        genome,
        experiment: typeof c.experiment === "string" && c.experiment ? c.experiment : null,
        arm: typeof c.arm === "string" && c.arm ? c.arm : null,
      }
    }),
  )

  // Filters / sorting --------------------------------------------------------------------------------
  const query = ref("")
  const statusFilter = ref<(typeof RUN_STATUSES)[number]>("all")
  const kindFilter = ref("all")
    const sortKey = ref<RunSortKey>("created")
  const sortDesc = ref(true)

  const kinds = computed(() => [...new Set(rows.value.map((r) => r.meta.representationLabel))])

  function sortValue(r: RunRow): number {
    switch (sortKey.value) {
      case "created":
        return r.run.created_at
      case "best":
        return r.best ?? -Infinity
      case "heldOut":
        return r.heldOut?.mean ?? -Infinity
      case "generations":
        return r.generations
      default:
        return r.duration
    }
  }
  const bySort = (a: number, b: number) => (sortDesc.value ? b - a : a - b)

  const matching = computed(() => {
    const q = query.value.trim().toLowerCase()
    return rows.value
      .filter((r) => {
        if (statusFilter.value !== "all" && r.run.status !== statusFilter.value) return false
        if (kindFilter.value !== "all" && r.meta.representationLabel !== kindFilter.value) return false
        if (!q) return true
        return [r.run.run_id, r.label, r.meta.title, r.meta.note, r.meta.selection, r.meta.benchmark, r.experiment, r.arm]
          .filter(Boolean)
          .some((s) => s!.toLowerCase().includes(q))
      })
      .sort((a, b) => bySort(sortValue(a), sortValue(b)))
  })


  const totals = computed(() => {
    const counts: Record<string, number> = {}
    for (const r of rows.value) if (r.experiment) counts[r.experiment] = (counts[r.experiment] ?? 0) + 1
    return counts
  })

  function group(experiment: string, members: RunRow[]): RunGroup {
    const statuses = members.map((r) => r.run.status)
    const count = (s: string) => statuses.filter((x) => x === s).length
    const bests = members.map((r) => r.best).filter((b): b is number => b !== null)
    return {
      experiment,
      rows: members,
      total: totals.value[experiment] ?? members.length,
      best: bests.length ? Math.max(...bests) : null,
      generations: members.reduce((sum, r) => sum + r.generations, 0),
      latest: Math.max(...members.map((r) => r.run.created_at)),
      // the state that most needs attention wins
      status: count("running") ? "running" : count("paused") ? "paused" : count("failed") ? "failed" : "completed",
      statusCounts: (["running", "paused", "failed", "completed"] as const)
        .filter((s) => count(s))
        .map((s) => `${count(s)} ${s}`)
        .join(" · "),
      arms: [...new Set(members.map((r) => r.arm).filter((a): a is string => a !== null))],
      algorithms: [...new Set(members.map((r) => r.meta.representationLabel))],
    }
  }

  const items = computed<RunItem[]>(() => {
    const groups = new Map<string, RunRow[]>()
    const out: RunItem[] = []
    for (const r of matching.value) {
      if (!r.experiment) out.push({ kind: "run", row: r })
      else if (groups.has(r.experiment)) groups.get(r.experiment)!.push(r)
      else groups.set(r.experiment, [r])
    }
    for (const [experiment, members] of groups) out.push({ kind: "group", group: group(experiment, members) })
    // A group sorts by its most extreme member in the current direction: its first (already sorted) row.
    const key = (item: RunItem) => sortValue(item.kind === "run" ? item.row : item.group.rows[0]!)
    return out.sort((a, b) => bySort(key(a), key(b)))
  })

  const expanded = ref(new Set<string>())
  function toggle(experiment: string) {
    const next = new Set(expanded.value)
    if (next.has(experiment)) next.delete(experiment)
    else next.add(experiment)
    expanded.value = next
  }
  // While searching, show the matching runs inside their groups rather than hiding them behind a click.
  const isOpen = (experiment: string) => expanded.value.has(experiment) || query.value.trim() !== ""

  function sortBy(key: RunSortKey) {
    if (sortKey.value === key) sortDesc.value = !sortDesc.value
    else {
      sortKey.value = key
      sortDesc.value = true
    }
  }

  // Summary tiles --------------------------------------------------------------------------------------
  const summary = computed(() => {
    const totalGens = rows.value.reduce((sum, r) => sum + r.generations, 0)
    const snake = rows.value.filter((r) => r.meta.game === "snake" && r.best !== null)
    const bestSnake = snake.length ? Math.max(...snake.map((r) => r.best!)) : null
    return {
      total: rows.value.length,
      running: rows.value.filter((r) => r.run.status === "running" && !r.stale).length,
      experiments: Object.keys(totals.value).length,
      totalGens,
      bestSnake,
      kinds: kinds.value.length,
    }
  })


  return { rows, kinds, query, statusFilter, kindFilter, sortKey, sortDesc, sortBy, matching, items, toggle, isOpen, summary }
}
