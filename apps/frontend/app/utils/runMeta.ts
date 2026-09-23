import type { GenerationStats, RunInfo } from "~/types/telemetry"

// Everything the UI derives from a run's free-form `config` dict (written by each jobs/*.py script)
// in one place, so the runs table, run detail page, and home page describe a run the same way.
// config is untyped on purpose (telemetry doesn't constrain it), hence the defensive reads.

export interface RunMeta {
  title: string
  game: string | null
  representation: string
  representationLabel: string
  selection: string | null
  variation: string | null
  benchmark: string | null
  populationSize: number | null
  targetGenerations: number | null
  network: string | null
  parameterCount: number | null
  note: string | null
  seedStrategy: string | null // "fixed:5" / "resample:5" (jobs/seeding.py); null for runs before it existed
  watchable: boolean
}

const REPRESENTATION_LABELS: Record<string, string> = {
  linear_gp: "Linear GP",
  neuroevolution: "Neuroevolution",
  neat: "NEAT",
  tree_gp: "Tree GP",
}

function str(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null
}

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function describeRun(run: RunInfo): RunMeta {
  const c = run.config ?? {}
  const game = str(c.game)
  const representation = str(c.representation) ?? "unknown"
  const representationLabel = REPRESENTATION_LABELS[representation] ?? capitalize(representation.replace(/_/g, " "))
  const benchmark = str(c.benchmark)
  const layers = Array.isArray(c.layer_sizes) ? (c.layer_sizes as unknown[]).filter((n): n is number => typeof n === "number") : null

  // Fully-connected weights + biases per layer, matching evolve.neuro.WeightVector's layout.
  const parameterCount = layers && layers.length > 1
    ? layers.slice(1).reduce((sum, size, i) => sum + size * (layers[i]! + 1), 0)
    : null

  const subject = game ? capitalize(game) : benchmark ? `f(x) = ${benchmark}` : "Run"

  return {
    title: `${subject} · ${representationLabel}`,
    game,
    representation,
    representationLabel,
    selection: str(c.selection),
    variation: str(c.variation),
    benchmark,
    populationSize: num(c.population_size),
    targetGenerations: num(c.generations),
    // A NEAT run has no fixed shape -- only what goes in and out; the hidden structure is what it evolves.
    network: layers && layers.length > 0 ? layers.join(" → ") : num(c.num_inputs) !== null && num(c.num_outputs) !== null ? `${c.num_inputs} → evolved → ${c.num_outputs}` : null,
    parameterCount,
    note: str(c.note),
    seedStrategy: str(c.seed_strategy) ?? (Array.isArray(c.training_seeds) ? `fixed:${c.training_seeds.length}` : null),
    watchable: game === "snake" || game === "checkers", // has a champion viewer: WatchChampion / CheckersWatch
  }
}

export function bestFitnessOf(run: RunInfo, history?: GenerationStats[]): number | null {
  const fromSummary = num(run.summary?.best_fitness)
  if (fromSummary !== null) return fromSummary
  if (history && history.length > 0) return Math.max(...history.map((h) => h.best_fitness))
  return null
}

/** A "running" run whose registry row hasn't been touched in a while probably died without saying so. */
export function isStale(run: RunInfo, now = Date.now() / 1000): boolean {
  return run.status === "running" && now - run.updated_at > 15 * 60
}

/** A run as the runs table shows it (pages/runs/index.vue, components/RunsTableRow.vue). */
export interface RunRow {
  run: RunInfo
  meta: RunMeta
  label: string // same naming as the leaderboard: "Snake · NEAT · 11 hidden · 73 conns"
  heldOut: { mean: number; rank: number; of: number; entrantId: string } | null
  generations: number
  best: number | null
  trend: number[]
  duration: number
  stale: boolean
  genome: string
  experiment: string | null // config.experiment: the comparison this run is one arm x seed of
  arm: string | null
}

/** Why a run has no leaderboard entry -- so its absence reads as a fact, not a bug. */
export function unrankedReason(run: RunInfo): string | null {
  const c = run.config ?? {}
  if (c.game !== "snake") return null
  if (c.experiment) return `comparison run (${String(c.experiment)}): aggregated in its experiment, not ranked individually`
  if (c.paradigm === "reinforcement_learning") return "an RL run: ranked once modelpack loads RL champions (docs/design/0010)"
  if (run.status === "running" || run.status === "paused") return "still training: evaluated once finished"
  return "not evaluated yet: run jobs/evaluate.py"
}

