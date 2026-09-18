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
  watchable: boolean
}

const REPRESENTATION_LABELS: Record<string, string> = {
  linear_gp: "Linear GP",
  neuroevolution: "Neuroevolution",
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
    network: layers && layers.length > 0 ? layers.join(" → ") : null,
    parameterCount,
    note: str(c.note),
    watchable: game === "snake",
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
