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
  paradigm: "evolution" | "reinforcement_learning"
  terms: RunTerms
  watchable: boolean
}

// What a run's GenerationStats *mean* (docs/design/0010 Decision 3): an RL run records training iterations, episode
// returns and policy entropy in the same fields evolution uses for generations, fitness and population diversity.
export interface RunTerms {
  unit: string // "generation" | "iteration"
  fitness: string // "fitness" | "return"
  diversity: string // "diversity" | "policy entropy"
  diversityTitle: string // its chart's heading
  diversityNote: string
  learner: string // "population" | "agent"
}

const EVOLUTION_TERMS: RunTerms = {
  unit: "generation",
  fitness: "fitness",
  diversity: "diversity",
  diversityTitle: "Population diversity",
  diversityNote: "Genotypic spread of the population -- a collapse toward zero is premature convergence.",
  learner: "population",
}
const RL_TERMS: RunTerms = {
  unit: "iteration",
  fitness: "return",
  diversity: "policy entropy",
  diversityTitle: "Policy entropy",
  diversityNote: "How undecided the agent's policy still is (nats) -- falling toward zero as it commits to its choices.",
  learner: "agent",
}

const REPRESENTATION_LABELS: Record<string, string> = {
  linear_gp: "Linear GP",
  neuroevolution: "Neuroevolution",
  neat: "NEAT",
  tree_gp: "Tree GP",
  // reinforcement learning (docs/design/0010)
  random: "Random agent",
  q_learning: "Q-learning",
  dqn: "DQN",
  reinforce: "REINFORCE",
  a2c: "A2C",
  ppo: "PPO",
  td_lambda: "TD(λ)",
}

function str(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null
}

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

export function capitalize(s: string): string {
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
  const paradigm = c.paradigm === "reinforcement_learning" ? "reinforcement_learning" : "evolution"

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
    paradigm,
    terms: paradigm === "reinforcement_learning" ? RL_TERMS : EVOLUTION_TERMS,
    // Has a champion viewer (WatchChampion / CheckersWatch). RL champions become watchable once modelpack exports
    // them (docs/design/0010 Phase 1).
    watchable: (game === "snake" || game === "checkers") && paradigm === "evolution",
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
