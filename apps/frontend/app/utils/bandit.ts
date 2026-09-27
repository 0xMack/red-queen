import type { EvaluationRecord } from "~/types/leaderboard"

// The multi-armed bandit (docs/design/0011), as the frontend describes it. The rules and the strategies are the Rust
// core (compiled to WebAssembly, `~/wasm/rl`); this is only what the page needs to *say* about them -- mirrored by
// hand from `games.bandit.SCENARIOS` and `jobs/evaluate_bandit.py`. Auto-imported (app/utils/).

export interface BanditScenario {
  id: string
  title: string
  arms: number
  budget: number
  /** Lamps the payouts depend on: 2 for a contextual scenario. */
  contexts: number
  /** Every payout is a win (1) or a loss (0). */
  binary: boolean
  lesson: string
  /** What goes wrong, and for whom -- the page's caption. */
  pitfall: string
}

export const BANDIT_SCENARIOS: BanditScenario[] = [
  {
    id: "classic",
    title: "Classic",
    arms: 5,
    budget: 100,
    contexts: 1,
    binary: true,
    lesson: "Five win-or-lose machines, one clearly best.",
    pitfall: "Greedy commits to the first machine that pays and never looks again.",
  },
  {
    id: "close-call",
    title: "Close call",
    arms: 5,
    budget: 100,
    contexts: 1,
    binary: true,
    lesson: "The best machine is only a little better than the rest.",
    pitfall: "Small differences need many pulls to see; in 100, nobody is sure.",
  },
  {
    id: "lucky-start",
    title: "Lucky start",
    arms: 5,
    budget: 100,
    contexts: 1,
    binary: false,
    lesson: "Noisy payouts: any machine can have a great pull.",
    pitfall: "Greedy trusts the first lucky payout -- no better than random.",
  },
  {
    id: "jackpot",
    title: "Jackpot",
    arms: 5,
    budget: 100,
    contexts: 1,
    binary: false,
    lesson: "The best machine rarely pays, but pays 50. Another always pays 0.8.",
    pitfall: "A rare jackpot hides in the averages; strategies sized for its spread explore forever.",
  },
  {
    id: "drifting",
    title: "Drifting",
    arms: 5,
    budget: 200,
    contexts: 1,
    binary: true,
    lesson: "Somewhere in pulls 50-70, the best machine breaks.",
    pitfall: "Averages and one-off optimism stay loyal to a machine that stopped paying.",
  },
  {
    id: "too-many-arms",
    title: "Too many arms",
    arms: 16,
    budget: 100,
    contexts: 1,
    binary: true,
    lesson: "Sixteen machines, a hundred pulls.",
    pitfall: "Trying every machine properly costs the whole budget -- UCB insists on it.",
  },
  {
    id: "two-lamps",
    title: "Two lamps",
    arms: 5,
    budget: 100,
    contexts: 2,
    binary: true,
    lesson: "A lamp lights red or blue before each pull; the best machine under one is the worst under the other.",
    pitfall: "A strategy that can't see the lamp keeps one row of values -- and every machine averages the same.",
  },
]

export const scenarioById = (id: string | undefined | null) => BANDIT_SCENARIOS.find((s) => s.id === id) ?? BANDIT_SCENARIOS[0]!

/** The strategies the page can put on stage without a leaderboard record: id, label, params (`name=value,...`). */
export const BANDIT_STRATEGIES: { id: string; strategy: string; params: string; label: string }[] = [
  { id: "greedy", strategy: "greedy", params: "", label: "Greedy" },
  { id: "epsilon", strategy: "epsilon_greedy", params: "epsilon=0.1", label: "ε-greedy (ε 0.1)" },
  { id: "epsilon-decay", strategy: "epsilon_greedy", params: "epsilon=0.3,decay=100", label: "ε-greedy, decaying" },
  { id: "epsilon-tracking", strategy: "epsilon_greedy", params: "epsilon=0.1,alpha=0.2", label: "ε-greedy, constant step" },
  { id: "optimistic", strategy: "optimistic", params: "", label: "Optimistic start" },
  { id: "ucb1", strategy: "ucb1", params: "", label: "UCB1 (c √2)" },
  { id: "ucb-tuned", strategy: "ucb1", params: "c=0.5", label: "UCB, tuned (c 0.5)" },
  { id: "thompson", strategy: "thompson", params: "", label: "Thompson sampling" },
  { id: "gradient", strategy: "gradient", params: "alpha=0.5", label: "Gradient bandit" },
  { id: "q-table", strategy: "q_table", params: "", label: "Q-table (Q-learning's agent)" },
  { id: "random", strategy: "random", params: "", label: "Random" },
]

/** A leaderboard record's strategy and settings, as a `BanditRun` takes them. */
export function entrantStrategy(r: EvaluationRecord): { strategy: string; params: string } | null {
  const b = r.metrics.bandit
  if (!b) return null
  return { strategy: b.strategy, params: Object.entries(b.params).map(([k, v]) => `${k}=${v}`).join(",") }
}

const SETTING_NAMES: Record<string, string> = { epsilon: "ε", alpha: "step", c: "c", decay: "decay over", initial: "starts at", baseline: "baseline" }

/** A strategy's settings in words: `{epsilon: 0.1, alpha: 0.2}` -> "ε 0.1 · step 0.2"; none -> "default settings". */
export function banditSettings(params: Record<string, number>): string {
  const parts = Object.entries(params).map(([k, v]) => `${SETTING_NAMES[k] ?? k} ${v}`)
  return parts.length ? parts.join(" · ") : "default settings"
}

/** One machine's colour on the floor: a fixed hue per machine, so a machine is recognisable across views. */
export const ARM_COLORS = [palette.queen400, palette.signal400, palette.gold400, palette.life400, palette.violet400, palette.teal400, palette.orange400, palette.pink400]
export const armColor = (arm: number) => ARM_COLORS[arm % ARM_COLORS.length]!
/** Machines are lettered A, B, C... -- a number reads as a payout. */
export const armName = (arm: number) => String.fromCharCode(65 + arm)

/** A payout as shown on a reel: wins and losses for a win/lose machine, one decimal otherwise. */
export function formatPayout(value: number, binary: boolean): string {
  if (binary) return value >= 1 ? "WIN" : "–"
  return value.toFixed(1)
}

export interface Beliefs {
  values: number[]
  spread: number[]
  counts: number[]
  /** Pull probabilities, for a strategy that chooses by chance; empty otherwise. */
  probabilities: number[]
}

/** `BanditRun.beliefs()` flattened `[values, spread, counts, probabilities]` -> Beliefs. */
export function parseBeliefs(flat: ArrayLike<number>, arms: number): Beliefs | null {
  if (flat.length === 0) return null
  const part = (i: number) => Array.from({ length: arms }, (_, a) => flat[i * arms + a]!)
  return { values: part(0), spread: part(1), counts: part(2), probabilities: flat.length >= 4 * arms ? part(3) : [] }
}

export interface RevealArm {
  kind: "bernoulli" | "gaussian" | "jackpot" | "fixed"
  mean: number
  p?: number
  sd?: number
  prize?: number
  value?: number
}

export interface Reveal {
  lamps: RevealArm[][]
  drift: { at: number; after: number[] } | null
}

/** How a machine really pays, in words: "wins 62% of the time", "pays 50 with 2% chance". */
export function describeArm(arm: RevealArm): string {
  switch (arm.kind) {
    case "bernoulli":
      return `wins ${Math.round((arm.p ?? 0) * 100)}% of pulls`
    case "gaussian":
      return `pays ${arm.mean.toFixed(2)} on average, ± ${arm.sd}`
    case "jackpot":
      return `pays ${arm.prize} with ${Math.round((arm.p ?? 0) * 100)}% chance`
    case "fixed":
      return `always pays ${arm.value}`
  }
}
