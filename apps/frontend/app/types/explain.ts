// Explainers: what a name on the page *means* -- an algorithm, a scenario, a representation, a metric -- opened from an
// ⓘ next to it. A concept (what ε-greedy is) is authored once in `app/data/explainers/`; an instance (this entrant's
// settings, how it was trained, how it did) is assembled from leaderboard records by `resolveExplainer`. Both render
// through the same card (the popover) and panel (the "More" view), so every kind shares one presentation.

export type ExplainKind = "algorithm" | "entrant" | "scenario" | "representation" | "metric"

/** What to explain, as it appears in the URL (`?explain=algorithm:epsilon-greedy`, `entrant:run:0ddb…`). */
export interface ExplainRef {
  kind: ExplainKind
  id: string
}

/** A static diagram, drawn by `ExplainVisual` (the card and the panel's head). */
export type VisualSpec =
  | { kind: "strategy"; mode: StrategyMode }
  | { kind: "art"; art: import("~/data/learnChapters").ChapterArtKind }
  | { kind: "scenario"; id: string }
  | { kind: "observer"; id: string }
  | { kind: "metric"; mode: MetricMode }

export type StrategyMode =
  | "random"
  | "greedy"
  | "epsilon"
  | "optimistic"
  | "ucb"
  | "thompson"
  | "gradient"
  | "q-table"
  | "q-lookahead"
  | "heuristic"
  | "search"
  | "first-legal"

export type MetricMode = "skill" | "held-out" | "points" | "interval" | "level" | "gap" | "regret" | "best-rate" | "cost"

/** Something live for the panel: a strategy playing a scenario (the real Rust core, in WebAssembly), or an observer
 *  watching a moving snake. */
export type LiveSpec =
  | { kind: "bandit"; strategy: string; params: string; scenario: string; observer?: string; label: string; pickScenario?: boolean; pickStrategy?: boolean }
  | { kind: "observer"; id: string }

export interface Fact {
  label: string
  value: string
  hint?: string
}

export interface ExplainLink {
  label: string
  to: string
  kind: "chapter" | "section" | "run" | "page" | "doc"
}

/** A row of results: skill per scenario, the best strategies on a scenario, an interface's entrants. */
export interface ResultRow {
  label: string
  value: number
  display: string
  /** Another explainer this row opens. */
  ref?: string
  tone?: "best" | "bad" | "self" | "muted"
  sub?: string
}

export interface ExplainResults {
  title: string
  caption?: string
  /** Bars scale to this. */
  max: number
  rows: ResultRow[]
}

/** An authored concept -- one per algorithm, scenario, representation, metric. */
export interface ExplainConcept {
  kind: Exclude<ExplainKind, "entrant">
  id: string
  title: string
  /** What sort of thing: "Bandit strategy", "Neuroevolution", "Observer · Snake". */
  family: string
  /** How it gets its behaviour: "learns within each game", "evolved", "trained by gradients", "hand-written". */
  paradigm?: string
  /** One or two sentences: the card. */
  summary: string
  /** The mechanism, step by step: the panel. */
  how?: string[]
  /** A few lines of the real rule, as code. */
  code?: string
  visual?: VisualSpec
  live?: LiveSpec
  good?: string[]
  bad?: string[]
  /** Chapter slug, and optionally a section heading in it (checked against `learnChapters`). */
  chapter?: { slug: string; section?: string; label?: string }[]
  links?: ExplainLink[]
  /** Other explainers worth opening next (`kind:id`). */
  related?: string[]
  /** What each setting means, by parameter name (bandit strategies). */
  params?: Record<string, { name: string; meaning: string }>
  facts?: Fact[]
}

/** What the card and the panel render: a concept, plus -- for an entrant or a representation seen on a game page --
 *  the instance layer. */
export interface ResolvedExplainer {
  ref: ExplainRef
  kind: ExplainKind
  eyebrow: string
  title: string
  summary: string
  visual?: VisualSpec
  live?: LiveSpec
  how?: string[]
  code?: string
  good?: string[]
  bad?: string[]
  /** Up to four at-a-glance facts: the card shows these. */
  facts: Fact[]
  /** "This entrant": its own settings, model and provenance. */
  instance?: { title: string; subtitle?: string; facts: Fact[]; notes?: string[]; chips?: string[] }
  results?: ExplainResults
  links: ExplainLink[]
  related: { ref: string; label: string }[]
}
