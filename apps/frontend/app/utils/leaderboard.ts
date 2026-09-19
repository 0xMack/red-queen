import type { EvaluationRecord } from "~/types/leaderboard"

// Shared by the game page's leaderboard pieces (docs/design/0007). Auto-imported (app/utils/).

export const LEVEL_COLORS: Record<number, string> = { 0: "#a78bfa", 1: "#60a5fa", 2: "#2dd4bf", 3: "#4ade80" }
export const HUMAN_COLOR = "#ff8fa3"

export function entrantColor(r: EvaluationRecord): string {
  if (r.entrant_kind === "baseline") return "#fbbf24"
  if (r.entrant_kind === "human") return HUMAN_COLOR
  return LEVEL_COLORS[r.metrics.model.observer_level] ?? "#a0a8ba"
}

/** Short, distinct label: champions get their run's short id (two runs can share a label). */
export function entrantLabel(r: EvaluationRecord): string {
  return r.run_id ? `${r.label} · ${shortId(r.run_id)}` : r.label
}

/**
 * Compact label for narrow lists: a champion is named by the representation it sees, since
 * "Neuroevolution 11 → 16 → …" truncates into ambiguity. Pair with entrantDetail().
 */
export function entrantShortLabel(r: EvaluationRecord): string {
  if (r.entrant_kind !== "champion") return r.label
  const observer = r.interface.split("/")[1]?.split("+")[0]?.replace(/\.v\d+$/, "") ?? "?"
  return `Evolved net · ${observer}`
}

/** Secondary line for entrantShortLabel: selection method and run id, e.g. "lexicase · e96e09ef". */
export function entrantDetail(r: EvaluationRecord): string {
  const selection = r.label.split(" · ")[1]
  return [selection, r.run_id ? shortId(r.run_id) : null].filter(Boolean).join(" · ") || r.entrant_kind
}

/** A baseline's name from its entrant id ("baseline:greedy" -> "greedy"), or null. */
export function baselineName(r: EvaluationRecord): string | null {
  return r.entrant_kind === "baseline" ? r.entrant_id.replace(/^baseline:/, "") : null
}

export function compactNumber(v: number): string {
  if (v >= 1e6) return `${(v / 1e6).toFixed(v >= 1e7 ? 0 : 1)}M`
  if (v >= 1e3) return `${(v / 1e3).toFixed(v >= 1e4 ? 0 : 1)}k`
  return `${Math.round(v * 100) / 100}`
}

/**
 * How often a single human score beats an entrant, across that entrant's held-out games: the share
 * of its games that scored lower, counting ties as half. Falls back to comparing with the mean for
 * records evaluated before per-game scores were stored.
 */
export function beatRate(humanScore: number, r: EvaluationRecord): number {
  const scores = r.metrics.quality.scores
  if (!scores || scores.length === 0) return humanScore > r.metrics.quality.mean ? 1 : 0
  let wins = 0
  for (const s of scores) wins += humanScore > s ? 1 : humanScore === s ? 0.5 : 0
  return wins / scores.length
}

// Personal Snake results for this browser only -- a visitor's history, not a leaderboard entry
// (submitting humans to the real leaderboard is docs/design/0007 step 4: shared challenge seeds).
export interface HumanHistory {
  best: number
  games: number[]
}

function historyKey(game: string) {
  return `redqueen:${game}:human`
}

export function loadHumanHistory(game: string): HumanHistory {
  try {
    const raw = localStorage.getItem(historyKey(game))
    if (raw) return JSON.parse(raw) as HumanHistory
  } catch {
    // Private mode / blocked storage: behave as a first visit.
  }
  return { best: 0, games: [] }
}

export function saveHumanGame(game: string, score: number): HumanHistory {
  const history = loadHumanHistory(game)
  const next = { best: Math.max(history.best, score), games: [...history.games.slice(-49), score] }
  try {
    localStorage.setItem(historyKey(game), JSON.stringify(next))
  } catch {
    // Not persisted -- still returned for this session.
  }
  return next
}
