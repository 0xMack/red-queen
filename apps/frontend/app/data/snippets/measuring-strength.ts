import type { Snippet } from "~/types/code"

// Measuring Strength's snippets. The measurement is Python (jobs/versus_stats.py, jobs/evaluate_versus.py,
// libs/games/src/games/checkers_openings.py); the chapter's own demos run the TypeScript twin (utils/versusStats.ts).
const STATS = "jobs/versus_stats.py"
const TS = "apps/frontend/app/utils/versusStats.ts"

export const ballot: Snippet = {
  pseudo: `// The ballot: every position three moves in that a deep material search calls level
openings ← every position reachable in 3 plies (a transposition counted once)
ballot ← [o in openings where material_search(o, depth 8) = 0]      // 174 of 216

// Every opening is played twice, the colours swapped: a game pair
for each opening o in the pairing's share of the ballot
    pair ← play(A as Red, B as Black, from o) + play(B as Red, A as Black, from o)   // 0 … 2 points for A`,
  python: {
    source: "libs/games/src/games/checkers_openings.py",
    code: `@cache
def ballot(plies: int = 3, depth: int = 8) -> tuple[Opening, ...]:
    """The openings every strength measurement plays (174 with the defaults), in a fixed order."""
    return tuple(opening for opening in all_openings(plies) if level_value(opening, depth) == 0.0)`,
  },
}

export const expected: Snippet = {
  pseudo: `// Elo: a rating difference is a prediction
function expected_score(d)             // A is d points stronger
    return 1 / (1 + 10^(−d / 400))    // 0 → 0.5, +200 → 0.76, +400 → 0.91`,
  python: {
    source: STATS,
    code: `def expected_score(elo: float) -> float:
    """Expected points per game for a player \`elo\` points stronger."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def elo_of(score: float, cap: float = 1e-4) -> float:
    """The Elo difference that expects \`score\` points per game."""
    score = min(max(score, cap), 1.0 - cap)
    return -400.0 * math.log10(1.0 / score - 1.0)`,
  },
  typescript: {
    source: TS,
    code: `export function expectedScore(elo: number): number {
  return 1 / (1 + 10 ** (-elo / 400))
}`,
  },
}

export const bradleyTerry: Snippet = {
  pseudo: `// Ratings from a whole round robin (Bradley–Terry, by Newton's method on log-strengths)
θ ← 0 for everyone                      // the anchor (Random) stays at 0
repeat until nothing moves
    p[i][j] ← 1 / (1 + e^(θj − θi))     // i's expected score against j
    gradient[i] ← points i scored − Σj games[i][j] · p[i][j]      // actual minus expected
    θ ← θ − Hessian⁻¹ · gradient
Elo[i] ← θi · 400 / ln 10`,
  python: {
    source: STATS,
    code: `for _ in range(100):
    p = 1.0 / (1.0 + np.exp(theta[None, :] - theta[:, None]))  # p[i, j]: P(i beats j)
    gradient = total - (games * p).sum(axis=1)                  # points scored minus points expected
    weight = games * p * p.T
    hessian = weight - np.diag(weight.sum(axis=1))
    step = np.linalg.solve(hessian[np.ix_(free, free)], -gradient[free])
    theta[free] += np.clip(step, -2.0, 2.0)
    if np.max(np.abs(step)) < 1e-10:
        break`,
  },
}

export const sprt: Snippet = {
  pseudo: `// Is A at least elo1 stronger (H1), or no stronger (H0)? Play pairs until the evidence says which.
lower ← ln(β / (1 − α))          // −2.94 with α = β = 0.05
upper ← ln((1 − β) / α)          // +2.94
for each opening, in a fixed shuffled order
    pairs.append(play_pair(A, B, opening) / 2)
    s0, s1 ← expected_score(elo0), expected_score(elo1)
    LLR ← N (s1 − s0)(2·mean(pairs) − s0 − s1) / (2·variance(pairs))
    if LLR ≥ upper: return "H1"
    if LLR ≤ lower: return "H0"
return "inconclusive"             // the ballot ran out first`,
  python: {
    source: STATS,
    code: `def llr(pairs: PairPoints, elo0: float, elo1: float, variance_floor: float = 1e-3) -> float:
    """Log-likelihood ratio of H1 (A is \`elo1\` stronger) over H0 (A is \`elo0\` stronger) given the game pairs so far,
    treating pair scores (0 ... 1) as normal: N (s1 - s0) (2 x̄ - s0 - s1) / (2 σ²)."""
    scores = [p / 2.0 for p in pairs]
    s0, s1 = expected_score(elo0), expected_score(elo1)
    mean = statistics.fmean(scores)
    variance = max(statistics.pvariance(scores), variance_floor)
    return len(scores) * (s1 - s0) * (2.0 * mean - s0 - s1) / (2.0 * variance)`,
  },
  typescript: {
    source: TS,
    code: `export function llr(pairs: number[], elo0: number, elo1: number, varianceFloor = 1e-3): number {
  const n = pairs.length
  if (!n) return 0
  const scores = pairs.map((p) => p / 2)
  const mean = scores.reduce((a, b) => a + b, 0) / n
  const variance = Math.max(scores.reduce((a, s) => a + (s - mean) ** 2, 0) / n, varianceFloor)
  const s0 = expectedScore(elo0)
  const s1 = expectedScore(elo1)
  return (n * (s1 - s0) * (2 * mean - s0 - s1)) / (2 * variance)
}`,
  },
}
