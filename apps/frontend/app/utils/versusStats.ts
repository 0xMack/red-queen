// The statistics of two-player strength (docs/design/0013), the TypeScript side of jobs/versus_stats.py: Elo, game
// pairs and the sequential probability ratio test. Used by the Checkers page (a human's performance rating) and the
// "Measuring Strength" chapter's demos. Keep the formulas in step with the Python.

/** Expected points per game for a player `elo` points stronger (a draw counts half). */
export function expectedScore(elo: number): number {
  return 1 / (1 + 10 ** (-elo / 400))
}

/** The Elo difference that expects `score` points per game; a perfect or zero score is capped (its Elo is infinite). */
export function eloOf(score: number, cap = 1e-4): number {
  const s = Math.min(Math.max(score, cap), 1 - cap)
  return -400 * Math.log10(1 / s - 1)
}

/** A performance rating: the Elo at which the expected points against these opponents equal the points actually
 *  scored -- how the leaderboard would rate a player from these games alone. One virtual draw against the average
 *  opponent keeps a perfect (or zero) score finite, as the leaderboard's one virtual draw per pairing does. */
export function performanceRating(games: { opponentElo: number; points: number }[]): number | null {
  if (!games.length) return null
  const mean = games.reduce((s, g) => s + g.opponentElo, 0) / games.length
  const all = [...games, { opponentElo: mean, points: 0.5 }]
  const scored = all.reduce((s, g) => s + g.points, 0)
  let lo = mean - 2000
  let hi = mean + 2000
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2
    const expected = all.reduce((s, g) => s + expectedScore(mid - g.opponentElo), 0)
    if (expected < scored) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

export const PAIR_OUTCOMES = [0, 0.5, 1, 1.5, 2] as const

/** Log-likelihood ratio of "A is elo1 stronger" over "A is elo0 stronger" from game-pair scores (0 ... 2), treating a
 *  pair's score as normal: N (s1 - s0)(2 x̄ - s0 - s1) / (2 σ²). */
export function llr(pairs: number[], elo0: number, elo1: number, varianceFloor = 1e-3): number {
  const n = pairs.length
  if (!n) return 0
  const scores = pairs.map((p) => p / 2)
  const mean = scores.reduce((a, b) => a + b, 0) / n
  const variance = Math.max(scores.reduce((a, s) => a + (s - mean) ** 2, 0) / n, varianceFloor)
  const s0 = expectedScore(elo0)
  const s1 = expectedScore(elo1)
  return (n * (s1 - s0) * (2 * mean - s0 - s1)) / (2 * variance)
}

/** Wald's stopping bounds on the LLR for error rates α (a false "stronger") and β (a missed one). */
export function sprtBounds(alpha: number, beta: number): { lower: number; upper: number } {
  return { lower: Math.log(beta / (1 - alpha)), upper: Math.log((1 - beta) / alpha) }
}

/** A seeded uniform generator (mulberry32), so a demo's run is reproducible and SSR-safe. */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** One simulated game pair between players `elo` apart, from the first's side: each game drawn with a probability that
 *  peaks (at `peakDraws`) between equals, else won with the probability that keeps the expected score at Elo's. The
 *  same model jobs/tests/test_versus_stats.py simulates with. */
export function simulatePair(rand: () => number, elo: number, peakDraws = 0.4): number {
  const s = expectedScore(elo)
  const draw = peakDraws * 4 * s * (1 - s)
  const win = (s - draw / 2) / (1 - draw)
  let points = 0
  for (let g = 0; g < 2; g++) {
    if (rand() < draw) points += 0.5
    else if (rand() < win) points += 1
  }
  return points
}
