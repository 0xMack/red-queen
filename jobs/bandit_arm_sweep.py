"""How many arms should the bandit game default to? (docs/design/0011, "The default table")

A pure-Python sketch that predates the Rust bandit core: it plays the core strategies on Bernoulli arms (means drawn
uniformly per game) for K arms and N pulls, and reports how far each gets from random (regret relative to random:
1.0 = no better, 0 = perfect), how often pure greedy ends up committed to the wrong arm, how often Thompson sampling
finds the right one, and how many arms greedy never tries. The default should be the smallest table where those
failures are common *and* the strategies separate, with few enough machines for a person to track.

    uv run python jobs/bandit_arm_sweep.py [games_per_cell]
"""

import math
import random
import statistics
import sys

STRATEGIES = ["random", "greedy", "eps0.1", "optimistic", "ucb1", "thompson"]


def choose(
    strategy: str, t: int, q: list[float], counts: list[int], beta: list[list[float]], rng: random.Random
) -> int:
    k = len(q)
    best = lambda score: max(range(k), key=lambda i: (score(i), rng.random()))  # ties broken at random
    if strategy == "random":
        return rng.randrange(k)
    if strategy == "greedy":
        return best(lambda i: q[i])
    if strategy == "eps0.1":
        return rng.randrange(k) if rng.random() < 0.1 else best(lambda i: q[i])
    if strategy == "optimistic":  # every untried arm is assumed to pay the maximum, 1
        return best(lambda i: q[i] if counts[i] else 1.0)
    if strategy == "ucb1":
        return t if t < k else best(lambda i: q[i] + math.sqrt(2 * math.log(t) / counts[i]))
    return best(lambda i: rng.betavariate(beta[i][0], beta[i][1]))  # thompson


def play(strategy: str, means: list[float], pulls: int, rng: random.Random) -> tuple[float, bool, int]:
    """(regret against always pulling the best arm, ended committed to the best arm, arms never pulled)."""
    k = len(means)
    counts, q, beta = [0] * k, [0.0] * k, [[1.0, 1.0] for _ in range(k)]
    for t in range(pulls):
        a = choose(strategy, t, q, counts, beta, rng)
        r = 1.0 if rng.random() < means[a] else 0.0
        counts[a] += 1
        q[a] += (r - q[a]) / counts[a]
        beta[a][0] += r
        beta[a][1] += 1 - r
    top = max(range(k), key=lambda i: means[i])
    regret = pulls * means[top] - sum(c * m for c, m in zip(counts, means, strict=True))
    return regret, counts.index(max(counts)) == top, counts.count(0)


def main(games: int) -> None:
    print(
        f"{'K':>3} {'N':>4} | "
        + " | ".join(f"{s:>10}" for s in STRATEGIES)
        + " || greedy wrong  thompson right  untried (greedy)"
    )
    for pulls in (60, 100, 150):
        for k in (3, 4, 5, 6, 8, 10):
            rng = random.Random(1000 * k + pulls)
            results: dict[str, list[tuple[float, bool, int]]] = {s: [] for s in STRATEGIES}
            for g in range(games):
                means = [rng.random() for _ in range(k)]
                for s in STRATEGIES:
                    results[s].append(play(s, means, pulls, random.Random(g * 7919 + STRATEGIES.index(s))))
            rnd = statistics.mean(r[0] for r in results["random"])
            row = " | ".join(f"{statistics.mean(r[0] for r in results[s]) / rnd:>10.2f}" for s in STRATEGIES)
            wrong = 1 - statistics.mean(r[1] for r in results["greedy"])
            right = statistics.mean(r[1] for r in results["thompson"])
            untried = statistics.mean(r[2] for r in results["greedy"])
            print(f"{k:>3} {pulls:>4} | {row} || {wrong:>12.0%}  {right:>14.0%}  {untried:>16.1f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
