"""The oracle for the Rust bandit strategies (docs/design/0011, rust/core/src/bandit.rs): each strategy's beliefs,
written from the textbook in plain Python, replayed over a game the Rust played. Given the same pulls and payouts,
the beliefs after every pull must match -- and a strategy that chooses deterministically must have chosen an arm
its own scores rank first. Test-only: the executable spec for the update rules.
"""

from __future__ import annotations

import math


class Beliefs:
    """Per-row tables for one strategy: counts, sample averages (or constant-alpha estimates), sums."""

    def __init__(
        self, strategy: str, arms: int, rows: int, binary: bool, scale: float, max_payout: float, params: dict
    ):
        self.strategy, self.arms = strategy, arms
        self.binary, self.scale = binary, scale
        self.params = params
        initial = params.get("initial", 0.0) if strategy in ("greedy", "epsilon_greedy") else 0.0
        if strategy == "q_table":
            initial = params.get("initial_q", 0.0)
        self.alpha = params.get("alpha", 0.0) if strategy not in ("q_table", "gradient") else None
        self.counts = [[0] * arms for _ in range(rows)]
        self.values = [[initial] * arms for _ in range(rows)]
        self.sums = [[0.0] * arms for _ in range(rows)]
        self.optimism = params.get("initial", max_payout)
        self.preferences = [[0.0] * arms for _ in range(rows)]
        self.baselines = [(0.0, 0)] * rows

    def probabilities(self, row: int) -> list[float]:
        h = self.preferences[row]
        top = max(h)
        exps = [math.exp(x - top) for x in h]
        total = sum(exps)
        return [e / total for e in exps]

    def update(self, row: int, arm: int, reward: float) -> None:
        self.counts[row][arm] += 1
        self.sums[row][arm] += reward
        if self.strategy == "q_table":
            alpha = self.params.get("alpha", 0.1)
            self.values[row][arm] += alpha * (reward - self.values[row][arm])  # gamma 0: no bootstrap
        else:
            step = self.alpha if self.alpha else 1.0 / self.counts[row][arm]
            self.values[row][arm] += step * (reward - self.values[row][arm])
        if self.strategy == "gradient":
            mean, n = self.baselines[row]
            baseline = 0.0 if self.params.get("baseline", 1.0) == 0 else (reward if n == 0 else mean)
            pi = self.probabilities(row)
            alpha = self.params.get("alpha", 0.1)
            for b in range(self.arms):
                self.preferences[row][b] += alpha * (reward - baseline) * ((1.0 if b == arm else 0.0) - pi[b])
            self.baselines[row] = (mean + (reward - mean) / (n + 1), n + 1)

    def scores(self, row: int) -> list[float] | None:
        """What a deterministic chooser ranks by; None for strategies that choose by chance."""
        counts, values = self.counts[row], self.values[row]
        if self.strategy == "greedy":
            return list(values)
        if self.strategy == "optimistic":
            return [(self.optimism + self.sums[row][a]) / (1 + counts[a]) for a in range(self.arms)]
        if self.strategy == "ucb1":
            if 0 in counts:
                return [1.0 if n == 0 else 0.0 for n in counts]
            t = sum(counts)
            c = self.params.get("c", math.sqrt(2))
            return [values[a] + c * self.scale * math.sqrt(math.log(t) / counts[a]) for a in range(self.arms)]
        return None

    def expected(self, row: int) -> tuple[list[float], list[float], list[int], list[float]]:
        """(values, spread, counts, probabilities) as the Rust strategy reports them."""
        counts, values = self.counts[row], self.values[row]
        zeros = [0.0] * self.arms
        if self.strategy in ("random", "greedy", "epsilon_greedy", "q_table"):
            return list(values), zeros, list(counts), []
        if self.strategy == "optimistic":
            return self.scores(row), zeros, list(counts), []
        if self.strategy == "ucb1":
            t = max(sum(counts), 1)
            c = self.params.get("c", math.sqrt(2))
            spread = [0.0 if n == 0 else c * self.scale * math.sqrt(math.log(t) / n) for n in counts]
            return list(values), spread, list(counts), []
        if self.strategy == "thompson":
            if self.binary:
                means, spreads = [], []
                for a in range(self.arms):
                    wins = self.sums[row][a]
                    alpha, beta = 1 + wins, 1 + counts[a] - wins
                    n = alpha + beta
                    means.append(alpha / n)
                    spreads.append(math.sqrt(alpha * beta / (n * n * (n + 1))))
                return means, spreads, list(counts), []
            means = [values[a] if counts[a] else 0.0 for a in range(self.arms)]
            spreads = [self.scale / math.sqrt(counts[a]) if counts[a] else self.scale for a in range(self.arms)]
            return means, spreads, list(counts), []
        if self.strategy == "gradient":
            return list(self.preferences[row]), zeros, list(counts), self.probabilities(row)
        raise ValueError(self.strategy)
