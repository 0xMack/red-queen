"""The statistics of two-player strength (docs/design/0013), for any game: Elo, game pairs, Bradley-Terry ratings and
the sequential probability ratio test. Nothing here knows about Checkers -- results come in as points.

- A **game pair** is one opening played twice with the colours swapped; its score is 0, 1/2, 1, 3/2 or 2 points (the
  *pentanomial* outcomes). Pairs, not games, are the independent samples: both games of a lopsided opening favour the
  same colour.
- **Elo** is a scale for P(win): a player rated `d` points above another is expected to score
  1 / (1 + 10^(-d/400)) per game (a draw counting half).
- **Bradley-Terry** fits one rating per player to a whole round robin -- the ratings under which every player's
  expected score equals its actual one -- so a strong player gains nothing from beating weak ones it was expected to
  beat. Reported in Elo, anchored so one named player is 0.
- **SPRT** (Wald's sequential test, in the generalized form over pair scores that chess engine testing uses): is A at
  least `elo1` stronger than B, or not stronger at all (`elo0`)? Stops as soon as the evidence says which, with error
  rates fixed in advance.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

# A pairing's results, from the first player's side: the points it scored in each game pair (0 ... 2).
PairPoints = Sequence[float]

PENTANOMIAL = (0.0, 0.5, 1.0, 1.5, 2.0)


def expected_score(elo: float) -> float:
    """Expected points per game for a player `elo` points stronger."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def elo_of(score: float, cap: float = 1e-4) -> float:
    """The Elo difference that expects `score` points per game (a perfect or zero score is capped: its Elo is
    infinite)."""
    score = min(max(score, cap), 1.0 - cap)
    return -400.0 * math.log10(1.0 / score - 1.0)


def pentanomial(pairs: PairPoints) -> list[int]:
    """How many pairs scored 0, 1/2, 1, 3/2 and 2 points."""
    return [sum(1 for p in pairs if abs(p - outcome) < 1e-9) for outcome in PENTANOMIAL]


def pair_elo(pairs: PairPoints) -> dict[str, float]:
    """A's Elo advantage over B from their game pairs, with a 95% interval (normal approximation over pair scores)."""
    scores = [p / 2.0 for p in pairs]
    n = len(scores)
    mean = statistics.fmean(scores)
    se = statistics.pstdev(scores) / math.sqrt(n) if n > 1 else 0.5
    return {
        "elo": round(elo_of(mean), 1),
        "lo": round(elo_of(mean - 1.96 * se), 1),
        "hi": round(elo_of(mean + 1.96 * se), 1),
        "score": round(mean, 4),
        "pairs": n,
    }


# --- Bradley-Terry ---------------------------------------------------------------------------------------------------


def bradley_terry(
    results: Mapping[tuple[Hashable, Hashable], tuple[float, float]],
    anchor: Hashable,
    prior_draws: float = 1.0,
) -> dict[Hashable, float]:
    """Elo ratings from `results[(a, b)] = (points a scored against b, games played)`, one key per pairing. Each pairing
    also counts `prior_draws` virtual drawn games, so a perfect score has a finite rating; `anchor` is rated 0.

    Maximum likelihood by Newton's method on log-strengths: the log-likelihood is concave, and with the virtual draws
    strictly so once the anchor is fixed, so it converges in a handful of steps."""
    players = sorted({p for pair in results for p in pair}, key=repr)
    index = {p: i for i, p in enumerate(players)}
    k = len(players)
    wins = np.zeros((k, k))  # wins[i, j]: points i scored against j
    games = np.zeros((k, k))
    for (a, b), (points, n) in results.items():
        i, j = index[a], index[b]
        wins[i, j] += points + prior_draws / 2
        wins[j, i] += n - points + prior_draws / 2
        games[i, j] += n + prior_draws
        games[j, i] += n + prior_draws
    free = [i for i in range(k) if players[i] != anchor]
    theta = np.zeros(k)  # natural-log strengths; the anchor stays at 0
    total = wins.sum(axis=1)
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(theta[None, :] - theta[:, None]))  # p[i, j]: P(i beats j)
        gradient = total - (games * p).sum(axis=1)
        weight = games * p * p.T
        hessian = weight - np.diag(weight.sum(axis=1))
        step = np.linalg.solve(hessian[np.ix_(free, free)], -gradient[free])
        step = np.clip(step, -2.0, 2.0)  # Newton overshoots from far away on a logistic; ~870 Elo a step at most
        theta[free] += step
        if np.max(np.abs(step)) < 1e-10:
            break
    scale = 400.0 / math.log(10.0)
    return {p: float(theta[index[p]] * scale) for p in players}


def rate(
    pairings: Mapping[tuple[Hashable, Hashable], PairPoints],
    anchor: Hashable,
    replicates: int = 200,
    seed: int = 0,
) -> dict[Hashable, dict[str, float]]:
    """Bradley-Terry Elo for every player, with a bootstrap 95% interval: each replicate resamples every pairing's game
    pairs with replacement and refits. `pairings[(a, b)]` is a's points in each game pair against b (2 games each)."""

    def fit(sample: Mapping[tuple[Hashable, Hashable], PairPoints]) -> dict[Hashable, float]:
        return bradley_terry({key: (sum(pairs), 2.0 * len(pairs)) for key, pairs in sample.items()}, anchor)

    point = fit(pairings)
    rng = random.Random(seed)
    draws: dict[Hashable, list[float]] = {p: [] for p in point}
    for _ in range(replicates):
        sample = {key: rng.choices(pairs, k=len(pairs)) for key, pairs in pairings.items()}
        for player, elo in fit(sample).items():
            draws[player].append(elo)
    out = {}
    for player, elo in point.items():
        ordered = sorted(draws[player])
        lo = ordered[int(0.025 * (len(ordered) - 1))] if ordered else elo
        hi = ordered[math.ceil(0.975 * (len(ordered) - 1))] if ordered else elo
        out[player] = {"elo": round(elo, 1), "lo": round(lo, 1), "hi": round(hi, 1)}
    return out


# --- SPRT ------------------------------------------------------------------------------------------------------------


def llr(pairs: PairPoints, elo0: float, elo1: float, variance_floor: float = 1e-3) -> float:
    """Log-likelihood ratio of H1 (A is `elo1` stronger) over H0 (A is `elo0` stronger) given the game pairs so far,
    treating pair scores (0 ... 1) as normal: N (s1 - s0) (2 x̄ - s0 - s1) / (2 σ²)."""
    scores = [p / 2.0 for p in pairs]
    if not scores:
        return 0.0
    s0, s1 = expected_score(elo0), expected_score(elo1)
    mean = statistics.fmean(scores)
    variance = max(statistics.pvariance(scores), variance_floor)
    return len(scores) * (s1 - s0) * (2.0 * mean - s0 - s1) / (2.0 * variance)


@dataclass
class Sprt:
    """A running SPRT: feed it game pairs (`add`) until `verdict` is set. `min_pairs` guards the start, where a few
    identical pairs have no variance to speak of."""

    elo0: float = 0.0
    elo1: float = 50.0
    alpha: float = 0.05
    beta: float = 0.05
    min_pairs: int = 20
    pairs: list[float] = field(default_factory=list)

    @property
    def lower(self) -> float:
        return math.log(self.beta / (1.0 - self.alpha))

    @property
    def upper(self) -> float:
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def llr(self) -> float:
        return llr(self.pairs, self.elo0, self.elo1)

    @property
    def verdict(self) -> str | None:
        """H1 (A is at least elo1 stronger), H0 (A is no stronger than elo0), or None while undecided."""
        if len(self.pairs) < self.min_pairs:
            return None
        value = self.llr
        return "H1" if value >= self.upper else "H0" if value <= self.lower else None

    def add(self, pair_points: float) -> str | None:
        self.pairs.append(pair_points)
        return self.verdict

    def summary(self) -> dict:
        return {
            "verdict": self.verdict or "inconclusive",
            "llr": round(self.llr, 3),
            "bounds": [round(self.lower, 3), round(self.upper, 3)],
            "elo0": self.elo0,
            "elo1": self.elo1,
            "alpha": self.alpha,
            "beta": self.beta,
            "pentanomial": pentanomial(self.pairs),
            **pair_elo(self.pairs),
        }
