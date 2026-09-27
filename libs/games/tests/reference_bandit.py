"""The oracle for the Rust bandit (docs/design/0011): the game written from rust/core/src/bandit.rs's specification in
plain Python -- streams, draws, scenarios, pulls and expected regret. `test_bandit.py` holds the Rust to it, pull for
pull. Test-only: the executable spec, not a second implementation for anything to run.
"""

from __future__ import annotations

from reference_snake import Pcg32

_MASK64 = (1 << 64) - 1


def _mix(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def stream_seed(seed: int, k: int) -> int:
    return _mix(seed ^ _mix((k + 0x9E3779B97F4A7C15) & _MASK64))


def uniform(rng: Pcg32) -> float:
    return rng.next_u32() / 4294967296.0


def normal(rng: Pcg32) -> float:
    total = 0.0
    for _ in range(12):
        total += uniform(rng)
    return total - 6.0


def shuffle(items: list, rng: Pcg32) -> None:
    for i in range(len(items) - 1, 0, -1):
        j = rng.bounded(i + 1)
        items[i], items[j] = items[j], items[i]


# A payout is (kind, a, b): bernoulli (p), gaussian (mean, sd), jackpot (p, prize), fixed (value).
def mean(payout) -> float:
    kind, a, b = payout
    return {"bernoulli": a, "gaussian": a, "jackpot": a * b, "fixed": a}[kind]


def draw(payout, rng: Pcg32) -> float:
    kind, a, b = payout
    if kind == "bernoulli":
        return 1.0 if uniform(rng) < a else 0.0
    if kind == "gaussian":
        return a + b * normal(rng)
    if kind == "jackpot":
        return b if uniform(rng) < a else 0.0
    return a


def ladder(rng: Pcg32) -> list:
    arms = [("bernoulli", base - 0.05 + 0.1 * uniform(rng), 0.0) for base in (0.15, 0.30, 0.45, 0.60, 0.75)]
    shuffle(arms, rng)
    return arms


def best_index(arms: list) -> int:
    best = 0
    for i, arm in enumerate(arms):
        if mean(arm) > mean(arms[best]):
            best = i
    return best


def worst_index(arms: list) -> int:
    worst = 0
    for i, arm in enumerate(arms):
        if mean(arm) < mean(arms[worst]):
            worst = i
    return worst


ARMS = {"too-many-arms": 16}
BUDGET = {"drifting": 200}


class ReferenceBandit:
    def __init__(self, scenario: str, seed: int):
        self.scenario, self.seed = scenario, seed
        setup = Pcg32(stream_seed(seed, 0))
        self.k = ARMS.get(scenario, 5)
        self.budget = BUDGET.get(scenario, 100)
        self.drift = None
        if scenario == "classic":
            self.arms = [ladder(setup)]
        elif scenario == "close-call":
            arms = [("bernoulli", 0.55, 0.0)] + [("bernoulli", 0.42 + 0.08 * uniform(setup), 0.0) for _ in range(4)]
            shuffle(arms, setup)
            self.arms = [arms]
        elif scenario == "lucky-start":
            arms = [("gaussian", m, 3.0) for m in (4.0, 4.75, 5.5, 6.25, 7.0)]
            shuffle(arms, setup)
            self.arms = [arms]
        elif scenario == "jackpot":
            arms = [("jackpot", 0.02, 50.0), ("fixed", 0.8, 0.0)]
            arms += [("bernoulli", 0.3 + 0.3 * uniform(setup), 0.0) for _ in range(3)]
            shuffle(arms, setup)
            self.arms = [arms]
        elif scenario == "drifting":
            before = ladder(setup)
            at = 50 + setup.bounded(21)
            after = list(before)
            after[best_index(before)] = before[worst_index(before)]
            self.drift = (at, after)
            self.arms = [before]
        elif scenario == "too-many-arms":
            self.arms = [[("bernoulli", 0.05 + 0.75 * uniform(setup), 0.0) for _ in range(self.k)]]
        elif scenario == "two-lamps":
            red = ladder(setup)
            self.arms = [red, [("bernoulli", 0.9 - mean(arm), 0.0) for arm in red]]
        else:
            raise ValueError(scenario)
        self.arm_rngs = [Pcg32(stream_seed(seed, 100 + a)) for a in range(self.k)]
        self.lamp_rng = Pcg32(stream_seed(seed, 1))
        self.pulls = 0
        self.total = self.expected = self.best_expected = self.random_expected = 0.0
        self.counts = [0] * self.k
        self.best_pulls = 0
        self.lamp = self._next_lamp()

    def _next_lamp(self) -> int:
        if len(self.arms) == 1:
            return 0
        return 0 if uniform(self.lamp_rng) < 0.5 else 1

    def current(self) -> list:
        if self.drift is not None and self.pulls >= self.drift[0]:
            return self.drift[1]
        return self.arms[self.lamp]

    def means(self) -> list[float]:
        return [mean(a) for a in self.current()]

    def pull(self, arm: int) -> tuple[float, bool]:
        current = self.current()
        best = best_index(current)
        reward = draw(current[arm], self.arm_rngs[arm])
        self.counts[arm] += 1
        self.total += reward
        self.expected += mean(current[arm])
        self.best_expected += mean(current[best])
        average = 0.0
        for a in current:
            average += mean(a)
        self.random_expected += average / len(current)
        if arm == best:
            self.best_pulls += 1
        self.pulls += 1
        self.lamp = self._next_lamp()
        return reward, self.pulls >= self.budget
