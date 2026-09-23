"""Tabular Q-learning and SARSA with n-step returns, written out independently in plain Python (Sutton & Barto,
2nd ed., sections 6.4-6.5 and 7.1-7.2) -- the oracle for rust/core/src/tabular.rs's update rule.

Same arithmetic order as the Rust (returns accumulated back to front: G = r + gamma * G), so the two tables must be
*equal*, not merely close.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

# (state, action, reward, next_state, done, truncated, next_action)
Transition = tuple[int, int, float, int, bool, bool, int | None]


def replay(
    sarsa: bool,
    states: int,
    actions: int,
    transitions: Iterable[Transition],
    alpha: float = 0.1,
    gamma: float = 0.95,
    n_step: int = 1,
    initial_q: float = 0.0,
) -> list[float]:
    q = [[initial_q] * actions for _ in range(states)]
    pending: deque[tuple[int, int, float]] = deque()

    def greedy_value(state: int) -> float:
        return max(q[state])  # the value of the first maximum is the maximum

    def update_oldest(bootstrap: float) -> None:
        g = bootstrap
        for _, _, reward in reversed(pending):
            g = reward + gamma * g
        state, action, _ = pending.popleft()
        q[state][action] += alpha * (g - q[state][action])

    for state, action, reward, next_state, done, truncated, next_action in transitions:
        pending.append((state, action, reward))
        if sarsa and next_action is not None:
            value_next = q[next_state][next_action]
        else:
            value_next = greedy_value(next_state)
        if done:
            while pending:
                update_oldest(0.0)
        elif truncated:
            while pending:
                update_oldest(value_next)
        elif len(pending) == n_step:
            update_oldest(value_next)
    return [v for row in q for v in row]
