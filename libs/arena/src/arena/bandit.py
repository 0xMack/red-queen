"""The bandit's evaluation protocol (docs/design/0011).

Protocol `bandit.skill.v1`: HELD_OUT games (seeds SEED_BASE onward) of every scenario. A game's **skill** is how much
better than pulling at random a strategy did, in expected payout: 0 = no better than random, 100 = the best arm on
every pull (the core's `Bandit::skill`; expected, so it scores the choices, not the dice). The leaderboard ranks by
skill on RANKED, the default game. Changing any of that means a new protocol version, not an edit.
"""

GAME = "bandit"
PROTOCOL = "bandit.skill.v1"
HELD_OUT = 500
SEED_BASE = 10_000
RANKED = "classic"
# Training runs' held-out curve (an evolved strategy, every few generations): kept apart from the leaderboard's games.
MONITOR_SEEDS: tuple[int, ...] = tuple(range(20_000, 20_200))
