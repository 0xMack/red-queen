# 0013 — Measuring two-player strength: openings, paired games, Elo and SPRT

Status: **Implemented** (2026-09-30): the ballot, paired games, Elo ratings on the versus leaderboard (protocol
`checkers.versus.v2`), and head-to-head SPRT for comparing training regimes. See "Implementation notes".
Relates to: [0006](0006-multiagent-games-and-strategy-framework.md) (the match framework),
[0007](0007-representations-leaderboards-and-tradeoffs.md) (leaderboards and protocols; it planned a rating),
[0010](0010-reinforcement-learning.md) (Checkers self-play, whose experiments outgrew the old yardstick).

## Context

The next thing to explore is *training regimes* for two-player games: opponent pools, league play, population-based
self-play, search-improved targets. Before comparing any of them, the yardstick has to be able to tell them apart,
and by 2026-09-30 it no longer could:

- **Points per game against the field saturates.** The versus leaderboard's leader scored 0.938, undefeated over 380
  games. Most of those points came from beating entrants that are hundreds of rating points weaker, so the number says
  more about who else is on the board than about the leader. A stronger network can only add a few hundredths.
- **The fixed-field experiment report is nearly out of opponents.** `checkers_selfplay_experiment.py` scores each
  run against material-2/3/4 and one evolved network. The TD-Leaf arm already scores 0.92 against the evolved network.
- **The samples are small and their structure is ignored.** Twenty to sixty games per opponent, each counted as an
  independent win/draw/loss, although the two colours of one pairing share luck and most strong games are draws.

Three changes, which suit any two-player game:

1. **A fixed ballot of balanced openings, each played twice with colours swapped** (a *game pair*). This gives every
   comparison varied positions and cancels opening luck.
2. **Elo ratings (Bradley–Terry) instead of points per game** on the leaderboard. A rating is set by the games an
   entrant played against opponents near its own strength, so it doesn't saturate as the field fills up with weaker
   players.
3. **Head-to-head SPRT for "is regime A better than regime B"**: play A against B over the ballot, score game pairs
   (the pentanomial model), and stop once the evidence is conclusive either way. The question stops depending on the
   field.

Not done here (see the conversation that produced this doc): a frozen-champion gauntlet and a hand-crafted evaluator
to strengthen the field; an endgame tablebase to measure evaluators move by move; an external engine. The external
engine is ruled out for now because of rules compatibility (our 40-moves-without-capture draw) and Windows DLL
plumbing.

## Decisions

| Question | Decision |
|---|---|
| Openings | **Every 3-ply position** reachable from the start (216, deduplicated by transposition), keeping those that material search at **8 plies** scores as level for the side to move: **174**. This is the same idea as tournament checkers' three-move ballot, derived from our own rules. It lives in `games.checkers_openings`, as move indices (move order is part of the specified game, docs/design/0009) |
| Opening in an env | `Checkers(opening=...)`: `reset()` returns to the opening, not to the start, because `play_match` resets its env (this also fixes `--opening-plies` in the evolution jobs, which had been silently discarded; see below) |
| Unit of evidence | A **game pair**: one opening, both colours. Its score (0, ½, 1, 1½ or 2 points) is one pentanomial sample |
| Rating model | Bradley–Terry on points (a draw is half a win each way), fitted by minorization–maximization, reported on the Elo scale (400·log₁₀). **One virtual draw per pairing** keeps a perfect score finite. Anchored **random = 0**: random is always an entrant, and it is well connected at the bottom (it takes points from first-legal and material-1) |
| Rating interval | Bootstrap: resample each pairing's game pairs, refit, 2.5–97.5 percentiles (200 replicates) |
| Leaderboard | Protocol `checkers.versus.v2`: 12 openings per pairing (24 games; the old protocol played 20), each pairing taking the next 12 in the ballot so the round robin covers all of it. `quality.mean` is the Elo rating, so every leaderboard component ranks by it unchanged; points per game, the pentanomial counts and the rating interval sit in `metrics.versus` |
| Comparing regimes | `jobs/checkers_sprt.py` (any two entrants) and `checkers_selfplay_experiment.py h2h` (each arm against its baseline arm, same seed, pooled over seeds; `--full` plays the whole ballot for an estimate): the pentanomial GSPRT (H0: Elo ≤ elo0, H1: Elo ≥ elo1; defaults **0 and 50**, α = β = 0.05, **first look after 20 pairs**; see the notes for why not 20 Elo and 10 pairs) over the ballot in a fixed shuffled order. The ballot running out is a result too: **inconclusive**, with the Elo estimate and its interval |

## Why these statistics

- **Pairs, not games.** Two games of one opening with colours swapped are correlated: a lopsided opening favours the
  same colour both times. Scoring the pair cancels that, and the pair-score variance, not the game variance, is what
  the SPRT and the interval need. Treating correlated games as independent overstates confidence.
- **Bradley–Terry, not points per game.** P(i beats j) = rᵢ / (rᵢ + rⱼ). The maximum-likelihood ratings are the ones
  under which every entrant's expected score equals its actual score. A leader that beats a weak field can't gain
  rating from it, because the model already expected those wins.
- **SPRT, not a fixed number of games.** Wald's sequential test answers "is A at least elo1 better, or no better at
  all?" with error rates fixed in advance, and stops as soon as the log-likelihood ratio crosses a bound. Clear
  differences take a handful of pairs; close ones take all the pairs they need. Its LLR uses the normal approximation
  over pair scores, the one chess engine testing uses (the generalized SPRT, Van den Bergh). Pair score x̄ ∈ [0, 1]
  per pair, variance σ²:
  LLR ≈ N (s₁ − s₀)(2x̄ − s₀ − s₁) / (2σ²), where sₖ = 1 / (1 + 10^(−eloₖ/400)).

## Implementation notes

Built (2026-09-30): `games.checkers_openings` (the ballot, pinned by a digest test), `Checkers(opening=...)`,
`jobs/versus_stats.py` (Elo, pentanomial, Bradley–Terry by Newton's method with a clipped step, bootstrap, `Sprt`),
`evaluate_versus.py` protocol v2, `jobs/checkers_sprt.py`, `checkers_selfplay_experiment.py h2h`, the Checkers page on
Elo (a human's row is a performance rating, `utils/versusStats.ts`), and the Learn chapter "Measuring Strength"
(`/learn/measuring-strength`: the ballot, an Elo demo, the leaderboard two ways, a simulated SPRT lab, these results).

**A bug the ballot found.** `checkers_training.OpponentPool`'s `--opening-plies` stepped random moves onto a fresh
`Checkers` and handed it to `play_match`, which starts with `env.reset()` -- so every "opening" was thrown away and every
fitness game started from the standard position. No error, no test failure. `reset()` now returns to the env's opening,
and a test asserts on the first position a player actually sees.

**The ballot.** 216 positions three plies in; material search at 8 plies keeps 174 (at 6 plies 177, at 4 plies 185 --
deeper search finds more forced losses; the count happens to equal the 174 three-move openings of tournament
checkers, a coincidence of our rules, not the official list).

**Were the old games duplicates?** Measured before building: 20 games between the top two entrants from the standard
start were 19 distinct games, 16-20 in other pairings -- random tie-breaks already varied them. The ballot's value is
varied *positions* and cancelled opening luck, not de-duplication.

**Is the head-to-head fair?** The same strategy (`material-3`, which breaks ties at random) against itself over the
whole ballot, eight seed sets: +3.9 Elo [-5.7, +13.5] pooled. No first-player or seat bias; single runs of 174 pairs
ranged -4 to +36, which is what one run's interval (about ±29) predicts.

**The SPRT's own error rates, simulated** (4,000 tests per row, a draw rate that peaks at 40% between equals, 174-pair
budget, α = β = 0.05):

| True difference | elo1 | First look | Says stronger | Says no stronger | Runs out | Mean pairs |
|---|---|---|---|---|---|---|
| 0 | 50 | 10 pairs | 8.1% | 85.0% | 6.9% | 69 |
| 0 | 50 | **20 pairs** | **5.9%** | 87.5% | 6.6% | 73 |
| +25 | 50 | 20 | 39% | 37% | 24% | 99 |
| +60 | 50 | 20 | 96% | 2% | 2% | 57 |
| +150 | 50 | 20 | 100% | 0% | 0% | 22 |
| +20 | 20 | 20 | 13% | 1% | 86% | 167 |

So: the normal approximation over pair scores is anti-conservative when the variance comes from a handful of pairs
(hence the first look at 20), and one head-to-head over 174 pairs can't resolve 20 Elo (hence elo1 = 50 by default;
pooling five seeds' couples gives 870 pairs).

**The leaderboard on v2** (20 entrants, 12 openings per pairing, 456 games each): TD-Leaf 2 × 64 at 4 plies **927
[884, 982]**, at 3 plies 828, plain TD 2 × 64 at 4 plies 797, at 3 plies 727, material-4 663, ... material-2 426, first
legal 35, Random 0, material-1 -23. Points per game barely moved from v1 (the leader 0.938 → 0.923, now with 3 losses
from ballot openings); what changed is the scale: the top four span 0.14 points but 200 Elo.

**Every self-play comparison, head to head** (`h2h --full`: 174 openings × 5 seeds = 870 pairs, both networks at 3 plies;
"field" is the old report's points-per-game difference and paired p):

| Arm vs baseline | Field (p) | SPRT (pairs) | Elo, 870 pairs | Per seed |
|---|---|---|---|---|
| `ft-leaf3` vs `ft-td` | +0.104 (0.125) | H1 (25) | **+120 [108, 132]** | +103 +124 +127 +120 +126 |
| `ft-leaf2` vs `ft-td` | +0.067 (0.125) | H1 (40) | +81 [70, 93] | +82 +71 +81 +54 +119 |
| `pool-2x64-1m` vs `pool-h64-1m` | +0.162 (0.0625) | H1 (20) | +143 [131, 156] | +152 +135 +114 +173 +146 |
| `pool-h192-1m` vs `pool-2x64-1m` | -0.149 (0.0625) | H0 (20) | -151 [-163, -139] | -152 -117 -186 -171 -132 |
| `pool-h64-1m` vs `pool-1m` | +0.032 (0.5) | H1 (20) | **+55 [45, 66]** | +48 +18 +66 +45 +102 |
| `pool-2x32-1m` vs `pool-h64-1m` | +0.019 (0.69) | H1 (35) | **+36 [24, 47]** | +28 +12 +3 +77 +60 |
| `pool-h64` vs `sp-pool` | -0.027 (0.375) | H0 (40) | +16 [5, 28] | +27 -15 +2 +52 +16 |
| `pool-1m` vs `sp-pool` | -0.040 (0.5) | H0 (25) | -14 [-25, -3] | -6 +11 -23 -4 -49 |
| `sp-pool` vs `sp` | +0.004 (1.0) | H0 (20) | +5 [-6, 16] | +24 +21 -7 +22 -33 |
| `sp-lambda0` vs `sp` | -0.030 (0.56) | H0 (120) | -13 [-22, -3] | -15 +12 -16 -8 -37 |
| `sp-lambda1` vs `sp` | -0.253 (0.0625) | H0 (20) | -148 [-161, -136] | -186 -150 -152 -117 -140 |

- **TD-Leaf is decisive**, every seed; the old p = 0.125 was the yardstick. The SPRT needed 25 pairs.
- **Two "same plateau" results weren't**: a 64-wide layer beats a 16-wide one by +55 after 1M games, and two 32-wide
  layers beat one 64-wide by +36, all five seeds each (0010's Phase 4b is annotated). Against a field of material
  searchers the differences don't show -- possibly real non-transitivity, more likely the field's coarseness.
- **The early-stopped SPRT is a verdict, not an estimate**: `sp-pool` stopped at -52 after 20 pairs (full: +5), and
  `pool-2x32-1m` said H1 (>= 50) where the full estimate is +36 -- inside the indifference zone, where either answer is
  allowed. Use `--full` (or `pair_elo` over many pairs) to size an effect.
- **Pairs pool over seeds, regimes don't.** The pooled interval compares these five networks with those five; a regime
  claim needs the seeds to agree (`across_seeds` in the h2h report: mean, sd, count positive, and an exact sign-flip p,
  whose floor with five seeds is 0.0625 however many games are played).

Not done, in order of value: a frozen-champion gauntlet and a hand-crafted evaluator for a stronger field; an endgame
tablebase (4-5 pieces) to score evaluators move by move; the arena on the game page still plays from the standard
start.
