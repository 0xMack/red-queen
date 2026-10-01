# 0014 — Two-player training regimes: who a self-play learner plays, and population-based training

Status: **Done** (2026-10-01): six opponent regimes and population-based training, measured head to head. See "Results"
and "Conclusions"; the Learn chapter is `/learn/training-regimes`.
Relates to: [0010](0010-reinforcement-learning.md) (Checkers self-play, TD(λ) and TD-Leaf, the uniform opponent pool),
[0013](0013-measuring-two-player-strength.md) (the head-to-head measurement every comparison here uses),
[0008](0008-neat-and-tracked-comparisons.md) (tracked experiments), [0006](0006-multiagent-games-and-strategy-framework.md).

## Context

Self-play Checkers (0010) learns from games against itself, optionally with an **opponent pool**: every 5k games the
network is frozen into a pool of the last 10, and half the games are played against a uniformly random member. Measured
head to head (0013), that pool bought nothing over pure self-play (+5 Elo [-6, +16]) at 32 -> 16 -> 1. What *did* help
was capacity (2 x 64) and training through search (TD-Leaf). This doc asks the remaining question from 0013's list:
**does the choice of opponents matter**, and can **evolution tune the learner's settings** while gradients train its
weights?

## Regimes

All on the recipe that works (32 -> 64 -> 64 -> 1 tanh, TD(λ) 0.7, 200k self-play games, 5 seeds), compared head to
head with `checkers_selfplay_experiment.py h2h --full` against the arm each changes one thing from (`selfplay-v4`):

| Arm | Who the learner plays | Baseline |
|---|---|---|
| `g-pool` | itself, and half the time one of its last 10 frozen selves, uniformly (the recipe so far) | -- |
| `g-sp` | only itself | `g-pool` |
| `g-league` | every past self (a pool of 40 at 200k games): fictitious self-play | `g-pool` |
| `g-pfsp` | the last 10, drawn with weight `(1 - score)^2 + 0.01`: prioritized fictitious self-play (AlphaStar), games go to the past selves it still fails to beat | `g-pool` |
| `g-league-pfsp` | every past self, prioritized | `g-league` |
| `g-pool-80` | the pool 80% of the time | `g-pool` |
| `g-rs` | 8 independent learners with sampled settings (learning rate, λ, pool fraction, PFSP); the round robin's best kept -- random search | `g-pool` |
| `g-pbt` | the same population, but every 10k games the bottom quarter copy a top-quarter member's weights and settings, then perturb them -- population-based training | `g-rs` |

`g-rs` and `g-pbt` spend 8x the games of a single learner; `g-rs` is their equal-compute control. A PBT win over `g-rs`
is the evolved schedule; a `g-rs` win over `g-pool` is just "more tries".

## Implementation

- `libs/rl/rust/envs/src/selfplay.rs`: `pfsp` (the prioritization exponent; 0 = uniform, drawing exactly the random
  numbers it always did, so the determinism fixture is unchanged), a running score per pool member (an exponential
  average, rate 0.05, starting at 1/2), `pool_score` in each `train` call's stats (how the network fares against its past
  selves), and `set_param` for the settings PBT moves mid-run (learning rate, λ, pool fraction, PFSP) without touching
  the network, optimizer, pool or exploration schedule.
- `jobs/checkers_pbt_run.py`: the population, its round robins (2 ballot openings per pair, both seats, 3 plies) and
  exploit/explore; recorded as one run whose champion is each round's best member.

## Results

### Opponent regimes (`selfplay-v4`, 2 x 64, 200k games, 5 seeds; `h2h --full`: 174 openings x 5 seeds = 870 pairs, 3 plies)

| Arm vs baseline | Elo, 870 pairs | Per seed | Late pool score | Draw rate |
|---|---|---|---|---|
| `g-sp` vs `g-pool` | **-26 [-37, -16]** | -40 -43 -38 -63 +51 | -- | 0.51 |
| `g-league` vs `g-pool` | **-38 [-50, -28]** | -56 -55 -17 -47 -17 | 0.64 | 0.43 |
| `g-pfsp` vs `g-pool` | -11 [-22, +1] | -49 -13 +4 -21 +26 | 0.54 | 0.48 |
| `g-league-pfsp` vs `g-league` | -8 [-20, +3] | +4 -29 -18 -0 +1 | 0.60 | 0.45 |
| `g-pool-80` vs `g-pool` | **-36 [-47, -24]** | -63 -62 -22 -36 +3 | 0.55 | 0.49 |
| (`g-pool` itself) | | | 0.55 | 0.51 |

"Late pool score": the network's mean score against the pool members it played, over its last 50k games (`pool_score`).

- **The 10-member, 50% pool is a local optimum** among these: dropping it (-26, 4 of 5 seeds), playing it more (-36, 4 of
  5) and lengthening its memory to every past self (-38, all 5) each cost strength. At 32 -> 16 -> 1 the pool was +5
  (0010/0013); with the 2 x 64 network it matters.
- **More history means easier opponents, not better ones**: the league scores 0.64 against its pool vs 0.55 for the
  10-member pool -- games spent on selves it has left behind.
- **PFSP is within noise** (-11 and -8, seeds split). The network scores only ~0.55 against its last ten selves, so
  they're close in strength and there's little for prioritization to choose between; AlphaStar-style prioritization is
  for games with real strategy cycles, which Checkers at this level shows no sign of.

Compute: every arm ~1,000 s of active training per run (measured with ~40 runs sharing 32 logical cores; the regimes
cost the same per game).

### Population-based training

| Arm vs baseline | Elo, 870 pairs | Per seed |
|---|---|---|
| `g-rs` (8 members, best kept, 8x games) vs `g-pool` | -10 [-21, +1] | -31 -26 +23 -11 -4 |
| `g-pbt` vs `g-rs` (equal compute) | -1 [-13, +11] | +10 +31 -19 +13 -40 |

- **Neither population beat one learner with the defaults.** Best-of-8 random search is level with `g-pool` (8x the
  games), and PBT is level with random search.
- **Every PBT run collapsed to one lineage** (of 8) by round 20, but the final settings disagree across seeds (median
  λ 0.42-0.80, pool fraction 0.15-0.59, learning rate 5e-4-1.4e-3; PFSP on in 0-6 of 8): drift, not a discovered
  schedule.
- **Why**: the defaults are near a local optimum (nothing in the regime table beat them either), and selection was
  noise-limited -- each round robin gave a member 28 games (7 opponents x 2 openings x 2 seats), where the differences
  that matter (20-40 Elo) need hundreds of pairs (0013). PBT here mostly copies luck. A fair retry needs a much larger
  evaluation per round (or SPRT-style sequential evaluation between candidates), which multiplies its cost.
- The population runs took ~5 min per round of 80k games with the machine to themselves; `checkers_pbt_run.py` now
  checks for a pause every 1,000 games per member (it first checked once per round, which kept 8 cores busy for half an
  hour after the runs were paused).

## Conclusions

Of everything tried, the plain recipe -- half the games against the ten most recent past selves, settings left
alone -- won. Opponent choice and population-level tuning are not where the strength is at this scale; capacity (0010
Phase 4b) and training through search (TD-Leaf, Phase 4c: +120 Elo) are. Next candidates, in order: TD-Leaf from the
start rather than as a fine-tune; a search-guided policy plus value (AlphaZero-style); and a stronger fixed field
(frozen-champion gauntlet, hand-crafted evaluator) to confirm head-to-head gains against everyone.
