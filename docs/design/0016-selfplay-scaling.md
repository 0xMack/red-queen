# 0016 — Scaling the self-play recipe: a longer TD phase, or a bigger network?

Status: **Done** (2026-10-06). See "Results".
Relates to: [0010](0010-reinforcement-learning.md) Phase 4b (one hidden layer was the ceiling; 2 x 64 broke it) and 4c
(TD-Leaf), [0013](0013-measuring-two-player-strength.md) (head-to-head measurement), [0015](0015-td-leaf-from-the-start.md)
(the recipe: plain TD, then a TD-Leaf fine-tune; its conclusion names this probe).

## Context

The best recipe so far trains a 2 x 64 position evaluator by plain TD(λ) self-play, then fine-tunes it with 40k games of
TD-Leaf(λ) at 2 plies (0015). 0010's Phase 4c found 50k more games of plain TD did nothing for a 1M-game 2 x 64 network,
so that network looked plateaued. The question is what limits the recipe now:

1. **The TD phase's length**: does the fine-tune do better from a 1M-game network than from a 200k-game one?
2. **The network**: do deeper or wider networks keep improving, before and after the fine-tune?

## Experiment (`selfplay-v6`, the 10-member 50% pool, λ 0.7, 5 seeds, `h2h --full`)

Every arm reuses earlier runs where it can. An arm's baseline may now live in another experiment, `(experiment, arm)`, so
the 1M-game 2 x 64 networks (`selfplay-v2`) and 0015's 200k-game fine-tunes are compared without retraining.

| Arm | Training | Weights | Baseline |
|---|---|---|---|
| `l-1m-ft2` | `selfplay-v2/pool-2x64-1m` (1M games of TD), then 40k games of TD-Leaf 2 plies | 6.3k | `selfplay-v5/s-td-ft2` (the same after 200k) |
| `l-3x64-1m` | 32 -> 64 -> 64 -> 64 -> 1, 1M games of TD | 10.5k | `selfplay-v2/pool-2x64-1m` |
| `l-2x128-1m` | 32 -> 128 -> 128 -> 1, 1M games of TD | 20.7k | `selfplay-v2/pool-2x64-1m` |
| `l-3x64-ft2` | `l-3x64-1m`, then 40k games of TD-Leaf 2 plies | 10.5k | `l-1m-ft2` |
| `l-2x128-ft2` | `l-2x128-1m`, then 40k games of TD-Leaf 2 plies | 20.7k | `l-1m-ft2` |

## Results

Measured training cost (active seconds per run, mean of 5, ten runs sharing 32 logical cores): `l-3x64-1m` 4,850,
`l-2x128-1m` 9,272 (`pool-2x64-1m`: 3,443, in `selfplay-v2`); the fine-tunes 772 (2 x 64), 1,111 (3 x 64) and 1,744
(2 x 128). Each comparison is the arm's five final networks against its baseline's, seed for seed, over the whole ballot
from both seats (870 game pairs), both searching 3 plies.

| Comparison | Compute ratio | Elo, 870 pairs | Per seed |
|---|---|---|---|
| `l-1m-ft2` vs `s-td-ft2` (1M games of TD before the fine-tune, not 200k) | 2.4x | **+64 [51, 76]** | +92 +73 +64 +18 +73 |
| `l-3x64-1m` vs `pool-2x64-1m` (a third layer) | 1.4x | +36 [24, 47] | -10 +46 +57 +33 +51 |
| `l-2x128-1m` vs `pool-2x64-1m` (twice the width) | 2.7x | **+100 [87, 112]** | +84 +107 +134 +84 +89 |
| `l-3x64-ft2` vs `l-1m-ft2` (a third layer, fine-tuned) | 1.4x | +20 [8, 31] | -9 +43 +22 -1 +43 |
| `l-2x128-ft2` vs `l-1m-ft2` (twice the width, fine-tuned) | 2.6x | **+68 [56, 80]** | +41 +81 +57 +54 +107 |

- **Capacity was the limit, and width pays most.** Twice the width trained the same 1M games of
  plain TD is +100 on every seed -- the largest single step since TD-Leaf. A third layer of 64 helps much less (+36, 4 of 5
  seeds) at about half the extra weights and half the extra compute.
- **The fine-tune keeps most of the gain.** After 40k games of TD-Leaf both bigger networks still lead: 2 x 128 by +68 on
  every seed, 3 x 64 by +20 (3 of 5 seeds). TD-Leaf closes some of the gap, so part of what width buys, search-trained
  targets also teach -- but not most of it.
- **A longer TD phase pays.** The same fine-tune from 1M games of plain TD rather than 200k is +64 on every seed. "50k more
  games of TD did nothing" (0010, 4c) was the end of the 1M run's curve, not a reason to stop TD early.
- **Not settled**: whether a 2 x 64 network given 2.7x the games (the 2 x 128 run's compute) would match it. Plain TD's
  1M-game curve was flat for 2 x 64 (4c), which suggests not, but this experiment doesn't test it.

## Conclusions and next

Capacity in width is the cheapest strength left in the recipe, and the TD phase should run long. The new best recipe is
**2 x 128, 1M games of plain TD, then TD-Leaf**: about 11,000 active seconds per run, against 4,200 for the 2 x 64 one.

Next, in order of cost:

- **Put it on the leaderboard**: a 2 x 128 network trained by the recipe with seed 0 (and a 3-ply fine-tune, as the
  current #1 had), head to head against the current #1.
- **Keep scaling width** (2 x 256, ~4x 2 x 128's weights) until it stops paying, with a 2 x 64 compute-matched control
  (2.7M games) to separate capacity from training time.
- The larger build 0015 named: AlphaZero-style search targets (a policy head over Checkers moves, MCTS in the Rust core).

**Prepared (2026-10-06), not yet run:** the leaderboard candidate is one command, `jobs/checkers_recipe_run.py` (defaults:
2 x 128, 1M games, then 50k games of TD-Leaf at 3 plies, seed 0; `--td-run <id> --depth 4` adds the 4-ply entrant
without retraining the TD phase). The width arms are `selfplay-v7` in `checkers_selfplay_experiment.py`: `w-2x256-1m`,
`w-2x256-ft2` and the compute-matched control `w-2x64-2700k`. Expect several hours per 2 x 256 run, since 2 x 128 took
~9,300 s at 3.6x fewer weights. The AlphaZero learner is built ([0017](0017-alphazero-checkers.md)), with its arms in
`selfplay-v8`.
