# 0015 — TD-Leaf from the start: should self-play train through its search from game one?

Status: **Done** (2026-10-01). See "Results".
Relates to: [0010](0010-reinforcement-learning.md) Phase 4c (TD-Leaf as a fine-tune: +120 Elo), [0013](0013-measuring-two-player-strength.md)
(head-to-head measurement), [0014](0014-two-player-training-regimes.md) (the regimes that didn't help; its conclusion
names this as the next candidate).

## Context

TD-Leaf(λ) -- self-play that searches a few plies and learns at the principal variation's leaf -- is the largest gain
measured so far, but only ever as a *finishing step*: a network trained by plain TD(λ) for 1M games, then 50k more games
of TD-Leaf. Two questions follow:

1. **Equal games**: is TD-Leaf from scratch better than plain TD for the same number of self-play games? (It must be at
   least as informative per game; it costs ~4.5x per game at 2 plies, ~19x at 3.)
2. **Equal compute**: given a fixed budget, is it better to spend it on TD-Leaf from the start, on plain TD, or on plain TD
   followed by a TD-Leaf fine-tune?

## Experiment (`selfplay-v5`, 2 x 64 tanh, the 10-member 50% pool, λ 0.7, 5 seeds, `h2h --full`)

Measured cost per game (one process, early games): plain TD 3.6 ms, TD-Leaf 2 plies 16.1 ms, 3 plies 67 ms.

| Arm | Training | ~Compute | Baseline |
|---|---|---|---|
| `s-td` | plain TD, 200k games | 1x | -- |
| `s-leaf2` | TD-Leaf 2 plies from scratch, 200k games | ~4.5x | `s-td` (equal games) |
| `s-leaf2-40k` | TD-Leaf 2 plies from scratch, 40k games | ~1x | `s-td` (equal compute) |
| `s-td-ft2` | `s-td`, then 40k games of TD-Leaf 2 plies | ~2x | `s-td` |
| `s-leaf2-80k` | TD-Leaf 2 plies from scratch, 80k games | ~2x | `s-td-ft2` (equal compute) |

Two plies, not three: 3-ply TD-Leaf from scratch for 200k games would be ~4 hours per run.

## Results

Measured training cost (active seconds per run, mean of 5; up to 10 runs sharing 32 logical cores): `s-td` 764,
`s-leaf2-40k` 1,055, `s-td-ft2` 973 on top of its `s-td` parent (1,737 in all), `s-leaf2-80k` 2,156, `s-leaf2` 5,249.
TD-Leaf games run longer as the network improves (~103 plies vs 83, draws 0.67 vs 0.53), so "40k games = s-td's
compute" came out at 1.4x and "80k = the fine-tune's" at 1.24x -- both in TD-Leaf's favour.

| Comparison | Compute ratio | Elo, 870 pairs | Per seed |
|---|---|---|---|
| `s-leaf2` vs `s-td` (equal games) | 6.9x | **+57 [45, 69]** | +26 +34 +71 +79 +76 |
| `s-leaf2-40k` vs `s-td` (about equal compute) | 1.4x | -6 [-17, +5] | -41 +25 +6 -44 +24 |
| `s-td-ft2` vs `s-td` (TD, then TD-Leaf) | 2.3x | **+74 [62, 86]** | +42 +64 +48 +127 +92 |
| `s-leaf2-80k` vs `s-td-ft2` (about equal compute) | 1.24x | **-54 [-66, -42]** | -7 -46 -73 -89 -55 |
| `s-td-ft2` vs `s-leaf2` (direct) | 0.33x | -1 [-12, +10] | +22 -8 -28 +3 +5 |

- **TD-Leaf is worth more per game, not per second, from scratch.** For the same 200k games it beats plain TD by +57 on
  every seed, but at the same compute it is level (-6).
- **The fine-tune recipe wins on compute.** Plain TD first, then 40k games of TD-Leaf: +74 over plain TD, beats TD-Leaf from
  scratch with 24% more compute by 54 Elo (every seed), and matches 200k games of TD-Leaf from scratch (-1) at a third of
  its compute. Cheap 1-ply games teach the basics; search-trained games are worth their cost once there is a decent
  evaluator to search with.
- So: keep TD-Leaf as the finishing step (as 0010's 4c did), and spend its budget late.

## Conclusions and next

The ordering of the training recipe matters more than any opponent regime did (0014). The remaining big lever is what the
search contributes: TD-Leaf only uses the search to choose *which* position to learn at. An AlphaZero-style learner uses
search results (MCTS visit counts) as *targets* for a policy as well as a value -- a larger build (a policy head over
Checkers moves, MCTS in the Rust core, a WASM path for the page). Before that, a cheaper probe: does a longer schedule
(TD to 1M games, then TD-Leaf 3 plies) keep scaling, or has the 2 x 64 network's capacity become the ceiling?

**Answered by [0016](0016-selfplay-scaling.md)**: it keeps scaling -- the 1M-game TD phase beats 200k by +64 before the
same fine-tune, and a 2 x 128 network beats 2 x 64 by +100 (+68 after the fine-tune). Both the TD
phase's length and the network's capacity were limits.
