# 0021 — Quiescence search: never score a position mid-exchange

Status: **Built; pilot, `selfplay-v12` to `-v14` done, the quiet recipe leads `checkers.versus.v3`. Quiescence wins at
play time; the best recipe is plain TD then a 200k-game 1-ply quiet fine-tune. `selfplay-v15` (width) is running**
(2026-10-24).
Relates to: [0013](0013-measuring-two-player-strength.md) (game pairs, Elo), [0015](0015-td-leaf-from-the-start.md) and
[0016](0016-selfplay-scaling.md) (TD-Leaf, the recipe), [0020](0020-alphazero-continued.md) (its conclusion: at this
scale, alpha-beta over a TD network is what to improve).

## Context

Every Checkers player that searches (`material-N`, a trained `evaluator` at depth > 1) runs plain fixed-depth
alpha-beta (`checkers_strategies.rs`'s `search`). So does TD-Leaf training (`selfplay.rs`). Neither has a quiescence
search. In Checkers that matters more than in most games, because captures are mandatory: a position where the side to
move must capture is one where the next move is forced and changes the material. A search that stops there scores the
position as if nothing were about to happen. Two moves that both win a piece look equal at depth 1 even when one of them
is recaptured at once (`quiescence_sees_the_recapture_a_fixed_depth_misses` builds that position).

0020 closed the AlphaZero line: at 2 x 64 or 2 x 128, TD-Leaf with alpha-beta is the better use of self-play games. That
makes the search itself the cheapest thing left to improve, for both what plays and what learns.

## Decision

**A quiescence extension, off by default, in both searches.**

- At depth 0, if the side to move has a capture (`checkers::is_capture` on its first legal move; captures are
  mandatory, so all of them are), search on instead of scoring, for up to `quiescence` more plies. Checkers has no
  "stand pat": the mover can't decline the capture, so every capture is searched, not just the good ones.
- **Players:** `Strategy::with_quiescence(plies)`, reached from Python as `evaluator(..., quiescence=Q)` /
  `strategy("material-N", quiescence=Q)`, and from a head-to-head as `+qQ` on any player (`672890ba@3+q4`,
  `material-4+q4`). The experiment report `experiment.checkers-h2h` takes `params.quiescence` for both sides.
- **Training:** `td_lambda`'s `agent.quiescence`. TD-Leaf's principal-variation leaf is then always a quiet position.
  Any `quiescence` uses the searched loop, even at `search_depth` 1.
- `quiescence` 0 is today's search exactly: the golden runs and the determinism digests don't move, and both searches'
  minimax oracles are extended to check the extension too.
- **The leaderboard:** `checkers.versus.v2` is unchanged (a protocol's meaning is fixed). `checkers.versus.v3` is the
  same round robin with every searching entrant (a champion at its depth, `material-3`, `material-4`) extended by
  quiescence 4; its records carry `metrics.model.quiescence`, and the game page plays each entrant with it (the WASM
  `CheckersStrategy` takes `quiescence`). The page shows the latest protocol, so v3 replaces v2 there once it has run.

## Pilot: quiescence at play time (2026-10-23)

The five `selfplay-v6/l-2x128-ft2` networks (2 x 128, 1M games of TD, then 40k of TD-Leaf at 2 plies), each against
itself without the extension, 40 ballot openings from both seats (200 game pairs per row):

| Player | vs | Elo | 95% CI |
|---|---|---|---|
| 3 plies + quiescence 2 | 3 plies | **+130** | [107, 154] |
| 3 plies + quiescence 4 | 3 plies | **+136** | [113, 161] |
| 3 plies + quiescence 8 | 3 plies | **+134** | [111, 159] |
| 3 plies + quiescence 8 | 4 plies | -22 | [-45, 1] |

Time per move (one network, the positions six ballot games pass through, native):

| Search | ms per move |
|---|---|
| 3 plies | 1.73 |
| 3 plies + quiescence 2 / 4 / 8 | 2.10 / 2.17 / 2.19 |
| 4 plies | 6.85 |
| 4 plies + quiescence 4 | 8.29 |
| 5 plies | 22.25 |

- **Quiescence buys most of a ply for a quarter of the time.** +130-136 Elo at 3 plies, every seed, for ~25% more time
  per move. A whole extra ply (4 plies) is only ~20 Elo stronger still, and costs 4x.
- **2-4 plies of extension is enough.** The gain is flat from 2 to 8: exchanges in these games rarely run longer.
  4 is the default the experiments use.

## Next: `selfplay-v12`, training through it

TD-Leaf learns at its search's leaves. Without quiescence a leaf can be half-way through an exchange, so the target it
learns from is the static value of a position about to change, which is noise. `experiments/selfplay-v12.yaml`
fine-tunes the same 1M-game 2 x 128 parents as `selfplay-v6/l-2x128-ft2`, for the same 40k games:

| Arm | Training search | Baseline |
|---|---|---|
| `q-ft2-q4` | 2 plies + quiescence 4 | `selfplay-v6/l-2x128-ft2` (2 plies) |
| `q-ft1-q4` | 1 ply + quiescence 4 (does a quiet leaf stand in for a ply of search, at a quarter the cost?) | `selfplay-v6/l-2x128-ft2` |

Read twice: the report plays each arm against its baseline at 3 plies plain, and again with both sides at 3 plies +
quiescence 4. Training through quiescence could help the network either way, or only once it is searched the way it was
trained. Training cost is compared from each run's `cost.active_s`.

### `selfplay-v12` results (2026-10-24)

Each arm's five networks against `l-2x128-ft2`'s, seed for seed, the whole ballot from both seats (870 game pairs):

| Arm | Active s per run | Both at 3 plies | Both at 3 plies + quiescence 4 |
|---|---|---|---|
| `l-2x128-ft2` (baseline: 2 plies, no quiescence) | 1,744 | 0 | 0 |
| `q-ft2-q4` (2 plies + quiescence 4) | 2,028 | **-54 [-66, -42]**, 0 of 5 seeds positive | +11 [1, 22] |
| `q-ft1-q4` (1 ply + quiescence 4) | **594** | **-43 [-56, -31]**, 0 of 5 | **+42 [31, 52]**, 4 of 5 |

- **A network trained at quiet leaves must be played at quiet leaves.** Searched plainly, both arms lose: their
  networks never learned to score a position mid-exchange, because their search never asked them to, and a plain
  search hands them exactly those positions. Searched with quiescence, both win.
- **One ply plus quiescence is the better fine-tune, at a third of the compute.** `q-ft1-q4` beats the 2-ply TD-Leaf
  fine-tune by 42 Elo for 594 active seconds against 1,744. Adding the second ply (`q-ft2-q4`) costs more and helps less.
  A likely reason: at 1 ply, the quiet leaf is usually the position right after the move, so every position's target is
  its own quiet successor, and those targets are less noisy than a 2-ply minimax over a network that is still moving.

### The recipe on the leaderboard (2026-10-24)

`experiments/pipelines/checkers-recipe.yaml` ran: 2 x 128, 1M games of TD (9,195 active s), then 50k games of TD-Leaf at
3 plies (10,067 s), seed 0. The fine-tune was also submitted at `params.depth: 4` for the 4-ply entrant (training is
deterministic: the two runs' networks are byte-identical).

| Head to head (checkers.sprt, elo0 0, elo1 50) | Elo | Verdict |
|---|---|---|
| recipe at 4 plies vs `672890ba` (2 x 64, the old #1) at 4 plies | **+52 [13, 93]** | H1, 37 pairs |
| recipe at 3 plies + quiescence 4 vs `672890ba` at 4 plies | +40 [10, 71] | H1, 70 pairs |
| recipe at 4 plies + quiescence 4 vs `672890ba` at 4 plies + quiescence 4 | +75 [21, 133] | H1, 26 pairs |

On `checkers.versus.v2` the recipe's 4-ply entrant is the new **#1 (1024)**, ahead of `672890ba` (978); its 3-ply
entrant is #3 (904). The cheaper search with quiescence (3 + q4, ~2.2 ms per move) beats the old #1's 4 plies (~7 ms).

### `selfplay-v13` results (2026-10-24)

Both sides at 3 plies + quiescence 4, the whole ballot from both seats (870 game pairs):

| Comparison | Active s per run | Elo | Seeds positive |
|---|---|---|---|
| `q1-ft-200k` (quiet fine-tune, 200k games) vs v12's `q-ft1-q4` (40k) | 2,918 vs 594 | **+48 [38, 59]** | 5 of 5 |
| `q1-1m` (1M games of quiet TD from scratch) vs `selfplay-v6/l-2x128-1m` (1M of plain TD) | 15,279 vs 9,272 | **+144 [133, 156]** | 5 of 5 |
| `q1-1m` vs `q1-ft-200k` (cross-experiment, same seeds paired) | 15,279 vs 9,272 + 2,918 | **-45 [-55, -35]** | 0 of 5 |

- **The quiet fine-tune keeps paying:** five times the games is +48 more, on every seed.
- **Quiet training from scratch beats plain TD by a lot (+144)**, but it loses to plain TD followed by the quiet
  fine-tune, which also costs less. The same finding as 0015's for TD-Leaf: learn cheaply first, then fine-tune through
  the search. The best recipe is now **2 x 128, 1M games of plain TD, then a long 1-ply fine-tune with quiescence 4**,
  played at 3 plies + quiescence 4. Chained through the baselines it is ~+90 over 0016's `l-2x128-ft2` under the same
  search.

### `selfplay-v14` results (2026-10-24)

`q1-ft-1m`, the same quiet fine-tune for 1M games instead of 200k, against `q1-ft-200k`: **+16 [6, 27]** (both at
3 plies + quiescence 4, 870 game pairs), for 14,521 active s per run against 2,918. Five times the games buys a sixth of
what the first 200k did. At 2 x 128 the fine-tune has stopped paying; the gain left is likely capacity (`selfplay-v15`).

### The quiet recipe on the leaderboard (2026-10-24)

The recipe's TD network (`d428cb8f`, seed 0, 1M games of plain TD) with a 200k-game quiet fine-tune: `629f90bd` (the
3-ply entrant) and a byte-identical 4-ply twin. Against the TD-Leaf recipe's entrants (`6964337b` at 3 plies,
`a6b88c8b` at 4), every side with quiescence 4 (`checkers.sprt`, elo0 0, elo1 50):

| Head to head | Elo | Verdict |
|---|---|---|
| quiet at 3 plies vs TD-Leaf recipe at 3 plies | **+53 [13, 96]** | H1, 46 pairs |
| quiet at 4 plies vs TD-Leaf recipe at 4 plies | **+40 [9, 72]** | H1, 70 pairs |
| quiet at 3 plies vs TD-Leaf recipe at 4 plies | -53 [-104, -3] | H0, 20 pairs |

The quiet fine-tune is the stronger network at either depth, for 2,910 active s against the TD-Leaf fine-tune's
10,067. A whole ply still beats it. `checkers.versus.v3`, the whole field:

| # | Entrant | Elo (± 95%) |
|---|---|---|
| 1 | **quiet recipe, 2 x 128, 4 plies** (`a29018d6`) | **983 ± 43** |
| 2 | TD-Leaf recipe, 2 x 128, 4 plies (`a6b88c8b`) | 967 ± 43 |
| 3 | 0016's 2 x 64 TD-Leaf, 4 plies (`672890ba`) | 928 ± 43 |
| 4 | quiet recipe, 3 plies (`629f90bd`) | 904 ± 49 |
| 5 | TD-Leaf recipe, 3 plies (`6964337b`) | 877 ± 44 |
| 8 | the recipe's plain TD parent, 3 plies (`d428cb8f`) | 749 ± 40 |
| 10 | Material, 4 plies | 670 ± 40 |

The round robin orders them as the SPRTs did, but its intervals overlap at the top: a round robin spreads its games
over 25 entrants, and the head-to-heads are what separate neighbours.

### Next (running)

- **`selfplay-v15`**: the quiet fine-tune (200k games) on `selfplay-v7`'s 2 x 256 networks, against `q1-ft-200k`
  (2 x 128). v14 says the 2 x 128 fine-tune is nearly spent, so width is the next thing to try. Each run waits for
  its v7 parent.
- **`selfplay-v7`** (0016's width arms, plain TD) is still running; v15 reads its 2 x 256 arm.
