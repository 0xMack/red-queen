# 0021 — Quiescence search: never score a position mid-exchange

Status: **Built; pilot and `selfplay-v12` done: quiescence wins at play time, and a cheap 1-ply quiet fine-tune wins
when played that way; `selfplay-v13` is running** (2026-10-24).
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
- **Not changed:** `checkers.versus.v2`. A leaderboard protocol's meaning is fixed, so entrants searching with
  quiescence need a new protocol version (see "Next").

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

### Next (running)

- **`selfplay-v13`**: `q1-ft-200k`, v12's `q-ft1-q4` for 200k games instead of 40k (against `q-ft1-q4`), and `q1-1m`, the
  whole 1M-game TD phase trained at 1 ply + quiescence 4 from scratch (against `selfplay-v6/l-2x128-1m`, plain TD).
  Both are read at 3 plies + quiescence 4.
- **`selfplay-v7`** (0016's width arms) is queued behind it.
- **A leaderboard protocol that searches with quiescence** (`checkers.versus.v3`): v2 can't rank what this doc found,
  since its entrants search plainly. Under v3 every searching entrant (champions at their depth, `material-3`/`-4`)
  would play with quiescence 4, and a v13 network would enter as it is meant to play.
