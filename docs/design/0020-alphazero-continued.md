# 0020 — AlphaZero, continued: train it, make the search a player, then scale

Status: **Proposed; Phase 0 run** (2026-10-18). The AlphaZero fine-tune lost, because of its outcome-only value
target; see "Phase 0 results". Nothing from Phases 1-4 is built.
Relates to: [0017](0017-alphazero-checkers.md) (the learner this continues: built, never trained),
[0016](0016-selfplay-scaling.md) (the current best recipe, 2 x 128 TD then TD-Leaf, and its unrun follow-ups),
[0013](0013-measuring-two-player-strength.md) (h2h SPRT, Elo), [0015](0015-td-leaf-from-the-start.md) (fine-tuning beat
learning from scratch), [0007](0007-representations-leaderboards-and-tradeoffs.md) (interfaces, cost tradeoffs),
[0009](0009-client-side-inference-at-scale.md) (everything plays in the browser), [0019](0019-static-hosting.md) (the
site snapshot).

## Context

0017 built AlphaZero-style self-play for Checkers: one network with a value unit and 128 move logits, PUCT search with
root Dirichlet noise, and visit counts and outcomes as targets from a replay buffer. It also prepared one experiment,
`selfplay-v8`. As of this doc there are no AlphaZero runs at all. Every baseline v8 names is trained
(`selfplay-v2/pool-2x64-1m`, `selfplay-v5/s-leaf2`, `selfplay-v6/l-1m-ft2`). The other two jobs 0016 prepared,
`selfplay-v7` (width) and `experiments/pipelines/checkers-recipe.yaml` (the 2 x 128 leaderboard candidate), haven't
run either.

**What's wrong with the setup as prepared.** v8 ranks AlphaZero's *value head*, searched by alpha-beta, against TD
networks searched the same way. That measures AlphaZero's by-product. What AlphaZero actually produces is a policy and
the MCTS player built on it, and those can win even when the value head is a worse alpha-beta evaluator.

The run's only view of the player can't settle that either. `mcts_points` is the search's score against material-2, and
the 2 x 128 TD network already scores 0.93 on the fixed-opponent monitor. Against an opponent that weak, a flat
`mcts_points` most likely means the score has hit its ceiling, not that AlphaZero failed. Telling the two apart needs
the MCTS player against strong opponents, so this doc treats the player as part of the experiment, not a reward for
passing it.

## Two budgets

"Equal compute" means two different things here, and every comparison below names which it holds equal.

- **Training compute:** the run's measured `cost.active_s` (`arena.costs`).
- **Thinking time per move:** what a player spends choosing a move. Alpha-beta's is set by its depth and MCTS's by its
  simulation count. A matched comparison fixes a time per move, measured natively by `arena.costs`. The MCTS player
  gets the simulation count that fits that time, not a count picked by hand.

## Plan

Phases 0 and 1 run at the same time: Phase 0 is compute and Phase 1 is code. The first decision point (below) comes
after both.

### Phase 0 — train what's built, with no code changes

1. **Smoke run.** Push a tiny-budget `alphazero` spec through the scheduler with a scratch `REDQUEEN_DATA_DIR`. Check
   only what has a grounded expectation:
   - value loss and policy loss fall;
   - `mcts_points` rises above an untrained network's;
   - games end normally (`mean_plies`, and the win/draw split in `TrainStats`).
2. **Submit `selfplay-v8` without `z-az-200k`.** Move that arm to a later experiment (see Phase 2). Learning from
   scratch is 5 runs of ~2.5 h each: the most expensive arm and the least informative one now. 0015 already found
   fine-tuning ahead of learning from scratch, and any change Phase 2 keeps would make those runs out of date. What's
   left is `z-az-ft` (~30 min per run) and `z-az-ft-16k` (~12 min per run), plus the full-ballot h2h report.
3. **Read the draw rate before changing the target.** 0017 named mixing the search's root value into the target,
   `z' = (1 - m) z + m q`, as the fix if the outcome-only target is too noisy. The case for it is that strong play draws
   often, so `z` is mostly 0 and carries little signal. Nobody has measured that. v8's per-iteration `draw_rate` extra
   measures it.
   - If draws dominate the fine-tune's games, the mixed target becomes `selfplay-v9`. It needs a param with `m = 0` as
     today's update, so the golden runs and the `alphazero` digest hold. `az_gradients` takes the mixed target as an
     input, covered by its finite-difference check. The RL WASM is rebuilt, since `alphazero.rs` is part of it.
   - If draws don't dominate, the mixed target waits until a result points at it.

### Phase 0 results (2026-10-18)

**`selfplay-v8`** (5 seeds; each arm's five final networks against the baseline's, seed for seed, over the whole ballot
from both seats, both searching 3 plies):

| Arm | Against | Elo | Seeds positive |
|---|---|---|---|
| `z-az-ft` (40k games, ~2.4x compute) | `selfplay-v6/l-1m-ft2` | **-323 [-342, -304]** | 0 of 5 |
| `z-az-ft-16k` (16k games, ~1x compute) | `selfplay-v6/l-1m-ft2` | **-328 [-348, -310]** | 0 of 5 |

The fine-tunes also lost to the TD network they started from. Seeding is exact (`set_value_network` reproduces the
parent to 0.0), so the strength is lost in training. It is lost early: the 16k arm is no better than the 40k one.
Draws were 59-73% of self-play games throughout. The policy hardly learned: over 40k games its loss went only from
~1.63 to ~1.56, and its entropy at the start position stayed ~1.8 nats, close to uniform over ~8 moves. The fixed-opponent
monitor (held-out score, `mcts_points`) stayed at 0.6-0.9 and noticed none of this. That confirms it saturates.

**`selfplay-v8-diag`** (3 seeds, 4k games each, played against the parent `selfplay-v2/pool-2x64-1m`, the whole
ballot):

| Arm | What it tests | Elo vs the parent | Per seed |
|---|---|---|---|
| `d-repeat` | v8's settings | **-324 [-348, -302]** | -365 -252 -375 |
| `d-lr1e-4` | a tenth of the learning rate | **-88 [-103, -73]** | -34 -104 -130 |
| `d-vw10` | the value loss weighted 10x | **-330 [-356, -306]** | -350 -278 -373 |

- **The damage is reproducible and immediate.** 4k games already cost all of v8's loss.
- **The policy gradients aren't the cause.** With the value loss weighted 10x, it is just as bad.
- **A smaller step only slows it.** At a tenth of the learning rate it still loses on every seed, so the network is
  moved steadily *toward* something worse than what TD taught it. That points at what the value is trained on: game
  outcomes, two-thirds of them draws, from exploratory self-play (root noise, sampled early moves, random openings).

**Next: `selfplay-v9`**, at v8's learning rate, against the parent, to separate the target from the data:

- the mixed target `z' = (1 - m) z + m q` at `m = 0.5` and `m = 1` (the search's value alone). This needs the
  `m = 0`-default param from step 3 above;
- exploration off (`dirichlet_weight: 0`, `temperature_plies: 0`), which existing params cover. If outcomes from noisy
  play are the problem, this arm recovers.

Phase 1 waits for v9. A player built on this policy would be measuring a policy that barely learned.

**`selfplay-v9`** (3 seeds, 4k games each, played against the parent, the whole ballot; `d-repeat` is v8-diag's arm with
the same settings and `m = 0`):

| Arm | Change | Elo vs the parent | Per seed |
|---|---|---|---|
| `d-repeat` | (none: `m = 0`) | -324 [-348, -302] | -365 -252 -375 |
| `v-mix05` | `value_target_mix: 0.5` | **-98 [-114, -82]** | -138 -76 -81 |
| `v-mix1` | `value_target_mix: 1` (the search's value alone) | **-29 [-43, -14]** | -23 -26 -37 |
| `v-greedy` | no root noise or sampled moves (4 random opening plies instead) | **-430 [-465, -400]** | -372 -574 -397 |

- **The outcome target is the cause.** The loss shrinks steadily as the searched value replaces the outcome: -324,
  -98, -29.
- **Exploring wasn't the problem.** Without it the loss is worse. With less varied games, draws climbed to 83-91%, and
  the policy's loss fell fast (1.5 to 0.95 in 4k games), which is overfitting to repeated games.
- **No setting beat the parent.** The best, `m = 1`, still loses 29 Elo on every seed, and its value loss (~0.006)
  shows the network has learned to agree with a 50-simulation search of itself. That kind of target keeps what TD
  taught; it doesn't add to it. Throughout, the policy learned almost nothing (loss ~1.69 to ~1.65).

**Where this leaves the plan.** The outcome-only value target is ruled out at this scale. The open question is whether
a searched-value target *improves* on TD with more games, as TD-Leaf's fine-tune did over 40k. The next run is v8's
question asked again: `value_target_mix: 1` (and 0.75, keeping a little of the outcome) for 40k games against
`selfplay-v6/l-1m-ft2`. If it can't beat TD-Leaf at equal training compute, AlphaZero-style training doesn't pay at this
network size. The MCTS player (Phase 1) would then be justified only as a way to search TD networks, not as
AlphaZero's payoff.

### Phase 1 — the MCTS player, where strategies live

1. **The search goes into `libs/games/rust/core`, written once against a small policy-value trait** (a position and its
   legal moves in, a value and one prior per move out). Moving the network isn't needed.
   - **Training** keeps calling it with `redqueen_rl::nn::Mlp`. Root Dirichlet noise stays on the RL side, drawn from
     the RL crate's RNG and applied to the priors before the search sees them. The `alphazero` determinism digest holds
     by construction: the same network, the same random numbers, the same arithmetic.
   - **The leaderboard player** calls the same search with a `nets.rs` network. `nets.rs` gains the linear last layer
     it lacks (0017, "Not built"). Its outputs are checked against the training network's within floating-point
     tolerance, and its moves agree on fixture positions. Bit-for-bit equality isn't required, because nothing pins the
     player's arithmetic to a digest.

   One search implementation, so the search that trains can't drift from the search that plays.
2. **A new interface,** `checkers/board32.v1+policy128.v1`, registered in `games.interfaces`. The evaluator interface
   `board32.v1+evaluate1ply.v1` stays as it is: `snapshot()`'s value-head export still ranks under alpha-beta.
3. **An `mcts` strategy** with params `simulations` and `c_puct`. It has no root noise and always plays the most-visited
   move, so it is deterministic. `Strategy::scores` returns visit shares, so `VersusStage`'s candidate-move and
   evaluation-trace diagnostics work with no frontend changes. `simulations: 0` is the policy alone: the masked argmax.
4. **Loading the whole network.** Today only the run artifact `<champion_ref>-net` holds both heads.
   - `modelpack.champions.load_champion` learns that artifact kind.
   - The browser gets it the way it gets evaluators (`/runs/{id}/artifacts/{ref}/brain`, numbers through
     `evolve.networks.compiled`'s path) or as a package.
   - The `site` publish target (0019) includes whatever the page reads.
5. **Leaderboard entrants** at several simulation counts (e.g. 0, 16, 64, 256) on `checkers.versus.v2`, with
   `arena.costs` recording time per move. The result is a curve of Elo against milliseconds per move for one network.
6. **A control:** MCTS over a TD champion, with uniform priors and the TD network as the value. Expect it to be weak.
   With no learned policy, PUCT spreads its simulations evenly, and in a game this tactical (mandatory captures,
   multi-jump chains) it will probably lose to alpha-beta at equal time. That makes it a control, not an entrant
   expected to place well. It separates what the *policy* buys from what *MCTS* buys.
7. **Tests and builds.**
   - The relocated search reproduces the old one's visit counts on fixture positions.
   - The `alphazero` digest in `determinism.json` doesn't change.
   - Both WASM builds are redone (`uv run redqueen dev build-wasm`), since the games core changes.

### Decision point: after Phases 0 and 1

The question is whether AlphaZero training produced a stronger *player*. Phase 1's player measures that against strong
opponents:

- v8's fine-tuned networks as MCTS players, at a matched time per move;
- against the TD-Leaf fine-tune of the same parent (`selfplay-v6/l-1m-ft2`) searched by alpha-beta;
- full-ballot h2h, 5 seeds.

v8's own report, value head against value head under alpha-beta, is read beside it.

- **If both lose** at equal training compute, the next step is diagnosis, not Phases 2 and 3: the target (the draw
  rate decides whether that's v9), the simulation count, the network size.
- **If either wins,** go on to Phase 2.

### Phase 2 — make search cheap enough to scale

At 50 simulations a game costs ~17x plain TD (0017, "Cost"). Each change below is an off-by-default param and an
experiment arm, so the golden runs and digests hold. Each is measured, not assumed:

- **Playout-cap randomisation** (KataGo; Wu, 2019). A full search on a fraction of moves (e.g. 25%) and a cheap one on
  the rest; only fully searched positions become policy targets.
- **Gumbel AlphaZero** (Danihelka et al., ICLR 2022). Sequential halving over Gumbel-sampled root moves improves the
  policy reliably with very few simulations (8-16), which suits a CPU budget. It is a bigger change than the others:
  it replaces the root's Dirichlet noise, and the policy target becomes the completed-Q policy instead of visit counts.
- **Tree reuse between moves.** Keep the chosen child's subtree.
- **Batched leaf evaluation with virtual loss.** This is probably worth the least on a small CPU MLP, where a batch of
  forward passes isn't much faster than one at a time, so it's measured last.

**Gate: a non-inferiority SPRT** (`arena.versus_stats.Sprt`). A is the changed learner trained on its lower compute; B
is the unmodified learner. With `elo0 = -20` and `elo1 = 0`, accepting H1 means "no more than ~20 Elo worse, for less
compute", and the change is kept. Accepting H0 means the saving cost strength, and it's dropped. An undecided test after
the full ballot counts as dropped.

The from-scratch arm deferred from v8 (`z-az-200k`, against `selfplay-v5/s-leaf2`) runs here, once, with Phase 2's
kept changes. If none were kept, it's dropped.

### Phase 3 — scale to the frontier

1. **Run `selfplay-v7` and `checkers-recipe` first** (see "Compute schedule"). AlphaZero should be compared with the
   best recipe (2 x 128), not v8's 2 x 64 parent, and that recipe's TD phase is the natural seed
   (`set_value_network`).
2. **A pipeline** (`experiments/pipelines/`): the recipe's 1M games of TD on 2 x 128, then an AlphaZero fine-tune with
   Phase 2's kept changes (and v9's target, if it ran and won). It sits beside the recipe's TD-Leaf fine-tune of the
   same parent, at equal training compute.
3. **Two comparisons,** each a full-ballot h2h SPRT:
   - the value head, under alpha-beta, against the TD-Leaf fine-tune, at equal training compute;
   - the MCTS player against the current leaderboard #1, at equal training compute *and* equal time per move.
4. If either wins, it becomes the leaderboard's new top entrant.

### Phase 4 — the site

- **The Checkers page.** MCTS entrants on `/games/checkers` run in the games core's WASM. Visit shares appear as the
  candidate-move diagnostics.
- **The Learn chapter.** `/learn/alphazero` is built around a trained network: a simulations slider moves it from
  instinct (0) to deliberation (hundreds), and the reader plays it on `VersusStage`. It does not train AlphaZero from
  scratch live. That takes on the order of 10^5 games, far past a reader's patience even for a tiny network in WASM.
  How training went is shown with recorded curves from the real runs (`redqueen export`). A short live *fine-tune* of
  the trained network in a worker, the `SelfPlayTrainer` pattern, is optional. It goes in only if a measured browser run
  shows a visible change within a minute or two. Phase 1's Elo-against-time curve is the cited result. Prerequisites:
  self-play and measuring strength.
- **Explainers.** Registry entries for the algorithm, the `mcts` entrant and the new interface (`data/explainers/`).
  Without them their ⓘ silently doesn't render.

## Compute schedule

Everything trains on one machine (Ryzen 9 7950X, 16 cores / 32 threads). One self-play run uses one process, and 0016
measured ten runs sharing the machine. Only v8's times and the 2 x 128 TD run are measured. The others are scaled from
0016's numbers by weight count and games, so treat them as rough.

| Order | Job | Runs | Per run | Run-hours |
|---|---|---|---|---|
| 1 | `selfplay-v8` without `z-az-200k` | 10 | ~30 min (`z-az-ft`), ~12 min (`-16k`) | ~3.5 |
| 2 | `checkers-recipe` (2 x 128: 1M games TD, then 50k games TD-Leaf at 3 plies, then a 4-ply fine-tune) | 1 pipeline | ~2.6 h TD (measured 9,300 s), ~1-2 h fine-tunes | ~4-5 |
| 3 | `selfplay-v7`: `w-2x256-1m` | 5 | ~8-10 h (~4x 2 x 128's weights) | ~40-50 |
| 3 | `selfplay-v7`: `w-2x256-ft2` | 5 | ~2-3 h | ~10-15 |
| 3 | `selfplay-v7`: `w-2x64-2700k` (compute-matched control) | 5 | ~2.6 h (2.7x 2 x 64's 3,443 s) | ~13 |
| 4 | Phase 2 arms, then the deferred from-scratch arm | per arm 5 | depends on the kept changes | sized when Phase 2 is designed |
| 5 | Phase 3 pipeline: the 2 x 128 TD parent is reused from (2); only the AlphaZero fine-tune is new | 1-5 | ~1-3 h | ~5-15 |

With ~10 workers (`python -m scheduler --workers 10`), (1) finishes within about an hour of wall-clock. (2) and (3)
overlap. (3) is bounded by its longest 2 x 256 run, so most of a day, and it can run while Phase 1 is being written.
Nothing in Phase 3 waits on more than (2).

## Not in scope

- **MuZero and other learned-model planners.** Checkers' rules are known exactly, and the WASM core already simulates
  them.
- **Tree-parallel search across threads.** Throughput comes from the scheduler running many self-play runs at once,
  not from making one run faster.
- **Exact move encoding** (adding the landing square). 0017 measured fewer than 1 in 1,000 legal moves sharing an
  index.
