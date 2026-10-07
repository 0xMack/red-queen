# 0017 — AlphaZero-style self-play for Checkers: search results as training targets

Status: **Built, not yet trained** (2026-10-06). The learner, its tests and the experiment arms exist; no run has been
trained, so there are no results yet. See "Experiment (prepared)".
Relates to: [0010](0010-reinforcement-learning.md) Phase 4 (TD(λ) self-play) and 4c (TD-Leaf),
[0013](0013-measuring-two-player-strength.md) (head-to-head measurement), [0015](0015-td-leaf-from-the-start.md) (which named
this the larger build), [0016](0016-selfplay-scaling.md) (the current recipe).

## Context

Every Checkers learner so far learns a **value** only. TD(λ) trains the positions that greedy play passes through.
TD-Leaf trains the leaves its alpha-beta search relies on: the search picks *which* position learns, but the target is
still a TD target built from the network's own estimates. 0015 ended on the remaining lever, which is what the search
contributes. AlphaZero (Silver et al., 2017) makes the search the *teacher*:

- One network gives a **value** and a **policy** (a probability per legal move).
- Every self-play move is a **Monte-Carlo tree search** (PUCT) guided by both heads.
- The search's **visit counts** are the policy's target, and the **game's result** is the value's.

A stronger network makes a stronger search, which makes better targets. No other algorithm here has that loop.

## Decisions

1. **Where it lives: `libs/rl/rust/envs/src/alphazero.rs`**, beside `selfplay.rs`. It needs both the RL core (the MLP,
   Adam, the RNG streams) and the games crate (Checkers rules). The games crate can't depend on the RL core, so the
   MCTS can't go into `checkers_strategies.rs` yet (see "Not built").
2. **Moves as indices: start square x first-step direction, 128 outputs.** A Checkers move is a whole turn, and a
   capture chain is one move, so there is no fixed move list to index. (start square 0..32 in `playable_squares` order,
   direction of the first step 0..4) covers every move. Two chains that start alike and branch later share an index:
   they split its probability equally, and their visits are summed into its target. In 200 random games that happened
   to fewer than 1 in 1,000 legal moves (`move_indices_are_in_range_and_almost_never_collide`). A more exact encoding
   (e.g. the landing square too) isn't worth a larger head for that.
3. **One network, a linear output layer of 1 + 128 units.** The value is `tanh(o_0)`, from the mover's side as
   everywhere else. The policy is a softmax over the *legal* moves' logits only, so illegal moves get no probability and
   no gradient. The hidden layers are tanh. Because the value unit's tanh can be moved into the export, **the trunk plus
   the value unit is exactly a `checkers/board32.v1+evaluate1ply.v1` evaluator** (an `evolve.WeightVector`, tanh on every
   layer). `snapshot()` exports that, so an AlphaZero run's champion goes straight into the versus leaderboard, the h2h
   SPRT, packaging and the page, as a TD champion does. `network_json()` is the whole two-headed network, stored once
   per run as `<champion_ref>-net`.
4. **Search (PUCT), as published, with these choices.** At each node: `Q + c_puct P sqrt(N) / (1 + n)`, Q 0 for an
   unvisited move, ties to the first move. The tree is not reused between moves. A finished game is scored exactly
   (+1 / -1 / 0). A position with one legal move isn't searched or learned from. In self-play the root's priors get
   Dirichlet noise (α 1.0, weight 0.25: α scaled to Checkers' ~8 legal moves, as AlphaZero's 0.3 was to chess's ~35).
   Moves are drawn in proportion to visits for the first `temperature_plies` (10) and are the most visited after that.
5. **Learning from a replay buffer.** Each searched position is stored with its visit distribution and, once the game
   ends, its outcome from its mover's side. After every game the learner takes `updates_per_game` (4) Adam steps on
   minibatches of 64 from the last 50k positions. The loss is `(v - z)^2` plus the cross-entropy against the visits.
   There is no L2 (the TD learners have none either) and no mixing of the search's root value into the value target. The
   gradient is a pure function (`az_gradients`), checked by finite differences.
6. **A TD network can seed it.** `set_value_network` copies a TD champion's hidden layers and output unit into the
   trunk and value unit, with a zero (uniform) policy. That gives the same choice 0015 measured for TD-Leaf: learn from
   scratch, or fine-tune what cheap plain TD already learned.
7. **Comparable by construction.** Runs go through `jobs/checkers_selfplay_run.py --algorithm alphazero`
   (`config.representation = "alphazero"`) and `checkers_selfplay_experiment.py` arms (`Arm.algorithm`). So the
   head-to-head SPRT compares an AlphaZero value head with a TD network *under the same alpha-beta search*. That answers
   "is a value learned this way a better evaluator?". It does not answer "is the MCTS player stronger?", which needs
   the search in the games crate (see "Not built"). The run's extras log the search player directly: `mcts_points`,
   its score against material-2 at held-out iterations.
8. **Determinism.** An `alphazero` digest joins `determinism.json` (Gamma draws for the noise, the tree search, the
   two-headed update). The native build, Node and `/dev/rl` reproduce it.

## Cost (measured, one process, 2 x 64, an untrained network)

| | ms per game |
|---|---|
| plain TD | 2.7 |
| TD-Leaf, 2 plies (0015) | ~16 |
| AlphaZero, 25 simulations | 22 |
| AlphaZero, 50 simulations | 46 |
| AlphaZero, 100 simulations | 82 |

At 50 simulations a game costs ~17x plain TD's and ~3x TD-Leaf's at 2 plies. Games will lengthen as the network
improves, as TD-Leaf's did (0015).

## Experiment (prepared: `selfplay-v8`, 5 seeds, `h2h --full`)

| Arm | Training | ~Compute vs baseline | Baseline |
|---|---|---|---|
| `z-az-ft` | `selfplay-v2/pool-2x64-1m` (1M games of TD), then 40k games of AlphaZero | ~2.4x | `selfplay-v6/l-1m-ft2` (the same parent, 40k games of TD-Leaf 2 plies) |
| `z-az-ft-16k` | the same, 16k games | ~1x | `selfplay-v6/l-1m-ft2` |
| `z-az-200k` | AlphaZero from scratch, 200k games | ~1.75x | `selfplay-v5/s-leaf2` (TD-Leaf from scratch, 200k games) |

```bash
uv run python -u jobs/checkers_selfplay_experiment.py run --name selfplay-v8 --arms z-az-ft,z-az-ft-16k --seeds 0-4
uv run python -u jobs/checkers_selfplay_experiment.py h2h --name selfplay-v8 --full
```

The fine-tune arms are ~30 min (`z-az-ft`) and ~12 min (`z-az-ft-16k`) per run alone, so start with them. `z-az-200k` is
~2.5 hours per run.

What would count: `z-az-ft` beating `l-1m-ft2` at about equal compute (`z-az-ft-16k`) would make search targets the
new finishing step. Losing at equal games would say this value target (the outcome only) is too noisy at this scale.
0010 found λ = 1 (Monte-Carlo returns) failed for TD. If so, the obvious next arm mixes the root's searched value into
the target.

## Not built (yet)

- **The MCTS player on the leaderboard and the page.** It needs the search to live where strategies do
  (`libs/games/rust/core`). That means either moving a small MLP forward pass into the games crate, which `nets.rs`
  nearly has (it lacks only a linear last layer), or a strategy crate depending on both. Do it once `selfplay-v8` shows
  the value head (or `mcts_points`) is worth it.
- **A Learn demo.** The WASM build has the digest, not a trainer. A `SelfPlayLab`-style AlphaZero demo is the same
  pattern as `SelfPlayTrainer`.
- Tree reuse between moves, batched leaf evaluation, and a value target mixed with the search's value: speed and
  variance improvements to measure, not assume.
