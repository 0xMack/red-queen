# 0011 — Multi-armed bandits: a game for exploration, exploitation, and the step up to Q-tables

Status: **Proposed** (2026-09-27). Nothing implemented yet except the arm-count sweep below.
Relates to: [0006](0006-multiagent-games-and-strategy-framework.md) (a game in `libs/games`, one page layout),
[0007](0007-representations-leaderboards-and-tradeoffs.md) (interfaces, held-out protocols, leaderboards),
[0009](0009-client-side-inference-at-scale.md) (Rust core → PyO3 + WASM), [0010](0010-reinforcement-learning.md)
(the RL ladder this sits underneath, the `libs/rl` core it reuses).

## Context

The reinforcement-learning chapters are steep in one specific place: the jump from "an agent keeps a table of
values" to "Snake's 11 features make 2,048 rows". A reader has to take on the update rule, exploration, discounting,
*and* how an observation becomes a row, all at once, on a game where one move's value depends on everything after it.

A **multi-armed bandit** is the same machinery with the hard parts removed and added back one at a time. Pull one of
K slot machines, get a payout, repeat for a fixed budget of pulls. There is one situation, so the Q-table is **one
row**; nothing you do changes what comes next, so there is no discount and no Bellman backup. What's left is the
part of RL that is hardest to *feel* from Snake: every pull spent learning about an arm is a pull not spent on the
best arm you know. Everyone who plays it runs into exploration vs. exploitation personally, before any algorithm
does.

Decided with the user (2026-09-27):

| Question | Decision |
|---|---|
| The game | K animated slot machines with hidden payout distributions fixed at the start of a game, a fixed budget of pulls, score = total payout |
| Shape | **Three levels** that climb to Q-learning: plain bandit (1 row), contextual bandit (a row per observed context), sequential (a pull changes what comes next). **Levels 1 and 2 first**; Level 3 later, if Q-learning on Snake doesn't already cover it |
| Scenarios | Several payout scenarios, each built to expose a particular method's pitfall |
| Rewards | Bernoulli (win / lose, reads as a slot machine) *and* continuous payouts, chosen per scenario |
| Human play | A real mode: race the algorithms on the same seed |
| Default size | **5 arms, 100 pulls** -- see "The default table" |

## The three levels

| Level | The player sees | The Q-table | The lesson |
|---|---|---|---|
| **1. Bandit** | K machines, N pulls | **1 row × K** | Estimating values from noisy samples; exploring vs. exploiting. `Q(a) ← Q(a) + α (r − Q(a))` *is* Q-learning, with no next state (γ = 0) |
| **2. Contextual bandit** | Each round a lamp shows red or blue, and the best machine depends on the colour | **a row per context × K** | *What you observe picks the row.* One lamp → 2 rows; a lamp and a bell → 4 rows; Snake's 11 yes/no features → 2,048. A cue the agent can't observe makes two situations share one row (aliasing) |
| **3. Sequential** | Pulling a machine changes which machines light up next | the same table, now with a future | Why a pull can be worth making for where it leads: γ and the Bellman update |

Level 2 is the missing rung in the current chapters. The reader adds a feature and watches the table double; hides
one and watches two situations collapse into a row that can't be right for both.

## The default table: 5 arms, 100 pulls

The default has to be small enough to take in at a glance and on a phone, and large enough that the pitfalls
actually happen. Measured with `jobs/bandit_arm_sweep.py` (a pure-Python sketch, 1,500 games per cell; Bernoulli
arms with means drawn uniformly each game). The strategy columns show regret relative to pulling at random (1.0 = no
better than random, 0 = perfect):

| K | N | random | greedy | ε-greedy 0.1 | optimistic | UCB1 | Thompson | greedy ends on the wrong arm | Thompson ends on the right arm | arms greedy never tries |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 100 | 1.00 | 0.52 | 0.30 | 0.24 | 0.41 | 0.20 | 47% | 86% | 1.4 |
| 4 | 100 | 1.00 | 0.56 | 0.34 | 0.20 | 0.47 | 0.22 | 59% | 83% | 2.3 |
| **5** | **100** | 1.00 | **0.58** | **0.36** | **0.21** | **0.53** | **0.25** | **65%** | **80%** | **3.3** |
| 6 | 100 | 1.00 | 0.57 | 0.39 | 0.21 | 0.56 | 0.27 | 68% | 79% | 4.2 |
| 8 | 100 | 1.00 | 0.61 | 0.41 | 0.22 | 0.63 | 0.31 | 77% | 71% | 6.2 |
| 10 | 100 | 1.00 | 0.65 | 0.42 | 0.24 | 0.67 | 0.34 | 83% | 67% | 8.1 |

Why 5 arms and 100 pulls:

- **The pitfalls show up in most games, not rare ones.** With 5 arms, pure greedy ends committed to the wrong
  machine in 65% of games and never tries 3.3 of the 5 (with 3 arms it's 47% and 1.4 — too often fine).
- **The strategies separate clearly.** Greedy 0.58 → ε-greedy 0.36 → Thompson 0.25 is a spread a reader sees in a
  single race, and there is still headroom (Thompson finds the best arm 80% of the time, not always).
- **A person can keep up.** Five is about as many machines as a player can hold impressions of without a notebook,
  so the game is hard for the right reason. Beyond ~6 the human game becomes bookkeeping, and every strategy slides
  toward "not enough pulls to try everything" -- a separate lesson, which gets its own scenario rather than being
  the default.
- **It fits.** Five machines sit in one row at desktop widths and still fit a 390px phone (~66px each, compact).
  100 pulls is 20 per arm on average: enough samples to start telling arms apart, and a game lasts a minute or two.
- **60 pulls is too short, 150 flattens it.** At N = 60 everything is noisy; at 150 ε-greedy and the rest converge
  toward each other, and a race stops being close.

Two results in the sweep are lessons in themselves, and belong in the chapter:

- **UCB1 barely beats greedy at this budget** (0.53 vs 0.58). Its bonus, √(2 ln t / n), is built for long horizons;
  in 100 pulls it spends too much re-checking bad arms. "Theoretically optimal" and "good at this game" are different
  claims.
- **Optimistic initial values look like the best strategy here** (0.21) -- because payouts are 0/1 and "assume every
  untried arm pays 1" is a very good guess when means are uniform. The *Deceptive optimism* scenario is there to show
  when that guess costs you.

Scenarios override K and N where their lesson needs it (only *Too many arms* changes K).

## Scenarios

Each game draws its arm values from the seed *within* a scenario's structure (not uniformly), so every seed shows the
scenario's pitfall. The "who fails" column is a set of hypotheses to verify in Phase 1, not established results.

| Scenario | K × N | Payouts | Who should fail, and why |
|---|---|---|---|
| **Classic** (default) | 5 × 100 | Bernoulli, means spread over 0.2–0.8 | Greedy (commits early); the baseline race |
| **Close call** | 5 × 100 | Bernoulli, best 0.55, others 0.45–0.52 | Everyone -- small gaps need many samples. Thompson degrades most gracefully |
| **Lucky start** | 5 × 100 | Gaussian, equal-ish means, high noise | Greedy locks onto whichever arm paid well first |
| **Jackpot** | 5 × 100 | One arm pays 50 with p = 0.02 (mean 1.0); one pays 0.8 every time; the rest less | Anything trusting a small sample writes the jackpot arm off; the sample average hides a heavy tail |
| **Drifting** | 5 × 200 | Means swap partway through (hidden switch) or random-walk | Sample averages (α = 1/n) and UCB stay loyal to the old best arm. A **constant α** tracks it -- *the* reason Q-learning uses a learning rate instead of an average |
| **Too many arms** | 16 × 100 | Bernoulli | You can't try everything once and still exploit. UCB and optimistic starts burn the budget on trial pulls |
| **Deceptive optimism** | 5 × 100 | All arms poor (means 0.05–0.25) | Optimistic initial values spend most of the budget being disappointed by every arm in turn |
| **Two lamps** (Level 2) | 5 × 100 | A red/blue lamp each round; the best arm differs by colour | A plain bandit agent (1 row) averages the two situations and is wrong in both; the contextual agent (2 rows) isn't |

## Strategies, and where each leads

| Strategy | What the reader sees | Connects to |
|---|---|---|
| Random, greedy | the two baselines | -- |
| ε-greedy (fixed, decaying) | the explore/exploit dial | the ε in the Q-learning lab |
| Optimistic initial values | exploration from optimism, not randomness | the Q-learning lab's "optimistic start" |
| UCB1 | exploration as an uncertainty bonus, drawn as an error bar per machine | -- |
| Thompson sampling | each machine's belief as a curve; the pull is a draw from the curves | usually the winner, and the most visual |
| Gradient bandit (softmax preferences + a baseline) | preferences, not values | **REINFORCE with a baseline** in miniature -- sets up the policy-gradients chapter |
| Tabular Q-learning (`libs/rl`'s, unchanged) | the Snake lab's agent on a 1-row table | literally the same `Trainer` as the Q-learning chapter |
| Evolved strategy (stretch) | a GA tuning ε schedule / α / initial value across many games | the one evolution-vs-RL comparison that fits on a single page |

## The game, as a player sees it

- **Machines:** an animated slot machine per arm -- a short reel spin, a payout flash, and a tally of the player's own
  wins and losses under each. The animation shortens with speed and drops to a plain readout when algorithms race
  fast, or races become unwatchable.
- **What the player knows:** pull counts and tallies are shown; running averages are *not*, by default -- keeping
  estimates in your head is the experience. A "show my estimates" toggle turns them on (and turns the player into a
  greedy agent with a notebook, which is itself a point).
- **The reveal:** at the end, each machine's true value is shown next to what the player -- and each algorithm --
  believed, with where the pulls went. The reveal is the lesson.
- **Race mode:** the player and three or four strategies on the *same seed*, side by side: a pull-count strip per arm
  per player, and a regret curve over pulls.
- **Score:** total payout, with **regret** (payout lost against always pulling the best arm) shown alongside. Regret
  is comparable across seeds and scenarios, and it's what the leaderboard ranks by.

## Architecture (reusing what exists)

- **Game core:** `libs/games/rust/core/src/bandit.rs` -- arms, scenarios, the pull budget, PCG32-seeded like every
  other game, so a seed is the same game in jobs, on the leaderboard and in the browser. Python face `games.bandit`,
  WASM `BanditGame`; a pure-Python oracle in `tests/reference_bandit.py`, parity-tested like the other games.
- **Interfaces (doc 0007):** `bandit/none.v1+arm.v1` (Level 1: an empty observation, action = arm index) and
  `bandit/lamp.v1+arm.v1` (Level 2: the lamp colour as a one-hot) -- each with a `Discretizer` so tabular agents run
  unchanged: one row for `none.v1`, two for `lamp.v1`.
- **Strategies:** in `libs/rl` (`rust/core/src/bandit.rs`): ε-greedy, optimistic, UCB1, Thompson (Beta for Bernoulli,
  Gaussian for continuous), gradient bandit -- checked against a plain-Python oracle (`reference_bandit_agents.py`),
  joining the determinism fixture (native, Node, `/dev/rl`). Tabular Q-learning needs nothing new.
- **Leaderboard:** `jobs/evaluate_bandit.py`, protocol `bandit.regret.v1`: mean regret on held-out seeds **per
  scenario**, since scenarios aren't comparable to each other. Baselines (random, greedy) always included.
- **Frontend:** a `GameModule` (`app/games/bandit.ts`) on the shared game page -- `BanditStage` (machines, race mode),
  a regret `ScoreSpec` (lower is better), a scenario picker. The Learn chapter's demos are built from the `lab/` kit;
  a strategy exposes its per-arm estimates/uncertainty the way `Strategy::scores` exposes a Checkers bot's.
- **Learn:** a new chapter, "Multi-armed bandits: exploration vs. exploitation", opening Part III (Reinforcement
  learning) before Q-learning; the Q-learning chapter then refers back ("you've seen this update, with one row").

## Phases

1. **Core and scenarios** -- `bandit.rs` in the games core with the scenarios above, the Python oracle and parity
   tests, the WASM build. Done when every scenario's structure holds across seeds (e.g. the Jackpot arm really has
   the highest mean) and native = WASM = oracle.
2. **Strategies and the leaderboard** -- the agents in `libs/rl`, their oracle, determinism digests; the evaluation
   job; each scenario's hypothesis above checked and the table's "who fails" column replaced with measurements.
3. **The game page** -- slot machines, human play, race mode, the reveal; the scenario picker; mobile layout.
4. **The Learn chapter** -- Level 1 and 2 labs (a strategy picker, the 1-row → 2-row table, belief curves), the
   scenario results as `ArmResults` figures; the Q-learning chapter updated to point back.
5. **Stretch** -- Level 3 (sequential), and the evolved strategy.

## Open questions

- **Level 3's shape.** A few "rooms" of machines where a pull decides the next room is the smallest MDP that needs
  γ -- but it may duplicate what Q-learning on Snake teaches. Decide after the chapter exists.
- **Human leaderboard.** Humans race algorithms on the same seed; whether human scores go on the real leaderboard
  waits on doc 0007 step 4, like Snake's.
- **Continuous-payout visuals.** A slot machine reads naturally as win/lose; Gaussian payouts need a payout readout
  that doesn't look like a bug. Prototype both in Phase 3.
