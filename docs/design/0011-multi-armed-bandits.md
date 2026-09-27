# 0011 — Multi-armed bandits: a game for exploration, exploitation, and the step up to Q-tables

Status: **Implemented** (2026-09-27): all three levels, the evolved strategy, the leaderboard, the game page and the
Learn chapter. See "Implementation notes" for what was measured and what changed from the plan below.
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
  untried arm pays 1" is a very good guess when means are uniform. (A *Deceptive optimism* scenario was planned to
  show when that guess costs you; measured, it didn't -- see the notes. *Drifting* shows optimism's real weakness.)

Scenarios override K and N where their lesson needs it (only *Too many arms* changes K).

## Scenarios

Each game draws its arm values from the seed *within* a scenario's structure (not uniformly), so every seed shows the
scenario's pitfall. As built (the "who fails" column is now measured -- skill, 0 = random, 100 = the best arm every
pull, on 500 held-out games; the full table is in "Implementation notes"):

| Scenario | K × N | Payouts | Who fails, measured |
|---|---|---|---|
| **Classic** (default) | 5 × 100 | Bernoulli, a jittered ladder 0.15-0.75, shuffled | Greedy (39) against optimism (78); UCB1's textbook c = √2 ties greedy (39) |
| **Close call** | 5 × 100 | Bernoulli, best 0.55, others in [0.42, 0.50) | Everyone: the best is 22 |
| **Lucky start** | 5 × 100 | Gaussian, means 4-7, sd 3 | Greedy: 0.5, no better than random |
| **Jackpot** | 5 × 100 | 50 with p = 0.02 (mean 1.0); always 0.8; three Bernoulli 0.3-0.6 | Nearly everyone: pulling the steady arm alone scores 46; the best strategy 31; optimism / UCB / Thompson ~5, their exploration sized to the jackpot's spread |
| **Drifting** | 5 × 200 | Classic arms; at a pull in 50-70 the best arm *breaks* (pays what the worst does) | Staying loyal scores -13; sample-average ε-greedy 43 vs constant step (0.2) 55; optimism 49 (explores once); tuned UCB 59 |
| **Too many arms** | 16 × 100 | Bernoulli 0.05-0.80 | UCB1 (19) must try all 16 first; ε-greedy 51, optimism 62 |
| **Two lamps** (Level 2) | 5 × 100 | Red lamp: the classic ladder; blue: `0.9 - p` for each arm | Every strategy that can't see the lamp: ~0 (every arm averages 0.45). Seeing it: Thompson 51, optimism 68 |
| **Detour** (Level 3) | 5 × 100 | Red room: the ladder, but one machine pays 0 and opens the gold room; gold: every machine pays 3 w.p. 0.4-0.8, back to red | Every strategy that values only the payout: ≤ 10. Q-learning with γ 0.9: 55 |

## Strategies, and where each leads

| Strategy | What the reader sees | Connects to |
|---|---|---|
| Random, greedy | the two baselines | -- |
| ε-greedy (fixed, decaying) | the explore/exploit dial | the ε in the Q-learning lab |
| Optimistic initial values | exploration from optimism, not randomness | the Q-learning lab's "optimistic start" |
| UCB1 | exploration as an uncertainty bonus, drawn as an error bar per machine | -- |
| Thompson sampling | each machine's belief as a curve; the pull is a draw from the curves | the most visual -- but *not* the winner at 100 pulls (see the notes) |
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

## Implementation notes

### Levels 1 and 2 (2026-09-27)

What was built:

- **Game core** (`libs/games/rust/core/src/bandit.rs`, `games.bandit`, oracle `tests/reference_bandit.py`): 7 scenarios.
  Each arm draws payouts from its own PCG32 stream (`stream_seed(seed, 100 + arm)`), the lamps from another, so the n-th
  pull of an arm pays the same for every player whatever else they pulled -- the race is fair by construction, and the
  oracle matches the Rust pull for pull. Only `+ - * /` (normals are Irwin-Hall), so WASM matches bit for bit.
- **Score**: *skill* = (expected payout − random's) / (best's − random's), per game -- 0 = no better than random,
  100 = the best arm every pull. Expected (the pulled arms' means), so it scores choices, not luck. Efficiency
  (expected / best) was the first score and was dropped: on *Close call*, random already gets 87% of the best, so
  every strategy looked alike.
- **Strategies** (`libs/rl/rust/core/src/bandit.rs`): online learners over a `rows × arms` table -- random, greedy,
  ε-greedy (constant or decaying ε, sample average or constant step), optimistic (one imaginary pull at the maximum
  payout), UCB1, Thompson (Beta by order statistic -- integer shapes, no transcendental; Gaussian otherwise), gradient
  bandit, and `q_table` -- `QTableAgent` itself with γ = 0. A bandit learns *within* a game (arms are redrawn every
  game), so strategies don't use the `Trainer`: `envs::bandit::BanditRun` couples one to one game. Oracle:
  `tests/reference_bandit_agents.py`, belief for belief after every pull, on four scenarios. A `bandit` digest joins
  the determinism fixture.
- **Interfaces**: `bandit/none.v1+arm.v1` (one row) and `bandit/lamp.v1+arm.v1` (a row per colour), registered in
  `games.interfaces` so the game page's representation cards explain them.
- **Leaderboard**: `jobs/evaluate_bandit.py`, protocol `bandit.skill.v1`, 500 held-out games per scenario (seeds
  10,000-10,499), ranked on *Classic*; every scenario's result (and *Two lamps* seeing the lamp) in `metrics.bandit`.
  The whole evaluation takes about a second.
- **Frontend**: `app/games/bandit.ts` on the shared game page; `components/bandit/` (`SlotMachine`, `BanditFloor`,
  `BanditTable` -- the table a strategy keeps, `BanditTape`, `BeliefCurves`, `BanditRace`, `BanditReveal`,
  `BanditPlayer`, the stages, `BanditScenarioMatrix` via the new `GameModule.Insights` hook); the chapter
  `/learn/multi-armed-bandits` with `lab/BanditLab`. Everything runs the rl WASM module on the main thread (a pull is
  microseconds).

Skill by strategy and scenario (500 held-out games each):

| strategy | classic | close-call | lucky-start | jackpot | drifting | too-many-arms | two-lamps (blind) | two-lamps (sees lamp) |
|---|---|---|---|---|---|---|---|---|
| Optimistic start | 77.9 | 22.2 | 66.1 | 4.2 | 49.3 | 61.8 | 0.4 | 68.0 |
| UCB, tuned (c 0.5) | 70.8 | 17.6 | 64.7 | 5.4 | 59.4 | 43.2 | 0.0 | 59.1 |
| Thompson sampling | 64.4 | 13.0 | 53.5 | 4.0 | 47.3 | 39.5 | 0.3 | 50.5 |
| ε-greedy, decaying | 64.1 | 20.8 | 54.2 | 31.2 | 43.5 | 55.9 | 0.3 | 52.7 |
| ε-greedy (ε 0.1) | 57.1 | 15.0 | 43.9 | 21.9 | 42.8 | 51.0 | -0.1 | 46.1 |
| Gradient bandit (α 0.5) | 53.1 | 11.8 | 50.0 | 24.0 | 40.6 | 24.8 | 0.2 | 33.2 |
| ε-greedy, constant step 0.2 | 51.7 | 13.9 | 16.7 | 9.6 | 55.1 | 47.0 | 0.1 | 42.4 |
| UCB1 (c √2) | 39.1 | 6.5 | 52.8 | 3.8 | 36.9 | 19.0 | -0.6 | 30.0 |
| Greedy | 38.8 | 3.6 | 0.5 | -6.2 | 13.6 | 34.9 | 0.1 | 35.9 |
| Q-table (Q-learning's agent) | 31.4 | 4.9 | 12.2 | -8.0 | 41.1 | 23.7 | -0.2 | 18.5 |
| Random | -0.4 | -0.2 | -0.1 | 0.0 | -0.2 | 0.4 | -0.3 | -0.3 |

What the measurements changed:

- **Thompson sampling is not the winner at 100 pulls.** Optimism beats it on every scenario; with 0/1 payouts, "assume
  every untried arm pays 1" costs one pull per arm and nothing more.
- **"Deceptive optimism" (every arm poor) was dropped**: optimism stayed the *best* strategy there. Optimism's real
  weakness is that it explores once -- *Drifting* shows it (49 vs tuned UCB's 59).
- **Drifting was redesigned twice.** Swapping the best and worst arm late (pulls 80-120) rewarded neither tracking nor
  averaging -- a tracker had to *rediscover* a formerly bad arm, which a constant step is slow at. Now the best arm
  breaks early (50-70) and the runner-up takes over, and a constant step visibly wins (55 vs 43).
- **Two lamps mirrors blue** (`0.9 - p`) instead of reshuffling: with a reshuffle a blind player could still find an
  arm decent under both colours (15.7 vs 22.1 regret -- a weak lesson); mirrored, every arm averages the same and blind
  play is exactly random.
- **UCB1's textbook constant over-explores in a short game** (39, tied with greedy); c = 0.5 scores 71. Both are on the
  leaderboard, because the gap is the lesson.
- **The gradient bandit needs a large step here** (α 0.1, the textbook default, scores 15 on *Classic*; 0.5 scores 53):
  100 pulls is not enough for small steps.

### Level 3: the detour (2026-09-27)

The open question above ("a few rooms where a pull decides the next room") was settled by a prototype first: two rooms
are enough, and the lesson is sharper than on Snake because the whole table is on screen.

- **`Detour`**: the red room is the classic ladder except that one machine (the detour, a `Fixed` 0) lights the gold
  room; every gold machine pays 3 with probability 0.4-0.8 (means 1.2-2.4) and leads back to red. Detour-then-gold
  earns ~1.2 a pull against the best red machine's 0.75. Skill can't use "the best arm of this situation" any more (the
  best red *payout* isn't the best move), so the game computes its yardsticks by backward induction over the budget --
  the best possible play's and random play's expected totals from the red room -- and a pull is "best" when it's what
  the best possible play does then. The oracle repeats the dynamic program.
- **Strategies** gained `update_to(row, arm, reward, next_row)`: only `q_table` uses it, bootstrapping
  `r + γ max Q(next room)` -- exactly `QTableAgent`'s transition with `done` false. Every other strategy is a bandit
  strategy and ignores where a pull led. `q_table` takes `gamma` (default 0, so nothing changed for the other scenarios:
  with γ 0 the bootstrap term is 0 whatever it is).
- Evaluated seeing the room (`lamp.v1`); a new entrant, *Q-learning, looking ahead* (γ 0.9, optimistic 10, α 0.5,
  ε 0).

| Q-learning's γ (optimistic 10, α 0.5) | 0 | 0.5 | 0.8 | 0.9 | 0.99 |
|---|---|---|---|---|---|
| Skill on Detour | 5.7 | 28.1 | 33.3 | 54.8 | -1.0 |

Every other strategy scores ≤ 10.5; playing the best red machine perfectly and never taking the detour scores 21.5.
γ 0.99 collapses because the optimistic initial values echo between the rooms faster than 100 pulls wear them down --
the chapter says so rather than hiding the dial's top end.

### The evolved strategy (2026-09-27)

`jobs/bandit_evolve_run.py`: ε-greedy's four settings (ε, its decay window, the step α, the initial estimate), squashed
from an `evolve.WeightVector` of 4 numbers, evolved by `evolve()` (32 genomes, 40 generations, tournament k 3, Gaussian
mutation σ 0.4) on 100 fresh training games per scenario per generation (seeds 1-9,999); the champion's skill on a fixed
monitor set (20,000-20,199) every 5 generations is the run's held-out score. Recorded like any evolution run
(`representation: evolved_bandit`); `evaluate_bandit.py` enters every completed one's final champion (`run:<id>`). The
run page shows its settings and plays it (`BanditRunChampion`).

| | classic | close-call | lucky-start | jackpot | drifting | too-many-arms | detour |
|---|---|---|---|---|---|---|---|
| evolved on classic (ε 0.001, decay 27, α 0.049, initial 0.588) | **80.4** | 17.1 | 4.8 | **42.0** | 67.3 | 66.5 | 14.6 |
| evolved on classic + close-call + drifting + too-many-arms | 78.5 | **22.8** | 6.0 | 35.5 | **71.8** | **68.3** | 13.9 |
| best hand-set strategy | 77.9 | 22.2 | 66.1 | 31.2 | 59.4 | 61.8 | 54.8 |

Evolution turned exploration off and the start optimistic -- it rediscovered optimistic initial values and tuned them.
Training fitness and the monitor score track each other (78.8 vs 78.6): with fresh games every generation there is
nothing to memorize. The evolved settings are the best in the table on every scenario they -- or their siblings -- were
evolved on, and fail wherever the payout scale differs (*Lucky start*, where 0.59 is pessimistic) or a future matters
(*Detour*): the specialization lesson, in one row.
