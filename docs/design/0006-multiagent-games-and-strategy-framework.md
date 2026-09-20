# 0006 — A general strategy/match framework, exercised first via checkers

## Context

Every game so far (`reach1d`, `snake`) is single-agent: one decision-maker, `evolve.simulation
.Environment`'s `reset()/step(action) -> (observation, reward, done)`, a small fixed action space.
That shape was never actually tested against anything else — it just happened to fit both games
built so far.

The ask: add a two-player board game (checkers) not primarily to have a checkers game, but to find
out whether `libs/games`, `apis/backend`, and `apps/frontend`'s "shared, generic" pieces actually
are generic, or just happen to work for Snake. And explicitly: build the "pit strategy A against
strategy B" mechanic as reusable infrastructure, not a checkers-specific feature — static heuristics,
evolved genomes (GP/neuroevolution), classifiers, and humans should all be able to play any
multi-agent game this repo adds, in any combination (bot vs. bot, human vs. bot, pass-and-play),
with more games and eventually more-than-two-player simulations expected later.

This doc pins down that framework's shape before writing checkers-specific code, the same way
0001/0002/0005 did for their respective pieces.

## What actually has to change

Checked each existing "shared" piece against what a two-player, large-state-dependent-action-space
game needs:

- **`Renderable`/`render_state()`** (`games/rendering.py`) — holds up with *no protocol change*.
  `render_state()`'s `cells: dict[(x, y): label]` already just needs a richer label vocabulary
  (`"black_man"`, `"red_king"`, ...) instead of `snake.py`'s `"head"/"body"/"food"`. The JSON-safety
  flattening (`apis/backend`'s `json_safe_render_state`, the Pyodide worker's `render_state_for_js`)
  is generic over label *values*, not label *meaning* — zero changes needed there either.
- **`evolve.simulation.Environment`** — does *not* hold up. `step(action) -> (obs, reward, done)`
  assumes one agent, one reward stream, and an action space small/fixed enough to hand a genome
  directly (Snake: always exactly 3 choices). Checkers needs whose-turn-is-it, a legal-move set that
  depends on board state, and per-player outcomes. New protocol, below — `games.checkers.Checkers`
  implements it structurally without importing `evolve`, same dependency direction as `Environment`.
- **Strategy representation** — holds up with *no new abstraction*, just a wider signature. Every
  genome-driven game already adapts a genome to an action space via an injected callable
  (`jobs/snake_neuro_run.py`'s `act(genome, observation) -> action`). Add `legal_moves` to that
  signature and the same pattern covers every backend the user asked for — see below.
- **`ActionRequest.action: float`** (`apis/backend`) — does not hold up; a move isn't a scalar. New
  request shape for multi-agent sessions, below. `ActionRequest` stays as-is for Snake — no reason
  to destabilize a working, tested contract to unify with something not proven yet.
- **`GridBoard.vue`** — genuinely unknown yet, and worth saying so rather than assuming: it currently
  bakes in Snake-specific animation (segments gliding between adjacent cells, directional eyes) that
  doesn't map onto checkers (pieces are removed on capture, kings get crowned in place — a different
  animation vocabulary entirely). Whether checkers reuses `GridBoard.vue` with a pluggable per-label
  renderer, or needs its own component sharing only the checkerboard-pattern idea, is an open
  question to resolve by building it, not to guess at here.

## The protocol: `MultiAgentEnvironment`

New module, `evolve/match.py` — sibling to `evolve/simulation.py`, same relationship
(`games.checkers` implements the shape, never imports `evolve`):

```python
Move = Any

class MultiAgentEnvironment(Protocol):
    def reset(self) -> Observation: ...
    def legal_moves(self) -> list[Move]: ...
    def current_player(self) -> int: ...
    def step(self, move: Move) -> tuple[Observation, dict[int, float], bool]:
        """Returns (observation, {player_index: reward}, done). Only players who should receive a
        signal this step need a key -- e.g. a non-terminal step in checkers has no natural
        per-move reward, so the dict can be empty; a terminal step sets the winner's reward to
        +1.0, the loser's to -1.0 (0.0 each for a draw)."""
        ...
    def winner(self) -> int | None:
        """Player index, or None if the game is ongoing or ended in a draw -- callers that need to
        tell "ongoing" from "drawn" apart use `done` (from the last `step()`) alongside this."""
        ...
```

Player-count-generic on purpose (`current_player() -> int`, rewards keyed by index) even though
only two players are built now — costs nothing to design in now, avoids a breaking rename later once
"simulations with many bots" (explicitly requested) actually gets built. `play_match()` below is
written for exactly 2 for now; generalizing it is future work, not blocked on this protocol.

## Strategy representation: no new class hierarchy

```python
Strategy = Callable[[Observation, list[Move]], Move]
```

- **Static/heuristic** — a plain function, e.g. "capture if any capture is legal, else random legal
  move." No genome involved. Needed early, per the request, to bootstrap/sanity-check the other
  backends against something non-trivial but not itself learned.
- **Evolved (GP or neuroevolution)** — `functools.partial(act, genome)`, exactly today's Snake
  pattern, just with `legal_moves` added to `act`'s signature. One real design difference worth
  flagging: Snake's network outputs one score per *fixed* action (3 outputs, argmax). Checkers'
  legal-move count varies by position, so the network instead scores *(observation, candidate move)*
  pairs and `act` picks the best-scoring legal move — `WeightVector.forward()` doesn't change, only
  how `act()` uses it.
- **Classifier** (e.g. scikit-learn) — same shape again: a function encoding each legal move as a
  feature vector, calling `model.predict`/`predict_proba`, returning the best-scored move.
- **Human** — not a Python `Strategy` at all. A human's move arrives through the API/UI directly;
  "human" means the session accepts an externally-supplied move instead of calling a `Strategy`.

## Match running and evolutionary fitness

```python
@dataclass(frozen=True, slots=True)
class MatchResult:
    winner: int | None
    moves_played: int

def play_match(env: MultiAgentEnvironment, strategies: dict[int, Strategy], max_moves: int = 200) -> MatchResult:
    """Alternates env.current_player() through `strategies`, applying each chosen move via step(),
    until done or max_moves. The "pit any two strategies against each other" mechanic the user
    asked to make first-class -- reusable across every future multi-agent game, not checkers-only."""
```

```python
class MatchFitnessEvaluator:
    """FitnessEvaluator via match outcomes against a fixed pool of reference opponents -- the
    multi-agent analogue of SimulationFitnessEvaluator's fixed benchmark environments. One fitness
    value per opponent (win=1.0/draw=0.0/loss=-1.0, or a margin-aware score), so LexicaseSelection
    works unchanged -- a genome that beats a strong opponent and loses to a weak one shouldn't look
    the same as the reverse, exactly the tension lexicase already exists to address
    (docs/design/0003). Plays each opponent from both first-move and second-move to cancel out
    first-move advantage in the fitness signal itself, not just in the rules."""

    def __init__(
        self,
        env_factory: Callable[[], MultiAgentEnvironment],
        opponents: Sequence[Strategy],
        act: Callable[[Genome, Observation, list[Move]], Move],
        max_moves: int = 200,
    ): ...
    def evaluate(self, genome: Genome) -> list[float]: ...
```

Structurally this is `SimulationFitnessEvaluator` with "N fixed environments" replaced by "N fixed
reference opponents" — same per-test-case contract, same selection strategies, no new machinery
needed upstream. Whether match outcomes alone (sparse: only the terminal step has nonzero reward)
are enough signal for evolution to bootstrap from, or need shaping (e.g. material-count difference
per move, the checkers analogue of Snake's distance-to-food shaping) is exactly the kind of thing
this project has twice now found out empirically rather than guessed at in advance
(docs/CODING_GUIDELINES.md) — expect the same here, not a guaranteed first-try success.

## `games.checkers.Checkers`

Standard American checkers rules: men move/capture diagonally forward only, kings both directions,
**mandatory captures** (a real rules detail that materially changes strategy, not optional), and
multi-jump chains (must continue capturing with the same piece if another capture is immediately
available). Win: opponent has no legal moves (no pieces, or every piece blocked). Draw: reuse
Snake's `max_steps_without_food`-style pattern — a move-count limit without a capture, rather than
implementing full threefold-repetition detection.

Observation: hand-engineered features, not a raw board dump — applying this session's own Snake
lesson from the start instead of re-learning it. Candidate features per candidate move (since `act`
scores moves, not board states alone): resulting piece counts (own/opponent, men/kings), whether the
move is itself a capture and how many pieces it captures, whether it exposes the moved piece to a
recapture, distance-to-king-row. Exact feature set is an implementation detail to settle while
building `act()`, not something to over-specify here.

`render_state()`: `{width: 8, height: 8, cells: {(x, y): "black_man" | "black_king" | "red_man" |
"red_king"}, current_player: int, winner: int | None}` — same shape family as `snake.py`'s, wider
label vocabulary only.

## `apis/backend` additions

- `MoveRequest{move_index: int}` — selects from the `legal_moves` list the session state already
  carries, rather than encoding a move as coordinates. Generalizes across future games with
  enumerable legal moves without needing a per-game action encoding in the API layer.
- A multi-agent session state (parallel to `GameSessionState`, not a retrofit of it) carrying
  `render_state`, `legal_moves`, `current_player`, `winner`, `done` — enough for a frontend to render
  the board, highlight legal destinations, and know whose move it is without recomputing rules
  client-side against the API.
- `GameSessionStore`/`game_sessions.py` needs a parallel per-seat mode: each of the two seats is
  either `"human"` (accepts a posted `MoveRequest`) or a named `Strategy` (auto-plays on its turn).
  Bot-vs-bot is then just "both seats are strategies" — the session auto-advances without waiting for
  a request, which is new; today's `GameSessionStore` only ever reacts to a client's `POST .../actions`.

## Frontend

Deliberately not designed in detail here (see `GridBoard.vue` question above) — settling it by
building `apps/frontend`'s checkers page is more honest than speccing pixels sight-unseen. Needed
regardless of how the board renders: click-a-piece → highlight its legal destinations (filtered from
the session's `legal_moves`) → click a destination → post `MoveRequest`; an opponent-choice control
(static heuristic / a specific trained run's genome / human pass-and-play / bot-vs-bot auto-play).

## Incremental plan

Same phasing principle as doc 0005 (prove the contract server-side/REST before moving to Pyodide):

1. ✅ `evolve/match.py` (`MultiAgentEnvironment`, `play_match`, `MatchFitnessEvaluator`) +
   `games/checkers.py` (rules, `render_state`, `simulate`, tests) — pure Python, no API/frontend.
   Validated both ways: (a) two static heuristics play a full match through `play_match()` (a
   greedy-capture strategy beats a random one in 56 moves); (b) a small `WeightVector` population
   evolved via `MatchFitnessEvaluator` against a randomized opponent shows a real, if noisy,
   improving trend (mean fitness ~0.03 → clearer with more opponent samples: ~0.09 → ~0.21 over 40
   generations). The framework holds up structurally with **zero changes** to `Renderable`/
   `render_state()` (just a wider label vocabulary) and **zero new class hierarchy** for strategies
   (the existing `act(genome, observation) -> action` pattern just gained a `legal_moves` parameter)
   — the two places that genuinely needed something new, `MultiAgentEnvironment` and
   `MoveRequest`-not-`ActionRequest`, were exactly the two flagged as such before writing any code.
   `libs/evolve/tests/test_match.py` (a toy Nim environment, keeping `evolve`'s tests
   games-independent) and `libs/games/tests/test_checkers.py` (17 tests total) cover this
   permanently; the evolutionary-improvement check was an ad hoc validation run, not committed as a
   test (too slow/noisy to assert on reliably) or a job (no telemetry wiring needed yet — that's
   real work for whenever a trained checkers champion needs to be served to a frontend, mirroring
   `jobs/snake_neuro_run.py`).
2. ~~`apis/backend`: `MoveRequest`, the multi-agent session state, per-seat sessions, bot-vs-bot
   auto-play.~~ **Skipped, superseded by doc 0009.** Once the game core runs in the browser as WebAssembly
   there is nothing for a server session to do: a move is `CheckersGame.step(index)`, a bot's turn is a
   function call. The `MoveRequest{move_index}` idea survives as the WASM interface (moves cross as
   indexes into `legalMoves()`, the same list and order Python sees). No REST surface was added.
3. ✅ `apps/frontend`: `/games/checkers`. Each seat is a human or a strategy, so human-vs-bot, bot-vs-bot
   and pass-and-play are one code path; an arena plays N games between any two strategies at full speed
   in the browser. **Resolves the `GridBoard.vue` question: it doesn't fit.** Checkers gets its own
   board component (`CheckersBoard`) -- pieces are captured and crowned, not glided, a human builds a
   multi-jump one landing at a time, and the board flips so the human's pieces sit at the bottom --
   sharing only the checkerboard idea and the `{width, height, cells}` shape. **Everything else is
   game-agnostic** (below), so the second two-player game only supplies rules, strategies and a board.
4. ✅ Client-side: done as part of 3 -- there was never a Pyodide checkers to port (0009 came first).
5. ✅ A run viewer, `CheckersWatch`: a checkers run's champion (newest, live, or any generation) playing an
   opponent on the same stage. It was missing at first (the run and watch pages were Snake-only:
   `useWatchSession` refused any `config.game` but `"snake"`, and the run page said "no viewer for this
   representation yet") -- an omission, not a decision.

## Results and findings (the framework, exercised)

**One real gap, found by training against it.** `Strategy = (observation, legal_moves) -> move` can't see
the environment, but lookahead opponents (`material-1/2`) and a genome that scores "the position each legal
move leads to" both need `env.simulate()`. The round robin got away with it by closing over one `env`;
`MatchFitnessEvaluator` builds a fresh env per match, so neither could plug in. Fix:
`StrategyFactory = env -> Strategy` and `MatchFitnessEvaluator(..., env_aware=True)` binding each strategy
to its match's env. Nothing else in the protocol needed to change -- `MultiAgentEnvironment`, `play_match`,
`render_state()`'s shape and the `Strategy` type all held up.

**Strategies are Rust, once.** The static players and the trained evaluator first existed twice (Python,
plus a TypeScript port for the browser), agreeing only in strength because each broke ties with its own
PRNG -- exactly what doc 0009 exists to prevent. They now live in `rust/core/src/checkers_strategies.rs`
(`Strategy` owns a PCG32; ties broken uniformly), exposed as `games._native.CheckersStrategy` and the WASM
`CheckersStrategy`; Python and TypeScript are thin faces. The original Python scoring survives as test
oracles: the Rust pick must be a top-scoring move across thousands of real positions
(`tests/test_checkers_strategies.py`). Side effect: evaluating a network in Rust instead of one Python
forward pass per candidate move made training ~4x faster, and the 1,200-game round robin takes 1 second.

**A game-agnostic two-player stack.** `VersusEngine` (one game's rules: legal moves as cell lists, `play`,
`position`, `notate`, `spawn`), `VersusStrategy`/`Bot`, `useVersusSession` (seats, click-to-move from the
engine's own legal moves, bot turns, history, arena, replay) and `VersusStage` (the page around a `#board`
slot). Checkers adds `utils/checkersEngine.ts` + `CheckersBoard`; the run viewer is the same stage with a
run's champion as a seat, which is the test that the split is real -- it needed no change to the session
beyond accepting a *changing* strategy list (one generation's champion swapped for another's while a game
is in progress). A move as "the cells it touches, in order" covers checkers' multi-jump chains and would
cover chess's from/to (promotion is the one thing it would need extending for).

**Evolving a checkers player.** `jobs/checkers_neuro_run.py`: a genome is a 32→H→1 *position evaluator*
(the observation encoding was designed for this), played one ply ahead; fitness is match outcomes against
random, 1-ply and 2-ply material players, both seats each, lexicase selection; held-out score is on games
whose opponent seeds training never used. A 100-generation run (population 60, hidden 12) reaches training
fitness +0.83 of a possible +1.0, and on 400 fresh games per opponent scores **+0.40 vs random, +0.34 vs
1-ply material, −0.85 vs 2-ply material** (win +1, loss −1). So it learned real strategy (it beats a
heuristic that itself only barely beats random) but a one-ply evaluator can't out-play a two-ply search;
that's the ceiling this setup hits, not a bug. A 300-generation, population-100, hidden-16 run finished with
a held-out score of −0.03, no better than the small run's: the champion stopped changing around generation
20-80 (its held-out score is identical from there on), so more of the same isn't the lever. What likely is:
opponents that improve (co-evolution / a hall of fame), deeper search, or richer inputs than 32 squares.
That is the next experiment, not built here.

**Measurement noise is large -- and it picked the wrong champion.** The held-out score used during training
(10 games per opponent) and even a 40-game record are too small: the same champion's score vs 1-ply material
read +0.7, +0.33, +0.05 and finally +0.34 as the sample and the tie-breaking PRNG changed. Choosing the
"best" champion by that noisy 3-opponent held-out score (+0.03 vs -0.03) picked the 100-generation run; the
round-robin leaderboard (`jobs/evaluate_versus.py`) later put the 300-generation `32→16→1` champion clearly
ahead (0.58 vs 0.46 points per game). The leaderboard, not a training-time monitor, is now the source of
truth for which champion is best -- and the page draws its players from it.

**One page for every game.** Checkers first got its own page (`CheckersPlay`) with no leaderboard, no
diagnostics and a hand-exported champion; Snake had all of that. Both now render through one `GamePage`
driven by a per-game `GameModule` (frontend README). The versus leaderboard follows doc 0007: a round robin
(`checkers.versus.v1`), entrants are the baselines plus every finished champion, score is points per game
against the field (win 1, draw ½) with a 95% interval, plus a head-to-head matrix. Glicko-2, which doc 0007
plans, is still future work; points per game has the drawback that adding an entrant shifts every score.
Diagnostics come from the strategy itself -- Rust `Strategy::scores` -- so they show what it actually
computed, not a reconstruction.

**A second attempt: search, NEAT, and a null result.** The one-ply evaluator can't out-play a two-ply search, so the
classic recipe was tried: the evolved network scores the *leaves* of an alpha-beta search (`--depth`), for a
fixed-topology network (`checkers_neuro_run.py`) and for NEAT (`checkers_neat_run.py`, the genome evaluated in
the Rust core through a compiled `GraphNet`). The fair opponent is then material search *at the same depth*
(`material-3`, `material-4`, now baselines) -- beating a shallower baseline than you search measures depth, not
learning. Result on the round-robin (16 entrants, 20 games per pair): `material-4` 0.90, `material-3` 0.79, the best
evolved champions 0.76 (statistical ties with `material-3`, and they lose to `material-4`), `material-2` 0.63. **No
evolved player beat its same-depth material baseline.** The evolved entrants that score well are runs whose
champion never changed -- the seeded material evaluator (half the population started as one), i.e. a material
search under another name -- and the run that did evolve away from its seed got worse (0.46; NEAT grew 3 hidden
nodes and scored 0.58).

Getting there took three fixes, each found by measuring rather than guessing (see docs/CODING_GUIDELINES.md):
mutating every weight of a 500-weight network wrecked every child (a **sparse mutation `rate`**); a
win/draw/loss fitness gave selection nothing to climb when most games are draws (a **material-margin score for
draws**, via `MatchFitnessEvaluator(scorer=...)`); and opponents seeded by index replayed the same games every
generation so training fitness reached +1 while held-out fell (**resample opponents per generation**). With all
three the population no longer memorizes or stalls -- but it still doesn't improve on material. Plausible next
steps, none tried: richer inputs than 32 signed squares (mobility, advancement, back-row -- what makes material
*insufficient*), self-play against a much larger and more varied hall of fame, choosing the reported champion by a
large fresh-game evaluation instead of one noisy generation (resampling makes the per-generation "best" a lucky
pick), and a longer horizon than 100 generations. What the attempt does leave: the search/graph machinery, the
fair baselines, and a leaderboard that says plainly where things stand.

**Run bookkeeping.** A job that is killed hard leaves its run "running" forever; `checkers_neuro_run.py` now
marks a run "failed" on any exception or Ctrl-C (a hard kill can't be caught).

## Explicitly out of scope for now

- N-player (>2) match running — the protocol allows for it, `play_match()` doesn't build it yet.
- Real RL (as distinct from evolution/ES) and classifier strategies — the `Strategy` shape supports
  them, but building one is separate work from building the framework.
- Chess — same framework; checkers has now exercised it (findings above), so it's unblocked.
