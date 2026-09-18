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
2. `apis/backend`: `MoveRequest`, the multi-agent session state, per-seat human/strategy sessions,
   bot-vs-bot auto-play. REST only, mirrors doc 0005 step 4.
3. `apps/frontend`: checkers board + opponent-selection UI, driven by the step-2 REST API. Resolves
   the `GridBoard.vue` question for real.
4. Pyodide client-side checkers (mirrors doc 0005 step 5) — bot-vs-bot and human-vs-bot fully
   client-side, reusing the existing `libs/games`/`libs/evolve` Pyodide-source-loading pattern.

## Explicitly out of scope for now

- N-player (>2) match running — the protocol allows for it, `play_match()` doesn't build it yet.
- Real RL (as distinct from evolution/ES) and classifier strategies — the `Strategy` shape supports
  them, but building one is separate work from building the framework.
- Chess — same framework, once checkers has exercised it for real.
