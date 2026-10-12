# evaluator

The evaluator workload (docs/design/0018): **one `EvalSpec` in, its result out.** A leaderboard protocol writes an
`EvaluationRecord` per entrant into the evaluation store (and prunes entrants that no longer qualify); a report returns
its result. Protocol definitions -- seed sets, caps, scoring -- live in `libs/arena` and are versioned: changing one
means a new protocol, never an edit.

```bash
uv run python -m evaluator jobs/evaluator/specs/checkers.versus.v2.yaml
uv run python -m evaluator jobs/evaluator/specs/checkers.sprt.yaml --set params.a=672890ba@4 --set params.b=material-4
```

`evaluator.evaluate(spec)` from Python. Specs: `specs/<protocol>.yaml`, one per protocol. The params models are in
`jobcore.protocols` (so the backend and CLI validate a spec without importing the evaluator); the runners are
`evaluator.protocols.*`. `jobs/tests/test_golden_eval.py` holds every runner to what its original script produced.

## Protocols

- **`snake.score.v2`** (was `evaluate.py`) — the leaderboard job (docs/design/0007). Runs every *finished* game run's final
  champion (under its recorded interface) and fixed baselines (random, greedy) on held-out seeds
  under protocol `snake.score.v1`, and writes an `EvaluationRecord` per entrant: score
  distribution with a 95% interval, the train-vs-held-out gap, per-decision inference latency
  (encode vs. decide), parameters, and training cost (measured, or estimated and labelled so for
  pre-cost-tracking runs). Re-run it after new runs finish. Changing seeds/caps/rules means bumping
  `PROTOCOL`, never editing it in place.
- **`checkers.versus.v2`** (was `evaluate_versus.py`) — the two-player leaderboard (docs/design/0007's "Versus", 0013): protocol
  `checkers.versus.v2`, a round robin over every finished checkers champion plus the fixed baselines, 12 ballot
  openings (`games.checkers_openings`) per pairing, each a **game pair** (both seats). An entrant's score
  (`quality.mean`) is an **Elo rating**: Bradley–Terry over every game, Random = 0, bootstrap 95% interval
  (`arena.versus_stats`); points per game, pentanomial pair counts and per-opponent W/D/L are in `metrics.versus`. Same
  `EvaluationRecord`s as `evaluate.py`, so the game page's leaderboard components need nothing game-specific. Ratings
  are relative to the field: re-run it whenever entrants change (it replaces the old records).
- **`bandit.skill.v1`** (was `evaluate_bandit.py`) — skill (0 random, 100 the best arm every pull) on 500 held-out
  games of every scenario, ranked on `classic`; the hand-set strategies plus every completed evolved run's champion.
- **`checkers.sprt`** (was `checkers_sprt.py`) — is player A stronger than B? Game pairs over the ballot (fixed shuffled order) until the SPRT
  decides; a player is a fixed strategy (`material-6`) or a run id prefix (`672890ba@4`). `experiment.checkers-h2h`
  runs the same test for every arm against its baseline arm, seed for seed (`full: true`: the whole ballot).
- **`checkers.static-round-robin`** (was `checkers_round_robin.py`) — four static Checkers strategies (random, first-legal, one- and two-ply
  material lookahead) played against each other, 200 games per pairing with seats alternating; the
  numbers the "Multi-Agent Games" Learn chapter cites (headline: one ply of lookahead ≈ random,
  because captures are mandatory; two plies wins ~95%).

Tests: `uv run pytest jobs/evaluator`.
