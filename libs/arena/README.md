# arena

Measurement as a domain (docs/design/0018): what a score *means*, independent of who runs it.

- `arena.snake`, `arena.checkers`, `arena.bandit` — each game's versioned evaluation protocol (seed sets, caps, board,
  scoring rule) and playing under it: `measure_quality`/`monitor_score` for Snake, game pairs over the ballot for
  Checkers. Training imports these for its held-out curves, leaderboard jobs for ranking, so both play by the same
  rules. **Never edit a protocol in place** — changing seeds, caps or rules means a new protocol version.
- `arena.seeding` — training-seed strategies (`fixed:N`, `resample:N`) and `TRAINING_POOL`, disjoint from every
  evaluation seed range (tested).
- `arena.versus_stats` — the statistics of two-player strength (docs/design/0013): Elo, pentanomial pair counts,
  Bradley–Terry ratings with bootstrap intervals, the SPRT. Has a TypeScript twin, `apps/frontend/app/utils/versusStats.ts`.
- `arena.costs` — `TrainingCostMeter` and `hardware_fingerprint()` (docs/design/0007: counters first, clocks second).

No I/O and no telemetry: depends only on `games`, `evolve` and numpy. Tests: `uv run pytest libs/arena`.
