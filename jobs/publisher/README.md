# publisher

The publisher workload (docs/design/0018): **one `PublishSpec` in, model packages in the model store out**
(docs/design/0009), or one of the frontend's data files. It writes the local store (`telemetry.models_dir()`); the eventual host, R2 or the HF Hub, is
still undecided. With `collect_garbage` (the default) it then deletes files no catalog references any more.

```bash
uv run python -m publisher jobs/publisher/specs/snake.yaml     # then re-run the snake.score.v2 evaluation
uv run python -m publisher jobs/publisher/specs/tinylm.yaml
uv run redqueen export ballot                                   # a frontend.* target, through the API
```

## Targets

- **`snake`** (was `publish_models.py snake`) — every champion the Snake leaderboard would rank, exported to ONNX as an
  fp64 and an fp32 variant, each checked against the trainer's forward pass and then played on the protocol's held-out
  games: a variant that doesn't choose the same move every time is published as its own entrant (`run:<id>@<variant>`),
  so visitors never watch a different game from the one the leaderboard scored.
- **`tinylm`** (was `publish_models.py tinylm`) — every checkpoint under `<data dir>/tinylm`, fp32 and int8, checked on
  the corpus's held-out text (`modelpack.lm`).
- **`scale-test`** (was `scale_test_package.py`) — TinyLM's architecture at GPT-2-small proportions with random weights,
  for the big-model path (`/dev/inference`): download, shards, caching, WebGPU. Its text is noise, and it's labelled so.

**Frontend data files** (`publisher/frontend.py`, was `jobs/export_*.py`). The Learn chapters draw these from real
results. They are generated, written under `apps/frontend/app/data/` (or `params.out`), and committed. Garbage
collection doesn't apply to them, since they aren't in the model store. Each target has a `redqueen export` command:

- **`frontend.ballot`** (`redqueen export ballot`) writes `checkersBallot.ts`: every 3-ply opening, its board, and
  material search's verdict. The level ones are the ballot.
- **`frontend.pbt-history`** (`export pbt-history [--experiment --arm --seed]`) writes `pbtHistory.ts`: one PBT
  run's population history.
- **`frontend.rl-curves`** (`export rl-curves [dqn-snake] [pg-snake]`) writes `recordings/<name>.json`: chosen runs
  of the tracked RL experiments as curves (held-out score, and mean Q or policy entropy), for what a live demo can't
  reliably show.
- **`frontend.rl-recording`** (`export rl-recording RUN`) writes `recordings/q-learning-snake.json`: a tabular run's
  curve and final Q-table, the Q-learning chapter's no-WebAssembly fallback.

Tests: `uv run pytest jobs/publisher`; the golden publish cases are in `jobs/tests/test_golden_eval.py`.
