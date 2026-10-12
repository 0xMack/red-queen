# publisher

The publisher workload (docs/design/0018): **one `PublishSpec` in, model packages in the model store out**
(docs/design/0009). It writes the local store (`telemetry.models_dir()`); the eventual host, R2 or the HF Hub, is
still undecided. With `collect_garbage` (the default) it then deletes files no catalog references any more.

```bash
uv run python -m publisher jobs/publisher/specs/snake.yaml     # then re-run the snake.score.v2 evaluation
uv run python -m publisher jobs/publisher/specs/tinylm.yaml
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

Tests: `uv run pytest jobs/publisher`; the golden publish cases are in `jobs/tests/test_golden_eval.py`.
