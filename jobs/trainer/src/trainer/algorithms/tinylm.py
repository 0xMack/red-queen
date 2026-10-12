"""Train TinyLM on Alice's Adventures in Wonderland and save a checkpoint (docs/design/0004, 0009).

Same model and schedule as notebooks/0006-tinylm-from-scratch.ipynb (the numbers the Transformers Learn chapter
cites), except that the last 5% of the corpus is held out: never trained on, used for a validation loss here and, by
jobs/publish_models.py, as the text an exported package is checked on (top-1 next-character agreement).

Writes <data dir>/tinylm/<checkpoint>.npz + .json, which is what the model store publishes -- and, since
docs/design/0018 stage 3c, records an ordinary run (`paradigm: supervised`) beside it, so a TinyLM run is on the runs
page like any other: one generation per `log_every` steps, fitness = minus the mean training loss of those steps
(higher is better, as everywhere), held-out score = minus the loss on 256 fixed held-out windows (the same windows the
checkpoint's `held_out_loss` is measured on), each generation's champion a small progress record, and the final
weights as the `<run_id>-weights` artifact. Monitoring never touches the training rng, so the checkpoint is exactly
what the standalone script trained.
"""

from __future__ import annotations

import io
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
from arena.costs import TrainingCostMeter
from jobcore import Sink, recorded_run
from jobcore.algorithms.language import TinyLMParams
from jobcore.specs import TrainSpec
from telemetry import GenerationStats, data_dir
from tinylm import Adam, CharTokenizer, TinyLM, cross_entropy, save
from tinylm.checkpoint import named_parameters

from trainer.registry import adapter, held_out_every

CORPUS = Path(__file__).parents[5] / "libs" / "tinylm" / "data" / "alice.txt"
HELD_OUT_FRACTION = 0.05
HELD_OUT_WINDOWS = 256


def split_corpus() -> tuple[CharTokenizer, np.ndarray, np.ndarray]:
    """(tokenizer over the whole corpus, training ids, held-out ids) -- the vocabulary covers the held-out
    text too, so every held-out character is encodable."""
    text = CORPUS.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    ids = np.array(tokenizer.encode(text))
    cut = int(len(ids) * (1 - HELD_OUT_FRACTION))
    return tokenizer, ids[:cut], ids[cut:]


def windows(ids: np.ndarray, count: int, rng: np.random.Generator, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    starts = rng.integers(0, len(ids) - seq_len - 1, size=count)
    x = np.stack([ids[s : s + seq_len] for s in starts])
    y = np.stack([ids[s + 1 : s + seq_len + 1] for s in starts])
    return x, y


class _Counters:
    """What TrainingCostMeter reads (`episodes`, `steps`): training steps and the characters they predicted."""

    def __init__(self) -> None:
        self.episodes = 0
        self.steps = 0


@adapter("tinylm", "text")
def train_tinylm(spec: TrainSpec, params: TinyLMParams, sink: Sink) -> str:
    steps, every = spec.budget_amount, held_out_every(spec, 5)
    tokenizer, train, held_out = split_corpus()
    rng = np.random.default_rng(spec.seed)
    model_config = {
        "d_model": params.d_model,
        "n_heads": params.n_heads,
        "n_layers": params.n_layers,
        "d_hidden": params.d_hidden,
    }
    model = TinyLM(vocab_size=tokenizer.vocab_size, max_seq_len=params.seq_len, rng=rng, **model_config)
    optimizer = Adam(model.parameters(), lr=params.lr)
    parameters = sum(p.data.size for p in model.parameters())
    # The same 256 held-out windows every time (their own rng): the curve compares like with like.
    held_x, held_y = windows(held_out, HELD_OUT_WINDOWS, np.random.default_rng(1), params.seq_len)
    generations = math.ceil(steps / params.log_every)

    config = {
        "representation": "tinylm",
        "game": "text",
        "paradigm": "supervised",
        "corpus": "libs/tinylm/data/alice.txt",
        "held_out_fraction": HELD_OUT_FRACTION,
        "model": model_config,
        "parameters": parameters,
        "seq_len": params.seq_len,
        "batch_size": params.batch_size,
        "lr": params.lr,
        "steps": steps,
        "log_every": params.log_every,
        "generations": generations,
        "held_out_every": every,
        "checkpoint": params.checkpoint,
        "rng_seed": spec.seed,
        **spec.tags,
    }
    counters = _Counters()
    cost = TrainingCostMeter(population_size=1, fitness=counters)

    with recorded_run(config, sink) as run:
        control = run.control_callback(cost)
        started = time.perf_counter()
        losses: list[float] = []
        for generation in range(generations):
            window: list[float] = []
            for _ in range(min(params.log_every, steps - generation * params.log_every)):
                x, y = windows(train, params.batch_size, rng, params.seq_len)
                loss = cross_entropy(model(x), y)
                window.append(float(loss.data))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses += window
            counters.episodes += len(window)
            counters.steps += len(window) * params.batch_size * params.seq_len
            last = generation == generations - 1
            held = float(cross_entropy(model(held_x), held_y).data) if generation % every == 0 or last else None
            champion_ref = f"{run.run_id}-gen{generation}"
            progress = {"type": "tinylm_progress", "step": len(losses), "checkpoint": params.checkpoint}
            run.artifacts.put_program(champion_ref, json.dumps(progress).encode("utf-8"))
            mean = statistics.fmean(window)
            run.metrics.record_generation(
                GenerationStats(
                    run_id=run.run_id,
                    island_id=None,
                    generation=generation,
                    timestamp=time.time(),
                    best_fitness=-min(window),
                    mean_fitness=-mean,
                    worst_fitness=-max(window),
                    diversity=0.0,  # one model, no population: nothing to measure
                    champion_ref=champion_ref,
                    held_out_score=None if held is None else -held,
                    extras={"step": float(len(losses)), "train_loss": mean, "characters": float(counters.steps)},
                )
            )
            cost.on_generation(None)
            control(None)
            if held is not None:
                print(f"step {len(losses):5d}  loss {mean:.3f}  held-out {held:.3f}", flush=True)
        train_s = time.perf_counter() - started

        val_loss = float(cross_entropy(model(held_x), held_y).data)
        path = data_dir() / "tinylm" / params.checkpoint
        save(
            model,
            tokenizer,
            path,
            name=params.checkpoint,
            corpus="libs/tinylm/data/alice.txt",
            held_out_fraction=HELD_OUT_FRACTION,
            training={"steps": steps, "batch_size": params.batch_size, "lr": params.lr, "seconds": round(train_s, 1)},
            train_loss=round(float(np.mean(losses[-50:])), 4),
            held_out_loss=round(val_loss, 4),
            parameters=parameters,
        )
        weights = io.BytesIO()
        np.savez(weights, **named_parameters(model))
        run.artifacts.put_program(f"{run.run_id}-weights", weights.getvalue())
        run.set_summary(
            {
                "best_fitness": -round(float(np.mean(losses[-50:])), 4),
                "held_out_score": -round(val_loss, 4),
                "checkpoint": f"tinylm/{params.checkpoint}",
                "cost": cost.summary(),
            }
        )
    return run.run_id
