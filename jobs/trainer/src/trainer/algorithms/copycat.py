"""`copycat.svgformer` (docs/design/0024, P2): supervised training of `svgformer`'s transformer on the `synthetic` stream,
recorded as an ordinary run (`paradigm: supervised`, as `tinylm` is).

A generation is `log_every` optimizer steps: fitness is minus the mean training loss, and every `held_out_every`
generations the held-out score is the mean fidelity (dB) of greedy, grammar-constrained SVGs on `synthetic/validate`,
scored exactly as `copycat.clean.v2` scores an entrant. Validate is a train-side group (0024 "Experiments"): decisions are
made on it, never on held-out images. The champion is the checkpoint best on validate, saved as the `<run_id>-weights`
artifact (a `torch.save` of `{"config", "state_dict"}`).

PyTorch is the `svgformer[torch]` extra, imported only when this runs, so the trainer works without it.
"""

from __future__ import annotations

import copy
import io
import json
import platform
import time
from typing import Any

from arena.copycat import INTERFACE
from arena.costs import TrainingCostMeter
from jobcore import Sink, recorded_run
from jobcore.algorithms.copycat import SvgFormerParams
from jobcore.specs import TrainSpec
from telemetry import GenerationStats, data_dir

from trainer.registry import adapter, held_out_every

SETS_ROOT = ("experiments", "copycat-sets")  # under the data dir: where copycat.report builds its groups


class _Counters:
    """What TrainingCostMeter reads: optimizer steps (`episodes`) and images trained on (`steps`)."""

    def __init__(self) -> None:
        self.episodes = 0
        self.steps = 0


def hardware(device: str) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {"device": device, "torch": torch.__version__, "python": platform.python_version()}
    if device == "cuda":
        out |= {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}
    else:
        out |= {"cpu": platform.processor() or platform.machine()}
    return out


@adapter("copycat.svgformer", "copycat")
def train_svgformer(spec: TrainSpec, params: SvgFormerParams, sink: Sink) -> str:
    import torch
    from arena.copycat_sets import group_targets
    from svgformer.train import TrainConfig, build, evaluate, model_config, train

    device = ("cuda" if torch.cuda.is_available() else "cpu") if params.device == "auto" else params.device
    if device == "cpu":
        torch.use_deterministic_algorithms(True)  # a CPU run is bit-reproducible on one machine (0024 Decision 4)
    cfg = TrainConfig(
        model=params.model,
        steps=spec.budget_amount,
        batch=params.batch,
        lr=params.lr,
        warmup=params.warmup,
        weight_decay=params.weight_decay,
        clip=params.clip,
        precision=params.precision,
        workers=params.workers,
        log_every=params.log_every,
        severities=tuple(params.severities),
        seed=spec.seed,
        number_sigma=params.number_sigma,
        number_features=params.number_features,
        patch_features=params.patch_features,
        fp32_head=params.fp32_head,
    )
    every = held_out_every(spec, 4)
    generations = -(-cfg.steps // cfg.log_every)
    targets = group_targets("synthetic/validate", data_dir().joinpath(*SETS_ROOT))[: params.validate_images]
    torch.manual_seed(spec.seed)
    model = build(cfg, device)
    architecture = model_config(cfg).to_dict()

    config = {
        "representation": "svgformer",
        "game": "copycat",
        "paradigm": "supervised",
        "interface": INTERFACE,
        "data": {"stream": "synthetic", "corruption": "corrupt.v1", "severities": list(cfg.severities)},
        "model": {"size": params.model, **architecture},
        "parameters": model.parameters_count(),
        "training": cfg.to_dict(),
        "validate": {"group": "synthetic/validate", "images": len(targets), "decoding": "greedy"},
        "generations": generations,
        "held_out_every": every,
        "hardware": hardware(device),
        "rng_seed": spec.seed,
        **spec.tags,
    }
    counters = _Counters()
    cost = TrainingCostMeter(population_size=1, fitness=counters)
    best: dict[str, Any] = {"fidelity": None, "state": None, "generation": None, "loss": None}

    with recorded_run(config, sink) as run:
        control = run.control_callback(cost)
        started = time.perf_counter()

        def on_generation(model, g) -> None:
            counters.episodes = g.step
            best["loss"] = g.loss
            counters.steps += g.samples
            last = g.generation == generations - 1
            held = None
            extras: dict[str, float] = {
                "step": float(g.step),
                "train_loss": g.loss,
                "token_accuracy": g.token_accuracy,
                "images_per_second": g.samples / g.seconds if g.seconds else 0.0,
            }
            if g.generation % every == 0 or last:
                result = evaluate(model, targets, device, precision=cfg.precision)
                held = result["fidelity"]
                extras |= {f"validate_{s}": v for s, v in result["by_severity"].items()}
                extras |= {
                    "validate_layer_recovery": result["layer_recovery"],
                    "validate_shape_ratio": result["shape_ratio"],
                    "validate_finished": result["finished"],
                    "validate_tokens": result["tokens"],
                }
                if best["fidelity"] is None or held > best["fidelity"]:
                    best.update(fidelity=held, generation=g.generation, state=copy.deepcopy(model.state_dict()))
            ref = f"{run.run_id}-gen{g.generation}"
            run.artifacts.put_program(ref, json.dumps({"type": "svgformer_progress", "step": g.step}).encode("utf-8"))
            run.metrics.record_generation(
                GenerationStats(
                    run_id=run.run_id,
                    island_id=None,
                    generation=g.generation,
                    timestamp=time.time(),
                    best_fitness=-g.loss,
                    mean_fitness=-g.loss,
                    worst_fitness=-g.loss,
                    diversity=0.0,  # one model, no population
                    champion_ref=ref,
                    held_out_score=None if held is None else round(held, 4),
                    extras={k: round(float(v), 6) for k, v in extras.items()},
                )
            )
            cost.on_generation(None)
            control(None)
            line = f"step {g.step:6d}  loss {g.loss:.4f}  accuracy {g.token_accuracy:.3f}"
            print(line + ("" if held is None else f"  validate {held:.2f} dB"), flush=True)

        train(cfg, device, on_generation, model)
        train_s = time.perf_counter() - started
        weights = io.BytesIO()
        torch.save({"config": architecture, "state_dict": best["state"] or model.state_dict()}, weights)
        run.artifacts.put_program(f"{run.run_id}-weights", weights.getvalue())
        summary = {
            "best_fitness": None if best["loss"] is None else -round(best["loss"], 4),
            "held_out_score": None if best["fidelity"] is None else round(best["fidelity"], 4),
            "champion_generation": best["generation"],
            "train_seconds": round(train_s, 1),
            "cost": cost.summary(),
        }
        if device == "cuda":
            summary["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        run.set_summary(summary)
    return run.run_id
