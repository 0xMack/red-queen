"""Supervised training (docs/design/0024, P2): teacher-forced cross-entropy on the clean SVG's tokens, AdamW with warmup
and a cosine decay, bf16 autocast on the GPU. Held-out curves decode greedily under the grammar and score each SVG
exactly as `copycat.clean.v2` does (`arena.copycat.score`).

No telemetry here: `train` reports each generation (a fixed number of optimizer steps) to a callback, which the trainer's
`copycat.svgformer` adapter records as a run.
"""

from __future__ import annotations

import dataclasses
import math
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from arena.copycat import Target, score
from arena.copycat_sets import SEVERITIES

from svgformer import tokens as T
from svgformer.data import collate, png_to_array, synthetic_sample
from svgformer.decode import generate, to_images
from svgformer.model import SIZES, Config, SvgFormer


class SyntheticBatches(torch.utils.data.IterableDataset):
    """Whole batches of the `synthetic` stream (`svgformer.data`). Worker w of W makes batches w, w + W, ..., batch k from
    samples k * batch .. k * batch + batch - 1; a DataLoader takes from workers in turn, so for a given W the batches
    arrive in order and every run on one stream seed sees the same data. (At module level: Windows spawns workers, and
    they unpickle it.)"""

    def __init__(self, stream_seed: int, batch: int, max_tokens: int, severities: tuple[str, ...], start: int = 0):
        self.stream_seed, self.batch, self.max_tokens, self.severities, self.start = (
            stream_seed, batch, max_tokens, severities, start,
        )  # fmt: skip

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        k = self.start + worker
        while True:
            samples = [
                synthetic_sample(self.stream_seed, k * self.batch + j, self.severities) for j in range(self.batch)
            ]
            rgb8, tokens = collate([s for s in samples if len(s.tokens) <= self.max_tokens])
            yield torch.from_numpy(rgb8), torch.from_numpy(tokens)
            k += workers


@dataclass(frozen=True)
class TrainConfig:
    model: str = "S"
    steps: int = 20_000
    batch: int = 64
    lr: float = 1e-3  # 3e-4 learned colours not at all in 6k steps; 1e-3 reached 18.5 dB (0024, P2)
    warmup: int = 1000
    weight_decay: float = 0.01
    clip: float = 1.0
    precision: str = "bf16"  # or "fp32"
    workers: int = 24
    log_every: int = 500  # optimizer steps per generation
    severities: tuple[str, ...] = SEVERITIES
    seed: int = 0
    number_sigma: float = 0.0  # soft labels: a coordinate's target is a Gaussian this many grid steps wide (0: exact)
    number_features: bool = True
    patch_features: bool = True
    fp32_head: bool = False

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["severities"] = list(self.severities)
        return out


@dataclass
class Generation:
    generation: int
    step: int
    loss: float  # mean training loss over the generation's steps
    token_accuracy: float  # of the generation's last batch, non-PAD targets
    seconds: float
    samples: int
    extras: dict[str, float] = field(default_factory=dict)


def lr_at(cfg: TrainConfig, step: int) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def model_config(cfg: TrainConfig) -> Config:
    return dataclasses.replace(
        SIZES[cfg.model],
        number_features=cfg.number_features,
        patch_features=cfg.patch_features,
        fp32_head=cfg.fp32_head,
    )


def build(cfg: TrainConfig | Config, device: torch.device | str) -> SvgFormer:
    config = model_config(cfg) if isinstance(cfg, TrainConfig) else cfg
    return SvgFormer(config).to(device)


def loss_and_accuracy(
    model: SvgFormer, images: torch.Tensor, tokens: torch.Tensor, number_sigma: float = 0.0
) -> tuple[torch.Tensor, float, float]:
    """The training loss, token accuracy and the plain cross-entropy (what curves compare, whatever the loss). With
    `number_sigma`, a coordinate's target is a Gaussian over the grid around its value instead of one exact bin: a near
    miss is nearly right, which a 1,280-way classification otherwise can't tell."""
    logits = model(images, tokens[:, :-1]).float()
    targets = tokens[:, 1:]
    real = targets != T.PAD
    log_p = torch.log_softmax(logits, dim=-1)
    hard = -log_p.gather(-1, targets[..., None])[..., 0]
    plain = hard[real].mean()
    loss = plain
    if number_sigma > 0:
        number = real & (targets >= T.NUM0)
        index = torch.arange(T.GRID, device=logits.device, dtype=torch.float32)
        centre = (targets - T.NUM0).clamp(min=0).float()[..., None]
        q = torch.exp(-0.5 * ((index - centre) / number_sigma) ** 2)
        q = q / q.sum(-1, keepdim=True)
        soft = -(q * log_p[..., T.NUM0 :]).sum(-1)
        loss = torch.where(number, soft, hard)[real].mean()
    accuracy = float(((logits.argmax(-1) == targets) & real).sum() / real.sum())
    return loss, accuracy, float(plain.detach())


def train(
    cfg: TrainConfig,
    device: torch.device | str,
    on_generation: Callable[[SvgFormer, Generation], bool | None] | None = None,
    model: SvgFormer | None = None,
) -> SvgFormer:
    """Trains for `cfg.steps`; `on_generation(model, generation)` after every `cfg.log_every` steps (return True to stop)."""
    torch.manual_seed(cfg.seed)
    model = model or build(cfg, device)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "pos" not in n]
    other = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "pos" not in n)]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": other, "weight_decay": 0.0}],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )
    # spawned workers import numpy afresh: one thread each, or 16 workers' BLAS pools fight over the cores
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    loader = torch.utils.data.DataLoader(
        SyntheticBatches(cfg.seed, cfg.batch, model.config.max_tokens, cfg.severities),
        batch_size=None,
        num_workers=cfg.workers,
        persistent_workers=cfg.workers > 0,
        prefetch_factor=4 if cfg.workers > 0 else None,
        pin_memory=str(device).startswith("cuda"),
    )
    autocast = str(device).startswith("cuda") and cfg.precision == "bf16"
    batches = iter(loader)
    step, generation = 0, 0
    while step < cfg.steps:
        model.train()
        started, losses, samples = time.perf_counter(), [], 0
        accuracy = 0.0
        for _ in range(min(cfg.log_every, cfg.steps - step)):
            rgb8, tokens = next(batches)
            images = rgb8.to(device, non_blocking=True).float() / 255
            tokens = tokens.to(device, non_blocking=True)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(cfg, step)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                loss, accuracy, plain = loss_and_accuracy(model, images, tokens, cfg.number_sigma)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step()
            losses.append(plain)
            samples += tokens.shape[0]
            step += 1
        report = Generation(
            generation, step, statistics.fmean(losses), accuracy, time.perf_counter() - started, samples
        )
        generation += 1
        if on_generation is not None and on_generation(model, report):
            break
    return model


def evaluate(
    model: SvgFormer,
    targets: list[Target],
    device: torch.device | str,
    batch: int = 100,
    precision: str = "bf16",
) -> dict[str, Any]:
    """Greedy, grammar-constrained SVGs for every target, scored as `copycat.clean.v2` scores an entrant."""
    model.eval()
    autocast = str(device).startswith("cuda") and precision == "bf16"
    rows, lengths, finished = [], [], 0
    for start in range(0, len(targets), batch):
        chunk = targets[start : start + batch]
        images = to_images(np.stack([png_to_array(t.png) for t in chunk]), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            sequences = generate(model, images)
        for target, sequence in zip(chunk, sequences, strict=True):
            finished += sequence[-1] == T.END
            lengths.append(len(sequence))
            rows.append(score(target, T.decode(sequence, target.width)))
    by_severity = {
        s: statistics.fmean(r["fidelity"] for r in rows if r["severity"] == s)
        for s in SEVERITIES
        if any(r["severity"] == s for r in rows)
    }
    return {
        "fidelity": statistics.fmean(r["fidelity"] for r in rows),
        "by_severity": by_severity,
        "layer_recovery": statistics.fmean(r["layer_recovery"] for r in rows),
        "shape_ratio": statistics.fmean(r["shape_ratio"] for r in rows),
        "finished": finished / len(rows),
        "tokens": statistics.fmean(lengths),
        "rows": rows,
    }
