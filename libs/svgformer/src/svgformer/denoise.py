"""A learned denoiser in front of the MDL tracer (docs/design/0025 Decision 1): a small convolutional U-Net maps a
`corrupt.v1` image to the clean SVG's render, which the tracer then traces.

- **Pairs** come from svgformer's own stream (`svgformer.data.mixed_draw`: `synthetic`, and the training artwork with
  `artwork_share`), so they are free and never touch `icons` or `broad`. The input is the corrupted image; the target is
  the clean SVG's render as premultiplied RGBA -- the subject alone, with no background.
- **Why RGBA, not the clean pipeline's image** (0025 D0): the tracer needs a background it can tell apart from the
  subject, and a clean image's off-white is drawn at random, so a model can't predict it from the input. Predicting the
  subject and its coverage, then compositing over a background chosen to differ from the subject's colours
  (`composite`), gives the tracer that background without the model guessing a random one.
- **Loss:** L1 on the four channels, weighted up on edge pixels (`edge_weight`), where 0024 found most of svgformer's
  error.

Needs PyTorch (the `torch` extra).
"""

from __future__ import annotations

import io
import math
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from arena.copycat_sets import SEVERITIES, SIZE, corrupt_pixels
from games.copycat import Svg
from torch import Tensor, nn
from torch.nn import functional as F

from svgformer.data import ArtworkPool, mixed_draw

# off-white candidates for the background the tracer sees: the clean pipeline's range (0.86-1.0 per channel), and a few
# tints for subjects that are themselves near grey
BACKGROUNDS = np.array(
    [[g, g, g] for g in (0.86, 0.90, 0.94, 0.98, 1.0)]
    + [[0.96, 0.90, 0.90], [0.90, 0.96, 0.90], [0.90, 0.90, 0.96], [0.96, 0.96, 0.88]]
)


def pair(
    stream_seed: int,
    i: int,
    severities: tuple[str, ...] = SEVERITIES,
    pool: ArtworkPool | None = None,
    artwork_share: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Sample `i` of the stream as (corrupted RGB (3, H, W) uint8, clean premultiplied RGBA (4, H, W) uint8, severity)."""
    svg, seed, severity = mixed_draw(stream_seed, i, severities, pool, artwork_share)
    pixels, _ = corrupt_pixels(svg, seed, severity)
    corrupted = np.asarray(pixels.convert("RGB")).transpose(2, 0, 1).copy()
    clean = np.frombuffer(Svg(svg).render(), np.float64).reshape(SIZE, SIZE, 4)
    target = np.round(np.clip(clean, 0, 1) * 255).astype(np.uint8).transpose(2, 0, 1).copy()
    return corrupted, target, severity


def background_for(rgba: np.ndarray) -> np.ndarray:
    """The candidate background (RGB, 0..1) farthest from the subject's colours: the one whose nearest well-covered
    subject pixel (5th percentile of distance, so a few stray pixels don't decide) is farthest. `rgba` is (H, W, 4)
    premultiplied, 0..1."""
    alpha = rgba[..., 3]
    covered = alpha > 0.5
    if not covered.any():
        return BACKGROUNDS[0]
    colours = rgba[covered][:, :3] / alpha[covered][:, None]
    if len(colours) > 4096:
        colours = colours[:: len(colours) // 4096]
    distance = np.linalg.norm(colours[None, :, :] - BACKGROUNDS[:, None, :], axis=-1)  # (candidates, pixels)
    return BACKGROUNDS[int(np.argmax(np.percentile(distance, 5, axis=1)))]


def composite(rgba: np.ndarray, under: np.ndarray | None = None) -> bytes:
    """A denoised (H, W, 4) premultiplied 0..1 image over `under` (default `background_for`), as straight RGBA8 with
    alpha 255: what the tracer reads."""
    under = background_for(rgba) if under is None else under
    rgb = rgba[..., :3] + under * (1 - rgba[..., 3:])
    rgb8 = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return np.concatenate([rgb8, np.full((*rgb8.shape[:2], 1), 255, np.uint8)], -1).tobytes()


@dataclass(frozen=True)
class Config:
    width: int = 32  # channels at full resolution, doubled at each level down
    levels: int = 4  # resolutions: 128, 64, 32, 16 at 4
    blocks: int = 2  # convolutions per level and side

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _conv(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.SiLU())


class _Stage(nn.Module):
    def __init__(self, c_in: int, c_out: int, blocks: int):
        super().__init__()
        self.layers = nn.Sequential(_conv(c_in, c_out), *[_conv(c_out, c_out) for _ in range(blocks - 1)])
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x) + self.skip(x)


class Denoiser(nn.Module):
    """A U-Net: corrupted RGB (B, 3, H, W) in 0..1 -> premultiplied RGBA (B, 4, H, W) in 0..1. Colour is predicted
    straight and multiplied by coverage, so premultiplied colour never exceeds its alpha."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        widths = [config.width * 2**k for k in range(config.levels)]
        self.stem = _conv(3, widths[0])
        self.down = nn.ModuleList()
        c = widths[0]
        for w in widths:
            self.down.append(_Stage(c, w, config.blocks))
            c = w
        self.pool = nn.AvgPool2d(2)
        self.middle = _Stage(c, c, config.blocks)
        self.up = nn.ModuleList()
        for w in reversed(widths[:-1]):
            self.up.append(_Stage(c + w, w, config.blocks))
            c = w
        # the head also sees the input itself, so fine detail can be copied rather than rebuilt through the levels
        self.head = nn.Sequential(nn.Conv2d(c + 3, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, 4, 1))

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, images: Tensor) -> Tensor:
        x = self.stem(images * 2 - 1)
        skips = []
        for k, stage in enumerate(self.down):
            x = stage(x)
            if k < len(self.down) - 1:
                skips.append(x)
                x = self.pool(x)
        x = self.middle(x)
        for stage in self.up:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = stage(torch.cat([x, skips.pop()], 1))
        out = self.head(torch.cat([x, images * 2 - 1], 1)).float()
        alpha = torch.sigmoid(out[:, 3:])
        return torch.cat([torch.sigmoid(out[:, :3]) * alpha, alpha], 1)

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        torch.save({"config": self.config.to_dict(), "state": self.state_dict()}, buffer)
        return buffer.getvalue()

    @classmethod
    def load(cls, blob: bytes, device: torch.device | str = "cpu") -> Denoiser:
        saved = torch.load(io.BytesIO(blob), map_location=device, weights_only=True)
        model = cls(Config(**saved["config"])).to(device)
        model.load_state_dict(saved["state"])
        return model.eval()


class PairBatches(torch.utils.data.IterableDataset):
    """Whole batches of `pair`s, worker w of W making batches w, w + W, ... (as `svgformer.train.SyntheticBatches`), so a
    run on one stream seed sees the same data for a given worker count."""

    def __init__(self, stream_seed: int, batch: int, severities, pool=None, artwork_share=0.0, start: int = 0):
        self.stream_seed, self.batch, self.severities = stream_seed, batch, severities
        self.pool, self.artwork_share, self.start = pool, artwork_share, start

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        k = self.start + worker
        while True:
            pairs = [
                pair(self.stream_seed, k * self.batch + j, self.severities, self.pool, self.artwork_share)
                for j in range(self.batch)
            ]
            yield torch.from_numpy(np.stack([p[0] for p in pairs])), torch.from_numpy(np.stack([p[1] for p in pairs]))
            k += workers


def edges(target: Tensor) -> Tensor:
    """(B, 1, H, W) 1 where the target changes against a 4-neighbour (any channel), dilated by a pixel."""
    dx = (target[..., :, 1:] - target[..., :, :-1]).abs().amax(1, keepdim=True) > 1 / 64
    dy = (target[..., 1:, :] - target[..., :-1, :]).abs().amax(1, keepdim=True) > 1 / 64
    e = torch.zeros_like(target[:, :1])
    e[..., :, 1:] += dx
    e[..., :, :-1] += dx
    e[..., 1:, :] += dy
    e[..., :-1, :] += dy
    return (F.max_pool2d(e, 3, 1, 1) > 0).float()


def loss_of(predicted: Tensor, target: Tensor, edge_weight: float) -> Tensor:
    weight = 1 + edge_weight * edges(target)
    return ((predicted - target).abs() * weight).sum() / (weight.sum() * predicted.shape[1])


@dataclass(frozen=True)
class TrainConfig:
    width: int = 32
    levels: int = 4
    blocks: int = 2
    steps: int = 20_000
    batch: int = 32
    lr: float = 1e-3
    warmup: int = 500
    weight_decay: float = 0.01
    clip: float = 1.0
    precision: str = "bf16"
    workers: int = 24
    log_every: int = 500
    severities: tuple[str, ...] = SEVERITIES
    seed: int = 0
    edge_weight: float = 4.0
    artwork: str | None = None
    artwork_share: float = 0.0

    def model_config(self) -> Config:
        return Config(self.width, self.levels, self.blocks)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["severities"] = list(self.severities)
        return out


@dataclass
class Generation:
    generation: int
    step: int
    loss: float
    seconds: float
    samples: int
    extras: dict[str, float] = field(default_factory=dict)


def lr_at(cfg: TrainConfig, step: int) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train(
    cfg: TrainConfig,
    device: torch.device | str,
    on_generation: Callable[[Denoiser, Generation], bool | None] | None = None,
    model: Denoiser | None = None,
) -> Denoiser:
    """Trains for `cfg.steps`; `on_generation(model, generation)` after every `cfg.log_every` steps (return True to stop)."""
    torch.manual_seed(cfg.seed)
    model = model or Denoiser(cfg.model_config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.99))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    pool = None
    if cfg.artwork_share > 0:
        if not cfg.artwork:
            raise ValueError("artwork_share > 0 needs the artwork pool (svgformer.build's icons.jsonl)")
        pool = ArtworkPool.load(cfg.artwork, 10**9)
    loader = torch.utils.data.DataLoader(
        PairBatches(cfg.seed, cfg.batch, cfg.severities, pool, cfg.artwork_share),
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
        for _ in range(min(cfg.log_every, cfg.steps - step)):
            corrupted, clean = next(batches)
            images = corrupted.to(device, non_blocking=True).float() / 255
            target = clean.to(device, non_blocking=True).float() / 255
            for group in optimizer.param_groups:
                group["lr"] = lr_at(cfg, step)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                predicted = model(images)
            loss = loss_of(predicted, target, cfg.edge_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step()
            losses.append(float(loss.detach()))
            samples += images.shape[0]
            step += 1
        report = Generation(generation, step, statistics.fmean(losses), time.perf_counter() - started, samples)
        generation += 1
        if on_generation is not None and on_generation(model, report):
            break
    return model


def denoise(model: Denoiser, rgb: np.ndarray, device: str, precision: str = "bf16", batch: int = 64) -> np.ndarray:
    """(N, 3, H, W) uint8 corrupted images -> (N, H, W, 4) premultiplied float64 0..1."""
    autocast = device.startswith("cuda") and precision == "bf16"
    out = []
    model.eval()
    for start in range(0, len(rgb), batch):
        images = torch.from_numpy(np.ascontiguousarray(rgb[start : start + batch])).to(device).float() / 255
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out.append(model(images).float().permute(0, 2, 3, 1).cpu().numpy().astype(np.float64))
    return np.concatenate(out)
