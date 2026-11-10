"""The model (docs/design/0024 Decision 1): a vision transformer reads the image's patches, and a causal transformer writes
the SVG's tokens (`svgformer.tokens`) with cross-attention to them. Pre-norm blocks, learned positions, no dropout (the
data is unlimited).

Written with plain modules and `scaled_dot_product_attention` rather than `nn.Transformer`, so the decoder step with its
KV cache is explicit: that is what decoding here and the ONNX export later (Decision 6) both need.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from svgformer import tokens as T


@dataclass(frozen=True)
class Config:
    size: int = 128  # the image's side, px
    patch: int = 8  # px per patch side: (128 / 8)^2 = 256 patch tokens
    d: int = 192
    heads: int = 6
    encoder_layers: int = 4
    decoder_layers: int = 4
    mlp: int = 4  # the MLP's width, in multiples of d
    max_tokens: int = 1024
    # Geometry priors (0024 P2, off the books: without them coordinates ignored the image). Both put a number token's
    # value and a patch's place in one frame, the token grid's (the canvas plus its margin, as 0..1), as Fourier features.
    number_features: bool = (
        True  # a number token's embedding and output row include a learned map of its value's features
    )
    patch_features: bool = True  # each patch's position includes a learned map of its centre's features
    frequencies: int = (
        10  # sin/cos at 1, 2, 4, ... 2^(k-1) cycles across the frame: the finest is about the grid's step
    )
    # The output logits in fp32 even under bf16 autocast: 1,280 coordinate logits that differ in the last bits of a
    # bf16 product (0024 E1: fp32 training beat bf16 by 0.8 dB).
    fp32_head: bool = False

    @property
    def patches(self) -> int:
        return (self.size // self.patch) ** 2

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def fourier(v: Tensor, k: int) -> Tensor:
    """(..., ) values in 0..1 -> (..., 2k) sin/cos features at 1, 2, 4, ... cycles."""
    angles = v[..., None] * (2 * math.pi * 2.0 ** torch.arange(k, dtype=v.dtype))
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def frame(px: Tensor, size: int) -> Tensor:
    """Canvas px -> the token grid's frame, 0..1 over the canvas plus a margin of size / 8 a side (`tokens.Grid`)."""
    margin = size / 8
    return (px + margin) / (size + 2 * margin)


SIZES = {
    "XS": Config(d=64, heads=2, encoder_layers=2, decoder_layers=2),  # tests and the golden case
    "S": Config(),
    "M": Config(d=256, heads=8, encoder_layers=6, decoder_layers=6),
    "L": Config(size=256, patch=16, d=352, heads=8, encoder_layers=8, decoder_layers=8, max_tokens=4096),
}


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)

    def split(self, x: Tensor) -> Tensor:  # (B, N, d) -> (B, heads, N, d / heads)
        b, n, d = x.shape
        return x.view(b, n, self.heads, d // self.heads).transpose(1, 2)

    def keys_values(self, x: Tensor) -> tuple[Tensor, Tensor]:
        k, v = self.kv(x).chunk(2, dim=-1)
        return self.split(k), self.split(v)

    def forward(self, x: Tensor, k: Tensor, v: Tensor, causal: bool = False) -> Tensor:
        q = self.split(self.q(x))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        b, h, n, e = y.shape
        return self.out(y.transpose(1, 2).reshape(b, n, h * e))


def _mlp(d: int, k: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, k * d), nn.GELU(), nn.Linear(k * d, d))


class EncoderBlock(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.attn = Attention(c.d, c.heads)
        self.mlp = _mlp(c.d, c.mlp)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, *self.attn.keys_values(h))
        return x + self.mlp(self.norm2(x))


class DecoderBlock(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(c.d), nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.self_attn = Attention(c.d, c.heads)
        self.cross_attn = Attention(c.d, c.heads)
        self.mlp = _mlp(c.d, c.mlp)

    def forward(self, x: Tensor, memory: tuple[Tensor, Tensor]) -> Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, *self.self_attn.keys_values(h), causal=True)
        x = x + self.cross_attn(self.norm2(x), *memory)
        return x + self.mlp(self.norm3(x))

    def step(self, x: Tensor, memory: tuple[Tensor, Tensor], cache: list[Tensor | None]) -> Tensor:
        """One new position (x is (B, 1, d)); `cache` holds this block's past keys and values and is extended."""
        h = self.norm1(x)
        k, v = self.self_attn.keys_values(h)
        if cache[0] is not None:
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
        cache[0], cache[1] = k, v
        x = x + self.self_attn(h, k, v)  # the one query attends to every past position: no mask needed
        x = x + self.cross_attn(self.norm2(x), *memory)
        return x + self.mlp(self.norm3(x))


class SvgFormer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        c = self.config = config
        self.patch_embed = nn.Linear(3 * c.patch * c.patch, c.d)
        self.patch_pos = nn.Parameter(torch.zeros(1, c.patches, c.d))
        self.encoder = nn.ModuleList(EncoderBlock(c) for _ in range(c.encoder_layers))
        self.encoder_norm = nn.LayerNorm(c.d)
        self.token_embed = nn.Embedding(T.VOCAB, c.d)
        self.token_pos = nn.Parameter(torch.zeros(1, c.max_tokens, c.d))
        self.decoder = nn.ModuleList(DecoderBlock(c) for _ in range(c.decoder_layers))
        self.decoder_norm = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, T.VOCAB, bias=False)
        grid = T.Grid(c.size)
        values = frame(torch.tensor([grid.value(i) for i in range(T.GRID)], dtype=torch.float32), c.size)
        self.register_buffer("number_basis", fourier(values, c.frequencies), persistent=False)
        side = c.size // c.patch
        centres = frame((torch.arange(side, dtype=torch.float32) + 0.5) * c.patch, c.size)
        ys, xs = torch.meshgrid(centres, centres, indexing="ij")
        basis = torch.cat([fourier(xs.reshape(-1), c.frequencies), fourier(ys.reshape(-1), c.frequencies)], dim=-1)
        self.register_buffer("patch_basis", basis[None], persistent=False)
        self.number_in = nn.Linear(2 * c.frequencies, c.d, bias=False) if c.number_features else None
        self.number_out = nn.Linear(2 * c.frequencies, c.d, bias=False) if c.number_features else None
        self.patch_in = nn.Linear(4 * c.frequencies, c.d, bias=False) if c.patch_features else None
        self.apply(self._init)
        nn.init.normal_(self.patch_pos, std=0.02)
        nn.init.normal_(self.token_pos, std=0.02)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def load(cls, blob: bytes, device: torch.device | str = "cpu") -> SvgFormer:
        """A `<run>-weights` artifact (`torch.save` of `{"config", "state_dict"}`), in eval mode on `device`. Tensors are
        mapped there, since a checkpoint saved on the GPU names `cuda:0` and won't load as-is without one."""
        import io

        saved = torch.load(io.BytesIO(blob), weights_only=True, map_location=device)
        model = cls(Config(**saved["config"])).to(device)
        model.load_state_dict(saved["state_dict"])
        return model.eval()

    def encode(self, images: Tensor) -> Tensor:
        """images: (B, 3, H, W) in [0, 1] -> patch tokens (B, patches, d)."""
        c = self.config
        b = images.shape[0]
        x = images.unfold(2, c.patch, c.patch).unfold(3, c.patch, c.patch)  # (B, 3, h, w, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(b, c.patches, 3 * c.patch * c.patch)
        x = self.patch_embed(x * 2 - 1) + self.patch_pos
        if self.patch_in is not None:
            x = x + self.patch_in(self.patch_basis)
        for block in self.encoder:
            x = block(x)
        return self.encoder_norm(x)

    def memories(self, encoded: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Each decoder block's cross-attention keys and values, computed once per image."""
        return [block.cross_attn.keys_values(encoded) for block in self.decoder]

    def forward(self, images: Tensor, tokens: Tensor) -> Tensor:
        """Teacher forcing: logits (B, N, VOCAB) for the token after each of `tokens` (B, N)."""
        memories = self.memories(self.encode(images))
        x = self.embed(tokens) + self.token_pos[:, : tokens.shape[1]]
        for block, memory in zip(self.decoder, memories, strict=True):
            x = block(x, memory)
        return self.logits(self.decoder_norm(x))

    def embed(self, tokens: Tensor) -> Tensor:
        x = self.token_embed(tokens)
        if self.number_in is not None:
            number = (tokens >= T.NUM0) & (tokens < T.NUM0 + T.GRID)
            features = self.number_basis[(tokens - T.NUM0).clamp(0, T.GRID - 1)]
            x = x + number[..., None] * self.number_in(features)
        return x

    def logits(self, h: Tensor) -> Tensor:
        if self.config.fp32_head and torch.is_autocast_enabled(h.device.type):
            with torch.autocast(h.device.type, enabled=False):
                return self._logits(h.float())
        return self._logits(h)

    def _logits(self, h: Tensor) -> Tensor:
        out = self.head(h)
        if self.number_out is not None:
            numbers = h @ self.number_out(self.number_basis).T  # (..., GRID)
            out = torch.cat([out[..., : T.NUM0], out[..., T.NUM0 :] + numbers], dim=-1)
        return out

    def step(
        self, token: Tensor, position: int, memories: list[tuple[Tensor, Tensor]], caches: list[list[Tensor | None]]
    ) -> Tensor:
        """Logits (B, VOCAB) for the token after `token` (B,) at `position`, extending `caches`."""
        x = self.embed(token)[:, None] + self.token_pos[:, position : position + 1]
        for block, memory, cache in zip(self.decoder, memories, caches, strict=True):
            x = block.step(x, memory, cache)
        return self.logits(self.decoder_norm(x))[:, 0]
