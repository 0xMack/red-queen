"""Saving and loading a trained TinyLM: named weights (`.npz`) plus the config and vocabulary needed to
rebuild it (`.json`). Names follow the model's structure (`blocks.0.attn.query.weight`), which is also
what `modelpack`'s exporter reads -- so a checkpoint is the hand-off between training (numpy,
autodiff) and export (ONNX), docs/design/0009."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tinylm.model import TinyLM
from tinylm.tokenizer import CharTokenizer


@dataclass(frozen=True)
class TinyLMConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_heads: int
    n_layers: int
    d_hidden: int


def config_of(model: TinyLM) -> TinyLMConfig:
    first = model.blocks[0]
    return TinyLMConfig(
        vocab_size=model.token_embedding.weight.data.shape[0],
        max_seq_len=model.max_seq_len,
        d_model=model.token_embedding.weight.data.shape[1],
        n_heads=first.attn.n_heads,
        n_layers=len(model.blocks),
        d_hidden=first.mlp.fc1.weight.data.shape[1],
    )


def named_parameters(model: TinyLM) -> dict[str, np.ndarray]:
    """Every learnable array by structural name -- the one naming both checkpoints and the exporter use."""
    named = {
        "token_embedding.weight": model.token_embedding.weight.data,
        "position_embedding.weight": model.position_embedding.weight.data,
        "ln_final.gamma": model.ln_final.gamma.data,
        "ln_final.beta": model.ln_final.beta.data,
        "head.weight": model.head.weight.data,
        "head.bias": model.head.bias.data,
    }
    for i, block in enumerate(model.blocks):
        p = f"blocks.{i}"
        named |= {
            f"{p}.ln1.gamma": block.ln1.gamma.data,
            f"{p}.ln1.beta": block.ln1.beta.data,
            f"{p}.ln2.gamma": block.ln2.gamma.data,
            f"{p}.ln2.beta": block.ln2.beta.data,
            f"{p}.mlp.fc1.weight": block.mlp.fc1.weight.data,
            f"{p}.mlp.fc1.bias": block.mlp.fc1.bias.data,
            f"{p}.mlp.fc2.weight": block.mlp.fc2.weight.data,
            f"{p}.mlp.fc2.bias": block.mlp.fc2.bias.data,
        }
        for proj in ("query", "key", "value", "out_proj"):
            layer = getattr(block.attn, proj)
            named[f"{p}.attn.{proj}.weight"] = layer.weight.data
            named[f"{p}.attn.{proj}.bias"] = layer.bias.data
    return named


def build(config: TinyLMConfig, weights: dict[str, np.ndarray] | None = None, seed: int = 0) -> TinyLM:
    """A TinyLM of `config` -- randomly initialized, then overwritten with `weights` if given."""
    model = TinyLM(**asdict(config), rng=np.random.default_rng(seed))
    if weights is not None:
        targets = named_parameters(model)
        missing = set(targets) - set(weights)
        if missing:
            raise ValueError(f"checkpoint is missing {sorted(missing)[:3]}...")
        for name, array in targets.items():
            if weights[name].shape != array.shape:
                raise ValueError(f"{name}: checkpoint {weights[name].shape} vs model {array.shape}")
            array[...] = weights[name]  # the Tensors hold these arrays: write in place
    return model


def save(model: TinyLM, tokenizer: CharTokenizer, path: Path | str, **metadata) -> None:
    """Writes `<path>.npz` (weights) and `<path>.json` (config, vocabulary, metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".npz"), **named_parameters(model))
    vocab = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]
    path.with_suffix(".json").write_text(
        json.dumps({"config": asdict(config_of(model)), "vocab": vocab, **metadata}, indent=2), encoding="utf-8"
    )


def load(path: Path | str) -> tuple[TinyLM, CharTokenizer, dict]:
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path.with_suffix(".npz")) as archive:
        weights = {name: archive[name] for name in archive.files}
    model = build(TinyLMConfig(**meta["config"]), weights)
    tokenizer = CharTokenizer("".join(meta["vocab"]))
    if [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)] != meta["vocab"]:
        raise ValueError("vocabulary doesn't round-trip (duplicate characters?)")
    return model, tokenizer, meta
