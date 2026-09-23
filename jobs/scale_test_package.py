"""Publish a deliberately large model package to exercise the big-model path end to end
(docs/design/0009 plan step 5): many weight shards, the explicit download-confirmation click, persistent
browser storage, the memory checks in device matching, and WebGPU on a model where the GPU should win.

The model is TinyLM's architecture at GPT-2-small proportions (12 layers, d_model 768, 12 heads,
~85M parameters, ~340 MB in fp32, ~85 MB in int8) with *random* weights: its text is noise, and it's
labelled a scale test everywhere it appears. What it measures is the infrastructure -- download,
integrity checks, caching, session creation, per-token latency -- not language modelling. Published to
the `scale-test` catalog, which only the /dev/inference page reads.

Run with: uv run python jobs/scale_test_package.py [d_model] [n_layers]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from modelpack import CatalogEntry, LocalModelStore
from modelpack.lm import LMConfig, build_lm_package
from tinylm.checkpoint import TinyLMConfig, build, named_parameters
from tinylm_run import split_corpus

MODELS_DIR = Path(__file__).parent / "run-data" / "models"


def main(d_model: int = 768, n_layers: int = 12) -> None:
    tokenizer, _, held_out = split_corpus()
    config = TinyLMConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=128,
        d_model=d_model,
        n_heads=d_model // 64,
        n_layers=n_layers,
        d_hidden=4 * d_model,
    )
    started = time.perf_counter()
    model = build(config, seed=0)
    weights = named_parameters(model)
    parameters = sum(w.size for w in weights.values())
    print(
        f"{parameters:,} parameters ({parameters * 4 / 1e6:,.0f} MB fp32), built in {time.perf_counter() - started:.0f}s"
    )

    windows = held_out[: 16 * config.max_seq_len].reshape(16, config.max_seq_len)
    package = build_lm_package(
        weights,
        LMConfig(**config.__dict__),
        [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)],
        lambda w: model(w).data,
        windows,
        label=f"Scale test · {parameters / 1e6:.0f}M params (random weights)",
        description=(
            f"TinyLM's architecture at {n_layers} layers x d_model {d_model}, randomly initialized -- it writes noise. "
            "It exists to test downloading, caching and running a model this size in a browser, not to write text."
        ),
        provenance={"champion_ref": "scale-test"},
    )
    store = LocalModelStore(MODELS_DIR)
    store.put(package)
    manifest = package.manifest
    catalog = store.catalog("scale-test")
    catalog.upsert(
        CatalogEntry(
            entrant_id=f"scale-test:{d_model}x{n_layers}",
            package_id=manifest.package_id,
            label=manifest.label,
            interface=manifest.interface,
            variants=[v.id for v in manifest.variants],
            download_bytes={v.id: v.requirements.download_bytes for v in manifest.variants},
        )
    )
    store.put_catalog(catalog)
    print(f"{manifest.label} -> {manifest.package_id[:12]} ({time.perf_counter() - started:.0f}s total)")
    for v in manifest.variants:
        print(
            f"    {v.id}: {v.requirements.download_bytes / 1e6:,.1f} MB in {len(v.shards)} shards, "
            f"largest tensor {v.requirements.max_tensor_bytes / 1e6:.1f} MB, backends {v.requirements.backends}, "
            f"top-1 agreement {v.parity.action_agreement}"
        )


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
