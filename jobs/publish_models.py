"""Publish leaderboard champions as model packages for client-side inference (docs/design/0009).

For every champion `evaluate.py` would rank: export it to ONNX (`modelpack`) as an fp64 and an fp32
variant, check each packaged artifact against the trainer's own forward pass, then play the
leaderboard protocol's held-out games with both and record how often they choose the same move. A
variant that doesn't agree on every decision would show visitors a different game than the one the
leaderboard scored, so it is published as its own entrant (`run:<id>@<variant>`) and evaluated in its
own right. (fp64 is WASM-only and agrees exactly; fp32 is what WebGPU can run, and for some champions
it flips a handful of near-tie decisions -- docs/design/0009.)

Writes to a `LocalModelStore` at jobs/run-data/models (served by apis/backend in development); the
game's catalog maps each entrant to its package. Re-running is idempotent: identical exports are
identical packages.

Also publishes every TinyLM checkpoint under jobs/run-data/tinylm (jobs/tinylm_run.py) to the
`tinylm` catalog: fp32 and int8 variants, checked on the corpus's held-out text (modelpack.lm).

Run with: uv run python jobs/publish_models.py [snake|tinylm]   (then re-run jobs/evaluate.py)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from evaluate import BOARD, HELD_OUT_SEEDS, MAX_STEPS, PROTOCOL, champion_entrants
from games import interfaces
from games.observation import Interface
from modelpack import (
    CatalogEntry,
    LocalModelStore,
    Package,
    PackagedModel,
    build_package,
    export_network_json,
    with_parity,
)
from run_context import RUN_DATA_DIR, TelemetryStores

MODELS_DIR = RUN_DATA_DIR / "models"
PARITY_SAMPLES = 256


@dataclass(frozen=True)
class Agreement:
    decisions: int
    agreed: int
    scores_match: bool

    @property
    def rate(self) -> float:
        return self.agreed / self.decisions if self.decisions else 1.0

    @property
    def exact(self) -> bool:
        return self.agreed == self.decisions and self.scores_match


def reference_games(interface: Interface, forward, seeds) -> list[tuple[list[list[float]], list[int], int]]:
    """Play each seed with the reference forward pass: (observations seen, decisions made, score)."""
    games = []
    for seed in seeds:
        game = interface.make_game(seed=seed, **BOARD)
        observation = game.reset()
        observations, decisions = [], []
        for _ in range(MAX_STEPS):
            action = interface.action.decode(forward(observation))
            observations.append(list(observation))
            decisions.append(action)
            observation, _reward, done = game.step(action)
            if done:
                break
        games.append((observations, decisions, game.score))
    return games


def measure_agreement(
    interface: Interface, reference_forward, packaged: PackagedModel, seeds=HELD_OUT_SEEDS
) -> tuple[Agreement, list[list[float]]]:
    """Decision agreement on the reference's own trajectories (one batched call per game), plus whether
    every game scores the same when the packaged model plays it end to end. Also returns the observations
    seen, as realistic parity samples."""
    games = reference_games(interface, reference_forward, seeds)
    decisions = agreed = 0
    for observations, reference_decisions, _ in games:
        outputs = packaged.run(np.asarray(observations))  # the variant casts to its own input dtype
        for row, expected in zip(outputs, reference_decisions, strict=True):
            decisions += 1
            agreed += interface.action.decode(row.tolist()) == expected
    packaged_scores = [s for _, _, s in reference_games(interface, packaged.forward, seeds)]
    scores_match = packaged_scores == [s for _, _, s in games]
    observations = [o for game_observations, _, _ in games for o in game_observations]
    return Agreement(decisions, agreed, scores_match), observations


def parity_samples(observations: list[list[float]], count: int = PARITY_SAMPLES) -> list[list[float]]:
    """An evenly spaced subset of real observations -- early, mid- and late-game states alike."""
    if len(observations) <= count:
        return observations
    step = len(observations) / count
    return [observations[int(i * step)] for i in range(count)]


VARIANT_DTYPES = ("float64", "float32")  # most exact first


def publish(
    entrant: dict[str, Any], raw: str, store: LocalModelStore
) -> tuple[list[CatalogEntry], Package, dict[str, Agreement]]:
    """Export, verify, measure agreement per variant, store. Returns the catalog entries: the entrant
    itself (served by every variant that plays its games identically) plus one `@<variant>` entrant per
    variant that doesn't."""
    interface = interfaces.get(entrant["interface"])
    exports = [export_network_json(raw, dtype) for dtype in VARIANT_DTYPES]
    run = entrant["run"]

    def package_with(samples=None) -> Package:
        return build_package(
            exports,
            label=entrant["label"],
            interface=interface.id,
            run_id=run.run_id,
            champion_ref=entrant["champion_ref"],
            samples=samples,
        )

    # First on random inputs, to have something runnable to measure agreement with; then rebuilt with
    # observations from the actual games as its parity samples (and fixture).
    draft = package_with()
    agreements: dict[str, Agreement] = {}
    observations: list[list[float]] = []
    for variant in draft.manifest.variants:
        model = PackagedModel(draft.manifest, variant.id, draft.blobs.__getitem__)
        agreements[variant.id], observations = measure_agreement(interface, exports[0].reference, model)
    package = package_with(parity_samples(observations))
    for variant_id, agreement in agreements.items():
        package = with_parity(
            package,
            variant_id,
            protocol=PROTOCOL,
            decisions=agreement.decisions,
            action_agreement=round(agreement.rate, 6),
            scores_match=agreement.scores_match,
        )
    store.put(package)

    manifest = package.manifest
    exact = [v.id for v in manifest.variants if agreements[v.id].exact]
    common = {
        "package_id": manifest.package_id,
        "interface": interface.id,
        "run_id": run.run_id,
        "champion_ref": entrant["champion_ref"],
    }
    entries = []
    if exact:
        # The package *is* the leaderboard entrant only through variants that play every game identically.
        entries.append(
            CatalogEntry(
                entrant_id=entrant["entrant_id"],
                label=entrant["label"],
                variants=exact,
                download_bytes={v: manifest.variant(v).requirements.download_bytes for v in exact},
                **common,
            )
        )
    for variant in manifest.variants:
        if variant.id not in exact:
            entries.append(
                CatalogEntry(
                    entrant_id=f"{entrant['entrant_id']}@{variant.id}",
                    label=f"{entrant['label']} ({variant.id} export)",
                    variants=[variant.id],
                    download_bytes={variant.id: variant.requirements.download_bytes},
                    **common,
                )
            )
    return entries, package, agreements


def main(game: str = "snake") -> None:
    registry, metrics, artifacts = TelemetryStores.open()
    store = LocalModelStore(MODELS_DIR)
    catalog = store.catalog(game)
    entrants = champion_entrants(registry, metrics, artifacts, game)
    # Only current entrants stay listed: a run that stopped qualifying (failed, deleted) drops out of the catalog too.
    current = {e["run"].run_id for e in entrants}
    catalog.entries = [e for e in catalog.entries if e.run_id is None or e.run_id in current]
    for entrant in entrants:
        started = time.perf_counter()
        raw = artifacts.get_program(entrant["champion_ref"]).decode("utf-8")
        entries, package, agreements = publish(entrant, raw, store)
        # A re-publish replaces this run's previous entries (e.g. an inexact export later fixed).
        catalog.entries = [e for e in catalog.entries if e.run_id != entrant["run"].run_id]
        for entry in entries:
            catalog.upsert(entry)
        print(f"{entrant['label']}  ->  {package.manifest.package_id[:12]}  ({time.perf_counter() - started:.1f}s)")
        for variant in package.manifest.variants:
            a = agreements[variant.id]
            print(
                f"    {variant.id}: {variant.requirements.download_bytes:>7,} B  max|Δ| {variant.parity.max_abs_error:.1e}  "
                f"agree {a.agreed}/{a.decisions}  scores {'match' if a.scores_match else 'DIFFER'}"
            )
    store.put_catalog(catalog)
    print(f"catalog: {store.catalog_path(game)} ({len(catalog.entries)} entries)")


def publish_tinylm(store: LocalModelStore) -> None:
    import tinylm
    from modelpack.lm import LMConfig, build_lm_package
    from tinylm.checkpoint import named_parameters
    from tinylm_run import SEQ_LEN, split_corpus

    catalog = store.catalog("tinylm")
    for checkpoint in sorted((RUN_DATA_DIR / "tinylm").glob("*.npz")):
        started = time.perf_counter()
        model, tokenizer, meta = tinylm.load(checkpoint.with_suffix(""))
        _, _, held_out = split_corpus()
        # Non-overlapping held-out windows, full context length -- text the checkpoint never saw.
        count = len(held_out) // SEQ_LEN
        windows = held_out[: count * SEQ_LEN].reshape(count, SEQ_LEN)
        config = LMConfig(**meta["config"])
        package = build_lm_package(
            named_parameters(model),
            config,
            [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)],
            lambda w, model=model: model(w).data,
            windows,
            label=f"TinyLM {meta['name']} · {meta['parameters']:,} params",
            description=(
                f"Character-level transformer, {config.n_layers} layers x {config.n_heads} heads, d_model {config.d_model}, "
                f"{config.max_seq_len}-character context; trained from scratch (libs/autodiff) on Alice in Wonderland, "
                f"held-out loss {meta['held_out_loss']}."
            ),
            provenance={"run_id": None, "champion_ref": checkpoint.name},
        )
        store.put(package)
        manifest = package.manifest
        catalog.upsert(
            CatalogEntry(
                entrant_id=f"tinylm:{meta['name']}",
                package_id=manifest.package_id,
                label=manifest.label,
                interface=manifest.interface,
                champion_ref=checkpoint.name,
                variants=[v.id for v in manifest.variants],
                download_bytes={v.id: v.requirements.download_bytes for v in manifest.variants},
            )
        )
        print(f"{manifest.label}  ->  {manifest.package_id[:12]}  ({time.perf_counter() - started:.1f}s)")
        for v in manifest.variants:
            print(
                f"    {v.id}: {v.requirements.download_bytes:>9,} B  max|Δlogit| {v.parity.max_abs_error:.1e}  "
                f"top-1 agreement {v.parity.action_agreement:.4f} over {v.parity.decisions} held-out positions"
            )
    store.put_catalog(catalog)


if __name__ == "__main__":
    which = sys.argv[1:] or ["snake", "tinylm"]
    if "snake" in which:
        main()
    if "tinylm" in which:
        publish_tinylm(LocalModelStore(MODELS_DIR))
    removed, freed = LocalModelStore(MODELS_DIR).collect_garbage()
    print(f"garbage collected {removed} superseded files ({freed / 1e6:.1f} MB)")
