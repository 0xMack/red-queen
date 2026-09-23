import json
import random
from dataclasses import replace

import numpy as np
import pytest
from evolve.neat import (
    InnovationTracker,
    NeatConfig,
    NeatGenome,
    initial_genome,
    mutate,
)
from evolve.neuro import random_weight_vector

from modelpack import (
    LocalModelStore,
    PackagedModel,
    build_package,
    export_network,
    with_parity,
)
from modelpack.exporters import neat_layers
from modelpack.packaging import sample_inputs, shard_initializers
from modelpack.store import Catalog, CatalogEntry


def _evolved_neat(seed: int, mutations: int = 60) -> NeatGenome:
    """A genome with real evolved structure: hidden nodes, skip connections, disabled genes."""
    rng = random.Random(seed)
    tracker = InnovationTracker(11 + 1 + 3)
    genome = initial_genome(11, 3, tracker, rng)
    config = NeatConfig(add_node_rate=0.3, add_connection_rate=0.5, toggle_rate=0.1)
    for _ in range(mutations):
        genome = mutate(genome, config, tracker, rng)
    return genome


def _max_error(network, package) -> float:
    model = PackagedModel(package.manifest, "fp32", package.blobs.__getitem__)
    samples = sample_inputs(network_inputs(network), count=200, seed=7)
    expected = np.asarray([network.forward(s) for s in samples])
    return float(np.max(np.abs(model.run(np.asarray(samples, dtype=np.float32)) - expected)))


def network_inputs(network) -> int:
    return network.num_inputs if isinstance(network, NeatGenome) else network.layer_sizes[0]


@pytest.mark.parametrize("sizes", [(11, 16, 3), (100, 24, 3), (4, 8, 8, 2), (3, 1)])
def test_weight_vector_package_matches_forward(sizes):
    network = random_weight_vector(sizes, random.Random(1))
    package = build_package(export_network(network), label="mlp")
    assert package.manifest.parameters == len(network.weights)
    assert _max_error(network, package) < 1e-5


@pytest.mark.parametrize("seed", range(8))
def test_neat_package_matches_forward(seed):
    genome = _evolved_neat(seed)
    hidden, _ = genome.complexity()
    package = build_package(export_network(genome), label="neat")
    assert _max_error(genome, package) < 1e-5
    if seed == 0:
        assert hidden > 0, "test genome should have evolved hidden structure"


def test_neat_layers_respect_dependencies():
    genome = _evolved_neat(3, mutations=120)
    seen = set(range(genome.num_inputs + 1))
    for group in neat_layers(genome):
        for _, incoming in group:
            assert all(source in seen for source, _ in incoming)
        seen |= {slot for slot, _ in group}


def test_minimal_neat_genome_exports():
    rng = random.Random(0)
    genome = initial_genome(11, 3, InnovationTracker(15), rng)
    assert _max_error(genome, build_package(export_network(genome), label="minimal")) < 1e-5


def test_sharding_splits_by_size_and_names_by_hash():
    network = random_weight_vector((100, 64, 64, 3), random.Random(2))
    exported = export_network(network)
    _, shards, contents = shard_initializers(exported.model, shard_bytes=20_000)
    assert len(shards) > 1
    for shard in shards:
        assert shard.path == f"weights/{shard.sha256}.bin"
        assert len(contents[shard.sha256]) == shard.bytes
    package = build_package(exported, label="sharded", shard_bytes=20_000)
    assert len(package.manifest.variants[0].shards) == len(shards)
    assert _max_error(network, package) < 1e-5


def test_package_id_is_content_addressed():
    network = random_weight_vector((11, 16, 3), random.Random(3))
    a = build_package(export_network(network), label="x")
    b = build_package(export_network(network), label="x")
    assert a.manifest.package_id == b.manifest.package_id
    c = with_parity(a, "fp32", action_agreement=1.0)
    assert c.manifest.package_id != a.manifest.package_id
    assert c.manifest.variant("fp32").parity.action_agreement == 1.0
    assert a.manifest.variant("fp32").parity.action_agreement is None  # original untouched


def test_broken_export_is_rejected():
    network = random_weight_vector((11, 16, 3), random.Random(4))
    exported = export_network(network)
    wrong = random_weight_vector((11, 16, 3), random.Random(5))
    with pytest.raises(ValueError, match="differs"):
        build_package(replace(exported, reference=wrong.forward), label="broken")


def test_local_store_round_trip(tmp_path):
    network = random_weight_vector((11, 16, 3), random.Random(6))
    package = build_package(export_network(network), label="stored", interface="snake/features.v1+relative3.v1")
    store = LocalModelStore(tmp_path)
    store.put(package)
    store.put(package)  # idempotent
    manifest = store.manifest(package.manifest.package_id)
    assert manifest == package.manifest
    assert (
        _max_error(
            network,
            type(package)(manifest=manifest, blobs={b.sha256: store.read_blob(b.sha256) for b in manifest.blobs()}),
        )
        < 1e-5
    )
    fixture = json.loads(store.read_blob(manifest.parity_fixture.sha256))
    assert len(fixture["inputs"]) == len(fixture["outputs"]) > 0

    catalog = Catalog(game="snake")
    catalog.upsert(
        CatalogEntry(entrant_id="run:a", package_id=manifest.package_id, label="a", interface=None, variants=["fp32"])
    )
    catalog.upsert(
        CatalogEntry(entrant_id="run:a", package_id=manifest.package_id, label="a2", interface=None, variants=["fp32"])
    )
    store.put_catalog(catalog)
    assert [e.label for e in store.catalog("snake").entries] == ["a2"]
    assert store.catalog("checkers").entries == []


@pytest.mark.parametrize("bad", ["../etc", "abc", "A" * 64])
def test_local_store_rejects_bad_keys(tmp_path, bad):
    store = LocalModelStore(tmp_path)
    with pytest.raises(ValueError):
        store.blob_path(bad)
    with pytest.raises(ValueError):
        store.catalog_path("../x")


def test_fp64_variant_is_exact_and_wasm_only():
    genome = _evolved_neat(1)
    package = build_package([export_network(genome, "float64"), export_network(genome, "float32")], label="both")
    fp64, fp32 = package.manifest.variants
    assert (fp64.id, fp32.id) == ("fp64", "fp32")
    assert fp64.requirements.backends == ["wasm"]
    assert "webgpu" in fp32.requirements.backends
    assert fp64.inputs[0].dtype == "float64" and fp32.inputs[0].dtype == "float32"
    assert fp64.parity.max_abs_error < 1e-12 < fp32.parity.max_abs_error
    samples = sample_inputs(11, count=50, seed=9)
    model = PackagedModel(package.manifest, "fp64", package.blobs.__getitem__)
    assert np.max(np.abs(model.run(np.asarray(samples)) - np.asarray([genome.forward(s) for s in samples]))) < 1e-12


def test_garbage_collection_keeps_only_what_catalogs_reference(tmp_path):
    store = LocalModelStore(tmp_path)
    old = build_package(export_network(random_weight_vector((11, 16, 3), random.Random(7))), label="old")
    new = build_package(export_network(random_weight_vector((11, 16, 3), random.Random(8))), label="new")
    store.put(old)
    store.put(new)
    catalog = Catalog(game="snake")
    catalog.upsert(
        CatalogEntry(
            entrant_id="run:a", package_id=new.manifest.package_id, label="a", interface=None, variants=["fp32"]
        )
    )
    store.put_catalog(catalog)

    removed, freed = store.collect_garbage()
    assert removed > 0 and freed > 0
    assert not store.manifest_path(old.manifest.package_id).exists()
    assert all(store.has_blob(b.sha256) for b in new.manifest.blobs())
    assert store.collect_garbage() == (0, 0)
