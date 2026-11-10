"""svgformer.data's artwork stream (docs/design/0024 E2): a pool of what fits, sampled reproducibly, and a mix that is
exactly the old synthetic stream when it has no artwork in it."""

import json

import pytest

from svgformer.data import ARTWORK_BASE, ArtworkPool, artwork_sample, mixed_sample, synthetic_sample

ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128"><circle cx="64" cy="64" r="40" fill="#2050c0"/>'
    '<rect x="44" y="44" width="40" height="40" fill="#f0c020"/></svg>'
)


def _pool(tmp_path):
    path = tmp_path / "icons.jsonl"
    rows = [
        {"size": 128, "clamped": 0, "tokens": 20, "svg": ICON},
        {"size": 128, "clamped": 0, "tokens": 5000, "svg": ICON},  # too long
        {"size": 256, "clamped": 0, "tokens": 20, "svg": ICON},  # wrong size
        {"size": 128, "clamped": 2, "tokens": 20, "svg": ICON},  # clamped
        {"size": 128, "clamped": 0, "tokens": 30, "svg": ICON.replace("#2050c0", "#c02020")},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return ArtworkPool.load(path, 1024)


def test_the_pool_keeps_what_fits(tmp_path):
    pool = _pool(tmp_path)
    assert len(pool) == 2
    assert pool.svg(1).count("#c02020") == 1
    with pytest.raises(ValueError):
        ArtworkPool.load(tmp_path / "icons.jsonl", 10)


def test_artwork_samples_are_reproducible_and_corrupted(tmp_path):
    pool = _pool(tmp_path)
    a, b = artwork_sample(pool, 1, 7), artwork_sample(pool, 1, 7)
    assert (a.image == b.image).all() and a.tokens == b.tokens and a.seed >= ARTWORK_BASE
    assert a.image.shape == (3, 128, 128)
    assert artwork_sample(pool, 2, 7).seed != a.seed


def test_the_mix(tmp_path):
    pool = _pool(tmp_path)
    for i in range(4):
        assert mixed_sample(0, i).tokens == synthetic_sample(0, i).tokens  # no pool: the old stream
        assert mixed_sample(0, i, pool=pool, artwork_share=0.0).seed == synthetic_sample(0, i).seed
        assert mixed_sample(0, i, pool=pool, artwork_share=1.0).seed >= ARTWORK_BASE
    drawn = [mixed_sample(0, i, pool=pool, artwork_share=0.25).seed >= ARTWORK_BASE for i in range(200)]
    assert 30 < sum(drawn) < 70
