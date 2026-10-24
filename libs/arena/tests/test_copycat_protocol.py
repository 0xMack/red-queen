"""copycat.clean.v1: the checked-in held-out set, and scoring against the clean SVG (docs/design/0022)."""

import io

import numpy as np
import pytest
from games.copycat import Svg
from PIL import Image

from arena import copycat, copycat_sets


def test_the_held_out_set_is_complete_and_canonical():
    targets = copycat.load_set()
    assert [t.seed for t in targets] == list(copycat.HELD_OUT_SEEDS)
    assert {t.severity: sum(x.severity == t.severity for x in targets) for t in targets} == dict.fromkeys(
        copycat_sets.SEVERITIES, 6
    )
    for target in targets:
        clean = Svg(target.clean)
        assert clean.canonical() == target.clean
        assert (clean.width, clean.height) == (target.width, target.height) == (128, 128)
        assert 2 <= clean.palette <= 6
        assert Image.open(io.BytesIO(target.png)).size == (128, 128)


def test_the_held_out_set_is_what_the_generator_makes():
    """The clean SVGs exactly; the images to within a level (Pillow's JPEG encoder is why they are checked in)."""
    for target in copycat.load_set()[:4]:  # one per severity
        clean, _parts = copycat_sets.synthetic(target.seed)
        assert clean == target.clean
        png, levels = copycat_sets.corrupt(clean, target.seed, target.severity)
        assert levels == target.levels
        ours = np.asarray(Image.open(io.BytesIO(png)), dtype=float)
        checked_in = np.asarray(Image.open(io.BytesIO(target.png)), dtype=float)
        assert np.abs(ours - checked_in).mean() < 1.0


def test_the_clean_svg_scores_perfectly():
    target = copycat.load_set()[0]
    scored = copycat.score(target, target.clean)
    assert scored["fidelity"] == copycat.PSNR_CAP
    assert scored["alpha_iou"] == scored["layer_recovery"] == 1.0
    assert all(scored[r] == 1.0 for r in copycat.RATIOS)
    assert scored["refused"] is None


def test_a_background_loses_over_black_and_a_refusal_scores_as_empty():
    target = copycat.load_set()[0]
    with_background = target.clean.replace(
        'viewBox="0 0 128 128">\n', 'viewBox="0 0 128 128">\n<rect width="128" height="128" fill="#ffffff"/>\n'
    )
    scored = copycat.score(target, with_background)
    assert scored["psnr_white"] == copycat.PSNR_CAP
    assert scored["psnr_black"] < 10
    assert scored["alpha_iou"] < 0.5
    refused = copycat.score(target, '<svg viewBox="0 0 128 128"><rect width="1" height="1" stroke="red"/></svg>')
    assert "stroke" in refused["refused"]
    assert refused["shapes"] == 0 and refused["layer_recovery"] == 0.0
    wrong_size = copycat.score(target, '<svg viewBox="0 0 64 64"/>')
    assert "64x64" in wrong_size["refused"]


def test_summaries_use_geometric_means_for_ratios():
    images = [
        {**dict.fromkeys(copycat.COLUMNS, 1.0), "severity": "clean", "shape_ratio": 10.0, "refused": None},
        {**dict.fromkeys(copycat.COLUMNS, 3.0), "severity": "heavy", "shape_ratio": 0.1, "refused": "no"},
    ]
    summary = copycat.summarize(images)
    assert summary["shape_ratio"] == pytest.approx(1.0)
    assert summary["fidelity"] == 2.0
    assert summary["refused"] == 1
    assert set(copycat.by_severity(images)) == {"clean", "heavy"}
