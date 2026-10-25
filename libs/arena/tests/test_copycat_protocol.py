"""copycat.clean.v1: the checked-in held-out set, and scoring against the clean SVG (docs/design/0022)."""

import io
import re

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


def test_the_icons_held_out_set_is_complete_and_what_the_split_makes():
    targets = copycat.load_set(set_name="icons")
    split = copycat_sets.icon_split("heldout")
    assert [t.seed for t in targets] == [seed for seed, _icon, _severity in split]
    assert {t.severity: sum(x.severity == t.severity for x in targets) for t in targets} == dict.fromkeys(
        copycat_sets.SEVERITIES, 6
    )
    for target, (_seed, icon, _severity) in zip(targets, split, strict=True):
        assert target.clean == (copycat_sets.ICONS / f"{icon}.svg").read_text(encoding="utf-8")
        assert Svg(target.clean).canonical() == target.clean
    # no icon is in both splits
    train = {icon for _seed, icon, _severity in copycat_sets.icon_split("train")}
    assert len(train) + len(split) == len(list(copycat_sets.ICONS.glob("*.svg")))
    assert not train & {icon for _seed, icon, _severity in split}


def test_twemoji_paths_become_the_subset():
    from arena import twemoji

    # relative commands, H/V, S, an arc with run-together flags, and a group's fill pushed down
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36"><g fill="#F00">'
        '<path d="M4 4h10v10H4z"/><path d="M20 20c2 0 4 2 4 4s-2 4-4 4a4 4 0 01-4-4z"/></g>'
        '<circle fill="#0f0" cx="18" cy="9" r="3"/></svg>'
    )
    out = Svg(twemoji.convert(svg))
    assert [(kind, fill) for kind, fill, _opacity in out.describe()] == [
        ("path", (255, 0, 0)),
        ("path", (255, 0, 0)),
        ("circle", (0, 255, 0)),
    ]
    # the square is 10 x 10 in 36, so about 35.6 px a side at 128 px (coordinates are written to a tenth)
    square = np.frombuffer(out.coverage(0), dtype=np.float64)
    assert square.sum() == pytest.approx(35.6**2, rel=1e-3)
    # every path command is one the subset has: the arc became cubics, the relative commands absolute
    for d in re.findall(r' d="([^"]*)"', out.canonical()):
        assert set(re.findall(r"[A-Za-z]", d)) <= set("MLCQZ")
    with pytest.raises(twemoji.Refused, match="transform"):
        twemoji.convert('<svg viewBox="0 0 36 36"><ellipse transform="rotate(45)" cx="1" cy="1" rx="1" ry="2"/></svg>')


def test_an_arc_is_a_circle_to_within_a_hair():
    from arena import twemoji

    # a full circle of radius 10 as two arcs: its area is pi r^2
    d = "M8 18a10 10 0 1 0 20 0a10 10 0 1 0-20 0z"
    out = Svg(twemoji.convert(f'<svg viewBox="0 0 36 36"><path d="{d}"/></svg>'))
    area = np.frombuffer(out.coverage(0), dtype=np.float64).sum()
    assert area == pytest.approx(np.pi * (10 * 128 / 36) ** 2, rel=2e-3)


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
