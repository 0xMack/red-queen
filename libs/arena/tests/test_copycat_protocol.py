"""copycat.clean.v1: the checked-in held-out set, and scoring against the clean SVG (docs/design/0022)."""

import io
import re

import numpy as np
import pytest
from games.copycat import Svg
from PIL import Image

from arena import copycat, copycat_artwork, copycat_sets, twemoji


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


def test_the_artwork_sets_are_pinned_to_exact_files_and_share_none():
    sources = copycat_artwork.sources()
    icons, broad = sources["icons"], sources["broad"]
    assert (len(copycat_artwork.file_names(icons)), icons["held_out"]) == (152, 24)
    assert (len(copycat_artwork.file_names(broad)), broad["held_out"]) == (616, 120)
    twemoji = next(f for f in broad["families"] if f["id"] == "twemoji")
    assert not set(twemoji["names"]) & set(icons["families"][0]["names"])
    for artwork in (icons, broad):
        for family in artwork["families"]:
            # a tag or a commit, never a branch that moves
            assert "/master" not in family["base"] and "/main" not in family["base"]
        split = copycat_sets.icon_split("heldout", artwork)
        train = copycat_sets.icon_split("train", artwork)
        assert len(split) == artwork["held_out"]
        assert not {i for _s, i, _v in train} & {i for _s, i, _v in split}
        severities = [severity for _seed, _icon, severity in split]
        assert {s: severities.count(s) for s in severities} == dict.fromkeys(
            copycat_sets.SEVERITIES, artwork["held_out"] // 4
        )
    # the checked-in data is only ever synthetic's
    assert [p.name for p in copycat_sets.DATA.iterdir() if p.is_dir()] == ["synthetic"]


def _small_set(tmp_path):
    """A set served from local files, so that nothing here needs the network."""
    sources = tmp_path / "served"
    sources.mkdir(parents=True)
    names = [f"a{i}" for i in range(8)]
    raw, clean = {}, {}
    for i, name in enumerate(names):
        body = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36"><circle fill="#a00" cx="18" cy="18" r="{i + 6}"/></svg>'
        (sources / f"{name}.svg").write_text(body, encoding="utf-8")
        raw[name] = body.encode("utf-8")
        clean[f"x-{name}.svg"] = twemoji.convert(body).encode("utf-8")
    return {
        "shuffle": "test",
        "held_out": 4,
        "families": [
            {"id": "x", "prefix": "x-", "base": sources.as_uri(), "names": names, "sha256": copycat_artwork.digest(raw)}
        ],
        "clean_sha256": copycat_artwork.digest(clean),
        "heldout_sha256": "",
    }, sources


def test_a_set_is_downloaded_once_built_and_refused_if_it_differs(tmp_path):
    artwork, served = _small_set(tmp_path)
    root = tmp_path / "root"
    with pytest.raises(copycat_artwork.SetNotFetched):
        copycat_artwork.ensure("small", root, download=False, artwork=artwork)
    clean = copycat_artwork.ensure("small", root, artwork=artwork)
    assert sorted(p.stem for p in clean.glob("*.svg")) == copycat_artwork.file_names(artwork)
    # built, it needs neither its sources nor permission to download
    for path in served.iterdir():
        path.unlink()
    assert copycat_artwork.ensure("small", root, download=False, artwork=artwork) == clean

    # a source that isn't what was pinned is refused, even if it's a perfectly good SVG
    artwork, served = _small_set(tmp_path / "other")
    (served / "a0.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36"><circle cx="1" cy="1" r="1"/></svg>',
        encoding="utf-8",
    )
    with pytest.raises(copycat_artwork.SetMismatch, match="pinned"):
        copycat_artwork.ensure("small", tmp_path / "other-root", artwork=artwork)


def test_held_out_images_are_refused_if_they_come_out_different(tmp_path):
    artwork, _ = _small_set(tmp_path)
    root = tmp_path / "root"
    with pytest.raises(copycat_artwork.SetMismatch, match="held-out"):
        copycat_artwork.held_out("small", root, artwork=artwork)
    where = root / "small" / "heldout"
    files = {p.name: p.read_bytes() for p in where.iterdir()}
    artwork["heldout_sha256"] = copycat_artwork.digest(files)
    assert copycat_artwork.held_out("small", root, artwork=artwork) == where
    targets = copycat.load_set("heldout", root=where, set_name="small")
    assert len(targets) == 4 and all(t.id.startswith("small/1000") for t in targets)
    assert all(Svg(t.clean).canonical() == t.clean for t in targets)


def test_a_polygon_and_a_rect_scale_from_any_square_view_box():
    from arena import twemoji

    out = Svg(
        twemoji.convert(
            '<svg viewBox="0 0 48 48"><rect x="12" y="12" width="12" height="24" fill="#f00"/>'
            '<g fill="#00f"><polygon points="24,12 36,12,36 36 24,36"/></g></svg>'
        )
    )
    assert [(kind, fill) for kind, fill, _opacity in out.describe()] == [
        ("rect", (255, 0, 0)),
        ("polygon", (0, 0, 255)),
    ]
    for part in (0, 1):
        assert np.frombuffer(out.coverage(part), dtype=np.float64).sum() == pytest.approx(32 * 64, rel=1e-3)
    with pytest.raises(twemoji.Refused, match="rounded"):
        twemoji.convert('<svg viewBox="0 0 48 48"><rect rx="2" width="4" height="4"/></svg>')
    with pytest.raises(twemoji.Refused, match="viewBox"):
        twemoji.convert('<svg viewBox="0 2 48 48"><circle r="4"/></svg>')


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
