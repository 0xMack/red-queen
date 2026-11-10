import json
import math
from pathlib import Path

import numpy as np
import pytest
from games.copycat import Svg

from svgformer import artwork
from svgformer.convert import Refused, convert

HEAD = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 16 16">'


def pixels(svg: str) -> np.ndarray:
    doc = Svg(svg)
    rgba = np.frombuffer(doc.render_rgba8(), dtype=np.uint8).reshape(doc.height, doc.width, 4).astype(float)
    alpha = rgba[..., 3:] / 255
    return rgba[..., :3] * alpha + 255 * (
        1 - alpha
    )  # over white: straight alpha's colour is arbitrary where alpha is ~0


def same_image(converted: str, reference: str, tolerance: float = 1.0) -> bool:
    """Mean absolute difference per channel, 0-255: the converter promises the source's image, to the subset's precision."""
    return float(np.abs(pixels(converted) - pixels(reference)).mean()) < tolerance


def subset(body: str, size: int = 128) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}">{body}</svg>'


@pytest.mark.parametrize(
    "source, reference",
    [
        # an even-odd hole wound the same way as its outline becomes a non-zero hole
        (
            f'{HEAD}<path fill-rule="evenodd" fill="#f00" d="M2 2h12v12H2zM5 5h6v6H5z"/></svg>',
            subset(
                '<rect x="16" y="16" width="96" height="24" fill="#ff0000"/><rect x="16" y="88" width="96" height="24"'
                ' fill="#ff0000"/><rect x="16" y="40" width="24" height="48" fill="#ff0000"/><rect x="88" y="40"'
                ' width="24" height="48" fill="#ff0000"/>'
            ),
        ),
        # <use> of a definition, under a group transform and with x/y
        (
            (
                f'{HEAD}<defs><circle id="c" r="3" fill="#00f"/></defs><g transform="translate(4 8)">'
                '<use xlink:href="#c"/></g><use href="#c" x="12" y="8"/></svg>'
            ),
            subset('<circle cx="32" cy="64" r="24" fill="#0000ff"/><circle cx="96" cy="64" r="24" fill="#0000ff"/>'),
        ),
        # style inheritance, and opacity on a group of one shape
        (
            f'{HEAD}<g style="fill:#00aa00" opacity=".5"><rect x="2" y="2" width="12" height="8"/></g></svg>',
            subset('<rect x="16" y="16" width="96" height="64" fill="#00aa00" fill-opacity="0.5"/>'),
        ),
        # a non-square viewBox is centred
        (
            (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="10 0 32 16"><rect x="10" y="0" width="32"'
                ' height="16" fill="rgb(0, 0, 255)"/></svg>'
            ),
            subset('<rect x="0" y="32" width="128" height="64" fill="#0000ff"/>'),
        ),
        # a skewed rect becomes a polygon; named colours; Illustrator's <switch>
        (
            (
                f'{HEAD}<switch><foreignObject requiredExtensions="x"/><g><rect width="8" height="8" fill="navy"'
                ' transform="matrix(1 0 0.5 1 0 4)"/></g></switch></svg>'
            ),
            subset('<polygon points="0,32 64,32 96,96 32,96" fill="#000080"/>'),
        ),
    ],
)
def test_conversions_render_as_their_source(source, reference):
    assert same_image(convert(source, 128), reference)


def test_rotated_and_rounded_shapes_stay_close():
    rotated = f'{HEAD}<rect x="4" y="4" width="8" height="8" rx="2" fill="#123456" transform="rotate(90 8 8)"/></svg>'
    reference = f'{HEAD}<rect x="4" y="4" width="8" height="8" rx="2" fill="#123456"/></svg>'
    assert same_image(convert(rotated, 128), convert(reference, 128))


@pytest.mark.parametrize(
    "source, reason",
    [
        ('<path d="M0 0L5 5L0 5Z" stroke="#000" stroke-dasharray="1 1"/>', "dashed stroke"),
        ('<rect width="4" height="4" fill="#f00" stroke="#000" opacity=".5"/>', "opacity on a filled, stroked shape"),
        ('<path d="M0 0L5 5L0 5Z" fill="url(#g)"/>', "gradient"),
        ('<path d="M0 0L5 5L0 5Z" clip-path="url(#c)"/>', "clip-path"),
        ('<text x="1" y="1">a</text>', "element text"),
        ('<g opacity=".5"><rect width="4" height="4"/><rect x="2" width="4" height="4"/></g>', "opacity on a group"),
        ('<path d="M0 0L9 9L9 0L0 9Z" fill-rule="evenodd"/>', "crosses itself"),
        ('<rect width="4" height="4" fill="none"/>', "draws nothing"),
    ],
)
def test_what_isnt_exact_is_refused_with_a_reason(source, reason):
    with pytest.raises(Refused, match=reason):
        convert(f"{HEAD}{source}</svg>")


def test_iconify_icons_become_documents(tmp_path: Path):
    data = {
        "prefix": "t",
        "width": 24,
        "height": 24,
        "icons": {"a": {"body": '<path d="M2 2h20v20H2z" fill="#f00"/>'}, "b": {"body": "", "hidden": True}},
        "aliases": {"c": {"parent": "a"}},
    }
    (tmp_path / "t.json").write_text(json.dumps(data), encoding="utf-8")
    icons = list(artwork.iconify_icons(tmp_path / "t.json"))
    assert [name for name, _ in icons] == ["a"]
    assert same_image(
        convert(icons[0][1], 128), subset('<rect x="10.67" y="10.67" width="106.67" height="106.67" fill="#ff0000"/>')
    )


def test_the_pinned_sources_are_licensed_for_training():
    src = artwork.sources()
    sets = src["iconify"]["sets"]
    assert all(m["licence"] in artwork.LICENCES for m in sets.values())
    assert not set(sets) & set(artwork.EXCLUDED)
    assert len(src["sha256"]) == len(sets) + 1  # every download is pinned


def test_shapes_entirely_off_the_canvas_are_dropped():
    svg = convert(f'{HEAD}<rect width="8" height="8" fill="#f00"/><circle cx="-40" cy="5" r="3"/></svg>', 128)
    assert Svg(svg).shapes == 1


@pytest.mark.parametrize(
    "source, reference",
    [
        # a butt-capped line is a rect; a square cap adds half the width at each end
        (
            '<line x1="2" y1="8" x2="14" y2="8" stroke="#000" stroke-width="2"/>',
            '<rect x="16" y="56" width="96" height="16" fill="#000000"/>',
        ),
        (
            '<line x1="3" y1="8" x2="13" y2="8" stroke="#000" stroke-width="2" stroke-linecap="square"/>',
            '<rect x="16" y="56" width="96" height="16" fill="#000000"/>',
        ),
        # a stroked square (miter joins) is a frame: the fill, then the stroke over it
        (
            '<rect x="4" y="4" width="8" height="8" fill="#ff0000" stroke="#0000ff" stroke-width="2"/>',
            (
                '<rect x="24" y="24" width="80" height="80" fill="#0000ff"/><rect x="40" y="40" width="48" height="48"'
                ' fill="#ff0000"/>'
            ),
        ),
        # a stroke scales with a non-uniform transform: 1 user unit wide, 2 tall, so the vertical line is 16 px wide
        (
            '<g transform="scale(2 1)"><line x1="4" y1="2" x2="4" y2="14" stroke="#000"/></g>',
            '<rect x="56" y="16" width="16" height="96" fill="#000000"/>',
        ),
        # a stroked circle is a ring
        (
            '<circle cx="8" cy="8" r="5" fill="none" stroke="#00aa00" stroke-width="2"/>',
            (
                '<path d="M16 64A48 48 0 1 0 112 64A48 48 0 1 0 16 64ZM32 64A32 32 0 1 1 96 64A32 32 0 1 1 32 64Z"'
                ' fill="#00aa00"/>'
            ),
        ),
    ],
)
def test_strokes_are_outlined(source, reference):
    # the reference goes through convert too, for its arcs: on a 128 viewBox that is only arcs to cubics
    assert same_image(convert(f"{HEAD}{source}</svg>", 128), convert(subset(reference), 128))


def test_strokes_use_the_compact_outline_when_it_is_close_enough():
    stats: dict[str, int] = {}
    convert(
        f'{HEAD}<circle cx="8" cy="8" r="5" fill="none" stroke="#000"/><path d="M2 2Q8 14 14 2" fill="none"'
        ' stroke="#000" stroke-linejoin="round" stroke-linecap="round"/><polyline points="2,14 8,9 14,14" fill="none"'
        ' stroke="#000" stroke-width="1.5"/></svg>',
        128,
        stats,
    )
    assert stats == {"curves": 3}


def test_lengths_take_absolute_units():
    svg = convert(
        '<svg xmlns="http://www.w3.org/2000/svg" width="12pt" height="12pt"><rect width="12pt" height="6pt"/></svg>'
    )
    assert same_image(svg, subset('<rect x="0" y="0" width="128" height="64" fill="#000000"/>'))


def test_a_dense_polyline_with_round_joins_outlines_compactly():
    # the reference is a union of a circle per vertex; rendered at 1x its overlaps count twice at the edges and it comes
    # out ~7% fat, which once failed every compact outline of a polyline like this one
    points = " ".join(
        f"{8 + 5 * math.cos(2 * math.pi * k / 60):.3f},{8 + 5 * math.sin(2 * math.pi * k / 60):.3f}" for k in range(60)
    )
    stats: dict[str, int] = {}
    convert(
        f'{HEAD}<polygon points="{points}" fill="none" stroke="#000" stroke-width=".8" stroke-linejoin="round"/></svg>',
        128,
        stats,
    )
    assert stats == {"curves": 1}
