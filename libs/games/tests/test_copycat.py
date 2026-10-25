"""games.copycat: the Python face of the Copy Cat core (the rasterizer and parser have their own Rust tests)."""

import struct

import pytest

from games.copycat import Svg, hill_climb, layer_recovery, trace

TWO_PARTS = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 8">'
    '<rect width="8" height="8" fill="#ff0000"/>'
    '<circle cx="12" cy="4" r="4" fill="#0000ff" opacity="0.5"/>'
    "</svg>"
)


def test_counts_and_canonical_form():
    svg = Svg(TWO_PARTS)
    assert (svg.width, svg.height, svg.shapes, svg.nodes, svg.palette) == (16, 8, 2, 8, 2)
    assert svg.describe() == [("rect", (255, 0, 0), 1.0), ("circle", (0, 0, 255), 0.5)]
    assert Svg(svg.canonical()).canonical() == svg.canonical()
    assert svg.turning == pytest.approx(4 * 3.141592653589793, abs=1e-9)


def test_refuses_with_the_reason():
    with pytest.raises(ValueError, match="stroke"):
        Svg('<svg viewBox="0 0 4 4"><rect width="1" height="1" stroke="red"/></svg>')


def test_renders():
    svg = Svg(TWO_PARTS)
    pixels = struct.unpack(f"<{16 * 8 * 4}d", svg.render())
    assert pixels[0:4] == (1.0, 0.0, 0.0, 1.0)  # inside the rect
    centre = (4 * 16 + 12) * 4
    assert pixels[centre : centre + 4] == (0.0, 0.0, 0.5, 0.5)  # inside the half-transparent circle
    rgba8 = svg.render_rgba8()
    assert rgba8[centre : centre + 4] == bytes([0, 0, 255, 128])
    coverage = struct.unpack(f"<{16 * 8}d", svg.coverage(1))
    assert coverage[0] == 0.0 and coverage[4 * 16 + 12] == 1.0
    with pytest.raises(ValueError):
        svg.coverage(2)


def test_layer_recovery_and_the_hill_climb():
    clean = Svg(TWO_PARTS)
    assert layer_recovery(clean, clean) == [1.0, 1.0]
    climbed = hill_climb(clean.render_rgba8(), 16, 8, shapes=6, candidates=20, patience=20, seed=1)
    assert climbed.shapes > 1 and climbed.describe()[0][0] == "rect"  # the background first
    assert (
        climbed.canonical()
        == hill_climb(clean.render_rgba8(), 16, 8, shapes=6, candidates=20, patience=20, seed=1).canonical()
    )
    with pytest.raises(ValueError):
        hill_climb(b"\0" * 12, 16, 8, shapes=1)


def test_the_structured_pipeline_recovers_the_parts():
    clean = Svg(
        TWO_PARTS.replace('viewBox="0 0 16 8"', 'viewBox="0 0 64 32"').replace(
            '<rect width="8" height="8" fill="#ff0000"/><circle cx="12" cy="4" r="4" fill="#0000ff" opacity="0.5"/>',
            '<rect x="6" y="6" width="24" height="20" fill="#d03030"/><circle cx="44" cy="16" r="11" fill="#2050c0"/>',
        )
    )
    # over an opaque background, as an image arrives
    pixels = bytearray(clean.render_rgba8())
    for i in range(0, len(pixels), 4):
        a = pixels[i + 3] / 255
        for c, bg in zip(range(3), (245, 240, 230), strict=True):
            pixels[i + c] = round(pixels[i + c] * a + bg * (1 - a))
        pixels[i + 3] = 255
    traced = trace(bytes(pixels), 64, 32)
    assert [kind for kind, _fill, _opacity in traced.describe()] == ["rect", "circle"]
    assert min(layer_recovery(clean, traced)) > 0.95
    assert traced.render_rgba8()[3] == 0  # the background is transparent
    with pytest.raises(ValueError):
        trace(b"\0" * 12, 64, 32)
