"""Copy Cat (docs/design/0022): recover the clean, layered SVG behind a generated logo or icon.

An entrant turns an image into an SVG; the protocol reads it in a restricted subset and renders it with the game core's
own rasterizer (rust/core/src/raster.rs, copycat.rs), so a score is the same number natively and in the browser. This is
the Python face of that core: parse, count, render, and the two pieces of scoring that need whole shapes (layer recovery)
or the core's speed (the `primitive`-style hill climb baseline). The subset and the canonical form are documented in
copycat.rs.

Images cross the boundary as bytes (no numpy here): `render()` is premultiplied RGBA as float64, `render_rgba8()` and
`hill_climb`'s and `trace`'s inputs are 8-bit RGBA, all row-major.
"""

from __future__ import annotations

from games import _native


class Svg:
    """An SVG in the subset. `Svg(text)` raises ValueError, with the reason, for anything outside it."""

    def __init__(self, text: str):
        self._core = _native.CopyCatDocument.parse(text)

    @classmethod
    def _wrap(cls, core) -> Svg:
        svg = cls.__new__(cls)
        svg._core = core
        return svg

    @property
    def width(self) -> int:
        return self._core.width

    @property
    def height(self) -> int:
        return self._core.height

    @property
    def shapes(self) -> int:
        return self._core.shapes

    @property
    def nodes(self) -> int:
        """On-curve points: a path's commands, a polygon's vertices, 4 for a rect, circle or ellipse."""
        return self._core.nodes

    @property
    def palette(self) -> int:
        """Distinct fill colours."""
        return self._core.palette

    @property
    def turning(self) -> float:
        """Total absolute turning angle along every flattened outline, radians (2 pi per convex shape)."""
        return self._core.turning

    def canonical(self) -> str:
        """The canonical serialization: coordinates to 0.1 px, a viewBox of the image's size, nothing but the subset."""
        return self._core.canonical()

    def describe(self) -> list[tuple[str, tuple[int, int, int], float]]:
        """(kind, fill, opacity) per shape, bottom first."""
        return self._core.describe()

    def render(self) -> bytes:
        """Premultiplied RGBA in 0..1, float64, `width * height * 4` values."""
        return self._core.render()

    def render_rgba8(self) -> bytes:
        """Straight-alpha 8-bit RGBA."""
        return self._core.render_rgba8()

    def coverage(self, index: int) -> bytes:
        """Shape `index` alone (nothing above it, opacity ignored): its coverage, float64, `width * height` values."""
        return self._core.coverage(index)

    def outlines(self) -> list[list[list[tuple[float, float]]]]:
        """Per shape, its flattened outline: closed polylines."""
        return self._core.outlines()


def layer_recovery(clean: Svg, output: Svg) -> list[float]:
    """For each part of `clean`, its best whole-shape IoU with any shape of `output`, each rendered alone."""
    return _native.copycat_layer_recovery(clean._core, output._core)


def hill_climb(
    rgba: bytes, width: int, height: int, *, shapes: int, candidates: int = 100, patience: int = 100, seed: int = 0
) -> Svg:
    """The `primitive`-style hill climb (rust/core/src/copycat_climb.rs): a background rect, then up to `shapes`
    opaque shapes, each the best of `candidates` random ones climbed until `patience` mutations in a row fail."""
    return Svg._wrap(_native.copycat_hill_climb(rgba, width, height, shapes, candidates, patience, seed))


def trace(
    rgba: bytes,
    width: int,
    height: int,
    *,
    colors: int = 16,
    merge_delta: float = 0.08,
    background_delta: float = 0.1,
    pixel_bits: float = 12.0,
    edge_bits: float = 2.0,
    min_area: int = 30,
    seed: int = 0,
) -> Svg:
    """The structured pipeline (rust/core/src/copycat_trace.rs): noise, background, an OKLab palette merged by
    description length, cleaned labels, parts in an order decided by code length, each fitted with the shortest shape --
    a transparent SVG with one `<g>` per part."""
    return Svg._wrap(
        _native.copycat_trace(
            rgba, width, height, colors, merge_delta, background_delta, pixel_bits, edge_bits, min_area, seed
        )
    )
