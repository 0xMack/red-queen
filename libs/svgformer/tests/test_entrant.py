"""svgformer.entrant's pick: the tracer's SVG only when it fits the input twice as well (numpy only, no PyTorch)."""

import pytest
from games.copycat import Svg

from svgformer.entrant import misfit, pick


def _over_white(svg: str) -> bytes:
    rgba = Svg(svg).render_rgba8()
    out = bytearray(rgba)
    for i in range(0, len(rgba), 4):
        a = rgba[i + 3] / 255
        for k in range(3):
            out[i + k] = round(rgba[i + k] * a + 255 * (1 - a))
        out[i + 3] = 255
    return bytes(out)


def _rect(x: float, width: float = 12) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect x="{x}" y="8" width="{width}" '
        'height="14" fill="#c03020"/></svg>'
    )


def test_misfit_is_zero_for_the_image_s_own_svg():
    image = _over_white(_rect(10))
    assert misfit(_rect(10), image, 32, 32) == pytest.approx(0, abs=1e-5)
    assert misfit(_rect(12), image, 32, 32) > 0.01


def test_the_tracer_needs_twice_the_fit():
    image = _over_white(_rect(10))
    near, far = _rect(10.5), _rect(14)
    assert pick(far, _rect(10), image, 32, 32) == "tracer"
    assert pick(_rect(10), far, image, 32, 32) == "model"
    a, b = misfit(near, image, 32, 32), misfit(_rect(10.6), image, 32, 32)
    assert a < b < 2 * a  # better, but not twice as good
    assert pick(_rect(10.6), near, image, 32, 32) == "model"
