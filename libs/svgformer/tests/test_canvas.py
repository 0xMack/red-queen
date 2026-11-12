import dataclasses

import numpy as np
import pytest
from arena.copycat_sets import SIZE, synthetic
from games.copycat import Svg

from svgformer import grammar
from svgformer import tokens as T
from svgformer.canvas import CHANNELS, inputs, prefix_sample, prefix_svg, rgba_array


def test_a_prefix_sample_is_the_next_shape_of_a_drawn_prefix():
    for i in range(20):
        s = prefix_sample(0, i)
        again = prefix_sample(0, i)  # a pure function of (stream seed, i)
        assert (s.image == again.image).all() and s.tokens == again.tokens
        assert s.image.shape == (CHANNELS, SIZE, SIZE) and s.image.dtype == np.uint8
        assert grammar.accepts(s.tokens)
        assert 0 <= s.shapes <= s.total
        if s.shapes == s.total:
            assert s.tokens == [T.BOS, T.END]
        else:
            assert len(T._shapes(s.tokens)) == 1
        alpha = s.image[6]
        assert (alpha.max() == 0) == (s.shapes == 0)  # nothing drawn on an empty canvas


def test_the_residual_is_what_the_canvas_has_not_drawn():
    svg, _ = synthetic(1_000_000)
    shapes = T._shapes(T.encode(svg, SIZE).tokens)
    canvas = rgba_array(Svg(prefix_svg(shapes)).render_rgba8(), SIZE, SIZE)
    under = np.full((SIZE, SIZE, 3), 240, np.uint8)
    alpha = canvas[..., 3:] / 255
    target = np.rint(canvas[..., :3] * alpha + under * (1 - alpha)).astype(np.uint8)
    x = inputs(target, under, canvas)
    assert np.abs(x[7:].astype(int) - 128).max() <= 1  # all drawn: no residual
    empty = inputs(target, under, np.zeros((SIZE, SIZE, 4), np.uint8))
    assert (empty[3:6] == under.transpose(2, 0, 1)).all() and empty[6].max() == 0


torch = pytest.importorskip("torch")

from svgformer.canvas import draw
from svgformer.decode import generate
from svgformer.model import SIZES, SvgFormer


@pytest.fixture
def model():
    torch.manual_seed(0)
    return SvgFormer(dataclasses.replace(SIZES["XS"], canvas=True)).eval()


def test_one_shape_decoding_writes_one_shape_or_none(model):
    images = torch.rand(6, CHANNELS, SIZE, SIZE)
    for sequence in generate(model, images, max_tokens=256, temperature=1.0, one_shape=True):
        if sequence[-1] == T.END:
            assert grammar.accepts(sequence)
            assert len(T._shapes(sequence)) <= 1


def test_drawing_only_keeps_shapes_that_improve_the_fit(model):
    svg, _ = synthetic(1_000_000)
    rgba = Svg(svg).render_rgba8()
    px = rgba_array(rgba, SIZE, SIZE).astype(float)
    a = px[..., 3:] / 255
    opaque = np.concatenate([np.rint(px[..., :3] * a + 245 * (1 - a)), np.full((SIZE, SIZE, 1), 255)], -1)
    image = (opaque.astype(np.uint8).tobytes(), SIZE, SIZE)
    drawn = draw(model, [image], "cpu", samples=2, max_shapes=4)[0]
    assert Svg(drawn).shapes <= 4
    loose = draw(model, [image], "cpu", samples=2, max_shapes=4, verify=False)[0]
    assert Svg(loose).shapes >= Svg(drawn).shapes
