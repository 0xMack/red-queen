"""svgformer.denoise (docs/design/0025 Decision 1): pairs drawn from svgformer's own stream, a target the corruption can't
leak into, a background the tracer can tell from the subject, and a U-Net that keeps premultiplied colour legal."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arena.copycat_sets import SIZE, corrupt_pixels
from games.copycat import Svg

from svgformer.data import synthetic_draw, synthetic_sample
from svgformer.denoise import (
    BACKGROUNDS,
    Config,
    Denoiser,
    TrainConfig,
    background_for,
    composite,
    edges,
    loss_of,
    pair,
    train,
)


def test_pair_is_svgformers_sample_and_the_clean_render():
    corrupted, target, severity = pair(3, 17)
    sample = synthetic_sample(3, 17)
    assert severity == sample.severity and np.array_equal(corrupted, sample.image)
    svg, _, _ = synthetic_draw(3, 17)
    clean = np.frombuffer(Svg(svg).render(), np.float64).reshape(SIZE, SIZE, 4)
    assert target.shape == (4, SIZE, SIZE) and target.dtype == np.uint8
    assert np.abs(target.transpose(1, 2, 0) / 255 - clean).max() <= 0.5 / 255 + 1e-9
    assert np.array_equal(pair(3, 17)[1], target)  # a pure function of (seed, i)


def test_the_clean_severity_input_is_the_target_over_a_background():
    svg, seed, _ = synthetic_draw(0, 5)
    image = np.asarray(corrupt_pixels(svg, seed, "clean")[0].convert("RGB")) / 255
    _, target, _ = pair(0, 5, ("clean",))
    rgba = target.transpose(1, 2, 0) / 255
    under = image[0, 0]  # a corner is background in synthetic
    assert np.abs(rgba[..., :3] + under * (1 - rgba[..., 3:]) - image).max() < 3 / 255


def test_background_avoids_the_subjects_colours():
    white = np.ones((8, 8, 4))
    assert not np.allclose(background_for(white), 1.0)
    grey = np.concatenate([np.full((8, 8, 3), 0.86), np.ones((8, 8, 1))], -1)
    assert not np.allclose(background_for(grey), 0.86)
    assert np.allclose(background_for(np.zeros((8, 8, 4))), BACKGROUNDS[0])
    rgba = np.frombuffer(composite(np.zeros((4, 4, 4))), np.uint8).reshape(4, 4, 4)
    assert (rgba[..., 3] == 255).all() and (rgba[..., :3] == round(BACKGROUNDS[0][0] * 255)).all()


def test_denoiser_output_is_premultiplied_and_round_trips():
    torch.manual_seed(0)
    model = Denoiser(Config(width=8, levels=3, blocks=1))
    out = model(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 4, 32, 32)
    assert (out[:, :3] <= out[:, 3:] + 1e-6).all() and (out >= 0).all() and (out <= 1).all()
    again = Denoiser.load(model.to_bytes())
    assert torch.equal(again(torch.ones(1, 3, 32, 32)), model.eval()(torch.ones(1, 3, 32, 32)))


def test_edges_and_loss():
    target = torch.zeros(1, 4, 8, 8)
    target[..., 4:] = 1
    e = edges(target)
    assert e[0, 0, 0].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]  # the step at x = 4, dilated by a pixel
    assert loss_of(target, target, 4.0) == 0
    assert loss_of(torch.zeros_like(target), target, 0.0) == pytest.approx(0.5)


def test_train_runs_and_reports():
    seen = []
    cfg = TrainConfig(width=8, levels=2, blocks=1, steps=4, batch=2, workers=0, log_every=2, warmup=1)
    train(cfg, "cpu", lambda model, g: seen.append(g.step))
    assert seen == [2, 4]
