"""Policy gradients: the Rust update against the `libs/autodiff` oracle (reference_pg.py), and training through the
Trainer."""

import json

import numpy as np
import pytest
from reference_pg import gae, policy_loss_and_gradients

import rl
from rl import _native


@pytest.mark.parametrize(("discrete", "clip"), [(True, None), (True, 0.2), (False, None), (False, 0.2)])
def test_policy_gradients_match_autodiff(discrete, clip):
    rng = np.random.default_rng(20 + 2 * discrete + (clip is not None))
    layer_sizes = [5, 8, 6, 3 if discrete else 1]
    params = np.array(_native.mlp_init(layer_sizes, ["tanh", "tanh", "linear"], 4))
    params = params + rng.normal(0, 0.1, params.shape)
    n = 12
    observations = rng.normal(size=(n, 5))
    actions = rng.integers(0, 3, n).astype(float) if discrete else rng.normal(size=n)
    old_log_probs = rng.normal(-1.1, 0.4, n)  # wide enough that some PPO ratios leave [0.8, 1.2]
    advantages = rng.normal(size=n)
    log_std = None if discrete else -0.4
    loss, grads, entropy, kl, clip_fraction = _native.policy_gradients(
        layer_sizes,
        params.tolist(),
        log_std,
        observations.ravel().tolist(),
        actions.tolist(),
        old_log_probs.tolist(),
        advantages.tolist(),
        clip,
        0.03,
    )
    expected_loss, expected_grads = policy_loss_and_gradients(
        layer_sizes, params, log_std, observations, actions, old_log_probs, advantages, clip, 0.03
    )
    assert loss == pytest.approx(expected_loss, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(grads, expected_grads, rtol=1e-10, atol=1e-12)
    assert len(grads) == len(params) + (0 if discrete else 1)
    assert entropy > 0 and np.isfinite(kl)
    if clip is not None:
        assert 0 < clip_fraction < 1, "both clipped and unclipped samples are exercised"


def test_gae_matches_the_textbook_recursion():
    rng = np.random.default_rng(5)
    n = 40
    rewards, values, next_values = (rng.normal(size=n).tolist() for _ in range(3))
    done = [bool(x) for x in rng.random(n) < 0.08]
    cut = [d or bool(t) for d, t in zip(done, rng.random(n) < 0.05, strict=True)]  # some episodes truncated
    for lam in (0.0, 0.95, 1.0):
        got = _native.generalized_advantages(rewards, values, next_values, done, cut, 0.97, lam)
        expected = gae(rewards, values, next_values, done, cut, 0.97, lam)
        np.testing.assert_allclose(got[0], expected[0], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(got[1], expected[1], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("algorithm", ["reinforce", "a2c", "ppo"])
def test_policy_gradient_agents_train_through_the_trainer(algorithm):
    trainer = rl.Trainer(
        algorithm, "snake/egocentric.v1+relative3.v1", seed=0, params={"hidden": 16, "rollout_steps": 256}
    )
    stats = trainer.train(3_000)
    extras = stats["extras"]
    assert extras["updates"] > 0 and np.isfinite(extras["policy_loss"])
    assert 0 < stats["entropy"] <= np.log(3) + 1e-9, "the entropy of a distribution over three moves"
    snapshot = json.loads(trainer.snapshot())
    assert snapshot["type"] == "mlp" and snapshot["algorithm"] == algorithm
    assert snapshot["layer_sizes"] == [27, 16, 16, 3] and snapshot["activations"] == ["tanh", "tanh", "linear"]


def test_a_continuous_policy_on_reach1d_snapshots_its_spread():
    trainer = rl.Trainer("ppo", "reach1d", seed=0, params={"hidden": 16, "rollout_steps": 256}, max_episode_steps=200)
    stats = trainer.train(2_000)
    assert "action_std" in stats["extras"]
    assert "log_std" in json.loads(trainer.snapshot())
