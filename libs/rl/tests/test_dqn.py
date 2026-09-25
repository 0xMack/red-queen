"""DQN: the Rust update rule against the `libs/autodiff` oracle (reference_dqn.py), and training through the Trainer."""

import json

import numpy as np
import pytest
from reference_dqn import td_gradients

import rl
from rl import _native

INPUTS, HIDDEN, ACTIONS = 7, [12, 9], 3


@pytest.mark.parametrize(("dueling", "double"), [(False, False), (True, False), (False, True), (True, True)])
def test_td_gradients_match_autodiff(dueling, double):
    rng = np.random.default_rng(10 + 2 * dueling + double)
    online = [np.array(p) + rng.normal(0, 0.1, len(p)) for p in _native.dqn_init(INPUTS, HIDDEN, ACTIONS, dueling, 1)]
    target = [np.array(p) + rng.normal(0, 0.1, len(p)) for p in _native.dqn_init(INPUTS, HIDDEN, ACTIONS, dueling, 2)]
    size = 16
    batch = (
        rng.normal(size=(size, INPUTS)),
        rng.integers(0, ACTIONS, size),
        rng.normal(0, 3, size),  # wide enough that some td errors are past Huber's kink
        rng.choice([0.0, 0.95, 0.95**3], size),
        rng.normal(size=(size, INPUTS)),
        rng.uniform(0.2, 1.0, size),
    )
    loss, grads, td, q_mean = _native.dqn_td_gradients(
        INPUTS,
        HIDDEN,
        ACTIONS,
        dueling,
        double,
        [p.tolist() for p in online],
        [p.tolist() for p in target],
        tuple(np.asarray(x).ravel().tolist() for x in batch),
    )
    e_loss, e_grads, e_td, e_q = td_gradients(INPUTS, HIDDEN, ACTIONS, dueling, double, online, target, batch)
    assert np.any(np.abs(e_td) > 1) and np.any(np.abs(e_td) < 1), "both sides of the Huber kink are exercised"
    assert loss == pytest.approx(e_loss, rel=1e-12)
    assert q_mean == pytest.approx(e_q, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(td, e_td, rtol=1e-12, atol=1e-12)
    assert len(grads) == (3 if dueling else 1)
    for got, expected in zip(grads, e_grads, strict=True):
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_double_changes_the_target_only_through_the_argmax():
    rng = np.random.default_rng(3)
    online = [np.array(p) for p in _native.dqn_init(INPUTS, HIDDEN, ACTIONS, False, 1)]
    batch = tuple(
        np.asarray(x).ravel().tolist()
        for x in (
            rng.normal(size=(8, INPUTS)),
            rng.integers(0, ACTIONS, 8),
            rng.normal(size=8),
            np.full(8, 0.9),
            rng.normal(size=(8, INPUTS)),
            np.ones(8),
        )
    )
    params = [p.tolist() for p in online]
    # with the target network *being* the online one, choosing by either is the same
    single = _native.dqn_td_gradients(INPUTS, HIDDEN, ACTIONS, False, False, params, params, batch)
    double = _native.dqn_td_gradients(INPUTS, HIDDEN, ACTIONS, False, True, params, params, batch)
    assert single[0] == double[0] and single[2] == double[2]


def test_dqn_trains_through_the_trainer_and_snapshots_an_mlp():
    trainer = rl.Trainer(
        "dqn",
        "snake/features.v1+relative3.v1",
        seed=0,
        params={"learn_start": 500, "hidden": 32, "dueling": 1, "prioritized": 1, "n_step": 3},
    )
    stats = trainer.train(3_000)
    assert stats["total_steps"] == 3_000
    extras = stats["extras"]
    assert extras["updates"] == (3_000 - 500) // 4 + 1  # every 4th step from step 500 on
    assert 3_000 - 2 <= extras["replay_size"] <= 3_000  # every step, minus up to n-1 still pending
    assert np.isfinite(extras["td_loss"]) and np.isfinite(extras["q_mean"])
    snapshot = json.loads(trainer.snapshot())
    assert snapshot["type"] == "mlp" and snapshot["algorithm"] == "dqn"
    assert snapshot["layer_sizes"] == [11, 32, 32, 3]
    assert snapshot["activations"] == ["relu", "relu", "linear"]
    assert len(snapshot["params"]) == 11 * 32 + 32 + 32 * 32 + 32 + 32 * 3 + 3
    assert len(trainer.evaluate([1, 2], 200)) == 2


def test_dqn_rejects_what_it_cant_do():
    with pytest.raises(ValueError, match="no parameter"):
        rl.Trainer("dqn", "snake/features.v1+relative3.v1", seed=0, params={"alpha": 0.1})
    with pytest.raises(ValueError, match="discrete"):
        rl.Trainer("dqn", "reach1d", seed=0)
