import numpy as np
import pytest

from tinylm import Adam, CharTokenizer, TinyLM, cross_entropy, generate


def _tiny_model(rng, vocab_size=6, max_seq_len=4):
    return TinyLM(
        vocab_size=vocab_size, max_seq_len=max_seq_len, d_model=8, n_heads=2, n_layers=2, d_hidden=16, rng=rng
    )


def test_model_forward_shape():
    rng = np.random.default_rng(0)
    model = _tiny_model(rng)
    ids = rng.integers(0, 6, size=(3, 4))

    logits = model(ids)

    assert logits.shape == (3, 4, 6)


def test_model_rejects_a_sequence_longer_than_max_seq_len():
    rng = np.random.default_rng(1)
    model = _tiny_model(rng, max_seq_len=4)

    with pytest.raises(ValueError):
        model(rng.integers(0, 6, size=(1, 5)))


def test_all_parameters_receive_a_gradient_after_backward():
    rng = np.random.default_rng(2)
    model = _tiny_model(rng)
    ids = rng.integers(0, 6, size=(2, 4))

    loss = cross_entropy(model(ids), ids)
    loss.backward()

    assert all(np.any(p.grad != 0) for p in model.parameters())


def test_cross_entropy_matches_hand_computed_value_for_a_uniform_prediction():
    # logits are all equal -> softmax is uniform -> cross-entropy of a uniform distribution over
    # `vocab_size` classes is exactly ln(vocab_size), regardless of what the target is.
    from autodiff import Tensor

    vocab_size = 5
    logits = Tensor(np.zeros((1, 3, vocab_size)))
    targets = np.array([[0, 2, 4]])

    loss = cross_entropy(logits, targets)

    assert np.isclose(loss.data, np.log(vocab_size), atol=1e-6)


def test_adam_step_changes_parameters_and_reduces_loss():
    rng = np.random.default_rng(3)
    model = _tiny_model(rng)
    optimizer = Adam(model.parameters(), lr=1e-2)
    ids = rng.integers(0, 6, size=(4, 4))

    first_weight = model.token_embedding.weight.data.copy()
    losses = []
    for _ in range(20):
        loss = cross_entropy(model(ids), ids)
        losses.append(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert not np.allclose(model.token_embedding.weight.data, first_weight)
    assert losses[-1] < losses[0]


def test_generate_produces_the_requested_length_using_known_vocabulary():
    rng = np.random.default_rng(4)
    text = "the quick brown fox"
    tokenizer = CharTokenizer(text)
    model = _tiny_model(rng, vocab_size=tokenizer.vocab_size, max_seq_len=8)

    result = generate(model, tokenizer, prompt="the", max_new_tokens=10, rng=rng)

    assert len(result) == len("the") + 10
    assert set(result) <= set(text)
