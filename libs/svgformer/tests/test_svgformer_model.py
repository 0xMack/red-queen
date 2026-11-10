import numpy as np
import pytest
from games.copycat import Svg

from svgformer import grammar
from svgformer import tokens as T
from svgformer.data import collate, synthetic_sample

torch = pytest.importorskip("torch")

from svgformer.decode import generate, to_images
from svgformer.model import SIZES, SvgFormer
from svgformer.train import SyntheticBatches, loss_and_accuracy


@pytest.fixture
def model():
    torch.manual_seed(0)
    return SvgFormer(SIZES["XS"]).eval()


def test_the_stream_is_a_pure_function_of_its_seed():
    a, b = synthetic_sample(3, 17), synthetic_sample(3, 17)
    assert a.tokens == b.tokens and np.array_equal(a.image, b.image)
    assert synthetic_sample(3, 18).seed != a.seed and synthetic_sample(4, 17).seed != a.seed
    assert a.image.shape == (3, 128, 128) and a.image.dtype == np.uint8
    assert grammar.accepts(a.tokens)


def test_batches_split_across_workers_cover_the_same_samples():
    batches = SyntheticBatches(0, 2, 1024, ("clean",))
    first = next(iter(batches))
    rgb8, tokens = collate([synthetic_sample(0, 0, ("clean",)), synthetic_sample(0, 1, ("clean",))])
    assert torch.equal(first[1], torch.from_numpy(tokens)) and torch.equal(first[0], torch.from_numpy(rgb8))


def test_the_cached_decoder_step_matches_teacher_forcing(model):
    rgb8, tokens = collate([synthetic_sample(0, i) for i in range(2)])
    images, tokens = to_images(rgb8, "cpu"), torch.from_numpy(tokens)[:, :20]
    with torch.no_grad():
        full = model(images, tokens)
        memories = model.memories(model.encode(images))
        caches = [[None, None] for _ in model.decoder]
        stepped = torch.stack([model.step(tokens[:, i], i, memories, caches) for i in range(tokens.shape[1])], dim=1)
    assert torch.allclose(full, stepped, atol=1e-5)


def test_every_decoded_sequence_parses_even_from_random_weights(model):
    rgb8, _ = collate([synthetic_sample(0, i) for i in range(4)])
    images = to_images(rgb8, "cpu")
    for temperature in (0.0, 1.0):
        generator = torch.Generator().manual_seed(1)
        for sequence in generate(model, images, max_tokens=96, temperature=temperature, generator=generator):
            Svg(T.decode(sequence, 128))  # raises if the subset refuses it
            if sequence[-1] == T.END:
                assert grammar.accepts(sequence)


def test_it_can_memorise_a_batch(model):
    torch.manual_seed(0)
    model.train()
    rgb8, tokens = collate([synthetic_sample(0, i, ("clean",)) for i in range(4)])
    images, tokens = to_images(rgb8, "cpu"), torch.from_numpy(tokens)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        loss, accuracy, _ = loss_and_accuracy(model, images, tokens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert accuracy > 0.95
    model.eval()
    sequences = generate(model, images)
    assert sum(s == tokens[i][: len(s)].tolist() for i, s in enumerate(sequences)) >= 3


def test_a_saved_checkpoint_loads_back(model):
    import io

    weights = io.BytesIO()
    torch.save({"config": model.config.to_dict(), "state_dict": model.state_dict()}, weights)
    loaded = SvgFormer.load(weights.getvalue(), "cpu")
    rgb8, tokens = collate([synthetic_sample(0, 0)])
    images, tokens = to_images(rgb8, "cpu"), torch.from_numpy(tokens)[:, :10]
    with torch.no_grad():
        assert torch.equal(model(images, tokens), loaded(images, tokens))
