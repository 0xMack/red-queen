import random
from pathlib import Path

import pytest
from arena.copycat_sets import SPLITS, synthetic
from games.copycat import Svg

from svgformer import grammar
from svgformer import tokens as T

ARTWORK = Path(__file__).resolve().parents[3] / "data" / "experiments" / "copycat-sets"


def test_the_grid_is_an_eighth_of_a_pixel_at_128_and_a_quarter_at_256():
    assert T.Grid(128).step == 0.125
    assert T.Grid(256).step == 0.25
    grid = T.Grid(128)
    for i in (0, 1, 64, 1279):
        assert grid.index(grid.value(i)) == (i, False)
    assert grid.index(-100.0) == (0, True)
    assert grid.index(1000.0) == (T.GRID - 1, True)


def test_the_vocabulary_is_laid_out_once():
    assert len(set(T.NAMES)) == T.VOCAB == 3 + 5 + 5 + 2 + T.OPACITIES + T.COLOURS + T.GRID


@pytest.mark.parametrize("seed", list(SPLITS["train"]))
def test_synthetic_round_trips_and_the_grammar_accepts_it(seed):
    svg, _ = synthetic(seed)
    encoded = T.encode(svg, 128)
    assert encoded.clamped == 0
    assert grammar.accepts(encoded.tokens), T.describe(encoded.tokens)
    decoded = T.decode(encoded.tokens, 128)
    # quantized once, a decoded SVG encodes to the same tokens: the round trip is idempotent
    assert T.encode(decoded, 128).tokens == encoded.tokens
    assert Svg(decoded).shapes == Svg(svg).shapes


def test_opacity_and_holes_are_tokens():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">'
        '<path d="M10 10L100 10L100 100L10 100ZM30 30L30 80L80 80L80 30Z" fill="#336699" fill-opacity="0.5"/>'
        "</svg>"
    )
    encoded = T.encode(svg, 128)
    assert grammar.accepts(encoded.tokens)
    names = T.describe(encoded.tokens).split()
    assert names[:3] == ["BOS", "path", "M"] and names[-6:] == [
        "PATH_END",
        "COLOUR_13",
        "COLOUR_25",
        "COLOUR_38",
        "OPACITY_8",
        "END",
    ]
    assert names.count("M") == 2 and names.count("Z") == 2
    assert T.encode(T.decode(encoded.tokens, 128), 128).tokens == encoded.tokens


def _walk(rng: random.Random, cap: int) -> list[int]:
    """A random sequence the grammar allows, ending early at random (as a capped decode would)."""
    state = grammar.START
    out = []
    while state.kind != "done" and len(out) < cap:
        choices = sorted(grammar.allowed(state))
        # lean toward finishing shapes, so sequences are varied but not all cut off
        structural = [t for t in choices if not (T.is_number(t) or T.is_colour(t))]
        if structural and rng.random() < 0.3:
            token = rng.choice(structural)
        else:
            token = rng.choice(choices)
        out.append(token)
        state = grammar.advance(state, token)
    return out


def test_every_sequence_the_grammar_allows_decodes():
    rng = random.Random(0)
    complete = 0
    for _ in range(3000):
        sequence = _walk(rng, rng.choice([16, 64, 256]))
        svg = T.decode(sequence, 128)  # raises if the subset refuses it
        Svg(svg)
        complete += grammar.accepts(sequence)
    assert complete > 300  # plenty of the walks reached END


def test_the_grammar_refuses_what_wouldnt_decode():
    for bad in (
        ["BOS", "polygon", "NUM_1", "NUM_2", "POLY_END"],  # one point
        ["BOS", "path", "L"],  # no M first
        ["BOS", "path", "M", "NUM_1", "NUM_2", "Z"],  # nothing drawn
        ["BOS", "circle", "NUM_1", "NUM_2", "NUM_3", "END"],  # no colour
    ):
        assert not grammar.accepts([T.ID[n] for n in bad])


def test_masks_match_the_allowed_sets():
    state = grammar.advance(grammar.START, T.BOS)
    mask = grammar.mask(state)
    assert set(mask.nonzero()[0]) == set(grammar.allowed(state))


@pytest.mark.skipif(not ARTWORK.exists(), reason="the artwork sets aren't downloaded here")
@pytest.mark.parametrize("set_name", ["icons", "broad"])
def test_artwork_round_trips(set_name):
    files = sorted((ARTWORK / set_name / "clean").glob("*.svg"))
    if not files:
        pytest.skip(f"{set_name} isn't built here")
    refused = []
    for path in files:
        encoded = T.encode(path.read_text(encoding="utf-8"), 128)
        if not grammar.accepts(encoded.tokens):
            refused.append(path.name)
            continue
        assert T.encode(T.decode(encoded.tokens, 128), 128).tokens == encoded.tokens, path.name
    assert not refused, refused[:10]
