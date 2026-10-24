"""copycat.clean.v1's entrants: the VTracer adapter keeps VTracer's shapes, only written in the subset."""

from arena.copycat import load_set
from games.copycat import Svg

from evaluator.protocols.copycat_clean import ENTRANTS, drop_bottom, trace, vtracer_to_subset

VTRACER = """<?xml version="1.0" encoding="UTF-8"?>
<!-- Generator: visioncortex VTracer 0.6.12 -->
<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="64" height="64">
<path d="M0 0 C21.12 0 42.24 0 64 0 L64 64 L0 64 Z " fill="#F0F0F0" transform="translate(0,0)"/>
<path d="M0 0 C8.5 0 17 0 26 0 C26 8.5 26 -1.5e1 26 26 Z " fill="#143CC8" transform="translate(30,30.5)"/>
</svg>
"""


def test_the_vtracer_adapter_applies_translations():
    svg = Svg(vtracer_to_subset(VTRACER))
    assert (svg.width, svg.height, svg.shapes, svg.nodes) == (64, 64, 2, 7)
    assert '<path d="M30 30.5C38.5 30.5 47 30.5 56 30.5C56 39 56 15.5 56 56.5Z" fill="#143cc8"/>' in svg.canonical()
    assert drop_bottom(vtracer_to_subset(VTRACER)).count("<path") == 1


def test_every_entrant_writes_the_subset():
    target = load_set()[1]
    for key, (_label, _description, settings) in ENTRANTS.items():
        svg = Svg(trace(settings, target))  # raises if not in the subset
        assert (svg.width, svg.height) == (128, 128), key
