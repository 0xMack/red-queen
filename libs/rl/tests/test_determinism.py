"""The native build reproduces the checked-in determinism digests -- the same ones the WASM build must reproduce
in Node and in a browser (make_determinism_fixture.py)."""

import json

from make_determinism_fixture import FIXTURE, digests


def test_native_build_reproduces_the_fixture():
    assert digests() == json.loads(FIXTURE.read_text(encoding="utf-8")), (
        "the core's arithmetic changed: if that's intended, regenerate with make_determinism_fixture.py "
        "and rebuild the WASM"
    )
