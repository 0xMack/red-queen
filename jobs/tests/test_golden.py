"""Every training job still produces exactly what it did when the golden fixture was recorded (golden_cases.py): the
same config, curve and champion bytes. The safety net for porting jobs to the trainer workload (docs/design/0018)."""

import json
import os
import sys
from pathlib import Path

import pytest
from golden_cases import CASES, FIXTURE, TOLERANT, close, load_fixture, run_case

EXPECTED = load_fixture() if FIXTURE.exists() else None
# CI sets this and uploads the directory when the test fails: how a platform's fixture is recorded (golden_cases.py).
OUT = os.environ.get("REDQUEEN_GOLDEN_OUT")


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_run_is_reproduced_exactly(name):
    if EXPECTED is None and not OUT:
        pytest.skip(f"no golden fixture for {sys.platform} ({FIXTURE.name})")
    actual = run_case(name)
    if OUT:
        Path(OUT).mkdir(parents=True, exist_ok=True)
        (Path(OUT) / f"{name}.json").write_text(json.dumps(actual, indent=1, sort_keys=True), encoding="utf-8")
    assert EXPECTED is not None, f"no golden fixture for {sys.platform}: this run's digests are in {OUT}"
    assert name in EXPECTED, f"no golden digest for {name}: record it with golden_cases.py --record {name}"
    if name in TOLERANT:
        assert close(actual, EXPECTED[name]), f"{name} differs beyond 1e-9: {actual} != {EXPECTED[name]}"
    else:
        assert actual == EXPECTED[name]
