"""Every evaluation, report and publish job still produces what it did when golden_eval.py's fixture was recorded --
the safety net for porting them to the evaluator and publisher workloads (docs/design/0018 stage 4)."""

import json
import os
import sys
from pathlib import Path

import pytest
from golden_eval import EVAL_CASES, FIXTURE, load_fixture, matches, run_case

EXPECTED = load_fixture() if FIXTURE.exists() else None
OUT = os.environ.get("REDQUEEN_GOLDEN_OUT")  # CI uploads it on failure: how a platform's fixture is recorded


@pytest.mark.parametrize("name", sorted(EVAL_CASES))
def test_golden_evaluation_is_reproduced(name):
    if EXPECTED is None and not OUT:
        pytest.skip(f"no golden evaluation fixture for {sys.platform} ({FIXTURE.name})")
    actual = run_case(name)
    if OUT:
        Path(OUT).mkdir(parents=True, exist_ok=True)
        (Path(OUT) / f"eval-{name}.json").write_text(json.dumps(actual, indent=1, sort_keys=True), encoding="utf-8")
    assert EXPECTED is not None, f"no golden evaluation fixture for {sys.platform}: this run's digests are in {OUT}"
    assert name in EXPECTED, f"no golden digest for {name}: record it with golden_eval.py --record {name}"
    assert matches(name, actual, EXPECTED[name])
