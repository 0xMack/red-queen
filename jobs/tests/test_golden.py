"""Every training job still produces exactly what it did when the golden fixture was recorded (golden_cases.py): the
same config, curve and champion bytes. The safety net for porting jobs to the trainer workload (docs/design/0018)."""

import pytest
from golden_cases import CASES, load_fixture, run_case

FIXTURE = load_fixture()


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_run_is_reproduced_exactly(name):
    assert name in FIXTURE, f"no golden digest for {name}: record it with golden_cases.py --record {name}"
    assert run_case(name) == FIXTURE[name]
