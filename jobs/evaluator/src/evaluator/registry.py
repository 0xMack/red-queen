"""Which runner runs which protocol, and `evaluate()`: one EvalSpec in, its result out.

A runner is `(spec, params, sink) -> result`. A leaderboard protocol writes an `EvaluationRecord` per entrant into
the sink's evaluation store (and prunes entrants that no longer qualify) and returns a summary; a report returns its
result. `jobs/tests/test_golden_eval.py` holds every runner to what its original script produced.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jobcore import Sink, open_sink
from jobcore.specs import EvalSpec

Runner = Callable[[EvalSpec, Any, Sink], dict[str, Any]]
_RUNNERS: dict[str, Runner] = {}


def protocol(name: str) -> Callable[[Runner], Runner]:
    def register(fn: Runner) -> Runner:
        _RUNNERS[name] = fn
        return fn

    return register


def evaluate(spec: EvalSpec | dict[str, Any], sink: Sink | None = None) -> dict[str, Any]:
    """Validates `spec`, runs its protocol, returns the summary (leaderboard) or the result (report)."""
    import evaluator.protocols  # noqa: F401 -- registers every runner

    spec = spec if isinstance(spec, EvalSpec) else EvalSpec.model_validate(spec)
    _protocol, params = spec.resolve()
    try:
        run = _RUNNERS[spec.protocol]
    except KeyError:
        raise ValueError(f"no evaluator for protocol {spec.protocol}") from None
    return run(spec, params, sink or open_sink())
