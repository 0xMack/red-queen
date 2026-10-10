"""Which adapter runs which algorithm, and `train()`: one TrainSpec in, one recorded run (its id) out.

An adapter is `(spec, params, sink) -> run_id`: `params` is the spec's params already validated against the model
the algorithm registered in `jobcore.algorithms`. An adapter writes the run config the original script wrote -- key
for key, since the frontend and the leaderboard jobs read it -- which `jobs/tests/test_golden.py` holds it to.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jobcore import Sink, open_sink
from jobcore.specs import TrainSpec

Adapter = Callable[[TrainSpec, Any, Sink], Any]
_ADAPTERS: dict[tuple[str, str], Adapter] = {}


def adapter(algorithm: str, *games: str) -> Callable[[Adapter], Adapter]:
    def register(fn: Adapter) -> Adapter:
        for game in games:
            _ADAPTERS[(algorithm, game)] = fn
        return fn

    return register


def train(spec: TrainSpec | dict[str, Any], sink: Sink | None = None) -> str:
    """Validates `spec`, runs it, returns the run id (a pipeline stage that records several runs returns the last)."""
    import trainer.algorithms  # noqa: F401 -- registers every adapter

    spec = spec if isinstance(spec, TrainSpec) else TrainSpec.model_validate(spec)
    _algorithm, params = spec.resolve()
    try:
        run = _ADAPTERS[(spec.algorithm, spec.game)]
    except KeyError:
        raise ValueError(f"no trainer adapter for {spec.algorithm} on {spec.game}") from None
    return run(spec, params, sink or open_sink())


def held_out_every(spec: TrainSpec, default: int) -> int:
    return spec.held_out_every if spec.held_out_every is not None else default
