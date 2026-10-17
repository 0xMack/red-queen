"""Which function publishes which target, and `publish()`: one PublishSpec in, a summary of what was stored out.

A target function is `(spec, params, store) -> summary`, given the local model store (docs/design/0009: the store's
eventual host, R2 or the HF Hub, is still undecided, so the publisher writes the local one). With
`collect_garbage`, files no catalog references any more are deleted afterwards.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jobcore import Sink, open_sink
from jobcore.specs import PublishSpec
from modelpack import LocalModelStore
from telemetry import models_dir

Target = Callable[[PublishSpec, Any, Sink, LocalModelStore], dict[str, Any]]
_TARGETS: dict[str, Target] = {}


def target(name: str) -> Callable[[Target], Target]:
    def register(fn: Target) -> Target:
        _TARGETS[name] = fn
        return fn

    return register


def publish(spec: PublishSpec | dict[str, Any], sink: Sink | None = None) -> dict[str, Any]:
    import publisher.frontend
    import publisher.models
    import publisher.scale_test
    import publisher.site  # noqa: F401

    spec = spec if isinstance(spec, PublishSpec) else PublishSpec.model_validate(spec)
    _target, params = spec.resolve()
    try:
        run = _TARGETS[spec.target]
    except KeyError:
        raise ValueError(f"no publisher for target {spec.target}") from None
    store = LocalModelStore(models_dir())
    summary = run(spec, params, sink or open_sink(), store)
    if getattr(params, "collect_garbage", False):  # model-store targets; a frontend file isn't in the store
        removed, freed = store.collect_garbage()
        summary["garbage_collected"] = {"files": removed, "bytes": freed}
        print(f"garbage collected {removed} superseded files ({freed / 1e6:.1f} MB)")
    return summary
