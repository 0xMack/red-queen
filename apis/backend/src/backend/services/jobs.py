"""Jobs as the API offers them: submitted only once their spec validates, read back with their worker's log.

The API never runs a job (docs/design/0018, decision 2): it records one as `queued`, and the scheduler claims it. So a
submission's only work here is making sure the scheduler will never be handed something that can't run.
"""

from __future__ import annotations

from typing import Any

from jobcore.specs import EvalSpec, PipelineSpec, PublishSpec, TrainSpec, parse_spec
from telemetry import Job, JobStore, data_dir

SUBMITTABLE = {"train": TrainSpec, "evaluate": EvalSpec, "publish": PublishSpec, "pipeline": PipelineSpec}


class InvalidSpec(ValueError):
    """A spec that doesn't validate or resolve -- the client's to fix (422)."""


def submit(store: JobStore, spec: dict[str, Any], parent: str | None = None) -> Job:
    data = {**spec, "kind": spec.get("kind", "train")}
    if data["kind"] not in SUBMITTABLE:
        raise InvalidSpec(
            f"a job is a train, evaluate, publish or pipeline spec, not {data['kind']!r} (experiments: /experiments)"
        )
    try:
        parsed = parse_spec(data)
        parsed.resolve()
    except ValueError as e:
        raise InvalidSpec(str(e)) from None
    return store.submit(parsed.kind, parsed.model_dump(mode="json", exclude_defaults=False), parent)


def read_log(job: Job, offset: int = 0) -> dict[str, Any]:
    """The worker's log from byte `offset` on, and where to ask from next -- so a client follows a running job by
    polling. Empty (offset 0) until the scheduler has started the job."""
    if not job.log:
        return {"offset": 0, "text": "", "finished": job.status in {"succeeded", "failed", "cancelled"}}
    path = data_dir() / job.log
    data = path.read_bytes() if path.exists() else b""
    chunk = data[offset:] if offset <= len(data) else b""
    return {
        "offset": offset + len(chunk),
        "text": chunk.decode("utf-8", errors="replace"),
        "finished": job.status in {"succeeded", "failed", "cancelled"},
    }
