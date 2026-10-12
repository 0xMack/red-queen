"""Schedules as the API offers them: stored once their spec validates; the scheduler fires them (docs/design/0018)."""

from __future__ import annotations

from typing import Any

from jobcore.specs import ScheduleSpec
from telemetry import JobStore, Schedule

from backend.services.jobs import InvalidSpec


def put(store: JobStore, spec: dict[str, Any]) -> Schedule:
    try:
        parsed = ScheduleSpec.model_validate({**spec, "kind": "schedule"}).resolve()
    except ValueError as e:
        raise InvalidSpec(str(e)) from None
    return store.put_schedule(parsed.name, parsed.model_dump(mode="json"))
