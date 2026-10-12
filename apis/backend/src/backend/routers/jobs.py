"""Jobs, experiments, schedules and spec schemas (docs/design/0018, decisions 2 and 5) -- what the CLI and, later, MCP
drive.

POST /jobs records a validated train/evaluate/publish/pipeline spec as `queued`; the scheduler (not this API) claims and
runs it. POST /experiments stores an experiment spec, which the scheduler fans out into jobs; POST /schedules stores a
schedule, which the scheduler submits a job from whenever it falls due. GET /specs/schemas is every
payload's JSON Schema, so a client can build or check a spec without importing Python.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Query
from jobcore.specs import schemas
from telemetry import Job, JobKind, JobStatus, Schedule

from backend.dependencies import JobStoreDep, RunRegistryDep
from backend.services import experiments as experiment_service
from backend.services import jobs as job_service
from backend.services import schedules as schedule_service
from backend.services.jobs import InvalidSpec

router = APIRouter(prefix="/jobs", tags=["jobs"])
experiments_router = APIRouter(prefix="/experiments", tags=["jobs"])
schedules_router = APIRouter(prefix="/schedules", tags=["jobs"])
specs_router = APIRouter(prefix="/specs", tags=["jobs"])


@router.post("", status_code=201)
def submit_job(store: JobStoreDep, spec: Annotated[dict[str, Any], Body()]) -> Job:
    """A train, evaluate or publish spec (`kind` defaults to `train`): 422 if it doesn't validate and resolve."""
    try:
        return job_service.submit(store, spec)
    except InvalidSpec as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


@router.get("")
def list_jobs(
    store: JobStoreDep,
    status: JobStatus | None = None,
    kind: JobKind | None = None,
    parent: str | None = None,
) -> list[Job]:
    return store.list(status=status, kind=kind, parent=parent)


@router.get("/{job_id}")
def get_job(job_id: str, store: JobStoreDep) -> Job:
    try:
        return store.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}") from None


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, store: JobStoreDep) -> Job:
    """A queued job is cancelled at once; a running one is marked, and the scheduler stops its worker."""
    try:
        return store.request_cancel(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}") from None


@router.get("/{job_id}/logs")
def get_job_log(job_id: str, store: JobStoreDep, offset: Annotated[int, Query(ge=0)] = 0) -> dict[str, Any]:
    """The worker's log from byte `offset`, with the offset to ask from next: poll it to follow a running job."""
    return job_service.read_log(get_job(job_id, store), offset)


@experiments_router.post("", status_code=201)
def put_experiment(
    store: JobStoreDep, registry: RunRegistryDep, spec: Annotated[dict[str, Any], Body()]
) -> dict[str, Any]:
    """Stores (or replaces -- resubmitting resumes) an experiment spec; 422 if any arm x seed can't run."""
    try:
        experiment = experiment_service.put(store, spec)
    except InvalidSpec as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    return experiment_service.status(experiment, store, registry)


@experiments_router.get("")
def list_experiments(store: JobStoreDep) -> list[dict[str, Any]]:
    return [{"name": e.name, "created_at": e.created_at} for e in store.list_experiments()]


@experiments_router.get("/{name}")
def get_experiment(name: str, store: JobStoreDep, registry: RunRegistryDep) -> dict[str, Any]:
    """The arm x seed matrix: `completed` (a completed run, its id), a job's status (its id), or `pending`."""
    try:
        experiment = store.get_experiment(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no such experiment: {name}") from None
    return experiment_service.status(experiment, store, registry)


@experiments_router.get("/{name}/report")
def get_experiment_report(name: str) -> dict[str, Any]:
    try:
        return experiment_service.report(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'")) from None


@schedules_router.post("", status_code=201)
def put_schedule(store: JobStoreDep, spec: Annotated[dict[str, Any], Body()]) -> Schedule:
    """Stores (or replaces) a schedule spec; 422 if it has no trigger or its job can't run. Replacing one keeps when it
    last fired."""
    try:
        return schedule_service.put(store, spec)
    except InvalidSpec as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


@schedules_router.get("")
def list_schedules(store: JobStoreDep) -> list[Schedule]:
    return store.list_schedules()


@schedules_router.get("/{name}")
def get_schedule(name: str, store: JobStoreDep) -> Schedule:
    try:
        return store.get_schedule(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no such schedule: {name}") from None


@schedules_router.delete("/{name}", status_code=204)
def delete_schedule(name: str, store: JobStoreDep) -> None:
    """Stops it firing; the jobs it already submitted are untouched (GET /jobs?parent=schedule:<name>)."""
    try:
        store.delete_schedule(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no such schedule: {name}") from None


@specs_router.get("/schemas")
def get_schemas() -> dict[str, Any]:
    """JSON Schema of every payload kind, and of each algorithm's, protocol's and publish target's params."""
    return schemas()
