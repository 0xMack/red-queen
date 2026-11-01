"""Copy Cat's research log (docs/design/0023): `copycat.report`'s reports, read from `<data dir>/experiments/`.

- `GET /games/copycat/reports`: every report, oldest first, without per-image scores;
- `GET /games/copycat/reports/{name}`: one report, its per-image scores and its comparison with its baseline;
- `PATCH /games/copycat/reports/{name}`: mark a report kept or not kept;
- `GET /games/copycat/reports/{name}/images/{set}/{split}/{seed}`: one image, as that report and its baseline traced it.

Local only: `static_site.py` doesn't export these, because publishing is outward-facing and reports name rejected work.
Held-out images aren't served one by one, and a report says whether it looked at held-out: they are scored once per change
and never debugged on.
"""

import json
from pathlib import Path
from typing import Any, Literal

from arena import copycat
from arena.copycat_sets import group_targets
from fastapi import APIRouter, HTTPException
from jobcore import experiments_dir
from pydantic import BaseModel

from backend.routers.copycat import png_url

router = APIRouter(prefix="/games/copycat/reports", tags=["copycat"])

PROTOCOL = "copycat.report"

Decision = Literal["kept", "not_kept"]


class ReportGroup(BaseModel):
    mean: float
    worst: float
    columns: dict[str, float]
    by_severity: dict[str, float]


class ReportSummary(BaseModel):
    name: str
    commit: str | None
    note: str | None
    baseline: str | None
    decision: Decision | None
    written_at: float
    held_out: bool
    groups: dict[str, ReportGroup]


class ReportComparison(BaseModel):
    before: float
    after: float
    up: int
    down: int
    moved: dict[str, float]


class ReportDetail(ReportSummary):
    per_image: dict[str, dict[str, float]]
    compared: dict[str, ReportComparison]


class ReportDecision(BaseModel):
    decision: Decision | None


class ReportImage(BaseModel):
    group: str
    seed: int
    severity: str
    png: str
    clean: str
    fidelity: float | None
    svg: str | None
    baseline: str | None
    baseline_fidelity: float | None
    baseline_svg: str | None


def _path(name: str) -> Path:
    return experiments_dir() / f"{name}.json"


def _read(path: Path) -> dict[str, Any] | None:
    """A report, or None for anything else in the directory. Older reports have no date of their own: the file's."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(report, dict) or report.get("protocol") != PROTOCOL:
        return None
    report.setdefault("written_at", path.stat().st_mtime)
    return report


def _report(name: str) -> dict[str, Any]:
    # the name of a file in that directory and nothing else: a route parameter never reaches a path unchecked
    known = {p.stem for p in experiments_dir().glob("*.json")} if experiments_dir().is_dir() else set()
    report = _read(_path(name)) if name in known else None
    if report is None:
        raise HTTPException(status_code=404, detail=f"no Copy Cat report named {name}")
    return report


def _summary(name: str, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "commit": report.get("commit"),
        "note": report.get("note"),
        "baseline": report.get("baseline"),
        "decision": report.get("decision"),
        "written_at": report["written_at"],
        "held_out": any(group.endswith("/heldout") for group in report["groups"]),
        "groups": {
            group: {k: row[k] for k in ("mean", "worst", "columns", "by_severity")}
            for group, row in report["groups"].items()
        },
    }


def _svgs(name: str) -> dict[str, dict[str, str]]:
    path = experiments_dir() / f"{name}.svgs.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@router.get("")
def list_reports() -> list[ReportSummary]:
    if not experiments_dir().is_dir():
        return []
    reports = [(p.stem, _read(p)) for p in experiments_dir().glob("*.json")]
    summaries = [_summary(name, report) for name, report in reports if report is not None]
    return [ReportSummary(**s) for s in sorted(summaries, key=lambda s: s["written_at"])]


@router.get("/{name}")
def get_report(name: str) -> ReportDetail:
    report = _report(name)
    baseline = report.get("baseline")
    before = _read(_path(baseline)) if baseline else None
    return ReportDetail(
        **_summary(name, report),
        per_image={group: row["per_image"] for group, row in report["groups"].items()},
        compared=copycat.compare(before["groups"], report["groups"]) if before else {},
    )


@router.patch("/{name}")
def set_decision(name: str, body: ReportDecision) -> ReportSummary:
    """Whether a change stays: asked of a person, usually after the report was written, so it isn't only the job's."""
    report = _report(name)
    if body.decision is None:
        report.pop("decision", None)
    else:
        report["decision"] = body.decision
    # written_at is now set from the file's date, before writing the file changes it
    _path(name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return ReportSummary(**_summary(name, report))


@router.get("/{name}/images/{set_name}/{split}/{seed}")
def get_image(name: str, set_name: str, split: str, seed: int) -> ReportImage:
    group = f"{set_name}/{split}"
    report = _report(name)
    if split == "heldout" or group not in report["groups"]:
        raise HTTPException(status_code=404, detail=f"report {name} has no images to show for {group}")
    target = next((t for t in group_targets(group, experiments_dir() / "copycat-sets") if t.seed == seed), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"{group} has no image with seed {seed}")
    baseline = report.get("baseline")
    before = _read(_path(baseline)) if baseline else None

    def fidelity(of: dict[str, Any] | None) -> float | None:
        return of["groups"].get(group, {}).get("per_image", {}).get(str(seed)) if of else None

    return ReportImage(
        group=group,
        seed=seed,
        severity=target.severity,
        png=png_url(target),
        clean=target.clean,
        fidelity=fidelity(report),
        svg=_svgs(name).get(group, {}).get(str(seed)),
        baseline=baseline,
        baseline_fidelity=fidelity(before),
        baseline_svg=_svgs(baseline).get(group, {}).get(str(seed)) if before else None,
    )
