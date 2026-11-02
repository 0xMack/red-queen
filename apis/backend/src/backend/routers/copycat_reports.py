"""What a Copy Cat report's run (docs/design/0023, `config.report`) doesn't hold, read from `<data dir>/experiments/`:

- `GET /games/copycat/reports/{name}`: its per-image scores and its comparison with its baseline;
- `GET /games/copycat/reports/{name}/images/{set}/{split}/{seed}`: one image, as that report and its baseline traced it.

The run itself carries the note, the group means and the decision. Held-out images aren't served one by one: they are
scored once per change and never debugged on.
"""

import json
from pathlib import Path
from typing import Any

from arena import copycat
from arena.copycat_sets import group_targets
from fastapi import APIRouter, HTTPException
from jobcore import experiments_dir
from pydantic import BaseModel

from backend.routers.copycat import png_url

router = APIRouter(prefix="/games/copycat/reports", tags=["copycat"])

PROTOCOL = "copycat.report"


class ReportComparison(BaseModel):
    before: float
    after: float
    up: int
    down: int
    moved: dict[str, float]


class ReportDetail(BaseModel):
    name: str
    baseline: str | None
    has_svgs: bool
    per_image: dict[str, dict[str, float]]
    compared: dict[str, ReportComparison]


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
    """A report, or None for anything else in the directory."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return report if isinstance(report, dict) and report.get("protocol") == PROTOCOL else None


def _report(name: str) -> dict[str, Any]:
    # the name of a file in that directory and nothing else: a route parameter never reaches a path unchecked
    known = {p.stem for p in experiments_dir().glob("*.json")} if experiments_dir().is_dir() else set()
    report = _read(_path(name)) if name in known else None
    if report is None:
        raise HTTPException(status_code=404, detail=f"no Copy Cat report named {name}")
    return report


def _svgs_path(name: str) -> Path:
    return experiments_dir() / f"{name}.svgs.json"


def _svgs(name: str) -> dict[str, dict[str, str]]:
    path = _svgs_path(name)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@router.get("/{name}")
def get_report(name: str) -> ReportDetail:
    report = _report(name)
    baseline = report.get("baseline")
    before = _read(_path(baseline)) if baseline else None
    return ReportDetail(
        name=name,
        baseline=baseline,
        has_svgs=_svgs_path(name).exists(),
        per_image={group: row["per_image"] for group, row in report["groups"].items()},
        compared=copycat.compare(before["groups"], report["groups"]) if before else {},
    )


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
