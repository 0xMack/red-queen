"""The `site` target (backend.static_site): every file is what the API answers at that route, and only what a visitor's
pages ask for is copied -- no live runs, no unreferenced champions or models."""

import hashlib
import json

import pytest
from jobcore import experiments_dir
from telemetry import open_stores
from trainer import train

from publisher import publish

MODELS = "https://data.example.test/models"


def _api_path(file: str) -> str:
    """`api/runs/abc.json` -> `/runs/abc` (the route the file mirrors)."""
    return "/" + file.removeprefix("api/").removesuffix(".json")


@pytest.fixture
def site(tmp_path, monkeypatch):
    snake = train(
        {"game": "snake", "algorithm": "neuroevolution", "budget": {"generations": 2}, "params": {"population_size": 6}}
    )
    checkers = train(
        {
            "game": "checkers",
            "algorithm": "td_lambda",
            "budget": {"iterations": 1},
            "params": {"games_per_iteration": 4, "depth": 1, "monitor_games": 2, "agent": {"hidden": 4}},
        }
    )
    live = open_stores().registry.create_run({"game": "snake"})
    open_stores().registry.update_status(live, "running")
    reports = {name: _report(name, decision) for name, decision in REPORTS.items()}
    out = tmp_path / "site"
    summary = publish({"target": "site", "params": {"out": str(out), "models_base_url": MODELS}})
    return out, summary, {"snake": snake, "checkers": checkers, "live": live, **reports}


REPORTS = {
    "report-kept": "kept",
    "report-instructive": "instructive",
    "report-rejected": "not_kept",
    "report-open": None,
}


def _report(name, decision):
    """A Copy Cat report as `copycat.report` leaves it: its file, and its run."""
    root = experiments_dir()
    root.mkdir(exist_ok=True)
    groups = {"icons/train": {"mean": 1.0, "worst": 1.0, "columns": {}, "by_severity": {}, "per_image": {"1": 1.0}}}
    (root / f"{name}.json").write_text(json.dumps({"protocol": "copycat.report", "groups": groups}))
    registry = open_stores().registry
    run_id = registry.create_run({"game": "copycat", "report": name, **({"decision": decision} if decision else {})})
    registry.update_status(run_id, "completed")
    return run_id


def test_a_report_is_published_only_once_marked_kept_or_instructive(site):
    out, _, runs = site
    published = {r["run_id"] for r in json.loads((out / "api/runs.json").read_text(encoding="utf-8"))}
    assert {runs["report-kept"], runs["report-instructive"]} <= published
    assert not {runs["report-rejected"], runs["report-open"]} & published
    reports = {p.stem for p in (out / "api/games/copycat/reports").glob("*.json")}
    assert reports == {"report-kept", "report-instructive"}
    assert not (out / f"api/runs/{runs['report-rejected']}.json").exists()


def test_a_published_report_has_the_images_its_page_offers():
    from backend.static_site import LOWEST_IMAGES, _report_images

    scores = {str(seed): float(seed) for seed in range(LOWEST_IMAGES + 5)}
    report = {
        "has_svgs": True,
        "per_image": {"icons/train": scores, "icons/heldout": {"99": 0.0}},
        "compared": {"icons/train": {"moved": {str(LOWEST_IMAGES + 3): 1.0}}},
    }
    images = _report_images(report)
    assert len(images) == LOWEST_IMAGES + 1
    assert all(group == "icons/train" for group, _ in images)
    assert ("icons/train", str(LOWEST_IMAGES + 3)) in images
    assert _report_images({**report, "has_svgs": False}) == []


def test_every_file_is_what_the_api_answers(site, monkeypatch):
    """...after redaction, the one transformation (backend.static_site.redact)."""
    from backend.dependencies import _model_store, _stores
    from backend.main import app
    from backend.static_site import redact, redact_json
    from fastapi.testclient import TestClient

    out, summary, _ = site
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert summary["files"] == len(index) > 0
    monkeypatch.setenv("REDQUEEN_MODELS_BASE_URL", MODELS)
    _stores.cache_clear()
    _model_store.cache_clear()
    http = TestClient(app)
    for file, sha in index.items():
        data = (out / file).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha, file
        if not file.startswith("api/"):
            continue
        path = _api_path(file)
        if path.endswith("/package"):  # a POST, run ahead of time: the same export again answers the same
            assert redact_json(http.post(path).content) == data, file
        elif path == "/runs/summaries":  # the live run is filtered out of the copy
            continue
        else:
            response = http.get(path)
            assert response.status_code == 200, file
            if path == "/runs":
                public = [
                    r
                    for r in response.json()
                    if r["status"] != "running"
                    and ("report" not in r["config"] or r["config"].get("decision") in ("kept", "instructive"))
                ]
                assert redact(public) == json.loads(data)
            else:
                assert redact_json(response.content) == data, file


def test_only_what_pages_ask_for_is_copied(site):
    out, _, runs = site
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    published = {r["run_id"] for r in json.loads((out / "api/runs.json").read_bytes())}
    # not the run still training
    assert published == {runs["snake"], runs["checkers"], runs["report-kept"], runs["report-instructive"]}
    assert not any(runs["live"] in f for f in index)

    snake_history = json.loads((out / f"api/runs/{runs['snake']}/metrics/history.json").read_bytes())
    latest = snake_history[-1]["champion_ref"]
    artifacts = [f for f in index if f"/{runs['snake']}/artifacts/" in f]
    # only the latest champion (not generation 0's), with its package
    assert sorted(artifacts) == sorted(
        [
            f"api/runs/{runs['snake']}/artifacts/{latest}.json",
            f"api/runs/{runs['snake']}/artifacts/{latest}/package.json",
        ]
    )
    package = json.loads(
        (out / artifacts[1] if artifacts[1].endswith("package.json") else out / artifacts[0]).read_bytes()
    )
    assert package["base_url"] == MODELS
    manifest = f"models/manifests/{package['package_id']}.json"
    assert manifest in index
    blobs = json.loads((out / manifest).read_bytes())["variants"][0]["graph"]["sha256"]
    assert f"models/blobs/{blobs}" in index
    assert any(f.endswith("/brain.json") and runs["checkers"] in f for f in index)  # a Checkers champion's brain
    assert {f"api/games/{g}/leaderboard.json" for g in ("snake", "checkers", "bandit")} <= set(index)


def test_a_new_snapshot_replaces_the_old_one(site):
    out, _, _ = site
    stale = out / "api" / "runs" / "gone.json"
    stale.write_bytes(b"{}")
    publish({"target": "site", "params": {"out": str(out), "models_base_url": MODELS}})
    assert not stale.exists()


def test_no_machine_detail_is_published(site):
    """The training-cost meter records this computer's CPU, cores, OS and Python; the copy keeps only `engine`."""
    from arena.costs import hardware_fingerprint
    from backend.static_site import PUBLIC_HARDWARE_CLASS

    out, _, runs = site
    published = json.loads((out / f"api/runs/{runs['snake']}.json").read_bytes())
    hardware = published["summary"]["cost"]["hardware"]
    assert set(hardware) <= {"engine", "hardware_class"} and hardware["hardware_class"] == PUBLIC_HARDWARE_CLASS
    cpu = hardware_fingerprint()["cpu"].encode()
    index = json.loads((out / "index.json").read_bytes())
    assert not [f for f in index if cpu in (out / f).read_bytes()]


def test_a_local_identifier_fails_the_export(site, monkeypatch):
    """A file carrying the home directory (or username, hostname, CPU name...) is never kept."""
    from backend import static_site

    out, _, runs = site
    leak = "someones-desktop-7f3a"  # stands in for a username or hostname
    monkeypatch.setattr(static_site, "local_identifiers", lambda: [leak])
    real_write = static_site._Writer.write

    def write_with_leak(self, path, data):
        if path == f"api/runs/{runs['snake']}.json":
            data = data.replace(b'"game":"snake"', json.dumps({"game": "snake", "path": leak})[1:-1].encode(), 1)
        real_write(self, path, data)

    monkeypatch.setattr(static_site._Writer, "write", write_with_leak)
    with pytest.raises(static_site.PrivacyError, match=runs["snake"]):
        publish({"target": "site", "params": {"out": str(out), "models_base_url": MODELS}})
    assert not out.exists()  # nothing left for a deploy to pick up


def test_redaction_leaves_everything_else_alone():
    from backend.static_site import PUBLIC_HARDWARE_CLASS, redact_json

    program = b"r0 = r1 + r2\nr1 = r0 * x\n"  # a linear-GP champion: not JSON, published as is
    assert redact_json(program) == program
    untouched = b'{"label":"NEAT 33 \xe2\x86\x92 3","hardware_class":"kept: not inside a hardware block"}'
    assert redact_json(untouched) == untouched  # byte for byte, when there's nothing to redact
    record = {"metrics": {"training": {"hardware": {"cpu": "Some CPU", "os": "An OS", "engine": "rust"}}}}
    assert json.loads(redact_json(json.dumps(record).encode())) == {
        "metrics": {"training": {"hardware": {"engine": "rust", "hardware_class": PUBLIC_HARDWARE_CLASS}}}
    }
