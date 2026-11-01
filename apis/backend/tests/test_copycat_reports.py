"""The research log's routes read `copycat.report`'s files, old and new, and leave everything else in the directory alone."""

import json

import pytest
from fastapi.testclient import TestClient

from backend.main import app


def _group(mean, **per_image):
    return {
        "mean": mean,
        "worst": min(per_image.values()),
        "columns": {"fidelity": mean, "layer_recovery": 0.5},
        "by_severity": {"clean": mean},
        "per_image": per_image,
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REDQUEEN_DATA_DIR", str(tmp_path))
    root = tmp_path / "experiments"
    root.mkdir()
    old = {
        "protocol": "copycat.report",
        "commit": "a",
        "note": "old",
        "groups": {"icons/train": _group(20.0, a=20.0, b=20.0)},
    }
    (root / "old.json").write_text(json.dumps(old))
    new = {
        "protocol": "copycat.report",
        "commit": "b",
        "note": "new",
        "baseline": "old",
        "written_at": 2e10,
        "groups": {"icons/train": _group(21.0, a=22.0, b=20.0), "icons/heldout": _group(19.0, c=19.0)},
    }
    (root / "new.json").write_text(json.dumps(new))
    (root / "new.svgs.json").write_text(json.dumps({"icons/train": {"a": "<svg/>"}}))
    (root / "other.json").write_text(json.dumps({"protocol": "experiment.snake-report"}))
    return TestClient(app)


def test_the_list_is_every_report_oldest_first_without_per_image_scores(client):
    reports = client.get("/games/copycat/reports").json()
    assert [r["name"] for r in reports] == ["old", "new"]
    assert (reports[0]["decision"], reports[0]["held_out"], reports[1]["held_out"]) == (None, False, True)
    assert "per_image" not in reports[0]["groups"]["icons/train"]


def test_a_report_says_what_it_changed_against_its_baseline(client):
    report = client.get("/games/copycat/reports/new").json()
    assert report["per_image"]["icons/train"] == {"a": 22.0, "b": 20.0}
    row = report["compared"]["icons/train"]
    assert (row["before"], row["after"], row["up"], row["down"], row["moved"]) == (20.0, 21.0, 1, 0, {"a": 2.0})
    assert client.get("/games/copycat/reports/old").json()["compared"] == {}


@pytest.mark.parametrize("name", ["missing", "new.svgs", "other", "..%2Fsecret"])
def test_only_a_report_is_a_report(client, name):
    assert client.get(f"/games/copycat/reports/{name}").status_code == 404


def test_a_decision_is_kept_with_the_report_and_keeps_its_date(client):
    before = next(r for r in client.get("/games/copycat/reports").json() if r["name"] == "old")["written_at"]
    assert client.patch("/games/copycat/reports/old", json={"decision": "not_kept"}).json()["decision"] == "not_kept"
    after = client.get("/games/copycat/reports/old").json()
    assert (after["decision"], after["written_at"]) == ("not_kept", before)
    assert client.patch("/games/copycat/reports/old", json={"decision": None}).json()["decision"] is None
    assert client.patch("/games/copycat/reports/old", json={"decision": "maybe"}).status_code == 422


def test_held_out_images_are_not_served_one_by_one(client):
    assert client.get("/games/copycat/reports/new/images/icons/heldout/10050").status_code == 404
    assert client.get("/games/copycat/reports/new/images/synthetic/train/0").status_code == 404
