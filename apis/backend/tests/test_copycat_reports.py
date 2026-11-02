"""The report routes read `copycat.report`'s files, old and new, and leave everything else in the directory alone."""

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
        "groups": {"icons/train": _group(21.0, a=22.0, b=20.0), "icons/heldout": _group(19.0, c=19.0)},
    }
    (root / "new.json").write_text(json.dumps(new))
    (root / "new.svgs.json").write_text(json.dumps({"icons/train": {"a": "<svg/>"}}))
    (root / "other.json").write_text(json.dumps({"protocol": "experiment.snake-report"}))
    return TestClient(app)


def test_a_report_says_what_it_changed_against_its_baseline(client):
    report = client.get("/games/copycat/reports/new").json()
    assert report["per_image"]["icons/train"] == {"a": 22.0, "b": 20.0}
    assert report["has_svgs"] is True
    row = report["compared"]["icons/train"]
    assert (row["before"], row["after"], row["up"], row["down"], row["moved"]) == (20.0, 21.0, 1, 0, {"a": 2.0})
    old = client.get("/games/copycat/reports/old").json()
    assert (old["compared"], old["has_svgs"]) == ({}, False)


@pytest.mark.parametrize("name", ["missing", "new.svgs", "other", "..%2Fsecret"])
def test_only_a_report_is_a_report(client, name):
    assert client.get(f"/games/copycat/reports/{name}").status_code == 404


def test_held_out_images_are_not_served_one_by_one(client):
    assert client.get("/games/copycat/reports/new/images/icons/heldout/10050").status_code == 404
    assert client.get("/games/copycat/reports/new/images/synthetic/train/0").status_code == 404
