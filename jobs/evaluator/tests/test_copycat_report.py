"""copycat.report's comparison, which decides what a change is said to have done, and the run it records."""

from types import SimpleNamespace

import pytest
from arena.copycat import compare
from jobcore import open_sink

from evaluator.protocols import copycat_report


def _params(out):
    return SimpleNamespace(out=out, note="a note", baseline="before", decision=None)


def test_a_report_with_a_name_is_a_completed_run_and_holds_no_images(tmp_path, monkeypatch):
    monkeypatch.setenv("REDQUEEN_DATA_DIR", str(tmp_path))
    row = {"mean": 20.0, "worst": 10.0, "columns": {}, "by_severity": {}, "per_image": {"1": 20.0}}
    monkeypatch.setattr(copycat_report, "report", lambda params: {"groups": {"icons/heldout": row}})
    sink = open_sink()
    copycat_report.run(None, _params("change"), sink)
    run = sink.registry.list_runs()[0]
    assert run.status == "completed"
    assert (run.config["game"], run.config["experiment"], run.config["report"]) == (
        "copycat",
        "copycat.report",
        "change",
    )
    assert (run.config["baseline"], run.config["note"], "decision" in run.config) == ("before", "a note", False)
    assert "per_image" not in run.summary["groups"]["icons/heldout"]
    assert run.summary["held_out"] is True


def test_a_report_with_no_name_records_nothing_and_a_failed_one_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("REDQUEEN_DATA_DIR", str(tmp_path))
    sink = open_sink()
    monkeypatch.setattr(copycat_report, "report", lambda params: {"groups": {}})
    copycat_report.run(None, _params(None), sink)
    assert sink.registry.list_runs() == []

    def dies(params):
        raise RuntimeError("no")

    monkeypatch.setattr(copycat_report, "report", dies)
    with pytest.raises(RuntimeError):
        copycat_report.run(None, _params("change"), sink)
    assert sink.registry.list_runs()[0].status == "failed"


def group(mean, **per_image):
    return {"mean": mean, "per_image": per_image}


def test_a_comparison_counts_the_images_that_moved_largest_first():
    before = {"icons/train": group(20.0, a=10.0, b=20.0, c=30.0, d=20.0)}
    after = {"icons/train": group(21.0, a=15.0, b=19.0, c=30.01, d=20.0)}
    row = compare(before, after)["icons/train"]
    assert (row["before"], row["after"], row["up"], row["down"]) == (20.0, 21.0, 1, 1)
    assert list(row["moved"].items()) == [("a", 5.0), ("b", -1.0)]


def test_a_group_only_one_report_measured_is_left_out():
    after = {"icons/heldout": group(23.0, a=23.0), "icons/train": group(25.0, a=25.0)}
    assert list(compare({"icons/train": group(25.0, a=25.0)}, after)) == ["icons/train"]
