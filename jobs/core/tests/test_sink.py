import pytest
from telemetry import FileMetricsStore, SqliteRunRegistry

from jobcore import open_sink


def test_the_default_sink_is_the_local_stores_under_the_data_dir(tmp_path):
    sink = open_sink()
    assert isinstance(sink.registry, SqliteRunRegistry) and isinstance(sink.metrics, FileMetricsStore)
    run_id = sink.registry.create_run({"game": "snake"})
    assert SqliteRunRegistry(tmp_path / "runs.db").get_run(run_id).config == {"game": "snake"}


def test_an_unknown_sink_is_refused_by_name(monkeypatch):
    monkeypatch.setenv("REDQUEEN_SINK", "http://localhost:8000")
    with pytest.raises(ValueError, match="stage 5"):
        open_sink()
    assert open_sink("local").registry is not None
