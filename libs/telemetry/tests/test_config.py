from pathlib import Path

import pytest

from telemetry import data_dir, models_dir, open_stores, workspace_root


@pytest.fixture
def no_overrides(monkeypatch):
    for var in ("REDQUEEN_DATA_DIR", "REDQUEEN_RUN_DATA_DIR", "REDQUEEN_MODELS_DIR"):
        monkeypatch.delenv(var, raising=False)


def test_data_dir_defaults_to_the_workspace_root_from_any_subdirectory(no_overrides, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root / "libs" / "telemetry" / "src")
    assert workspace_root() == root
    assert data_dir() == root / "data"
    assert models_dir() == root / "data" / "models"


def test_data_dir_outside_the_workspace_needs_the_variable(no_overrides, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="REDQUEEN_DATA_DIR"):
        data_dir()


def test_overrides_and_the_old_name(no_overrides, monkeypatch, tmp_path):
    monkeypatch.setenv("REDQUEEN_RUN_DATA_DIR", str(tmp_path / "old"))
    assert data_dir() == tmp_path / "old"
    monkeypatch.setenv("REDQUEEN_DATA_DIR", str(tmp_path / "new"))
    assert data_dir() == tmp_path / "new"
    monkeypatch.setenv("REDQUEEN_MODELS_DIR", str(tmp_path / "models"))
    assert models_dir() == tmp_path / "models"


def test_open_stores_reads_the_data_dir_at_call_time(monkeypatch, tmp_path):
    monkeypatch.setenv("REDQUEEN_DATA_DIR", str(tmp_path))
    stores = open_stores()
    run_id = stores.registry.create_run({"game": "snake"})
    assert open_stores(tmp_path).registry.get_run(run_id).config == {"game": "snake"}
    assert (tmp_path / "runs.db").exists() and (tmp_path / "evaluations.db").exists()
