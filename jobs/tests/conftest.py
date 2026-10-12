import pytest


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """Every test's runs, metrics, evaluations and reports land in its own temporary directory (`telemetry.data_dir()`
    reads REDQUEEN_DATA_DIR at call time) -- never in the real data directory."""
    monkeypatch.setenv("REDQUEEN_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("REDQUEEN_MODELS_DIR", raising=False)
    monkeypatch.delenv("REDQUEEN_SINK", raising=False)
    return tmp_path
