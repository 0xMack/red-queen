import pytest


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """Every test's runs land in its own temporary directory, never the real data directory."""
    monkeypatch.setenv("REDQUEEN_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("REDQUEEN_SINK", raising=False)
    return tmp_path
