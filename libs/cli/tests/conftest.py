import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """`cli("runs", "list")` runs a command against the real backend (its TestClient) over a scratch data directory;
    returns (result, the job store, the run registry)."""
    from backend.dependencies import _job_store, _stores
    from backend.main import app as api
    from fastapi.testclient import TestClient
    from telemetry import open_job_store, open_stores

    from redqueen_cli.main import app
    from redqueen_cli.state import State

    monkeypatch.setenv("REDQUEEN_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("REDQUEEN_API_URL", raising=False)
    _stores.cache_clear()
    _job_store.cache_clear()
    http = TestClient(api)
    runner = CliRunner()

    def invoke(*args: str):
        return runner.invoke(app, list(args), obj=State(transport=http))

    invoke.jobs, invoke.registry = open_job_store(), open_stores().registry
    yield invoke
    _stores.cache_clear()
    _job_store.cache_clear()
