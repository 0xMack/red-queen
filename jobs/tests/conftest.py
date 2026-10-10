import sys
from pathlib import Path

import pytest

# jobs/ isn't a uv workspace package (see AGENTS.md) -- baseline_gp_run.py can import control.py
# because `uv run python jobs/<script>.py` puts the script's own directory on sys.path
# automatically. Tests need the same thing done explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """Every test's runs, metrics, evaluations and reports land in its own temporary directory (`telemetry.data_dir()`
    reads REDQUEEN_DATA_DIR at call time) -- never in the real data directory."""
    monkeypatch.setenv("REDQUEEN_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("REDQUEEN_MODELS_DIR", raising=False)
    return tmp_path
