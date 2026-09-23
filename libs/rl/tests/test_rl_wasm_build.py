"""The browser's copy of the RL core (apps/frontend/app/wasm/rl, checked in) must be built from the sources as they
are now. Rebuild with `uv run python libs/rl/build-wasm.py`."""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def test_rl_wasm_build_is_current():
    spec = importlib.util.spec_from_file_location("rl_build_wasm", HERE / "build-wasm.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    recorded = (build.OUT / "source-hash.txt").read_text(encoding="utf-8").strip()
    assert recorded == build.source_hash(), (
        "libs/rl or the games core changed since the WASM build -- run build-wasm.py"
    )
