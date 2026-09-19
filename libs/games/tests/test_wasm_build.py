"""The browser's copy of the game core (apps/frontend/app/wasm/games, checked in) must be built from the
Rust sources as they are now -- otherwise visitors would watch a different game than training plays.
Rebuild with `uv run python libs/games/build-wasm.py`."""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def test_wasm_build_is_current():
    spec = importlib.util.spec_from_file_location("build_wasm", HERE / "build-wasm.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    recorded = (build.OUT / "source-hash.txt").read_text(encoding="utf-8").strip()
    assert recorded == build.source_hash(), "libs/games/rust changed since the WASM build -- run libs/games/build-wasm.py"
