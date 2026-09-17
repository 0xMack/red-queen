# redqueen

C++ core with pybind11 bindings for linear genetic programming (LGP). Evolves populations of
register-machine programs against a dataset (currently exercised against Iris) and exposes a
`Population` class to Python for training and prediction.

## Layout

- `src/linear/` — C++ implementation (individuals, populations, dataset loading, utils)
- `examples/gp.py` — minimal usage example against the Iris dataset
- `data/` — sample datasets used by the examples
- `CMakeLists.txt` / `pyproject.toml` — pybind11 + scikit-build-core build config

## Build

From the repo root, as part of the `uv` workspace (see [AGENTS.md](../../AGENTS.md)):

```
uv sync --all-packages --extra examples
uv run python examples/gp.py
```

On Windows, run those through an MSVC dev environment (no `cl.exe` on PATH by default) —
`vcvarsall.bat x64` first, then the commands above; scikit-build-core drives the actual
CMake/pybind11 build.
