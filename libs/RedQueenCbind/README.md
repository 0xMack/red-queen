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

```
pip install -e .
```

(from this directory — scikit-build-core drives the CMake/pybind11 build)
