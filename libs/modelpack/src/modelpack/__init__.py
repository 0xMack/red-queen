"""Portable model packages for client-side inference (docs/design/0009).

Trainer object → `exporters` (ONNX graph + reference forward) → `packaging` (sharded, hashed,
parity-checked, sealed) → a `ModelStore` → a browser (ONNX Runtime Web) or an evaluation job
(`runtime.PackagedModel`, ONNX Runtime CPU).
"""

from modelpack.exporters import Exported, export_network, export_network_json
from modelpack.manifest import ModelManifest, Parity, Requirements, Variant
from modelpack.packaging import Package, build_package, with_parity
from modelpack.runtime import PackagedModel
from modelpack.store import Catalog, CatalogEntry, LocalModelStore, ModelStore

__all__ = [
    "Catalog",
    "CatalogEntry",
    "Exported",
    "LocalModelStore",
    "ModelManifest",
    "ModelStore",
    "Package",
    "PackagedModel",
    "Parity",
    "Requirements",
    "Variant",
    "build_package",
    "export_network",
    "export_network_json",
    "with_parity",
]
