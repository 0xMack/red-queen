import random

import pytest
from evolve.neuro import random_weight_vector
from fastapi.testclient import TestClient
from modelpack import Catalog, CatalogEntry, LocalModelStore, build_package, export_network

from backend.dependencies import _model_store
from backend.main import app


@pytest.fixture
def published(tmp_path):
    store = LocalModelStore(tmp_path)
    package = build_package(export_network(random_weight_vector((11, 16, 3), random.Random(0))), label="mlp")
    store.put(package)
    catalog = Catalog(game="snake")
    catalog.upsert(CatalogEntry(entrant_id="run:x", package_id=package.manifest.package_id, label="mlp", interface=None, variants=["fp32"]))
    store.put_catalog(catalog)
    app.dependency_overrides[_model_store] = lambda: store
    yield TestClient(app), package
    app.dependency_overrides.clear()


def test_catalog_points_at_this_backend_by_default(published):
    client, package = published
    body = client.get("/games/snake/models").json()

    assert body["base_url"] == "http://testserver/models"
    assert [e["package_id"] for e in body["entries"]] == [package.manifest.package_id]
    assert client.get("/games/checkers/models").json()["entries"] == []


def test_manifest_and_blobs_are_served_immutable(published):
    client, package = published
    manifest = client.get(f"/models/manifests/{package.manifest.package_id}.json")
    assert manifest.status_code == 200
    assert "immutable" in manifest.headers["cache-control"]
    assert manifest.json()["package_id"] == package.manifest.package_id

    for blob in package.manifest.blobs():
        response = client.get(f"/models/blobs/{blob.sha256}")
        assert response.status_code == 200
        assert response.content == package.blobs[blob.sha256]
        assert "immutable" in response.headers["cache-control"]


@pytest.mark.parametrize("path", ["/models/blobs/" + "0" * 64, "/models/blobs/..%2Fruns.db", "/models/manifests/nope.json"])
def test_unknown_or_malformed_keys_are_404(published, path):
    client, _ = published
    assert client.get(path).status_code == 404
