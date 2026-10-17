"""The R2 uploader's decisions (scripts/deploy_r2.py), without R2: what changed, how each file is served."""

import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy_r2.py"
spec = importlib.util.spec_from_file_location("deploy_r2", SCRIPT)
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def test_only_new_or_changed_files_upload_and_prune_deletes_the_rest():
    remote = {"api/runs.json": "a", "api/old.json": "b", "models/blobs/x": "c"}
    local = {"api/runs.json": "a2", "models/blobs/x": "c", "models/blobs/y": "d"}
    assert deploy.plan(local, remote, prune=False) == (["api/runs.json", "models/blobs/y"], [])
    assert deploy.plan(local, remote, prune=True)[1] == ["api/old.json"]
    assert deploy.plan(local, {}, prune=True) == (sorted(local), [])


def test_each_file_is_served_with_its_type_and_lifetime():
    assert deploy.content_type("api/runs.json") == "application/json"
    assert deploy.content_type("ort/1.30.0/ort-wasm-simd-threaded.mjs") == "text/javascript"
    assert deploy.content_type("ort/1.30.0/ort-wasm-simd-threaded.wasm") == "application/wasm"
    assert deploy.content_type("models/blobs/" + "a" * 64) == "application/octet-stream"
    assert deploy.cache_control("models/blobs/abc") == deploy.cache_control("ort/1.30.0/x.wasm") == deploy.IMMUTABLE
    assert deploy.cache_control("index.json") == "no-cache"
    assert "max-age=300" in deploy.cache_control("api/runs/abc.json")


def test_the_local_index_adds_onnx_runtime_under_its_version(tmp_path):
    snapshot, ort = tmp_path / "site", tmp_path / "ort"
    snapshot.mkdir()
    (snapshot / "index.json").write_bytes(json.dumps({"api/runs.json": "a"}).encode())
    (ort / "1.30.0").mkdir(parents=True)
    (ort / "1.30.0" / "ort-wasm-simd-threaded.wasm").write_bytes(b"wasm")
    index = deploy.local_index(snapshot, ort)
    assert index == {
        "api/runs.json": "a",
        "ort/1.30.0/ort-wasm-simd-threaded.wasm": hashlib.sha256(b"wasm").hexdigest(),
    }
    assert (
        deploy.source("ort/1.30.0/ort-wasm-simd-threaded.wasm", snapshot, ort)
        == ort / "1.30.0" / "ort-wasm-simd-threaded.wasm"
    )
    assert deploy.source("api/runs.json", snapshot, ort) == snapshot / "api/runs.json"


def test_the_bucket_is_readable_from_the_site_only():
    (rule,) = deploy.cors_rules()
    assert "https://redqueen.mackinations.ca" in rule["AllowedOrigins"] and "*" not in rule["AllowedOrigins"]
    assert rule["AllowedMethods"] == ["GET", "HEAD"]


def test_the_printed_policy_is_the_rule_the_dashboard_takes():
    assert json.loads(deploy.cors_policy_json()) == deploy.cors_rules()


def test_wrangler_gets_the_same_rule_in_the_cloudflare_api_shape():
    (rule,) = json.loads(deploy.cors_policy_json("wrangler"))["rules"]
    (s3,) = deploy.cors_rules()
    assert rule["allowed"] == {"origins": s3["AllowedOrigins"], "methods": s3["AllowedMethods"], "headers": ["*"]}
    assert rule["maxAgeSeconds"] == s3["MaxAgeSeconds"]


class FakeBucket:
    """head_object / upload_file over a dict, raising like botocore for a missing key."""

    def __init__(self, keys):
        self.keys, self.uploaded = set(keys), []

    def head_object(self, Bucket, Key):
        if Key not in self.keys:
            error = Exception("not found")
            error.response = {"Error": {"Code": "404"}}
            raise error

    def upload_file(self, filename, bucket, key, ExtraArgs):
        self.uploaded.append((key, ExtraArgs["ContentType"], ExtraArgs["CacheControl"]))


def test_a_site_deploy_uploads_only_the_onnx_runtime_files_the_bucket_lacks(tmp_path):
    ort = tmp_path / "ort"
    (ort / "1.31.0").mkdir(parents=True)
    for name in ("ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"):
        (ort / "1.31.0" / name).write_bytes(b"x")
    bucket = FakeBucket({"ort/1.31.0/ort-wasm-simd-threaded.mjs"})
    deploy.upload_missing_ort(bucket, "b", ort, dry_run=False, parser=None)
    assert bucket.uploaded == [("ort/1.31.0/ort-wasm-simd-threaded.wasm", "application/wasm", deploy.IMMUTABLE)]
    dry = FakeBucket(set())
    deploy.upload_missing_ort(dry, "b", ort, dry_run=True, parser=None)
    assert dry.uploaded == []
