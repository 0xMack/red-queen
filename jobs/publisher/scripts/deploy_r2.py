# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3>=1.35"]
# ///
"""Upload the public site's data to Cloudflare R2 (docs/design/0019): the snapshot (`redqueen publish site`) and ONNX
Runtime's files (`pnpm generate:static`), into the bucket served at https://redqueen-data.mackinations.ca.

Incremental: the bucket keeps the last upload's `index.json` (path -> sha256), and only files whose hash differs are
uploaded. `index.json` goes last, so an interrupted upload is redone next time. `--prune` deletes objects the new
index no longer lists.

The bucket also needs a CORS rule so the site can read it -- set once, in the dashboard (R2 -> redqueen-data ->
Settings -> CORS Policy): `--print-cors` prints the JSON to paste. The upload token is deliberately object-only (it
can't change bucket settings); `--configure-cors` applies the rule itself only with an Admin Read & Write token.

    uv run --script jobs/publisher/scripts/deploy_r2.py --snapshot data/site --ort apps/frontend/.output/ort [--dry-run]

Credentials come from the environment, never the repo -- an R2 API token's S3 credentials:
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY   (and R2_BUCKET, default `redqueen-data`)
Usually run through `redqueen dev deploy`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SITE_ORIGINS = ["https://redqueen.mackinations.ca", "http://localhost:3000", "http://localhost:8788"]
INDEX = "index.json"
TYPES = {".json": "application/json", ".mjs": "text/javascript", ".wasm": "application/wasm"}
IMMUTABLE = "public, max-age=31536000, immutable"


def content_type(path: str) -> str:
    return TYPES.get(Path(path).suffix, "application/octet-stream")


def cache_control(path: str) -> str:
    """Content-addressed or versioned files never change; API answers do, whenever the snapshot is republished."""
    if path.startswith(("models/", "ort/")):
        return IMMUTABLE  # blobs and manifests are named by their hash; ORT files by its version
    if path == INDEX:
        return "no-cache"
    return "public, max-age=300"  # api/: a new snapshot shows within five minutes


def local_index(snapshot: Path, ort: Path | None) -> dict[str, str]:
    """Every file to serve, path -> sha256: the snapshot's own index, plus ONNX Runtime's files under `ort/`."""
    index = json.loads((snapshot / INDEX).read_text(encoding="utf-8"))
    if ort is not None:
        for file in sorted(p for p in ort.rglob("*") if p.is_file()):
            index[f"ort/{file.relative_to(ort).as_posix()}"] = hashlib.sha256(file.read_bytes()).hexdigest()
    return index


def plan(local: dict[str, str], remote: dict[str, str], prune: bool) -> tuple[list[str], list[str]]:
    """What to upload (new or changed) and what to delete (gone from the new index, with --prune)."""
    uploads = sorted(path for path, sha in local.items() if remote.get(path) != sha)
    deletes = sorted(set(remote) - set(local)) if prune else []
    return uploads, deletes


def source(path: str, snapshot: Path, ort: Path | None) -> Path:
    if path.startswith("ort/"):
        assert ort is not None
        return ort / path.removeprefix("ort/")
    return snapshot / path


def cors_rules() -> list[dict]:
    return [
        {
            "AllowedOrigins": SITE_ORIGINS,
            "AllowedMethods": ["GET", "HEAD"],
            "AllowedHeaders": ["*"],
            "MaxAgeSeconds": 86400,
        }
    ]


def cors_policy_json() -> str:
    """The rule as the R2 dashboard's CORS Policy editor takes it (the same fields the S3 API does)."""
    return json.dumps(cors_rules(), indent=2)


def _client():
    import boto3

    missing = [v for v in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY") if not os.environ.get(v)]
    if missing:
        sys.exit(f"set {', '.join(missing)} (an R2 API token's S3 credentials; see docs/deploying.md)")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--snapshot", type=Path, default=Path("data/site"), help="`redqueen publish site`'s output")
    parser.add_argument("--ort", type=Path, default=None, help="`pnpm generate:static`'s .output/ort")
    parser.add_argument("--bucket", default=os.environ.get("R2_BUCKET", "redqueen-data"))
    parser.add_argument("--prune", action="store_true", help="delete objects the new index no longer lists")
    parser.add_argument("--dry-run", action="store_true", help="say what would change; upload nothing")
    parser.add_argument("--print-cors", action="store_true", help="print the CORS policy to paste in the dashboard")
    parser.add_argument("--configure-cors", action="store_true", help="apply the CORS rule (needs an admin token)")
    args = parser.parse_args(argv)

    if args.print_cors:
        print(cors_policy_json())
        return
    s3 = _client()
    if args.configure_cors:
        from botocore.exceptions import ClientError

        try:
            s3.put_bucket_cors(Bucket=args.bucket, CORSConfiguration={"CORSRules": cors_rules()})
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") != "AccessDenied":
                raise
            sys.exit(
                "this token can't change bucket settings (an Object Read & Write token can't, by design). Set the "
                f"policy in the dashboard instead -- R2 -> {args.bucket} -> Settings -> CORS Policy -> Add CORS policy "
                f"-- and paste:\n\n{cors_policy_json()}"
            )
        print(f"CORS: {args.bucket} readable from {', '.join(SITE_ORIGINS)}")
        return
    if not (args.snapshot / INDEX).exists():
        sys.exit(f"no snapshot at {args.snapshot} (run `redqueen publish site` first)")
    local = local_index(args.snapshot, args.ort)
    try:
        remote = json.loads(s3.get_object(Bucket=args.bucket, Key=INDEX)["Body"].read())
    except s3.exceptions.NoSuchKey:
        remote = {}
    uploads, deletes = plan(local, remote, args.prune)
    size = sum(source(p, args.snapshot, args.ort).stat().st_size for p in uploads)
    print(
        f"{args.bucket}: {len(uploads)} to upload ({size / 1e6:.1f} MB), {len(deletes)} to delete, {len(local)} in all"
    )
    if args.dry_run:
        for path in uploads[:20]:
            print(f"  + {path}")
        return

    def put(path: str) -> None:
        s3.upload_file(
            str(source(path, args.snapshot, args.ort)),
            args.bucket,
            path,
            ExtraArgs={"ContentType": content_type(path), "CacheControl": cache_control(path)},
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        for done, _ in enumerate(pool.map(put, uploads), 1):
            if done % 250 == 0:
                print(f"  {done}/{len(uploads)}", flush=True)
    for start in range(0, len(deletes), 1000):  # the S3 API deletes at most 1000 keys a call
        s3.delete_objects(Bucket=args.bucket, Delete={"Objects": [{"Key": k} for k in deletes[start : start + 1000]]})
    s3.put_object(
        Bucket=args.bucket,
        Key=INDEX,
        Body=json.dumps(local, indent=0).encode(),
        ContentType=content_type(INDEX),
        CacheControl=cache_control(INDEX),
    )
    print(f"{args.bucket}: uploaded {len(uploads)}, deleted {len(deletes)}; serving {len(local)} files")


if __name__ == "__main__":
    main()
