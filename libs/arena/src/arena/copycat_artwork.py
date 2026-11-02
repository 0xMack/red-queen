"""Copy Cat's artwork sets (`icons`, `broad`) aren't in the repo, only how to get them, in `copycat_sources.json`: for each set
its families (a pinned URL and the files to take from it), and digests of the files downloaded, of the SVGs converted from
them (`arena.twemoji`), and of its held-out images. `ensure` downloads a set on first use and builds it under `root`, and
refuses one that doesn't match: a score means something only on the same images.

The held-out digest also depends on Pillow's JPEG encoder (`corrupt.v1`), which `uv.lock` pins. If an upgrade moves it,
that's a new version of the set, not a new digest for this one.

    uv run python -m arena.copycat_artwork ROOT [--set icons|broad]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

SOURCES = Path(__file__).parent / "copycat_sources.json"
DOWNLOADERS = 8
ATTEMPTS = 3


class SetNotFetched(FileNotFoundError):
    pass


class SetMismatch(ValueError):
    pass


def sources() -> dict[str, Any]:
    return json.loads(SOURCES.read_text(encoding="utf-8"))


def digest(files: dict[str, bytes]) -> str:
    """One hash of some named files, whatever order they came in."""
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode("utf-8") + b"\0" + hashlib.sha256(files[name]).digest())
    return h.hexdigest()


def file_names(artwork: dict[str, Any]) -> list[str]:
    """Every clean SVG's name (without `.svg`) in a set, in name order."""
    return sorted(family["prefix"] + name for family in artwork["families"] for name in family["names"])


def _download(url: str, path: Path) -> None:
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                body = response.read()
            break
        except OSError:
            if attempt + 1 == ATTEMPTS:
                raise
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_bytes(body)
    temporary.replace(path)


def _fetch_family(family: dict[str, Any], where: Path) -> dict[str, bytes]:
    missing = [name for name in family["names"] if not (where / f"{name}.svg").exists()]
    with concurrent.futures.ThreadPoolExecutor(DOWNLOADERS) as pool:
        list(pool.map(lambda name: _download(f"{family['base']}/{name}.svg", where / f"{name}.svg"), missing))
    return {name: (where / f"{name}.svg").read_bytes() for name in family["names"]}


def ensure(set_name: str, root: Path, download: bool = True, artwork: dict[str, Any] | None = None) -> Path:
    """The directory of a set's clean SVGs, downloaded and converted under `root` if they aren't there yet."""
    from arena import twemoji

    artwork = artwork or sources()[set_name]
    clean = root / set_name / "clean"
    built = clean / ".built"
    if built.exists() and built.read_text(encoding="utf-8").strip() == artwork["clean_sha256"]:
        return clean
    if not download:
        raise SetNotFetched(f"{set_name} isn't downloaded: run `python -m arena.copycat_artwork {root}`")
    converted: dict[str, bytes] = {}
    for family in artwork["families"]:
        where = root / set_name / "sources" / family["id"]
        raw = _fetch_family(family, where)
        if digest(raw) != family["sha256"]:
            raise SetMismatch(f"{set_name}: {family['base']} isn't what was pinned")
        for name, body in raw.items():
            text = twemoji.convert(body.decode("utf-8"))
            converted[f"{family['prefix']}{name}.svg"] = text.encode("utf-8")
    if digest(converted) != artwork["clean_sha256"]:
        raise SetMismatch(f"{set_name}: converting the same sources gave different SVGs (arena.twemoji changed?)")
    clean.mkdir(parents=True, exist_ok=True)
    for name, body in converted.items():
        (clean / name).write_bytes(body)
    built.write_text(artwork["clean_sha256"], encoding="utf-8", newline="\n")
    print(f"{set_name}: {len(converted)} SVGs in {clean}")
    return clean


def held_out(set_name: str, root: Path, download: bool = True, artwork: dict[str, Any] | None = None) -> Path:
    """The directory of a set's held-out images, built from its clean SVGs if they aren't there yet."""
    from arena.copycat_sets import write_split

    artwork = artwork or sources()[set_name]
    where = root / set_name / "heldout"
    built = where / ".built"
    if built.exists() and built.read_text(encoding="utf-8").strip() == artwork["heldout_sha256"]:
        return where
    clean = ensure(set_name, root, download, artwork)
    if not download:
        raise SetNotFetched(f"{set_name}'s held-out images aren't built: run `python -m arena.copycat_artwork {root}`")
    write_split("heldout", where, set_name, clean, artwork)
    files = {p.name: p.read_bytes() for p in where.iterdir() if p.name != ".built"}
    if digest(files) != artwork["heldout_sha256"]:
        raise SetMismatch(f"{set_name}: its held-out images came out different (did Pillow's JPEG encoder change?)")
    built.write_text(artwork["heldout_sha256"], encoding="utf-8", newline="\n")
    return where


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="e.g. <data dir>/experiments/copycat-sets, where copycat.report looks")
    parser.add_argument("--set", choices=sorted(sources()), dest="set_name")
    args = parser.parse_args()
    for set_name in [args.set_name] if args.set_name else sorted(sources()):
        print(f"{set_name}: held-out images in {held_out(set_name, args.root)}")


if __name__ == "__main__":
    main()
