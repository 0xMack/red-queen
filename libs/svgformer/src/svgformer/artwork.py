"""The training artwork (docs/design/0024, phase B): openly licensed icon sets from Iconify and Openclipart's illustrations,
downloaded, checked against the digests pinned in `artwork_sources.json`, and converted to Copy Cat's subset by
`svgformer.convert`. None of it is in the repo -- other people's artwork, and hundreds of MB of it.

    uv run python -I -m svgformer.artwork fetch ROOT      # download and verify (resumes)
    uv run python -I -m svgformer.artwork pin ROOT        # record the digests of what was downloaded (once, by hand)
    uv run python -I -m svgformer.artwork build ROOT      # convert; refusal counts per source in ROOT/build.json

`ROOT` is e.g. `<data dir>/experiments/svgformer-artwork`. Downloads land in `ROOT/sources/`, its own directory, and are
only ever read as data (the Openclipart archive is read in place, never extracted).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
import tarfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SOURCES = Path(__file__).parent / "artwork_sources.json"
LICENCES = ("MIT", "Apache-2.0", "CC0-1.0", "CC-BY-4.0")
# the families the held-out `icons` and `broad` sets are drawn from: never trained on
EXCLUDED = ("twemoji", "flat-color-icons")
DOWNLOADERS = 8
ATTEMPTS = 3


class Mismatch(ValueError):
    pass


def sources() -> dict[str, Any]:
    return json.loads(SOURCES.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, path: Path) -> None:
    if path.exists():
        return
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                body = response.read()
            break
        except OSError:
            if attempt + 1 == ATTEMPTS:
                raise
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_bytes(body)
    temporary.replace(path)


def iconify_url(src: dict[str, Any], prefix: str) -> str:
    return f"https://raw.githubusercontent.com/iconify/icon-sets/{src['iconify']['commit']}/json/{prefix}.json"


def files(src: dict[str, Any], root: Path) -> dict[str, tuple[str, Path]]:
    """Every download: its id -> (url, where it goes)."""
    out = {
        f"iconify/{p}": (iconify_url(src, p), root / "sources" / "iconify" / f"{p}.json")
        for p in src["iconify"]["sets"]
    }
    clip = src["openclipart"]
    out["openclipart"] = (clip["url"], root / "sources" / "openclipart" / clip["file"])
    return out


def fetch(root: Path, verify: bool = True) -> None:
    src = sources()
    todo = files(src, root)
    with concurrent.futures.ThreadPoolExecutor(DOWNLOADERS) as pool:
        list(pool.map(lambda item: _download(*item), todo.values()))
    if not verify:
        return
    pinned = src.get("sha256", {})
    bad = [k for k, (_, path) in todo.items() if pinned.get(k) != sha256(path)]
    if bad:
        raise Mismatch(f"{len(bad)} downloads aren't what was pinned: {bad[:5]}")
    print(f"{len(todo)} downloads in {root / 'sources'}, all as pinned")


def pin(root: Path) -> None:
    src = sources()
    fetch(root, verify=False)
    src["sha256"] = {k: sha256(path) for k, (_, path) in sorted(files(src, root).items())}
    SOURCES.write_text(json.dumps(src, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"pinned {len(src['sha256'])} digests in {SOURCES}")


def iconify_icons(path: Path) -> Iterator[tuple[str, str]]:
    """A set's icons as standalone SVG documents: (name, svg). Aliases are skipped (they repeat an icon)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    width, height = data.get("width", 16), data.get("height", 16)
    left, top = data.get("left", 0), data.get("top", 0)
    for name, icon in sorted(data["icons"].items()):
        if icon.get("hidden"):
            continue
        w, h = icon.get("width", width), icon.get("height", height)
        x, y = icon.get("left", left), icon.get("top", top)
        body = icon["body"]
        # Iconify's own transformations (rotate in quarter turns, flips), as SVG transforms about the box's centre
        cx, cy = x + w / 2, y + h / 2
        steps = []
        if icon.get("hFlip") != data.get("hFlip", False):
            steps.append(f"translate({2 * cx} 0) scale(-1 1)")
        if icon.get("vFlip") != data.get("vFlip", False):
            steps.append(f"translate(0 {2 * cy}) scale(1 -1)")
        if turns := (icon.get("rotate", 0) + data.get("rotate", 0)) % 4:
            steps.insert(0, f"rotate({90 * turns} {cx} {cy})")
        if steps:
            body = f'<g transform="{" ".join(steps)}">{body}</g>'
        yield name, f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x} {y} {w} {h}">{body}</svg>'


CREATOR = re.compile(r"<dc:creator>.*?<dc:title>\s*(.*?)\s*</dc:title>", re.DOTALL)


def openclipart_svgs(path: Path) -> Iterator[tuple[str, str, str]]:
    """The archive's SVGs, read in place: (name, svg, author from its RDF metadata or ""). Nothing is extracted. (Its
    `.txt` files are the licence's text, the same for every image.)"""
    svgs: dict[str, str] = {}
    # one pass in archive order: a gzip stream can't seek back, so reading out of order re-decompresses from the start
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.isfile() and member.name.endswith(".svg") and (f := archive.extractfile(member)):
                svgs[member.name] = f.read().decode("utf-8", errors="replace")
    for name in sorted(svgs):
        m = CREATOR.search(svgs[name])
        yield name, svgs[name], (m.group(1).strip() if m else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fetch", "pin", "build"])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.root)
    elif args.command == "pin":
        pin(args.root)
    else:
        from svgformer.build import build

        build(args.root)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
