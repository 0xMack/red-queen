"""Build the training artwork from the downloads (`svgformer.artwork`): convert every icon and illustration to the subset,
drop duplicates and anything that looks like a held-out image, and hold out Openclipart's `illustrations` by author.

Writes under ROOT/build/: `icons.jsonl` (Iconify, 128 px) and `illustrations.jsonl` (Openclipart, 256 px) for training,
`illustrations-heldout.jsonl` (never trained on), one record a line: {source, name, licence, author, size, tokens, clamped, svg}. ROOT/build.json
counts what each source gave and why the rest was refused.
"""

from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from arena.copycat_sets import SPLITS, synthetic
from games.copycat import Svg

from svgformer import artwork, tokens
from svgformer.convert import Refused, convert

ICON_SIZE, ILLUSTRATION_SIZE = 128, 256
THUMB = 16
# a thumbnail this close (mean absolute difference, 0-255 grey) to a held-out image's is treated as the same image
NEAR = 3.0
HELDOUT_SEED = "svgformer-illustrations-v1"
HELDOUT_IMAGES = 240


def thumbnail(svg: str) -> np.ndarray:
    doc = Svg(svg)
    rgba = np.frombuffer(doc.render_rgba8(), dtype=np.uint8).reshape(doc.height, doc.width, 4).astype(np.float32)
    alpha = rgba[..., 3:] / 255
    grey = (rgba[..., :3] * alpha + 255 * (1 - alpha)).mean(axis=2)  # over white
    k = doc.width // THUMB
    return grey.reshape(THUMB, k, THUMB, k).mean(axis=(1, 3))


def _convert(job: tuple[str, str, str, int, str, str]) -> dict[str, Any]:
    source, name, text, size, licence, author = job
    try:
        strokes: dict[str, int] = {}
        svg = convert(text, size, strokes)
    except Refused as e:
        return {"source": source, "name": name, "refused": str(e).split(":")[0][:48]}
    except (ValueError, KeyError, ZeroDivisionError, OverflowError) as e:  # malformed numbers and the like
        return {"source": source, "name": name, "refused": f"malformed ({type(e).__name__})"}
    encoded = tokens.encode(svg, size)
    return {
        "source": source,
        "name": name,
        "licence": licence,
        "author": author,
        "size": size,
        "tokens": len(encoded.tokens),  # so training can filter by length without tokenizing everything
        "clamped": encoded.clamped,  # numbers outside the grid's margin: the decoded image differs there
        "svg": svg,
        "strokes": strokes,  # strokes outlined, by kind of outline (svgformer.stroke)
        "thumb": thumbnail(svg).round(2).tolist(),
    }


def _jobs(root: Path) -> list[tuple[str, str, str, int, str, str]]:
    src = artwork.sources()
    jobs = []
    for prefix, meta in src["iconify"]["sets"].items():
        for name, text in artwork.iconify_icons(root / "sources" / "iconify" / f"{prefix}.json"):
            jobs.append((f"iconify/{prefix}", name, text, ICON_SIZE, meta["licence"], meta["author"]))
    clip = src["openclipart"]
    for name, text, author in artwork.openclipart_svgs(root / "sources" / "openclipart" / clip["file"]):
        jobs.append(("openclipart", name, text, ILLUSTRATION_SIZE, clip["licence"], author))
    return jobs


def _heldout_thumbs(sets_root: Path) -> tuple[set[str], np.ndarray]:
    svgs = [synthetic(seed)[0] for split in ("heldout", "validate") for seed in SPLITS[split]]
    for set_name in ("icons", "broad"):
        clean = sets_root / set_name / "clean"
        if not clean.exists():
            raise FileNotFoundError(f"{clean} isn't built: run `python -m arena.copycat_artwork {sets_root}` first")
        svgs += [p.read_text(encoding="utf-8") for p in sorted(clean.glob("*.svg"))]
    digests = {hashlib.sha256(Svg(s).canonical().encode()).hexdigest() for s in svgs}
    return digests, np.stack([thumbnail(Svg(s).canonical()) for s in svgs])


def _author_rank(author: str) -> str:
    return hashlib.sha256(f"{HELDOUT_SEED}:{author}".encode()).hexdigest()


def build(root: Path, sets_root: Path | None = None, workers: int | None = None) -> dict[str, Any]:
    artwork.fetch(root)
    sets_root = sets_root or root.parent / "copycat-sets"
    heldout_digests, heldout_thumbs = _heldout_thumbs(sets_root)
    jobs = _jobs(root)
    print(f"converting {len(jobs)} files", flush=True)
    with concurrent.futures.ProcessPoolExecutor(workers or max(1, (os.cpu_count() or 2) - 1)) as pool:
        results = list(pool.map(_convert, jobs, chunksize=64))
    del jobs

    report: dict[str, Any] = collections.defaultdict(
        lambda: {"files": 0, "kept": 0, "strokes": collections.Counter(), "refused": collections.Counter()}
    )
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for r in results:
        entry = report[r["source"]]
        entry["files"] += 1
        if "refused" in r:
            entry["refused"][r["refused"]] += 1
            continue
        digest = hashlib.sha256(r["svg"].encode()).hexdigest()
        if digest in heldout_digests:
            entry["refused"]["a held-out image"] += 1
            continue
        if digest in seen:
            entry["refused"]["a duplicate"] += 1
            continue
        thumb = np.asarray(r.pop("thumb"), dtype=np.float32)
        if (np.abs(heldout_thumbs - thumb).mean(axis=(1, 2)) < NEAR).any():
            entry["refused"]["looks like a held-out image"] += 1
            continue
        seen.add(digest)
        entry["kept"] += 1
        entry["strokes"].update(r["strokes"])
        kept.append(r)

    # Openclipart's held-out illustrations: whole authors, in a seeded order, until there are enough images
    clip = [r for r in kept if r["source"] == "openclipart"]
    by_author = collections.defaultdict(list)
    for r in clip:
        by_author[r["author"] or "(unknown)"].append(r)
    held: set[int] = set()
    held_authors = []
    for author in sorted((a for a in by_author if a != "(unknown)"), key=_author_rank):
        if sum(len(by_author[a]) for a in held_authors) >= HELDOUT_IMAGES:
            break
        held_authors.append(author)
        held.update(id(r) for r in by_author[author])

    out = root / "build"
    out.mkdir(parents=True, exist_ok=True)
    files = {"icons.jsonl": [], "illustrations.jsonl": [], "illustrations-heldout.jsonl": []}
    for r in kept:
        if r["source"] != "openclipart":
            files["icons.jsonl"].append(r)
        elif id(r) in held:
            files["illustrations-heldout.jsonl"].append(r)
        else:
            files["illustrations.jsonl"].append(r)
    for name, records in files.items():
        with (out / name).open("w", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "sources": {
            k: {**v, "strokes": dict(v["strokes"]), "refused": dict(v["refused"].most_common())}
            for k, v in sorted(report.items())
        },
        "files": {name: len(records) for name, records in files.items()},
        "heldout_authors": held_authors,
    }
    (root / "build.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["files"]), flush=True)
    return summary
