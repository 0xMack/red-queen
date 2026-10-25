"""Twemoji SVGs into Copy Cat's SVG subset (docs/design/0022 Decision 1, the `icons` set): the 36 x 36 artwork scaled
to the 128 px canvas, a group's fill pushed down onto its shapes, and every path made absolute with only M/L/C/Q/Z
(H/V become L, S/T become C/Q, an arc becomes cubics). A file using anything the subset refuses -- a transform, a
stroke, opacity, an unknown element -- is refused with the reason, never approximated.

The checked-in `data/copycat/icons/clean/` was made once from a seeded sample of Twemoji v17.0.3's single-codepoint
SVGs under 6 KB (`SAMPLE_SEED`; 160 drawn, 152 converted):

    uv run python -m arena.twemoji DOWNLOADED_SVG_DIR OUT_DIR
"""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from games.copycat import Svg

from arena.copycat_sets import SIZE

SOURCE = "https://github.com/jdecked/twemoji/tree/v17.0.3/assets/svg"
SAMPLE_SEED = "copycat-icons-v1"
SCALE = SIZE / 36
SVG_NS = "{http://www.w3.org/2000/svg}"
NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
COMMAND = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)")
ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}
REFUSED_ATTRIBUTES = ("transform", "stroke", "stroke-width", "opacity", "fill-opacity", "clip-path", "mask", "style")

Point = tuple[float, float]


class Refused(ValueError):
    pass


def colour(value: str) -> str:
    if not re.fullmatch(r"#[0-9a-fA-F]{3}|#[0-9a-fA-F]{6}", value.strip()):
        raise Refused(f"fill {value!r}")
    value = value.strip().lower()
    return "#" + "".join(c * 2 for c in value[1:]) if len(value) == 4 else value


def _arc_numbers(body: str) -> list[float]:
    """An arc's numbers, where the two flags may be written without separators ("a1 1 0 01 2 2")."""
    out: list[float] = []
    pos = 0
    while pos < len(body):
        if body[pos] in " ,\t\r\n":
            pos += 1
        elif len(out) % 7 in (3, 4):
            out.append(float(body[pos]))
            pos += 1
        else:
            m = NUMBER.match(body, pos)
            if not m:
                raise Refused(f"arc numbers {body[pos : pos + 12]!r}")
            out.append(float(m.group(0)))
            pos = m.end()
    return out


def arc_to_cubics(p0: Point, rx: float, ry: float, phi: float, large: bool, sweep: bool, p1: Point) -> list[tuple]:
    """An SVG arc (endpoint form) as cubics of at most a quarter turn each, through its centre form (SVG 1.1 F.6.5)."""
    if rx == 0 or ry == 0 or p0 == p1:
        return [("L", p1)]
    rx, ry = abs(rx), abs(ry)
    cos_phi, sin_phi = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    dx, dy = (p0[0] - p1[0]) / 2, (p0[1] - p1[1]) / 2
    x1, y1 = cos_phi * dx + sin_phi * dy, -sin_phi * dx + cos_phi * dy
    excess = x1**2 / rx**2 + y1**2 / ry**2
    if excess > 1:
        rx, ry = rx * math.sqrt(excess), ry * math.sqrt(excess)
    num = rx**2 * ry**2 - rx**2 * y1**2 - ry**2 * x1**2
    den = rx**2 * y1**2 + ry**2 * x1**2
    k = math.sqrt(max(num / den, 0.0)) * (-1 if large == sweep else 1)
    cx1, cy1 = k * rx * y1 / ry, -k * ry * x1 / rx
    cx = cos_phi * cx1 - sin_phi * cy1 + (p0[0] + p1[0]) / 2
    cy = sin_phi * cx1 + cos_phi * cy1 + (p0[1] + p1[1]) / 2

    def angle(u: Point, v: Point) -> float:
        return math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])

    u = ((x1 - cx1) / rx, (y1 - cy1) / ry)
    start = angle((1, 0), u)
    turn = angle(u, ((-x1 - cx1) / rx, (-y1 - cy1) / ry))
    if not sweep and turn > 0:
        turn -= 2 * math.pi
    elif sweep and turn < 0:
        turn += 2 * math.pi
    n = max(1, math.ceil(abs(turn) / (math.pi / 2) - 1e-9))
    step = turn / n
    handle = 4 / 3 * math.tan(step / 4)

    def at(t: float) -> Point:
        x, y = rx * math.cos(t), ry * math.sin(t)
        return (cos_phi * x - sin_phi * y + cx, sin_phi * x + cos_phi * y + cy)

    def tangent(t: float) -> Point:
        x, y = -rx * math.sin(t), ry * math.cos(t)
        return (cos_phi * x - sin_phi * y, sin_phi * x + cos_phi * y)

    out: list[tuple] = []
    for i in range(n):
        a, b = start + i * step, start + (i + 1) * step
        pa, pb, ta, tb = at(a), at(b), tangent(a), tangent(b)
        end = p1 if i == n - 1 else pb
        out.append(
            (
                "C",
                (pa[0] + handle * ta[0], pa[1] + handle * ta[1]),
                (pb[0] - handle * tb[0], pb[1] - handle * tb[1]),
                end,
            )
        )
    return out


def absolute(d: str) -> list[tuple]:
    """A path's segments as absolute M/L/C/Q/Z."""
    out: list[tuple] = []
    cur = start = (0.0, 0.0)
    control: Point | None = None  # the last C or Q control point, for S / T
    previous = ""
    for cmd, body in COMMAND.findall(d):
        up, rel = cmd.upper(), cmd.islower()
        nums = _arc_numbers(body) if up == "A" else [float(x) for x in NUMBER.findall(body)]
        if up == "Z":
            out.append(("Z",))
            cur, control, previous = start, None, "Z"
            continue
        if not nums or len(nums) % ARITY[up]:
            raise Refused(f"{cmd} with {len(nums)} numbers")
        for k in range(0, len(nums), ARITY[up]):
            a = nums[k : k + ARITY[up]]
            base = cur if rel else (0.0, 0.0)

            def pt(x: float, y: float, base: Point = base) -> Point:
                return (x + base[0], y + base[1])

            kind = "L" if up == "M" and k else up
            if kind == "M":
                cur = start = pt(*a)
                out.append(("M", cur))
            elif kind in "LHV":
                if kind == "H":
                    cur = (a[0] + base[0], cur[1])
                elif kind == "V":
                    cur = (cur[0], a[0] + base[1])
                else:
                    cur = pt(*a)
                out.append(("L", cur))
            elif kind in "CS":
                if kind == "C":
                    c1, c2, p = pt(a[0], a[1]), pt(a[2], a[3]), pt(a[4], a[5])
                else:
                    c1 = (2 * cur[0] - control[0], 2 * cur[1] - control[1]) if previous == "C" and control else cur
                    c2, p = pt(a[0], a[1]), pt(a[2], a[3])
                out.append(("C", c1, c2, p))
                cur, control, previous = p, c2, "C"
                continue
            elif kind in "QT":
                if kind == "Q":
                    c, p = pt(a[0], a[1]), pt(a[2], a[3])
                else:
                    c = (2 * cur[0] - control[0], 2 * cur[1] - control[1]) if previous == "Q" and control else cur
                    p = pt(a[0], a[1])
                out.append(("Q", c, p))
                cur, control, previous = p, c, "Q"
                continue
            else:  # A
                p = pt(a[5], a[6])
                out.extend(arc_to_cubics(cur, a[0], a[1], a[2], bool(a[3]), bool(a[4]), p))
                cur = p
            control, previous = None, kind
    return out


def _path(d: str) -> str:
    def point(p: Point) -> str:
        return f"{p[0] * SCALE:.2f} {p[1] * SCALE:.2f}"

    return " ".join(seg[0] + " ".join(point(p) for p in seg[1:]) for seg in absolute(d))


def convert(text: str) -> str:
    """One Twemoji SVG as a canonical subset document. Raises Refused for what the subset can't say."""
    root = ET.fromstring(text)
    if root.get("viewBox", "").split() != ["0", "0", "36", "36"]:
        raise Refused(f"viewBox {root.get('viewBox')!r}")
    lines: list[str] = []

    def walk(element: ET.Element, inherited: str | None) -> None:
        for child in element:
            tag = child.tag.replace(SVG_NS, "")
            attrs = dict(child.attrib)
            for name in REFUSED_ATTRIBUTES:
                if name in attrs:
                    raise Refused(f"{tag} {name}")
            fill = colour(attrs["fill"]) if "fill" in attrs else inherited
            if tag == "g":
                walk(child, fill)
                continue
            fill = fill or "#000000"
            s = {k: float(attrs.get(k, 0)) * SCALE for k in ("cx", "cy", "r", "rx", "ry")}
            if tag == "path":
                lines.append(f'<path d="{_path(attrs["d"])}" fill="{fill}"/>')
            elif tag == "circle":
                lines.append(f'<circle cx="{s["cx"]:.2f}" cy="{s["cy"]:.2f}" r="{s["r"]:.2f}" fill="{fill}"/>')
            elif tag == "ellipse":
                lines.append(
                    f'<ellipse cx="{s["cx"]:.2f}" cy="{s["cy"]:.2f}" rx="{s["rx"]:.2f}" ry="{s["ry"]:.2f}" fill="{fill}"/>'
                )
            else:
                raise Refused(f"element {tag}")

    walk(root, None)
    body = "\n".join(lines)
    return Svg(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}">\n{body}\n</svg>').canonical()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="a directory of downloaded Twemoji SVGs")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    kept = 0
    for path in sorted(args.source.glob("*.svg")):
        try:
            canonical = convert(path.read_text(encoding="utf-8"))
        except Refused as e:
            print(f"refused {path.name}: {e}")
            continue
        (args.out / path.name).write_text(canonical, encoding="utf-8", newline="\n")
        kept += 1
    print(f"converted {kept}")


if __name__ == "__main__":
    main()
