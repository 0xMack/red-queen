"""Any reasonably plain SVG into Copy Cat's subset (docs/design/0022 Decision 2), for training data (docs/design/0024, phase B).

`arena.twemoji` converts the held-out sets and refuses anything it doesn't need; this one has to take what real icon sets
and illustrations are written in, so it does more, but only what is exact:

- any viewBox, scaled uniformly onto a `size` x `size` canvas and centred;
- `transform`s on any element, composed and applied to the points (a circle under a skew becomes a path);
- `style="..."` and presentation attributes, inherited as CSS inherits them; `currentColor` is black, as Iconify renders it;
- `opacity` on a shape, or on a group holding one shape, folded into `fill-opacity`;
- `fill-rule="evenodd"` made non-zero (the subset's rule) by reorienting each subpath by how deeply it is nested, which is
  exact when subpaths don't cross;
- rounded rects, polylines, arcs (as cubics), H/V/S/T, relative commands;
- strokes, as a filled outline painted after the fill (`svgformer.stroke`: joins, caps and miter limits as SVG draws them,
  checked against an exact outline by rendering, so a stroke is right to within 3% of its area).

It refuses -- with the reason, so refusals can be counted -- gradients and patterns, clip paths, masks, filters, dashes,
text, images, a `<use>` it can't place, style sheets, opacity on a group of several shapes or on a filled, stroked shape
(not the same as on each), and anything it doesn't recognise. A file that converts renders as its source did, to the
subset's precision.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from arena.twemoji import NUMBER, Refused, absolute
from games.copycat import Svg

from svgformer.stroke import Segment, outline

SVG_NS = "http://www.w3.org/2000/svg"
INHERITED = (
    "fill", "fill-opacity", "fill-rule", "stroke", "stroke-width", "stroke-opacity", "stroke-linejoin", "stroke-linecap",
    "stroke-miterlimit", "stroke-dasharray", "visibility",
)  # fmt: skip
UNITS = {"": 1.0, "px": 1.0, "pt": 96 / 72, "pc": 16.0, "mm": 96 / 25.4, "cm": 96 / 2.54, "in": 96.0}
NOT_INHERITED = ("opacity", "display", "clip-path", "mask", "filter")
DEFINITIONS = {
    "defs", "title", "desc", "metadata", "linearGradient", "radialGradient", "pattern", "clipPath", "mask", "symbol",
    "marker", "filter", "font", "font-face", "cursor", "script", "animate", "set",
}  # fmt: skip
CONTAINERS = {"g", "a", "svg"}
NAMED = {
    "black": "#000000", "white": "#ffffff", "red": "#ff0000", "lime": "#00ff00", "green": "#008000", "blue": "#0000ff",
    "yellow": "#ffff00", "cyan": "#00ffff", "aqua": "#00ffff", "magenta": "#ff00ff", "fuchsia": "#ff00ff",
    "silver": "#c0c0c0", "gray": "#808080", "grey": "#808080", "maroon": "#800000", "olive": "#808000",
    "purple": "#800080", "teal": "#008080", "navy": "#000080", "orange": "#ffa500", "brown": "#a52a2a",
    "pink": "#ffc0cb", "gold": "#ffd700", "darkgray": "#a9a9a9", "darkgrey": "#a9a9a9", "lightgray": "#d3d3d3",
    "lightgrey": "#d3d3d3", "dimgray": "#696969", "darkred": "#8b0000", "darkgreen": "#006400", "darkblue": "#00008b",
    "skyblue": "#87ceeb", "beige": "#f5f5dc", "tan": "#d2b48c", "violet": "#ee82ee", "indigo": "#4b0082",
    "khaki": "#f0e68c", "salmon": "#fa8072", "crimson": "#dc143c", "coral": "#ff7f50", "tomato": "#ff6347",
    "wheat": "#f5deb3", "ivory": "#fffff0", "whitesmoke": "#f5f5f5", "gainsboro": "#dcdcdc", "chocolate": "#d2691e",
    "sienna": "#a0522d", "peru": "#cd853f", "orchid": "#da70d6", "plum": "#dda0dd", "turquoise": "#40e0d0",
    "steelblue": "#4682b4", "royalblue": "#4169e1", "firebrick": "#b22222", "forestgreen": "#228b22",
    "limegreen": "#32cd32", "darkorange": "#ff8c00", "lightblue": "#add8e6", "lightgreen": "#90ee90",
    "goldenrod": "#daa520", "slategray": "#708090", "slategrey": "#708090", "darkslategray": "#2f4f4f",
}  # fmt: skip

Matrix = tuple[float, float, float, float, float, float]  # x' = a x + c y + e, y' = b x + d y + f
IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
Point = tuple[float, float]


def compose(m: Matrix, n: Matrix) -> Matrix:
    """The matrix of applying n first, then m (m x n)."""
    a, b, c, d, e, f = m
    p, q, r, s, t, u = n
    return (a * p + c * q, b * p + d * q, a * r + c * s, b * r + d * s, a * t + c * u + e, b * t + d * u + f)


def apply(m: Matrix, p: Point) -> Point:
    return (m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5])


TRANSFORM = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)")


def parse_transform(text: str) -> Matrix:
    m = IDENTITY
    rest = TRANSFORM.sub("", text).replace(",", " ").strip()
    if rest:
        raise Refused(f"transform {text[:40]!r}")
    for kind, body in TRANSFORM.findall(text):
        v = [float(x) for x in NUMBER.findall(body)]
        if kind == "matrix" and len(v) == 6:
            step = tuple(v)
        elif kind == "translate" and len(v) in (1, 2):
            step = (1.0, 0.0, 0.0, 1.0, v[0], v[1] if len(v) == 2 else 0.0)
        elif kind == "scale" and len(v) in (1, 2):
            step = (v[0], 0.0, 0.0, v[1] if len(v) == 2 else v[0], 0.0, 0.0)
        elif kind == "rotate" and len(v) in (1, 3):
            r = math.radians(v[0])
            cos, sin = math.cos(r), math.sin(r)
            step = (cos, sin, -sin, cos, 0.0, 0.0)
            if len(v) == 3:
                cx, cy = v[1], v[2]
                step = compose(compose((1.0, 0.0, 0.0, 1.0, cx, cy), step), (1.0, 0.0, 0.0, 1.0, -cx, -cy))
        elif kind == "skewX" and len(v) == 1:
            step = (1.0, 0.0, math.tan(math.radians(v[0])), 1.0, 0.0, 0.0)
        elif kind == "skewY" and len(v) == 1:
            step = (1.0, math.tan(math.radians(v[0])), 0.0, 1.0, 0.0, 0.0)
        else:
            raise Refused(f"transform {kind}({body.strip()[:30]})")
        m = compose(m, step)  # type: ignore[arg-type]
    return m


def colour(value: str) -> str | None:
    """A paint as #rrggbb, or None for `none`. Raises Refused for gradients, patterns and what it can't read."""
    v = value.strip().lower()
    if v in ("none", "transparent"):
        return None
    if v == "currentcolor":
        return "#000000"
    if v.startswith("url("):
        raise Refused("gradient or pattern fill")
    if re.fullmatch(r"#[0-9a-f]{6}", v):
        return v
    if re.fullmatch(r"#[0-9a-f]{3}", v):
        return "#" + "".join(ch * 2 for ch in v[1:])
    if m := re.fullmatch(r"rgb\(\s*([^)]*)\)", v):
        parts = [p.strip() for p in m.group(1).replace(",", " ").split()]
        if len(parts) == 3:
            channels = [float(p[:-1]) * 2.55 if p.endswith("%") else float(p) for p in parts]
            return "#" + "".join(f"{max(0, min(255, round(c))):02x}" for c in channels)
    if v in NAMED:
        return NAMED[v]
    raise Refused(f"colour {value.strip()[:20]!r}")


def length(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    v = value.strip()
    m = NUMBER.match(v)
    unit = v[m.end() :].strip() if m else None
    if not m or unit not in UNITS:
        raise Refused(f"length {v[:20]!r}")
    return float(m.group(0)) * UNITS[unit]


def fraction(value: str | None) -> float:
    if value is None:
        return 1.0
    v = value.strip()
    out = float(v[:-1]) / 100 if v.endswith("%") else float(v)
    return max(0.0, min(1.0, out))


def _style(element: ET.Element) -> dict[str, str]:
    props = {k: v for k, v in element.attrib.items() if "}" not in k}
    for item in props.pop("style", "").split(";"):
        if ":" in item:
            key, _, value = item.partition(":")
            props[key.strip()] = value.replace("!important", "").strip()
    return props


@dataclass
class Shape:
    svg: str  # one subset element, without its paint
    fill: str
    opacity: float

    def write(self) -> str:
        paint = f' fill="{self.fill}"'
        if self.opacity < 1:
            paint += f' fill-opacity="{self.opacity:.4f}"'
        return self.svg.replace("/>", paint + "/>")


def _fmt(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".") if not math.isnan(x) else "0"


def _signed_area(points: list[Point]) -> float:
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1], strict=True))


def _inside(p: Point, polygon: list[Point]) -> bool:
    x, y = p
    inside = False
    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def _flatten(subpath: list[tuple]) -> list[Point]:
    out: list[Point] = []
    cur: Point = (0.0, 0.0)
    for seg in subpath:
        if seg[0] in "ML":
            cur = seg[1]
            out.append(cur)
        elif seg[0] == "C":
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = cur, seg[1], seg[2], seg[3]
            for i in range(1, 9):
                t = i / 8
                u = 1 - t
                out.append(
                    (
                        u**3 * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t**3 * x3,
                        u**3 * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t**3 * y3,
                    )
                )
            cur = seg[3]
        elif seg[0] == "Q":
            (x0, y0), (x1, y1), (x2, y2) = cur, seg[1], seg[2]
            for i in range(1, 9):
                t = i / 8
                u = 1 - t
                out.append((u * u * x0 + 2 * u * t * x1 + t * t * x2, u * u * y0 + 2 * u * t * y1 + t * t * y2))
            cur = seg[2]
    return out


def _reverse(subpath: list[tuple]) -> list[tuple]:
    closed = subpath[-1][0] == "Z"
    segs = [s for s in subpath if s[0] != "Z"]
    start = segs[0][1]
    ends = [s[-1] for s in segs]
    if closed and ends[-1] != start:
        segs.append(("L", start))
        ends.append(start)
    out: list[tuple] = [("M", ends[-1])]
    for i in range(len(segs) - 1, 0, -1):
        seg, previous = segs[i], ends[i - 1]
        if closed and i == 1 and seg[0] == "L":
            break  # the closing line back to the start: Z draws it
        if seg[0] == "L":
            out.append(("L", previous))
        elif seg[0] == "C":
            out.append(("C", seg[2], seg[1], previous))
        else:
            out.append(("Q", seg[1], previous))
    if closed:
        out.append(("Z",))
    return out


def _subpaths(segments: list[tuple]) -> list[list[tuple]]:
    out: list[list[tuple]] = []
    for seg in segments:
        if seg[0] == "M" or not out:
            if seg[0] != "M":
                raise Refused("path data must start with M")
            out.append([seg])
        elif out[-1][-1][0] == "Z":
            # a command straight after Z starts from the subpath's start (SVG's implicit M)
            out.append([("M", out[-1][0][1]), seg])
        else:
            out[-1].append(seg)
    return [s for s in out if any(seg[0] in "LCQ" for seg in s)]


def _nonzero(subpaths: list[list[tuple]]) -> list[list[tuple]]:
    """Even-odd as non-zero: orient each subpath by its nesting depth (outer clockwise, a hole the other way, ...)."""
    polygons = [_flatten(s) for s in subpaths]
    out = []
    for i, (subpath, polygon) in enumerate(zip(subpaths, polygons, strict=True)):
        probe = polygon[0]
        depth = sum(_inside(probe, other) for j, other in enumerate(polygons) if j != i and len(other) >= 3)
        want = 1 if depth % 2 == 0 else -1
        area = _signed_area(polygon)
        out.append(_reverse(subpath) if area * want < 0 else subpath)
    return out


def _write_path(subpaths: list[list[tuple]]) -> str:
    def pt(p: Point) -> str:
        return f"{_fmt(p[0])} {_fmt(p[1])}"

    return " ".join(seg[0] + " ".join(pt(p) for p in seg[1:]) for s in subpaths for seg in s)


def _ellipse_d(cx: float, cy: float, rx: float, ry: float) -> str:
    return f"M{cx - rx} {cy}A{rx} {ry} 0 1 0 {cx + rx} {cy}A{rx} {ry} 0 1 0 {cx - rx} {cy}Z"


def _rounded_rect_d(x: float, y: float, w: float, h: float, rx: float, ry: float) -> str:
    rx, ry = min(rx, w / 2), min(ry, h / 2)
    return (
        f"M{x + rx} {y}L{x + w - rx} {y}A{rx} {ry} 0 0 1 {x + w} {y + ry}L{x + w} {y + h - ry}"
        f"A{rx} {ry} 0 0 1 {x + w - rx} {y + h}L{x + rx} {y + h}A{rx} {ry} 0 0 1 {x} {y + h - ry}"
        f"L{x} {y + ry}A{rx} {ry} 0 0 1 {x + rx} {y}Z"
    )


def _geometry(tag: str, a: dict[str, str], m: Matrix, evenodd: bool) -> str | None:
    """One shape's subset element (no paint) under the matrix m, or None if it draws nothing."""
    axis = abs(m[1]) < 1e-12 and abs(m[2]) < 1e-12
    if tag == "rect":
        x, y, w, h = (length(a.get(k)) for k in ("x", "y", "width", "height"))
        if w <= 0 or h <= 0:
            return None
        rx, ry = a.get("rx"), a.get("ry")
        rxv = length(rx if rx is not None else ry)
        ryv = length(ry if ry is not None else rx)
        if rxv > 0 and ryv > 0:
            return _geometry("path", {"d": _rounded_rect_d(x, y, w, h, rxv, ryv)}, m, False)
        if axis:
            (x0, y0), (x1, y1) = apply(m, (x, y)), apply(m, (x + w, y + h))
            return (
                f'<rect x="{_fmt(min(x0, x1))}" y="{_fmt(min(y0, y1))}" width="{_fmt(abs(x1 - x0))}"'
                f' height="{_fmt(abs(y1 - y0))}"/>'
            )
        corners = [apply(m, p) for p in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))]
        return '<polygon points="' + " ".join(f"{_fmt(px)},{_fmt(py)}" for px, py in corners) + '"/>'
    if tag in ("circle", "ellipse"):
        cx, cy = length(a.get("cx")), length(a.get("cy"))
        if tag == "circle":
            rx = ry = length(a.get("r"))
        else:
            rx, ry = length(a.get("rx", a.get("ry"))), length(a.get("ry", a.get("rx")))
        if rx <= 0 or ry <= 0:
            return None
        if axis:
            (px, py), sx, sy = apply(m, (cx, cy)), abs(m[0]) * rx, abs(m[3]) * ry
            if abs(sx - sy) < 1e-9:
                return f'<circle cx="{_fmt(px)}" cy="{_fmt(py)}" r="{_fmt(sx)}"/>'
            return f'<ellipse cx="{_fmt(px)}" cy="{_fmt(py)}" rx="{_fmt(sx)}" ry="{_fmt(sy)}"/>'
        return _geometry("path", {"d": _ellipse_d(cx, cy, rx, ry)}, m, False)
    if tag in ("polygon", "polyline"):
        numbers = [float(n) for n in NUMBER.findall(a.get("points", ""))]
        if len(numbers) % 2:
            numbers = numbers[:-1]
        points = [apply(m, p) for p in zip(numbers[::2], numbers[1::2], strict=True)]
        if len(points) < 3:
            return None
        if evenodd and _crosses(points):
            raise Refused("even-odd polygon that crosses itself")
        return '<polygon points="' + " ".join(f"{_fmt(px)},{_fmt(py)}" for px, py in points) + '"/>'
    if tag == "path":
        segments = []
        for seg in absolute(a.get("d", "")):
            segments.append((seg[0], *(apply(m, p) for p in seg[1:])))
        subpaths = _subpaths(segments)
        if not subpaths:
            return None
        if evenodd and len(subpaths) > 1:
            subpaths = _nonzero(subpaths)
        elif evenodd and _crosses(_flatten(subpaths[0])):
            raise Refused("even-odd path that crosses itself")
        return f'<path d="{_write_path(subpaths)}"/>'
    raise Refused(f"element {tag}")


def _crosses(points: list[Point]) -> bool:
    """Whether a closed polyline crosses itself (where even-odd and non-zero could differ). Quadratic, but outlines are short."""
    n = len(points)
    if n < 4:
        return False
    if n > 1200:
        return True  # too long to check: refuse rather than guess
    p = np.asarray(points, dtype=np.float64)
    a, b = p, np.roll(p, -1, axis=0)  # edge i runs a[i] -> b[i]

    def side(o: np.ndarray, d: np.ndarray, q: np.ndarray) -> np.ndarray:
        # the cross product (d - o) x (q - o), for every edge pair: o, d indexed [i, None], q [None, j]
        return (d[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1]) - (d[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0])

    ai, bi, aj, bj = a[:, None], b[:, None], a[None, :], b[None, :]
    d1, d2 = side(aj, bj, ai), side(aj, bj, bi)
    d3, d4 = side(ai, bi, aj), side(ai, bi, bj)
    proper = (d1 * d2 < 0) & (d3 * d4 < 0)
    i, j = np.indices((n, n))
    neighbours = (np.abs(i - j) <= 1) | (np.abs(i - j) == n - 1)
    return bool((proper & ~neighbours).any())


def _canvas(root: ET.Element, size: int) -> Matrix:
    box = root.get("viewBox")
    if box:
        v = [float(x) for x in NUMBER.findall(box)]
        if len(v) != 4 or v[2] <= 0 or v[3] <= 0:
            raise Refused(f"viewBox {box[:40]!r}")
        x, y, w, h = v
    else:
        # with no viewBox, user units are px, so the size is converted from its unit
        x, y = 0.0, 0.0
        if not root.get("width") or not root.get("height"):
            raise Refused("no viewBox and no size")
        w, h = length(root.get("width")), length(root.get("height"))
        if w <= 0 or h <= 0:
            raise Refused("no viewBox and no size")
    if root.get("preserveAspectRatio", "").strip() not in ("", "xMidYMid", "xMidYMid meet"):
        raise Refused("preserveAspectRatio")
    s = size / max(w, h)
    return (s, 0.0, 0.0, s, -x * s + (size - w * s) / 2, -y * s + (size - h * s) / 2)


def _local(tag: str) -> str | None:
    """An element's SVG name, or None for an element of another namespace (Inkscape's, Sodipodi's: not drawn)."""
    if tag.startswith("{"):
        ns, _, name = tag[1:].partition("}")
        return name if ns == SVG_NS else None
    return tag


def _user_subpaths(tag: str, a: dict[str, str]) -> list[tuple[list[Segment], bool]]:
    """A shape's outline in user space, for stroking: per subpath its lines and cubics, and whether it is closed."""
    if tag == "path":
        d = a.get("d", "")
    elif tag == "rect":
        x, y, w, h = (length(a.get(k)) for k in ("x", "y", "width", "height"))
        if w <= 0 or h <= 0:
            return []
        rx, ry = a.get("rx"), a.get("ry")
        rxv, ryv = length(rx if rx is not None else ry), length(ry if ry is not None else rx)
        d = _rounded_rect_d(x, y, w, h, rxv, ryv) if rxv > 0 and ryv > 0 else f"M{x} {y}H{x + w}V{y + h}H{x}Z"
    elif tag in ("circle", "ellipse"):
        cx, cy = length(a.get("cx")), length(a.get("cy"))
        if tag == "circle":
            rx = ry = length(a.get("r"))
        else:
            rx, ry = length(a.get("rx", a.get("ry"))), length(a.get("ry", a.get("rx")))
        if rx <= 0 or ry <= 0:
            return []
        d = _ellipse_d(cx, cy, rx, ry)
    elif tag in ("polygon", "polyline"):
        numbers = [float(n) for n in NUMBER.findall(a.get("points", ""))]
        pairs = list(zip(numbers[::2], numbers[1::2], strict=False))
        if len(pairs) < 2:
            return []
        d = "M" + " L".join(f"{x} {y}" for x, y in pairs) + ("Z" if tag == "polygon" else "")
    else:  # line
        d = f"M{length(a.get('x1'))} {length(a.get('y1'))}L{length(a.get('x2'))} {length(a.get('y2'))}"
    out: list[tuple[list[Segment], bool]] = []
    segs: list[Segment] = []
    cur = start = (0.0, 0.0)
    for op in absolute(d):
        kind = op[0]
        if kind == "M":
            if segs:
                out.append((segs, False))
            segs, cur, start = [], op[1], op[1]
        elif kind == "L":
            if op[1] != cur:
                segs.append((cur, op[1]))
            cur = op[1]
        elif kind in "CQ":
            if kind == "Q":  # as a cubic
                q, p = op[1], op[2]
                c1 = (cur[0] + 2 / 3 * (q[0] - cur[0]), cur[1] + 2 / 3 * (q[1] - cur[1]))
                c2 = (p[0] + 2 / 3 * (q[0] - p[0]), p[1] + 2 / 3 * (q[1] - p[1]))
                seg: Segment = (cur, c1, c2, p)
            else:
                seg = (cur, op[1], op[2], op[3])
            if any(q != cur for q in seg[1:]):
                segs.append(seg)
            cur = seg[-1]
        else:  # Z
            if cur != start:
                segs.append((cur, start))
            if segs:
                out.append((segs, True))
            segs, cur = [], start
    if segs:
        out.append((segs, False))
    return out


def convert(text: str, size: int = 128, stats: dict[str, int] | None = None) -> str:
    """A canonical subset document on a `size` x `size` canvas. Raises Refused, with the reason, for what isn't exact.
    `stats`, if given, counts the strokes outlined by each kind of outline (`svgformer.stroke`)."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise Refused(f"xml: {str(e)[:40]}") from None
    if _local(root.tag) != "svg":
        raise Refused("not an svg")

    ids = {el.get("id"): el for el in root.iter() if el.get("id")}

    def chosen(switch: ET.Element) -> list[ET.Element]:
        """A <switch>'s one rendered child: the first a renderer without extensions would take (Illustrator's files)."""
        for child in switch:
            tag = _local(child.tag) if isinstance(child.tag, str) else None
            conditional = any(k in child.attrib for k in ("requiredExtensions", "requiredFeatures", "systemLanguage"))
            if tag is not None and tag != "foreignObject" and not conditional:
                return [child]
        return []

    def walk(children: list[ET.Element], matrix: Matrix, inherited: dict[str, str], depth: int = 0) -> list[Shape]:
        if depth > 64:
            raise Refused("nested too deeply")
        out: list[Shape] = []
        for child in children:
            tag = _local(child.tag) if isinstance(child.tag, str) else None
            if tag is None or tag in DEFINITIONS:
                continue
            if tag == "style":
                if (child.text or "").strip():
                    raise Refused("style sheet")
                continue
            props = _style(child)
            if props.get("display", "").strip() == "none":
                continue
            for key in ("clip-path", "mask", "filter"):
                if props.get(key, "none").strip() not in ("none", ""):
                    raise Refused(key)
            here = dict(inherited)
            here.update({k: props[k] for k in INHERITED if k in props})
            m = compose(matrix, parse_transform(props["transform"])) if "transform" in props else matrix
            opacity = fraction(props.get("opacity"))
            if tag in CONTAINERS or tag in ("switch", "use"):
                if tag == "svg":
                    raise Refused("nested svg")
                if tag == "switch":
                    inner = walk(chosen(child), m, here, depth + 1)
                elif tag == "use":
                    href = child.get("href") or child.get("{http://www.w3.org/1999/xlink}href") or ""
                    target = ids.get(href[1:]) if href.startswith("#") else None
                    if target is None or _local(target.tag) in ("symbol", "svg"):
                        raise Refused("use of something it can't place")
                    shift = (1.0, 0.0, 0.0, 1.0, length(child.get("x")), length(child.get("y")))
                    # a <use> renders its target as if it were a <g> child, so the target's own transform still applies
                    inner = walk([target], compose(m, shift), here, depth + 1)
                else:
                    inner = walk(list(child), m, here, depth + 1)
                if opacity < 1 and len(inner) > 1:
                    raise Refused("opacity on a group of several shapes")
                for shape in inner:
                    shape.opacity *= opacity
                out.extend(s for s in inner if s.opacity > 0)
                continue
            if tag not in ("rect", "circle", "ellipse", "polygon", "polyline", "path", "line"):
                raise Refused(f"element {tag}")
            if here.get("visibility", "visible").strip() in ("hidden", "collapse"):
                continue
            stroke = colour(here.get("stroke", "none"))
            stroke_alpha = opacity * fraction(here.get("stroke-opacity"))
            width = length(here.get("stroke-width"), 1.0)
            stroked = stroke is not None and width > 0 and stroke_alpha > 0
            fill = None if tag == "line" else colour(here.get("fill", "#000000"))
            alpha = opacity * fraction(here.get("fill-opacity"))
            filled = fill is not None and alpha > 0
            if stroked and filled and opacity < 1:
                raise Refused("opacity on a filled, stroked shape")  # SVG fades the two together, not each
            if filled:
                evenodd = here.get("fill-rule", "nonzero").strip() == "evenodd"
                geometry = _geometry(tag, props, m, evenodd)
                if geometry is not None:
                    out.append(Shape(geometry, fill, alpha))
            if stroked:
                if here.get("stroke-dasharray", "none").strip() not in ("none", ""):
                    raise Refused("dashed stroke")
                if props.get("vector-effect", "none").strip() not in ("none", ""):
                    raise Refused("vector-effect")
                if filled and props.get("paint-order", "normal").strip() not in ("normal", "fill", "fill stroke"):
                    raise Refused("paint-order")
                join = here.get("stroke-linejoin", "miter").strip()
                cap = here.get("stroke-linecap", "butt").strip()
                if join not in ("miter", "round", "bevel", "miter-clip", "arcs") or cap not in (
                    "butt",
                    "round",
                    "square",
                ):
                    raise Refused(f"stroke-linejoin {join} / stroke-linecap {cap}")
                limit = float(here.get("stroke-miterlimit", "4"))
                result = outline(_user_subpaths(tag, props), width, join, cap, limit, m, size)
                if result is not None:
                    d, kind = result
                    if stats is not None:
                        stats[kind] = stats.get(kind, 0) + 1
                    out.append(Shape(f'<path d="{d}"/>', stroke, stroke_alpha))  # type: ignore[arg-type]
        return out

    root_props = _style(root)
    inherited = {k: root_props[k] for k in INHERITED if k in root_props}
    canvas = _canvas(root, size)
    if "transform" in root_props:
        canvas = compose(canvas, parse_transform(root_props["transform"]))
    shapes = walk(list(root), canvas, inherited)
    if not shapes:
        raise Refused("draws nothing")

    def document(shapes: list[Shape]) -> Svg:
        body = "\n".join(s.write() for s in shapes)
        try:
            return Svg(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}">\n{body}\n</svg>')
        except ValueError as e:
            raise Refused(f"subset: {str(e)[:60]}") from None

    # drop what is drawn entirely off the canvas (Inkscape's off-page layers): it changes no pixel, but would clamp
    visible = []
    for shape, rings in zip(shapes, document(shapes).outlines(), strict=True):
        points = [p for ring in rings for p in ring]
        if points and _overlaps(points, size):
            visible.append(shape)
    if not visible:
        raise Refused("draws nothing")
    return document(visible).canonical()


def _overlaps(points: list[Point], size: int) -> bool:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return max(xs) > 0 and min(xs) < size and max(ys) > 0 and min(ys) < size
