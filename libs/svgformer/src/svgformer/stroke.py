"""Strokes as filled outlines (docs/design/0024, phase B): the subset has no `stroke`, so a stroked shape's stroke becomes
a second shape, a filled path in the stroke's colour, painted after its fill as SVG paints them.

Two outlines are built for every stroke, both in user space (so a transform, even a skew, scales the stroke as SVG does) and
then mapped to the canvas:

- **the reference**, exact to its flattening: the union of a quad per flattened segment, a wedge at every flattened vertex,
  and the joins and caps, every piece wound the same way so the non-zero rule fills their union;
- **the compact outline**, what a person would write: each side offset as curves (a cubic split until it turns little, its
  control points pushed along the end normals), the inner side of a turn pivoting through the vertex, joins and caps as
  arcs. Short, but approximate: it is used only if it renders within `TOLERANCE` of the reference, and otherwise the
  offsets are flattened, and failing that the file is refused.

Dashes and `vector-effect` are refused by the caller.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from arena.twemoji import Refused, arc_to_cubics
from games.copycat import Svg

Point = tuple[float, float]
Matrix = tuple[float, float, float, float, float, float]
# a segment in user space: a line (p0, p1) or a cubic (p0, c1, c2, p3)
Segment = tuple[Point, ...]

TOLERANCE = 0.03  # the compact outline's coverage may differ from the reference's by this share of the stroke's area
ROOM = 0.5  # an inner join's offsets meet if that is within this share of the shorter neighbouring segment
SUPERSAMPLE = 4  # the reference's rendering, per pixel per axis
CURVE_TURN = math.pi / 8  # a cubic is split until its control polygon turns at most this much per piece


def _apply(m: Matrix, p: Point) -> Point:
    return (m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5])


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: Point, b: Point, k: float = 1.0) -> Point:
    return (a[0] + k * b[0], a[1] + k * b[1])


def _norm(v: Point) -> float:
    return math.hypot(v[0], v[1])


def _unit(v: Point) -> Point:
    n = _norm(v)
    return (v[0] / n, v[1] / n)


def _left(t: Point) -> Point:
    """The unit normal on the side `t` would turn toward with a positive cross product."""
    t = _unit(t)
    return (-t[1], t[0])


def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _tiny(v: Point) -> bool:
    return _norm(v) < 1e-9


def start_tangent(seg: Segment) -> Point:
    for q in seg[1:]:
        if not _tiny(d := _sub(q, seg[0])):
            return d
    return (1.0, 0.0)


def end_tangent(seg: Segment) -> Point:
    for q in reversed(seg[:-1]):
        if not _tiny(d := _sub(seg[-1], q)):
            return d
    return (1.0, 0.0)


def reverse(segs: list[Segment]) -> list[Segment]:
    return [tuple(reversed(s)) for s in reversed(segs)]


def _bezier(seg: Segment, t: float) -> Point:
    if len(seg) == 2:
        return _add(seg[0], _sub(seg[1], seg[0]), t)
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = seg
    u = 1 - t
    return (
        u**3 * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t**3 * x3,
        u**3 * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t**3 * y3,
    )


def _derivative(seg: Segment, t: float) -> Point:
    if len(seg) == 2:
        return _sub(seg[1], seg[0])
    p0, p1, p2, p3 = seg
    u = 1 - t
    d = (
        3 * u * u * (p1[0] - p0[0]) + 6 * u * t * (p2[0] - p1[0]) + 3 * t * t * (p3[0] - p2[0]),
        3 * u * u * (p1[1] - p0[1]) + 6 * u * t * (p2[1] - p1[1]) + 3 * t * t * (p3[1] - p2[1]),
    )
    if _tiny(d):  # a cusp or a doubled control point: the chord nearby is the direction
        a, b = _bezier(seg, max(0.0, t - 1e-3)), _bezier(seg, min(1.0, t + 1e-3))
        return _sub(b, a)
    return d


def _split(seg: Segment, t: float) -> tuple[Segment, Segment]:
    p0, p1, p2, p3 = seg
    a, b, c = _add(p0, _sub(p1, p0), t), _add(p1, _sub(p2, p1), t), _add(p2, _sub(p3, p2), t)
    d, e = _add(a, _sub(b, a), t), _add(b, _sub(c, b), t)
    f = _add(d, _sub(e, d), t)
    return (p0, a, d, f), (f, e, c, p3)


def _turning(seg: Segment) -> float:
    edges = [e for e in (_sub(seg[i + 1], seg[i]) for i in range(len(seg) - 1)) if not _tiny(e)]
    total = 0.0
    for a, b in itertools.pairwise(edges):
        total += abs(math.atan2(_cross(a, b), a[0] * b[0] + a[1] * b[1]))
    return total


def _length(seg: Segment) -> float:
    return sum(_norm(_sub(seg[i + 1], seg[i])) for i in range(len(seg) - 1))


class _Outliner:
    def __init__(self, h: float, join: str, cap: str, limit: float, m: Matrix):
        self.h, self.join, self.cap, self.limit, self.m = h, join, cap, limit, m
        self.scale = math.sqrt(abs(m[0] * m[3] - m[1] * m[2])) or 1.0  # canvas px per user unit, on average

    def steps(self, user_length: float, px: float) -> int:
        """Pieces to flatten a run of this user-space length into, about `px` canvas px each."""
        return max(1, min(256, math.ceil(user_length * self.scale / px)))

    # -- the compact outline: ops in user space, ("L", p) / ("C", c1, c2, p) ------------------------------------------

    def offset(self, seg: Segment, flat: bool) -> list[tuple]:
        h = self.h
        if len(seg) == 2:
            n = _left(_sub(seg[1], seg[0]))
            return [("L", _add(seg[1], n, h))]
        if flat:
            k = self.steps(_length(seg), 1.0)
            return [("L", _add(_bezier(seg, i / k), _left(_derivative(seg, i / k)), h)) for i in range(1, k + 1)]
        pieces = [seg]
        for _ in range(4):  # up to 16 pieces
            if all(_turning(p) <= CURVE_TURN for p in pieces):
                break
            pieces = [half for p in pieces for half in (_split(p, 0.5) if _turning(p) > CURVE_TURN else (p,))]
        ops = []
        for p in pieces:
            n0, n1 = _left(start_tangent(p)), _left(end_tangent(p))
            ops.append(("C", _add(p[1], n0, h), _add(p[2], n1, h), _add(p[3], n1, h)))
        return ops

    def arc(self, centre: Point, a: Point, b: Point) -> list[tuple]:
        """Around `centre` from a to b, turning the negative way (the outer side of a join on the left), as cubics."""
        if _tiny(_sub(a, b)):
            return []
        return arc_to_cubics(a, self.h, self.h, 0.0, False, False, b)

    def join_ops(
        self, p: Point, t0: Point, t1: Point, room: float = 0.0, ops: list[tuple] | None = None
    ) -> list[tuple]:
        """The ops from the offset end of one segment to the offset start of the next, around the vertex p. On the inner
        side of a turn the two offsets meet where they cross, if that is within `room` of p along both segments (the
        previous op, the last of `ops`, is moved to end there); otherwise the outline pivots through p, which non-zero
        still fills but which folds back on itself where segments are short."""
        h = self.h
        n0, n1 = _left(t0), _left(t1)
        a, b = _add(p, n0, h), _add(p, n1, h)
        cross, dot = _cross(_unit(t0), _unit(t1)), _unit(t0)[0] * _unit(t1)[0] + _unit(t0)[1] * _unit(t1)[1]
        if abs(cross) < 1e-9 and dot > 0:
            return [] if _tiny(_sub(a, b)) else [("L", b)]
        if cross > 0:  # the inner side of the turn
            half = math.acos(max(-1.0, min(1.0, n0[0] * n1[0] + n0[1] * n1[1]))) / 2
            if ops and half < math.pi / 2 - 1e-6 and h * math.tan(half) <= room:
                meet = _add(p, _unit(_add(n0, n1)), h / math.cos(half))
                last = ops[-1]
                shift = _sub(meet, last[-1])
                ops[-1] = (*last[:-1], meet) if last[0] == "L" else ("C", last[1], _add(last[2], shift), meet)
                return []
            return [("L", p), ("L", b)]
        if self.join == "round":
            if dot < 0 and abs(cross) < 1e-9:  # a reversal: half a turn, in two quarters
                mid = _add(p, _unit(t0), h)
                return self.arc(p, a, mid) + self.arc(p, mid, b)
            return self.arc(p, a, b)
        if self.join in ("miter", "miter-clip", "arcs"):
            bisector = _add(n0, n1)
            if not _tiny(bisector):
                half = math.acos(max(-1.0, min(1.0, n0[0] * n1[0] + n0[1] * n1[1]))) / 2
                ratio = 1 / max(math.cos(half), 1e-12)
                if ratio <= self.limit:
                    return [("L", _add(p, _unit(bisector), h * ratio)), ("L", b)]
        return [("L", b)]  # bevel, or a miter over its limit

    def side(self, segs: list[Segment], closed: bool, flat: bool) -> tuple[Point, list[tuple]]:
        start = _add(segs[0][0], _left(start_tangent(segs[0])), self.h)
        ops: list[tuple] = []
        chords = [_norm(_sub(s[-1], s[0])) for s in segs]
        for i, seg in enumerate(segs):
            ops += self.offset(seg, flat)
            if i + 1 < len(segs) or closed:
                j = (i + 1) % len(segs)
                # half of each neighbour, so the meets at both ends of a segment can't pass each other
                room = ROOM * min(chords[i], chords[j])
                if closed and j == 0:
                    room = 0.0  # the ring's start is fixed by its M: pivot there rather than move it
                ops += self.join_ops(seg[-1], end_tangent(seg), start_tangent(segs[j]), room, ops)
        return start, ops

    def cap_ops(self, p: Point, t: Point) -> list[tuple]:
        """From the left offset at an end (direction t, pointing out) round to the right offset."""
        h, n = self.h, _left(t)
        a, b, out = _add(p, n, h), _add(p, n, -h), _unit(t)
        if self.cap == "round":
            mid = _add(p, out, h)
            return self.arc(p, a, mid) + self.arc(p, mid, b)
        if self.cap == "square":
            return [("L", _add(a, out, h)), ("L", _add(b, out, h)), ("L", b)]
        return [("L", b)]

    def compact(self, subpaths: list[tuple[list[Segment], bool]], flat: bool) -> list[list[tuple]]:
        rings = []
        for segs, closed in subpaths:
            if closed:
                for run in (segs, reverse(segs)):
                    start, ops = self.side(run, True, flat)
                    rings.append([("M", start), *ops, ("Z",)])
            else:
                start, ops = self.side(segs, False, flat)
                back_start, back = self.side(reverse(segs), False, flat)
                ring = [("M", start), *ops, *self.cap_ops(segs[-1][-1], end_tangent(segs[-1]))]
                ring += back[:]  # the cap ended at the right offset of the end, which is where `back` starts
                ring += self.cap_ops(segs[0][0], _sub((0.0, 0.0), start_tangent(segs[0])))
                del back_start
                rings.append([*ring, ("Z",)])
        return [[(op[0], *(_apply(self.m, q) for q in op[1:])) for op in ring] for ring in rings]

    # -- the reference: a union of pieces, each a polygon in user space -----------------------------------------------

    def circle(self, p: Point) -> list[Point]:
        n = max(12, min(96, math.ceil(2 * math.pi * self.h * self.scale / 0.75)))
        return [
            (p[0] + self.h * math.cos(2 * math.pi * i / n), p[1] + self.h * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]

    def reference(self, subpaths: list[tuple[list[Segment], bool]]) -> list[list[Point]]:
        h = self.h
        pieces: list[list[Point]] = []
        for segs, closed in subpaths:
            for i, seg in enumerate(segs):
                k = 1 if len(seg) == 2 else self.steps(_length(seg), 0.35)
                pts = [_bezier(seg, j / k) for j in range(k + 1)]
                pts = [q for j, q in enumerate(pts) if j == 0 or not _tiny(_sub(q, pts[j - 1]))]
                normals = [_left(_sub(b, a)) for a, b in itertools.pairwise(pts)]
                for (a, b), n in zip(itertools.pairwise(pts), normals, strict=True):
                    pieces.append([_add(a, n, h), _add(b, n, h), _add(b, n, -h), _add(a, n, -h)])
                for j in range(1, len(normals)):  # the wedges between a curve's flattened segments, both sides
                    q, n0, n1 = pts[j], normals[j - 1], normals[j]
                    pieces.append([q, _add(q, n0, h), _add(q, n1, h)])
                    pieces.append([q, _add(q, n0, -h), _add(q, n1, -h)])
                if i + 1 < len(segs) or closed:
                    nxt = segs[(i + 1) % len(segs)]
                    pieces += self.reference_join(seg[-1], end_tangent(seg), start_tangent(nxt))
            if not closed:
                for p, t in (
                    (segs[-1][-1], end_tangent(segs[-1])),
                    (segs[0][0], _sub((0.0, 0.0), start_tangent(segs[0]))),
                ):
                    if self.cap == "round":
                        pieces.append(self.circle(p))
                    elif self.cap == "square":
                        n, out = _left(t), _unit(t)
                        pieces.append(
                            [_add(p, n, h), _add(_add(p, n, h), out, h), _add(_add(p, n, -h), out, h), _add(p, n, -h)]
                        )
        return [[_apply(self.m, q) for q in piece] for piece in pieces if len(piece) >= 3]

    def reference_join(self, p: Point, t0: Point, t1: Point) -> list[list[Point]]:
        if self.join == "round":
            return [self.circle(p)]
        # the outer side is the one the turn leans away from: flip the frame so it is always the negative-cross case
        sign = -1.0 if _cross(t0, t1) > 0 else 1.0
        n0, n1 = _left(t0), _left(t1)
        a, b = _add(p, n0, sign * self.h), _add(p, n1, sign * self.h)
        if self.join in ("miter", "miter-clip", "arcs"):
            bisector = _add(n0, n1)
            if not _tiny(bisector):
                half = math.acos(max(-1.0, min(1.0, n0[0] * n1[0] + n0[1] * n1[1]))) / 2
                ratio = 1 / max(math.cos(half), 1e-12)
                if ratio <= self.limit:
                    return [[p, a, _add(p, _unit(bisector), sign * self.h * ratio), b]]
        return [[p, a, b]]


def _signed_area(points: list[Point]) -> float:
    return 0.5 * sum(_cross(a, b) for a, b in zip(points, points[1:] + points[:1], strict=True))


def _fmt(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def _ops_d(rings: list[list[tuple]]) -> str:
    return " ".join(op[0] + " ".join(f"{_fmt(q[0])} {_fmt(q[1])}" for q in op[1:]) for ring in rings for op in ring)


def _polygons_d(polygons: list[list[Point]], k: float = 1.0, origin: Point = (0.0, 0.0)) -> str:
    """Polygons as path data, each wound positively, mapped by (p - origin) * k."""
    out = []
    for poly in polygons:
        if abs(_signed_area(poly)) < 1e-12:
            continue
        if _signed_area(poly) < 0:
            poly = poly[::-1]
        pts = " L".join(f"{_fmt((x - origin[0]) * k)} {_fmt((y - origin[1]) * k)}" for x, y in poly)
        out.append("M" + pts + " Z")
    return " ".join(out)


def _coverage(d: str, width: int, height: int | None = None) -> np.ndarray:
    height = height or width
    doc = Svg(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><path d="{d}" fill="#000000"/></svg>'
    )
    rgba = np.frombuffer(doc.render_rgba8(), dtype=np.uint8).reshape(height, width, 4)
    return rgba[..., 3].astype(np.float32) / 255


def _reference_coverage(polygons: list[list[Point]], size: int) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """The union's coverage over its bounding box on the canvas, rendered SUPERSAMPLE times finer and averaged down.

    The rasterizer accumulates signed area, so where pieces overlap at an anti-aliased edge it counts them twice: a union
    of overlapping pieces renders a little fat at 1x. Finer, that bias shrinks with the pixel."""
    xs = [x for poly in polygons for x, _ in poly]
    ys = [y for poly in polygons for _, y in poly]
    x0, y0 = max(0, math.floor(min(xs)) - 1), max(0, math.floor(min(ys)) - 1)
    x1, y1 = min(size, math.ceil(max(xs)) + 1), min(size, math.ceil(max(ys)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    k = SUPERSAMPLE
    d = _polygons_d(polygons, k, (x0, y0))
    if not d:
        return None
    fine = _coverage(d, (x1 - x0) * k, (y1 - y0) * k)
    return fine.reshape(y1 - y0, k, x1 - x0, k).mean(axis=(1, 3)), (x0, y0, x1, y1)


def outline(
    subpaths: list[tuple[list[Segment], bool]],
    width: float,
    join: str,
    cap: str,
    limit: float,
    m: Matrix,
    size: int,
) -> tuple[str, str] | None:
    """A stroke's outline as path data on the canvas, and which outline it is ("curves" or "flat"); None if the stroke
    covers nothing on the canvas. Raises Refused if neither compact outline renders close enough to the reference (the
    reference itself is never written: it renders a little fat, and it is far too long to learn from)."""
    o = _Outliner(width / 2, join, cap, limit, m)
    subpaths = [(segs, closed) for segs, closed in subpaths if segs]
    if not subpaths:
        return None
    polygons = [poly for poly in o.reference(subpaths) if abs(_signed_area(poly)) >= 1e-12]
    if not polygons:
        return None
    found = _reference_coverage(polygons, size)
    if found is None:
        return None
    reference, (x0, y0, x1, y1) = found
    area = float(reference.sum())
    if area < 1e-3:
        return None
    for kind in ("curves", "flat"):
        d = _ops_d(o.compact(subpaths, flat=kind == "flat"))
        try:
            mine = _coverage(d, size)[y0:y1, x0:x1]
        except ValueError:
            continue
        if float(np.abs(mine - reference).sum()) <= TOLERANCE * area:
            return d, kind
    raise Refused("stroke it can't outline compactly")
