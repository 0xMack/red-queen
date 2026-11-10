"""SVG as tokens (docs/design/0024 Decision 2): an SVG in Copy Cat's subset to a sequence of token ids, and back.

A sequence is `BOS`, then each shape in paint order (bottom first), then `END`. A shape is

    TYPE  geometry  colour colour colour  [OPACITY_k]

- `TYPE` is one of `rect`, `circle`, `ellipse`, `polygon`, `path`.
- A colour is three tokens of 6 bits a channel (`COLOUR_0` to `COLOUR_63`; level k is `round(k * 255 / 63)`).
- An opacity below 1 is one of 16 levels (`OPACITY_k` is k / 16); an opaque shape has none.
- Geometry is numbers, each one `NUM_i` token on a grid of 1,280 steps over the canvas plus a margin of 1/8 of it on every
  side (1/8 px at 128 px, 1/4 px at 256): `rect` x y width height, `circle` cx cy r, `ellipse` cx cy rx ry; `polygon`
  x y pairs then `POLY_END`; `path` commands (`M`, `L`, `Q`, `C`, `Z`), each followed by its x y pairs, then `PATH_END`.
  A second `M` after a `Z` starts a subpath, which is how holes are drawn.

The geometry comes before the paint, so a shape's colour is predicted knowing which region it is the colour of: read off
the image there. (Colour first was tried first: the model had to commit to a colour before saying where, and colours
were barely learned -- 0024, P2.) Groups are not
tokens: a decoded SVG puts each shape in its own group, as the MDL tracer writes them. Numbers outside the grid's range are
clamped to its ends and counted (`Encoded.clamped`).
"""

from __future__ import annotations

import dataclasses
import re

from games.copycat import Svg

GRID = 1280
COLOURS = 64
OPACITIES = 16

SPECIALS = ("PAD", "BOS", "END")
TYPES = ("rect", "circle", "ellipse", "polygon", "path")
COMMANDS = ("M", "L", "Q", "C", "Z")
ENDS = ("POLY_END", "PATH_END")

# The vocabulary: specials, types, commands, ends, opacities, colours, numbers -- in that order.
NAMES: tuple[str, ...] = (
    *SPECIALS,
    *TYPES,
    *COMMANDS,
    *ENDS,
    *(f"OPACITY_{k}" for k in range(OPACITIES)),
    *(f"COLOUR_{k}" for k in range(COLOURS)),
    *(f"NUM_{i}" for i in range(GRID)),
)
ID: dict[str, int] = {name: i for i, name in enumerate(NAMES)}
VOCAB = len(NAMES)

PAD, BOS, END = ID["PAD"], ID["BOS"], ID["END"]
POLY_END, PATH_END = ID["POLY_END"], ID["PATH_END"]
TYPE_ID = {t: ID[t] for t in TYPES}
COMMAND_ID = {c: ID[c] for c in COMMANDS}
OPACITY0 = ID["OPACITY_0"]
COLOUR0 = ID["COLOUR_0"]
NUM0 = ID["NUM_0"]

# How many numbers each primitive and each path command takes.
PRIMITIVE_NUMBERS = {"rect": 4, "circle": 3, "ellipse": 4}
COMMAND_NUMBERS = {"M": 2, "L": 2, "Q": 4, "C": 6, "Z": 0}


def is_number(token: int) -> bool:
    return NUM0 <= token < NUM0 + GRID


def is_colour(token: int) -> bool:
    return COLOUR0 <= token < COLOUR0 + COLOURS


def is_opacity(token: int) -> bool:
    return OPACITY0 <= token < OPACITY0 + OPACITIES


@dataclasses.dataclass(frozen=True)
class Grid:
    """The number grid for a square canvas of `size` px."""

    size: int

    @property
    def margin(self) -> float:
        return self.size / 8

    @property
    def step(self) -> float:
        return (self.size + 2 * self.margin) / GRID

    def index(self, value: float) -> tuple[int, bool]:
        """The nearest grid index, and whether it had to be clamped."""
        i = round((value + self.margin) / self.step)
        clamped = min(max(i, 0), GRID - 1)
        return clamped, clamped != i

    def value(self, index: int) -> float:
        return -self.margin + index * self.step


@dataclasses.dataclass
class Encoded:
    tokens: list[int]
    clamped: int


_ELEMENT = re.compile(r"<(rect|circle|ellipse|polygon|path)\b([^>]*)/>")
_ATTRIBUTE = re.compile(r'([a-z-]+)="([^"]*)"')
_PATH_PART = re.compile(r"[MLQCZ]|-?(?:\d+\.?\d*|\.\d+)")


def colour_level(channel: int) -> int:
    return round(channel * (COLOURS - 1) / 255)


def colour_value(level: int) -> int:
    return round(level * 255 / (COLOURS - 1))


def opacity_level(opacity: float) -> int | None:
    """None for an opaque shape (anything that rounds to 16/16)."""
    level = round(opacity * OPACITIES)
    return None if level >= OPACITIES else level


def encode(svg: str, size: int) -> Encoded:
    """`svg` (any text the subset accepts, on a `size x size` canvas) as tokens."""
    document = Svg(svg)
    if (document.width, document.height) != (size, size):
        raise ValueError(f"a {document.width} x {document.height} SVG on a {size} px grid")
    grid = Grid(size)
    tokens = [BOS]
    clamped = 0

    def number(text: str) -> None:
        nonlocal clamped
        i, was = grid.index(float(text))
        clamped += was
        tokens.append(NUM0 + i)

    for kind, attributes in _ELEMENT.findall(document.canonical()):
        attrs = dict(_ATTRIBUTE.findall(attributes))
        subpaths = _drawn_subpaths(attrs["d"]) if kind == "path" else []
        if kind == "path" and not subpaths:
            continue  # draws nothing
        tokens.append(TYPE_ID[kind])
        if kind == "rect":
            for name in ("x", "y", "width", "height"):
                number(attrs[name])
        elif kind == "circle":
            for name in ("cx", "cy", "r"):
                number(attrs[name])
        elif kind == "ellipse":
            for name in ("cx", "cy", "rx", "ry"):
                number(attrs[name])
        elif kind == "polygon":
            for pair in attrs["points"].split():
                x, y = pair.split(",")
                number(x)
                number(y)
            tokens.append(POLY_END)
        else:
            for part in (p for subpath in subpaths for p in subpath):
                if part in COMMAND_ID:
                    tokens.append(COMMAND_ID[part])
                else:
                    number(part)
            tokens.append(PATH_END)
        fill = attrs["fill"]
        tokens.extend(COLOUR0 + colour_level(int(fill[i : i + 2], 16)) for i in (1, 3, 5))
        level = opacity_level(float(attrs.get("fill-opacity", "1")))
        if level is not None:
            tokens.append(OPACITY0 + level)
    tokens.append(END)
    return Encoded(tokens, clamped)


def _drawn_subpaths(d: str) -> list[list[str]]:
    """A path's subpaths (each from its `M`), less any with no `L`, `Q` or `C` in it: those draw nothing."""
    subpaths: list[list[str]] = []
    for part in _PATH_PART.findall(d):
        if part == "M" or not subpaths:
            subpaths.append([])
        subpaths[-1].append(part)
    return [s for s in subpaths if any(p in ("L", "Q", "C") for p in s)]


def _shapes(tokens: list[int]) -> list[list[int]]:
    """The tokens of each complete shape (BOS, END and anything after END dropped)."""
    body = tokens[1:] if tokens and tokens[0] == BOS else list(tokens)
    if END in body:
        body = body[: body.index(END)]
    starts = [i for i, t in enumerate(body) if t in TYPE_ID.values()]
    if not starts:
        return []
    return [body[a:b] for a, b in zip(starts, [*starts[1:], len(body)], strict=True)]


def decode(tokens: list[int], size: int) -> str:
    """Tokens as a canonical SVG, each shape its own group. A shape cut off before it is complete (a sequence that hit
    its length cap) is dropped, so any prefix the grammar allows decodes."""
    grid = Grid(size)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}">']
    for shape in _shapes(tokens):
        element = _element(shape, grid)
        if element is not None:
            out.append(f"<g>{element}</g>")
    out.append("</svg>")
    return Svg("\n".join(out)).canonical()


def _number(token: int, grid: Grid) -> str:
    return f"{grid.value(token - NUM0):.4f}"


def _element(shape: list[int], grid: Grid) -> str | None:
    kind = NAMES[shape[0]]
    # the paint ends the shape: three colours, then an opacity or nothing
    opacity = None
    if len(shape) > 1 and is_opacity(shape[-1]):
        opacity, shape = shape[-1], shape[:-1]
    if len(shape) < 4 or not all(is_colour(t) for t in shape[-3:]):
        return None
    r, g, b = (colour_value(t - COLOUR0) for t in shape[-3:])
    style = f'fill="#{r:02x}{g:02x}{b:02x}"'
    if opacity is not None:
        style += f' fill-opacity="{(opacity - OPACITY0) / OPACITIES:.4f}"'
    rest = shape[1:-3]
    if kind in PRIMITIVE_NUMBERS:
        n = PRIMITIVE_NUMBERS[kind]
        if len(rest) != n or not all(is_number(t) for t in rest):
            return None
        v = [_number(t, grid) for t in rest]
        names = {
            "rect": ("x", "y", "width", "height"),
            "circle": ("cx", "cy", "r"),
            "ellipse": ("cx", "cy", "rx", "ry"),
        }
        if kind == "rect" and (grid.value(rest[2] - NUM0) <= 0 or grid.value(rest[3] - NUM0) <= 0):
            return None
        if kind != "rect" and any(grid.value(t - NUM0) <= 0 for t in rest[2:]):
            return None
        attributes = " ".join(f'{name}="{value}"' for name, value in zip(names[kind], v, strict=True))
        return f"<{kind} {attributes} {style}/>"
    if kind == "polygon":
        if not rest or rest[-1] != POLY_END:
            return None
        numbers = rest[:-1]
        if len(numbers) < 6 or len(numbers) % 2 or not all(is_number(t) for t in numbers):
            return None
        points = " ".join(
            f"{_number(x, grid)},{_number(y, grid)}" for x, y in zip(numbers[::2], numbers[1::2], strict=True)
        )
        return f'<polygon points="{points}" {style}/>'
    if not rest or rest[-1] != PATH_END:
        return None
    # the subset's path syntax: a command letter, then its numbers space-separated ("M1 2L3 4Z")
    d = ""
    for t in rest[:-1]:
        if t in COMMAND_ID.values():
            d += NAMES[t]
        else:
            d += ("" if d[-1:].isalpha() else " ") + _number(t, grid)
    return f'<path d="{d}" {style}/>'


def describe(tokens: list[int]) -> str:
    """The token names, for reading a sequence."""
    return " ".join(NAMES[t] for t in tokens)
