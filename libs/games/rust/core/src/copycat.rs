//! Copy Cat (docs/design/0022): the SVG subset every entrant's output is read in, its canonical form, its counts, and
//! its render through `raster`.
//!
//! **The subset** (Decision 2): `rect`, `circle`, `ellipse`, `polygon`, and `path` with `M`/`L`/`C`/`Q`/`Z` (either
//! case); a solid `fill` (`#rgb`, `#rrggbb` or `rgb(r, g, b)`; black if absent, as in SVG) with `opacity` /
//! `fill-opacity`; `<g>` for layers, without attributes beyond `id`. The root `<svg>` may carry `xmlns`, `version`,
//! `width`, `height` and a `viewBox` of `0 0 W H`. An XML declaration, comments and whitespace are skipped. Anything
//! else -- a stroke, a transform, a gradient, an unknown element or attribute -- is refused with the reason, never
//! approximated.
//!
//! **The canonical form** (`Document::canonical`) is what shapes, nodes and bytes are counted from, so formatting
//! can't win: coordinates to 0.1 px, absolute path commands, `fill` as `#rrggbb`, ids dropped, a `viewBox` of the
//! image's size. Parsing it gives back the same document, rounded.

use crate::raster::{coverage, ellipse, Canvas, Mask, Outline, PathBuilder, Point};

#[derive(Clone, Debug, PartialEq)]
pub enum Segment {
    Move(Point),
    Line(Point),
    Quad(Point, Point),
    Cubic(Point, Point, Point),
    Close,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Geometry {
    Rect { x: f64, y: f64, width: f64, height: f64 },
    Circle { cx: f64, cy: f64, r: f64 },
    Ellipse { cx: f64, cy: f64, rx: f64, ry: f64 },
    Polygon(Vec<Point>),
    Path(Vec<Segment>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    pub geometry: Geometry,
    pub fill: [u8; 3],
    /// `opacity` x `fill-opacity`, in 0..1.
    pub opacity: f64,
    /// The innermost `<g>` it is in, an index into `Document::groups`.
    pub group: Option<usize>,
}

/// A `<g>`: only its place in the tree.
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub parent: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Document {
    pub width: u32,
    pub height: u32,
    pub groups: Vec<Group>,
    /// In paint order, bottom first.
    pub shapes: Vec<Shape>,
}

impl Shape {
    pub fn outline(&self) -> Outline {
        match &self.geometry {
            Geometry::Rect { x, y, width, height } => {
                vec![vec![
                    (*x, *y),
                    (x + width, *y),
                    (x + width, y + height),
                    (*x, y + height),
                ]]
            }
            Geometry::Circle { cx, cy, r } => ellipse(*cx, *cy, *r, *r),
            Geometry::Ellipse { cx, cy, rx, ry } => ellipse(*cx, *cy, *rx, *ry),
            Geometry::Polygon(points) => {
                if points.len() > 2 {
                    vec![points.clone()]
                } else {
                    Vec::new()
                }
            }
            Geometry::Path(segments) => {
                let mut path = PathBuilder::new();
                for segment in segments {
                    match *segment {
                        Segment::Move(p) => path.move_to(p),
                        Segment::Line(p) => path.line_to(p),
                        Segment::Quad(c, p) => path.quad_to(c, p),
                        Segment::Cubic(c1, c2, p) => path.cubic_to(c1, c2, p),
                        Segment::Close => path.close(),
                    }
                }
                path.finish()
            }
        }
    }

    /// On-curve points of the outline it draws: a path's commands (a `Z` adds none), a polygon's vertices, 4 for a
    /// rect, and 4 for a circle or ellipse (the four Béziers it is drawn as).
    pub fn nodes(&self) -> usize {
        match &self.geometry {
            Geometry::Rect { .. } | Geometry::Circle { .. } | Geometry::Ellipse { .. } => 4,
            Geometry::Polygon(points) => points.len(),
            Geometry::Path(segments) => segments.iter().filter(|s| !matches!(s, Segment::Close)).count(),
        }
    }

    /// Total absolute turning angle along its flattened outline, in radians (2 pi for any convex shape; a jagged edge
    /// turns far more): smoothness, Decision 3. Zero-length edges are skipped.
    pub fn turning(&self) -> f64 {
        let mut total = 0.0;
        for polyline in self.outline() {
            let mut edges: Vec<Point> = Vec::with_capacity(polyline.len());
            for i in 0..polyline.len() {
                let (a, b) = (polyline[i], polyline[(i + 1) % polyline.len()]);
                let e = (b.0 - a.0, b.1 - a.1);
                if e.0 != 0.0 || e.1 != 0.0 {
                    edges.push(e);
                }
            }
            for i in 0..edges.len() {
                let (e, f) = (edges[i], edges[(i + 1) % edges.len()]);
                let cross = e.0 * f.1 - e.1 * f.0;
                let dot = e.0 * f.0 + e.1 * f.1;
                total += libm::atan2(cross.abs(), dot);
            }
        }
        total
    }

    /// Its coverage alone, opacity and everything above it ignored (layer recovery's whole-shape IoU).
    pub fn mask(&self, width: u32, height: u32) -> Mask {
        coverage(&self.outline(), width as usize, height as usize)
    }
}

impl Document {
    pub fn parse(text: &str) -> Result<Document, String> {
        Parser::new(text).document()
    }

    pub fn nodes(&self) -> usize {
        self.shapes.iter().map(Shape::nodes).sum()
    }

    /// Distinct fill colours.
    pub fn palette(&self) -> usize {
        let mut fills: Vec<[u8; 3]> = self.shapes.iter().map(|s| s.fill).collect();
        fills.sort_unstable();
        fills.dedup();
        fills.len()
    }

    pub fn turning(&self) -> f64 {
        self.shapes.iter().map(Shape::turning).sum()
    }

    pub fn render(&self) -> Canvas {
        let mut canvas = Canvas::new(self.width as usize, self.height as usize);
        for shape in &self.shapes {
            let rgb = shape.fill.map(|c| c as f64 / 255.0);
            canvas.fill(&shape.mask(self.width, self.height), rgb, shape.opacity);
        }
        canvas
    }

    /// The canonical serialization (see the module docs).
    pub fn canonical(&self) -> String {
        let mut out = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\">\n",
            self.width, self.height
        );
        let mut open: Vec<usize> = Vec::new();
        for shape in &self.shapes {
            let chain = self.chain(shape.group);
            let common = open.iter().zip(&chain).take_while(|(a, b)| a == b).count();
            for _ in common..open.len() {
                out.push_str("</g>\n");
            }
            open.truncate(common);
            for &g in &chain[common..] {
                out.push_str("<g>\n");
                open.push(g);
            }
            write_shape(&mut out, shape);
        }
        for _ in &open {
            out.push_str("</g>\n");
        }
        out.push_str("</svg>\n");
        out
    }

    /// The groups from the root down to `group`.
    fn chain(&self, mut group: Option<usize>) -> Vec<usize> {
        let mut chain = Vec::new();
        while let Some(g) = group {
            chain.push(g);
            group = self.groups[g].parent;
        }
        chain.reverse();
        chain
    }
}

/// Rounded to 0.1, without a trailing `.0` (and never `-0`).
pub fn number(v: f64) -> String {
    let tenths = (v * 10.0).round() as i64;
    let (sign, abs) = if tenths < 0 { ("-", -tenths) } else { ("", tenths) };
    if abs % 10 == 0 {
        format!("{sign}{}", abs / 10)
    } else {
        format!("{sign}{}.{}", abs / 10, abs % 10)
    }
}

fn write_shape(out: &mut String, shape: &Shape) {
    let n = number;
    match &shape.geometry {
        Geometry::Rect { x, y, width, height } => out.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"",
            n(*x),
            n(*y),
            n(*width),
            n(*height)
        )),
        Geometry::Circle { cx, cy, r } => {
            out.push_str(&format!("<circle cx=\"{}\" cy=\"{}\" r=\"{}\"", n(*cx), n(*cy), n(*r)))
        }
        Geometry::Ellipse { cx, cy, rx, ry } => out.push_str(&format!(
            "<ellipse cx=\"{}\" cy=\"{}\" rx=\"{}\" ry=\"{}\"",
            n(*cx),
            n(*cy),
            n(*rx),
            n(*ry)
        )),
        Geometry::Polygon(points) => {
            let points: Vec<String> = points.iter().map(|p| format!("{},{}", n(p.0), n(p.1))).collect();
            out.push_str(&format!("<polygon points=\"{}\"", points.join(" ")));
        }
        Geometry::Path(segments) => {
            let mut d = String::new();
            for segment in segments {
                let (command, points): (char, Vec<Point>) = match *segment {
                    Segment::Move(p) => ('M', vec![p]),
                    Segment::Line(p) => ('L', vec![p]),
                    Segment::Quad(c, p) => ('Q', vec![c, p]),
                    Segment::Cubic(c1, c2, p) => ('C', vec![c1, c2, p]),
                    Segment::Close => ('Z', vec![]),
                };
                d.push(command);
                let numbers: Vec<String> = points.iter().flat_map(|p| [n(p.0), n(p.1)]).collect();
                d.push_str(&numbers.join(" "));
            }
            out.push_str(&format!("<path d=\"{d}\""));
        }
    }
    let [r, g, b] = shape.fill;
    out.push_str(&format!(" fill=\"#{r:02x}{g:02x}{b:02x}\""));
    let percent = (shape.opacity * 100.0).round() as i64;
    if percent < 100 {
        out.push_str(&format!(" fill-opacity=\"{}\"", format_percent(percent)));
    }
    out.push_str("/>\n");
}

/// 0.01 steps: `0.5`, `0.07`, `0`.
fn format_percent(percent: i64) -> String {
    match percent {
        0 => "0".to_string(),
        p if p % 10 == 0 => format!("0.{}", p / 10),
        p => format!("0.{p:02}"),
    }
}

// --- parsing ---------------------------------------------------------------------------------------------------

struct Parser<'a> {
    text: &'a str,
    pos: usize,
}

type Attributes<'a> = Vec<(&'a str, &'a str)>;

enum Tag<'a> {
    Open {
        name: &'a str,
        attributes: Attributes<'a>,
        closed: bool,
    },
    Close(&'a str),
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Self {
        Parser { text, pos: 0 }
    }

    fn rest(&self) -> &'a str {
        &self.text[self.pos..]
    }

    fn skip_whitespace(&mut self) {
        let rest = self.rest();
        self.pos += rest.len() - rest.trim_start().len();
    }

    /// Skips whitespace, comments and an XML declaration; refuses any other text between tags.
    fn skip_misc(&mut self) -> Result<(), String> {
        loop {
            self.skip_whitespace();
            let rest = self.rest();
            if let Some(after) = rest.strip_prefix("<!--") {
                let end = after.find("-->").ok_or("unterminated comment")?;
                self.pos += 4 + end + 3;
            } else if let Some(after) = rest.strip_prefix("<?xml") {
                let end = after.find("?>").ok_or("unterminated XML declaration")?;
                self.pos += 5 + end + 2;
            } else if rest.is_empty() || rest.starts_with('<') {
                return Ok(());
            } else {
                return Err(format!("text content is not in the subset: {:?}", preview(rest)));
            }
        }
    }

    fn tag(&mut self) -> Result<Option<Tag<'a>>, String> {
        self.skip_misc()?;
        let rest = self.rest();
        if rest.is_empty() {
            return Ok(None);
        }
        if rest.starts_with("<!") {
            return Err(format!("not in the subset: {:?}", preview(rest)));
        }
        if let Some(after) = rest.strip_prefix("</") {
            let end = after.find('>').ok_or("unterminated end tag")?;
            self.pos += 2 + end + 1;
            return Ok(Some(Tag::Close(after[..end].trim())));
        }
        self.pos += 1;
        let name = self.name()?;
        let mut attributes = Vec::new();
        loop {
            self.skip_whitespace();
            let rest = self.rest();
            if rest.starts_with("/>") {
                self.pos += 2;
                return Ok(Some(Tag::Open {
                    name,
                    attributes,
                    closed: true,
                }));
            }
            if rest.starts_with('>') {
                self.pos += 1;
                return Ok(Some(Tag::Open {
                    name,
                    attributes,
                    closed: false,
                }));
            }
            if rest.is_empty() {
                return Err(format!("unterminated <{name}>"));
            }
            let key = self.name()?;
            self.skip_whitespace();
            if !self.rest().starts_with('=') {
                return Err(format!("attribute {key} on <{name}> has no value"));
            }
            self.pos += 1;
            self.skip_whitespace();
            let quote = self.rest().chars().next().filter(|c| *c == '"' || *c == '\'');
            let quote = quote.ok_or_else(|| format!("attribute {key} on <{name}> is not quoted"))?;
            let after = &self.rest()[1..];
            let end = after
                .find(quote)
                .ok_or_else(|| format!("unterminated attribute {key}"))?;
            let value = &after[..end];
            if value.contains('&') {
                return Err(format!("entities are not in the subset ({key} on <{name}>)"));
            }
            if attributes.iter().any(|(k, _)| *k == key) {
                return Err(format!("attribute {key} repeated on <{name}>"));
            }
            attributes.push((key, value));
            self.pos += 1 + end + 1;
        }
    }

    fn name(&mut self) -> Result<&'a str, String> {
        let rest = self.rest();
        let len = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, ':' | '-' | '_' | '.')))
            .unwrap_or(rest.len());
        if len == 0 {
            return Err(format!("expected a name at {:?}", preview(rest)));
        }
        self.pos += len;
        Ok(&rest[..len])
    }

    fn document(mut self) -> Result<Document, String> {
        let (attributes, closed) = match self.tag()? {
            Some(Tag::Open {
                name: "svg",
                attributes,
                closed,
            }) => (attributes, closed),
            Some(Tag::Open { name, .. }) => return Err(format!("the root element must be <svg>, not <{name}>")),
            _ => return Err("no <svg> element".to_string()),
        };
        let (width, height) = svg_size(&attributes)?;
        let mut doc = Document {
            width,
            height,
            groups: Vec::new(),
            shapes: Vec::new(),
        };
        if !closed {
            let mut stack: Vec<Option<usize>> = vec![None]; // the open elements' groups: svg, then each <g>
            loop {
                match self.tag()? {
                    None => return Err("unterminated <svg>".to_string()),
                    Some(Tag::Close(name)) => {
                        let expected = if stack.len() > 1 { "g" } else { "svg" };
                        if name != expected {
                            return Err(format!("</{name}> closes <{expected}>"));
                        }
                        stack.pop();
                        if stack.is_empty() {
                            break;
                        }
                    }
                    Some(Tag::Open {
                        name,
                        attributes,
                        closed,
                    }) => {
                        let parent = *stack.last().unwrap();
                        if name == "g" {
                            only(&attributes, "g", &["id"])?;
                            doc.groups.push(Group { parent });
                            if !closed {
                                stack.push(Some(doc.groups.len() - 1));
                            }
                        } else {
                            doc.shapes.push(shape(name, &attributes, parent)?);
                            if !closed {
                                // a shape may only be empty: <rect ...></rect>
                                match self.tag()? {
                                    Some(Tag::Close(end)) if end == name => {}
                                    _ => return Err(format!("<{name}> may not have content")),
                                }
                            }
                        }
                    }
                }
            }
        }
        self.skip_misc()?;
        if !self.rest().is_empty() {
            return Err("content after </svg>".to_string());
        }
        Ok(doc)
    }
}

fn preview(text: &str) -> &str {
    let end = text.char_indices().nth(24).map_or(text.len(), |(i, _)| i);
    &text[..end]
}

fn only(attributes: &Attributes, element: &str, allowed: &[&str]) -> Result<(), String> {
    for (key, _) in attributes {
        if !allowed.contains(key) {
            return Err(format!("attribute {key} on <{element}> is not in the subset"));
        }
    }
    Ok(())
}

fn get<'a>(attributes: &Attributes<'a>, key: &str) -> Option<&'a str> {
    attributes.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

fn svg_size(attributes: &Attributes) -> Result<(u32, u32), String> {
    for (key, value) in attributes {
        let ok = match *key {
            "xmlns" => *value == "http://www.w3.org/2000/svg",
            "version" | "width" | "height" | "viewBox" => true,
            _ => false,
        };
        if !ok {
            return Err(format!("attribute {key}=\"{value}\" on <svg> is not in the subset"));
        }
    }
    let pixels = |key: &str| -> Result<Option<u32>, String> {
        get(attributes, key)
            .map(|v| size(v.trim().strip_suffix("px").unwrap_or(v.trim())).ok_or(format!("bad {key} on <svg>: {v}")))
            .transpose()
    };
    let (width, height) = (pixels("width")?, pixels("height")?);
    let view_box = match get(attributes, "viewBox") {
        Some(v) => {
            let numbers = numbers(v)?;
            if numbers.len() != 4 || numbers[0] != 0.0 || numbers[1] != 0.0 {
                return Err(format!("the viewBox must be 0 0 W H, not {v}"));
            }
            let w = size_of(numbers[2]).ok_or(format!("bad viewBox width: {v}"))?;
            let h = size_of(numbers[3]).ok_or(format!("bad viewBox height: {v}"))?;
            Some((w, h))
        }
        None => None,
    };
    match (view_box, width, height) {
        (Some(vb), w, h) => {
            if w.is_some_and(|w| w != vb.0) || h.is_some_and(|h| h != vb.1) {
                return Err("width/height differ from the viewBox (a scale is not in the subset)".to_string());
            }
            Ok(vb)
        }
        (None, Some(w), Some(h)) => Ok((w, h)),
        _ => Err("<svg> needs a viewBox or both width and height".to_string()),
    }
}

/// A whole, positive pixel count.
fn size(text: &str) -> Option<u32> {
    text.parse::<f64>().ok().and_then(size_of)
}

fn size_of(v: f64) -> Option<u32> {
    ((1.0..=65_536.0).contains(&v) && v.fract() == 0.0).then_some(v as u32)
}

fn shape(name: &str, attributes: &Attributes, group: Option<usize>) -> Result<Shape, String> {
    const PAINT: [&str; 4] = ["id", "fill", "opacity", "fill-opacity"];
    let geometry_keys: &[&str] = match name {
        "rect" => &["x", "y", "width", "height"],
        "circle" => &["cx", "cy", "r"],
        "ellipse" => &["cx", "cy", "rx", "ry"],
        "polygon" => &["points"],
        "path" => &["d"],
        _ => return Err(format!("<{name}> is not in the subset")),
    };
    for (key, value) in attributes {
        if !(PAINT.contains(key) || geometry_keys.contains(key)) {
            return Err(format!("attribute {key}=\"{value}\" on <{name}> is not in the subset"));
        }
    }
    let length = |key: &str, default: Option<f64>| -> Result<f64, String> {
        match get(attributes, key) {
            None => default.ok_or(format!("<{name}> needs {key}")),
            Some(v) => {
                let t = v.trim();
                let t = t.strip_suffix("px").unwrap_or(t);
                t.parse::<f64>()
                    .ok()
                    .filter(|x| x.is_finite())
                    .ok_or(format!("bad {key} on <{name}>: {v}"))
            }
        }
    };
    let nonnegative = |key: &str, v: f64| -> Result<f64, String> {
        if v < 0.0 {
            Err(format!("negative {key} on <{name}>"))
        } else {
            Ok(v)
        }
    };
    let geometry = match name {
        "rect" => Geometry::Rect {
            x: length("x", Some(0.0))?,
            y: length("y", Some(0.0))?,
            width: nonnegative("width", length("width", None)?)?,
            height: nonnegative("height", length("height", None)?)?,
        },
        "circle" => Geometry::Circle {
            cx: length("cx", Some(0.0))?,
            cy: length("cy", Some(0.0))?,
            r: nonnegative("r", length("r", None)?)?,
        },
        "ellipse" => Geometry::Ellipse {
            cx: length("cx", Some(0.0))?,
            cy: length("cy", Some(0.0))?,
            rx: nonnegative("rx", length("rx", None)?)?,
            ry: nonnegative("ry", length("ry", None)?)?,
        },
        "polygon" => {
            let numbers = numbers(get(attributes, "points").unwrap_or(""))?;
            if numbers.len() % 2 != 0 {
                return Err("<polygon> points has an odd count of numbers".to_string());
            }
            Geometry::Polygon(numbers.as_chunks::<2>().0.iter().map(|p| (p[0], p[1])).collect())
        }
        _ => Geometry::Path(path_data(get(attributes, "d").unwrap_or(""))?),
    };
    let fill = match get(attributes, "fill") {
        None => [0, 0, 0],
        Some(v) => colour(v).ok_or(format!("fill=\"{v}\" is not a solid colour in the subset"))?,
    };
    let fraction = |key: &str| -> Result<f64, String> {
        match get(attributes, key) {
            None => Ok(1.0),
            Some(v) => v
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|x| x.is_finite())
                .map(|x| x.clamp(0.0, 1.0))
                .ok_or(format!("bad {key} on <{name}>: {v}")),
        }
    };
    Ok(Shape {
        geometry,
        fill,
        opacity: fraction("opacity")? * fraction("fill-opacity")?,
        group,
    })
}

fn colour(text: &str) -> Option<[u8; 3]> {
    let text = text.trim();
    if let Some(hex) = text.strip_prefix('#') {
        let digit = |i: usize| u8::from_str_radix(hex.get(i..i + 1)?, 16).ok();
        return match hex.len() {
            3 => Some([digit(0)? * 17, digit(1)? * 17, digit(2)? * 17]),
            6 => {
                let byte = |i: usize| u8::from_str_radix(hex.get(i..i + 2)?, 16).ok();
                Some([byte(0)?, byte(2)?, byte(4)?])
            }
            _ => None,
        };
    }
    let inner = text.strip_prefix("rgb(")?.strip_suffix(')')?;
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.len() != 3 {
        return None;
    }
    let mut rgb = [0u8; 3];
    for (slot, part) in rgb.iter_mut().zip(parts) {
        *slot = part.parse::<u8>().ok()?;
    }
    Some(rgb)
}

/// SVG number lists: separated by whitespace and/or one comma, or by nothing where a sign or a second decimal point
/// starts the next number (`1-2`, `.5.5`).
fn numbers(text: &str) -> Result<Vec<f64>, String> {
    let mut scanner = Scanner { text, pos: 0 };
    let mut out = Vec::new();
    loop {
        scanner.separators();
        if scanner.done() {
            return Ok(out);
        }
        out.push(scanner.number()?);
    }
}

struct Scanner<'a> {
    text: &'a str,
    pos: usize,
}

impl Scanner<'_> {
    fn done(&self) -> bool {
        self.pos >= self.text.len()
    }

    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.pos).copied()
    }

    fn separators(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r' | b',')) {
            self.pos += 1;
        }
    }

    fn number(&mut self) -> Result<f64, String> {
        let bytes = self.text.as_bytes();
        let start = self.pos;
        let mut i = self.pos;
        if matches!(bytes.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        let digits = |i: &mut usize| {
            let from = *i;
            while bytes.get(*i).is_some_and(u8::is_ascii_digit) {
                *i += 1;
            }
            *i > from
        };
        let whole = digits(&mut i);
        let mut fraction = false;
        if bytes.get(i) == Some(&b'.') {
            i += 1;
            fraction = digits(&mut i);
        }
        if !whole && !fraction {
            return Err(format!("expected a number at {:?}", preview(&self.text[start..])));
        }
        if matches!(bytes.get(i), Some(b'e' | b'E')) {
            let mut j = i + 1;
            if matches!(bytes.get(j), Some(b'+' | b'-')) {
                j += 1;
            }
            if digits(&mut j) {
                i = j;
            }
        }
        self.pos = i;
        self.text[start..i]
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .ok_or_else(|| format!("bad number {:?}", &self.text[start..i]))
    }
}

fn path_data(d: &str) -> Result<Vec<Segment>, String> {
    let mut scanner = Scanner { text: d, pos: 0 };
    let mut segments = Vec::new();
    let mut current = (0.0, 0.0);
    let mut start = (0.0, 0.0);
    let mut command: Option<u8> = None;
    loop {
        scanner.separators();
        let Some(next) = scanner.peek() else { break };
        if next.is_ascii_alphabetic() {
            scanner.pos += 1;
            command = Some(next);
            if matches!(next, b'Z' | b'z') {
                segments.push(Segment::Close);
                current = start;
                continue;
            }
        }
        let c = command.ok_or("path data must start with M")?;
        let relative = c.is_ascii_lowercase();
        let point = |scanner: &mut Scanner| -> Result<Point, String> {
            let x = scanner.number()?;
            scanner.separators();
            let y = scanner.number()?;
            Ok(if relative {
                (current.0 + x, current.1 + y)
            } else {
                (x, y)
            })
        };
        let segment = match c.to_ascii_uppercase() {
            b'M' => {
                let p = point(&mut scanner)?;
                start = p;
                // further pairs after a moveto are linetos (SVG 1.1, 8.3.2)
                command = Some(if relative { b'l' } else { b'L' });
                Segment::Move(p)
            }
            b'L' => Segment::Line(point(&mut scanner)?),
            b'Q' => {
                let c1 = point(&mut scanner)?;
                scanner.separators();
                Segment::Quad(c1, point(&mut scanner)?)
            }
            b'C' => {
                let c1 = point(&mut scanner)?;
                scanner.separators();
                let c2 = point(&mut scanner)?;
                scanner.separators();
                Segment::Cubic(c1, c2, point(&mut scanner)?)
            }
            other => return Err(format!("path command {} is not in the subset", other as char)),
        };
        if segments.is_empty() && !matches!(segment, Segment::Move(_)) {
            return Err("path data must start with M".to_string());
        }
        current = match segment {
            Segment::Move(p) | Segment::Line(p) | Segment::Quad(_, p) | Segment::Cubic(_, _, p) => p,
            Segment::Close => start,
        };
        segments.push(segment);
    }
    Ok(segments)
}

// --- scoring helpers ---------------------------------------------------------------------------------------------

/// Soft intersection-over-union of two masks: sum of min over sum of max. 0 when both are empty.
pub fn mask_iou(a: &Mask, b: &Mask) -> f64 {
    let x0 = a.x0.min(b.x0);
    let y0 = a.y0.min(b.y0);
    let x1 = (a.x0 + a.width).max(b.x0 + b.width);
    let y1 = (a.y0 + a.height).max(b.y0 + b.height);
    let (mut inter, mut union) = (0.0, 0.0);
    if !a.is_empty() && !b.is_empty() {
        let disjoint =
            a.x0 >= b.x0 + b.width || b.x0 >= a.x0 + a.width || a.y0 >= b.y0 + b.height || b.y0 >= a.y0 + a.height;
        if disjoint {
            return 0.0;
        }
    }
    for y in y0..y1 {
        for x in x0..x1 {
            let (p, q) = (a.at(x, y), b.at(x, y));
            inter += p.min(q);
            union += p.max(q);
        }
    }
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// Layer recovery (Decision 3): for each shape of `clean`, its best whole-shape IoU with any shape of `output`, each
/// rendered alone (occlusion and opacity ignored). High only if the parts were separated *and* their hidden portions
/// completed.
pub fn layer_recovery(clean: &Document, output: &Document) -> Vec<f64> {
    let candidates: Vec<Mask> = output
        .shapes
        .iter()
        .map(|s| s.mask(clean.width, clean.height))
        .filter(|m| m.area() > 0.0)
        .collect();
    clean
        .shapes
        .iter()
        .map(|part| {
            let mask = part.mask(clean.width, clean.height);
            candidates.iter().map(|c| mask_iou(&mask, c)).fold(0.0, f64::max)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r##"<?xml version="1.0" encoding="UTF-8"?>
<!-- a comment -->
<svg xmlns="http://www.w3.org/2000/svg" width="64px" height="48" viewBox="0 0 64 48">
  <rect x="1.04" y="2" width="20" height="10" fill="#f00"/>
  <g id="layer">
    <circle cx="30" cy="20.25" r="8" fill="rgb(0, 128, 255)" opacity="0.5" fill-opacity="0.5"></circle>
    <g>
      <ellipse cx="40" cy="30" rx="10" ry="4"/>
      <polygon points="1,2 3,4,5 6"/>
    </g>
  </g>
  <path d="m10 10 l5 0 5 5 q1 2 3 4 C1e1 2 3-4 .5.5 z M0 0 L1 1 Z" fill="#00Aa00"/>
</svg>"##;

    #[test]
    fn parses_the_subset() {
        let doc = Document::parse(SAMPLE).unwrap();
        assert_eq!((doc.width, doc.height), (64, 48));
        assert_eq!(doc.shapes.len(), 5);
        assert_eq!(doc.groups, vec![Group { parent: None }, Group { parent: Some(0) }]);
        assert_eq!(doc.shapes[0].fill, [255, 0, 0]);
        assert_eq!(doc.shapes[1].opacity, 0.25);
        assert_eq!(doc.shapes[1].group, Some(0));
        assert_eq!(doc.shapes[2].fill, [0, 0, 0]);
        assert_eq!(doc.shapes[3].group, Some(1));
        assert_eq!(
            doc.shapes[3].geometry,
            Geometry::Polygon(vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])
        );
        let Geometry::Path(segments) = &doc.shapes[4].geometry else {
            panic!()
        };
        assert_eq!(
            segments[..5],
            [
                Segment::Move((10.0, 10.0)),
                Segment::Line((15.0, 10.0)),
                Segment::Line((20.0, 15.0)),
                Segment::Quad((21.0, 17.0), (23.0, 19.0)),
                Segment::Cubic((10.0, 2.0), (3.0, -4.0), (0.5, 0.5)),
            ]
        );
        assert_eq!(doc.shapes[4].nodes(), 7);
        assert_eq!(doc.nodes(), 4 + 4 + 4 + 3 + 7);
        assert_eq!(doc.palette(), 4);
    }

    #[test]
    fn refuses_what_is_outside_the_subset() {
        let wrap =
            |inner: &str| format!("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 10 10\">{inner}</svg>");
        for (inner, reason) in [
            ("<rect width=\"1\" height=\"1\" stroke=\"red\"/>", "stroke"),
            ("<rect width=\"1\" height=\"1\" transform=\"scale(2)\"/>", "transform"),
            ("<rect width=\"1\" height=\"1\" fill=\"url(#g)\"/>", "solid colour"),
            ("<rect width=\"1\" height=\"1\" rx=\"1\"/>", "rx"),
            ("<g fill=\"red\"><rect width=\"1\" height=\"1\"/></g>", "fill"),
            ("<line x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\"/>", "<line>"),
            ("<text>hi</text>", "<text>"),
            ("<path d=\"M0 0 H5 V5 Z\"/>", "command H"),
            ("<path d=\"M0 0 A1 1 0 0 0 5 5\"/>", "command A"),
            ("<path d=\"L0 0\"/>", "start with M"),
            ("<circle cx=\"1\" cy=\"1\" r=\"-1\"/>", "negative r"),
            ("hello", "text content"),
        ] {
            let err = Document::parse(&wrap(inner)).unwrap_err();
            assert!(err.contains(reason), "{inner}: {err}");
        }
        assert!(Document::parse("<svg viewBox=\"0 0 10 10\" width=\"20\"></svg>").is_err());
        assert!(Document::parse("<svg viewBox=\"1 0 10 10\"></svg>").is_err());
        assert!(Document::parse("<svg></svg>").is_err());
        assert!(Document::parse("<svg viewBox=\"0 0 10 10\"><g></svg>").is_err());
    }

    #[test]
    fn canonical_form() {
        let doc = Document::parse(SAMPLE).unwrap();
        let text = doc.canonical();
        assert_eq!(
            text,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 64 48\">\n\
             <rect x=\"1\" y=\"2\" width=\"20\" height=\"10\" fill=\"#ff0000\"/>\n\
             <g>\n\
             <circle cx=\"30\" cy=\"20.3\" r=\"8\" fill=\"#0080ff\" fill-opacity=\"0.25\"/>\n\
             <g>\n\
             <ellipse cx=\"40\" cy=\"30\" rx=\"10\" ry=\"4\" fill=\"#000000\"/>\n\
             <polygon points=\"1,2 3,4 5,6\" fill=\"#000000\"/>\n\
             </g>\n\
             </g>\n\
             <path d=\"M10 10L15 10L20 15Q21 17 23 19C10 2 3 -4 0.5 0.5ZM0 0L1 1Z\" fill=\"#00aa00\"/>\n\
             </svg>\n"
        );
        assert_eq!(number(-0.04), "0");
        assert_eq!(number(-1.25), "-1.3");
        assert_eq!(number(0.05), "0.1");
    }

    #[test]
    fn round_trips() {
        let doc = Document::parse(SAMPLE).unwrap();
        let once = doc.canonical();
        let parsed = Document::parse(&once).unwrap();
        assert_eq!(parsed.canonical(), once);
        assert_eq!(parsed.shapes.len(), doc.shapes.len());
        for (a, b) in parsed.shapes.iter().zip(&doc.shapes) {
            assert_eq!(a.fill, b.fill);
            assert_eq!(a.nodes(), b.nodes());
            assert!((a.opacity - b.opacity).abs() <= 0.005);
            let (pa, pb) = (a.outline(), b.outline());
            for (p, q) in pa.iter().flatten().zip(pb.iter().flatten()) {
                assert!((p.0 - q.0).abs() <= 0.05 + 1e-9 && (p.1 - q.1).abs() <= 0.05 + 1e-9);
            }
        }
        // the group structure survives (indices renumbered, empty groups dropped)
        assert_eq!(
            parsed.shapes.iter().map(|s| s.group).collect::<Vec<_>>(),
            vec![None, Some(0), Some(1), Some(1), None]
        );
    }

    #[test]
    fn turning_measures_jaggedness() {
        let doc = Document::parse(
            "<svg viewBox=\"0 0 32 32\"><rect x=\"2\" y=\"2\" width=\"10\" height=\"10\"/>             <circle cx=\"20\" cy=\"20\" r=\"9\"/>             <polygon points=\"0,0 4,0 4,1 5,1 5,2 6,2 6,6 0,6\"/></svg>",
        )
        .unwrap();
        let tau = 2.0 * std::f64::consts::PI;
        assert!((doc.shapes[0].turning() - tau).abs() < 1e-12);
        assert!((doc.shapes[1].turning() - tau).abs() < 1e-9);
        // a staircase: two extra corners each way per step
        assert!((doc.shapes[2].turning() - 2.0 * tau).abs() < 1e-12);
    }

    #[test]
    fn renders_in_paint_order() {
        let doc = Document::parse(
            "<svg viewBox=\"0 0 4 2\"><rect width=\"4\" height=\"2\" fill=\"#ff0000\"/>\
             <rect x=\"2\" width=\"2\" height=\"2\" fill=\"#0000ff\" opacity=\"0.5\"/></svg>",
        )
        .unwrap();
        let canvas = doc.render();
        assert_eq!(&canvas.data[0..4], &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(&canvas.data[12..16], &[0.5, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn layer_recovery_wants_whole_parts() {
        let clean = Document::parse(
            "<svg viewBox=\"0 0 32 32\"><circle cx=\"12\" cy=\"12\" r=\"8\" fill=\"#ff0000\"/>\
             <rect x=\"14\" y=\"14\" width=\"12\" height=\"12\" fill=\"#0000ff\"/></svg>",
        )
        .unwrap();
        assert_eq!(layer_recovery(&clean, &clean), vec![1.0, 1.0]);
        // the circle cut out where the square covers it (here, its bounding box less the corner): not the part
        let cut = Document::parse(
            "<svg viewBox=\"0 0 32 32\"><polygon points=\"4,4 20,4 20,14 14,14 14,20 4,20\" fill=\"#ff0000\"/>             <rect x=\"14\" y=\"14\" width=\"12\" height=\"12\" fill=\"#0000ff\"/></svg>",
        )
        .unwrap();
        let recovered = layer_recovery(&clean, &cut);
        assert!(recovered[0] < 0.9 && recovered[0] > 0.5, "{recovered:?}");
        assert_eq!(recovered[1], 1.0);
    }
}
