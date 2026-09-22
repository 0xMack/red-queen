//! Snake -- the rules, its observers and its action adapter (docs/design/0007 interfaces), ported
//! from `libs/games/src/games/snake.py` (see that module's docstring for why the rules and the
//! observations are what they are). Behaviour matches the Python original step for step given the
//! same food placements; the only intended difference is the PRNG (`pcg::Pcg32`, not Mersenne
//! Twister), which is why seeds map to different food sequences than before (protocol
//! `snake.score.v2`, docs/design/0009). `libs/games/tests/test_native_parity.py` holds the two
//! implementations to that.

use std::collections::VecDeque;

use crate::pcg::Pcg32;

/// Clockwise, so a right turn (+1) is the next direction and a left turn (-1) the previous one:
/// RIGHT, DOWN, LEFT, UP.
pub const DIRECTIONS: [(i32, i32); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

pub type Cell = (i32, i32);

/// `egocentric.v1`'s rays as (ahead, right) unit steps in the head's frame: left, front-left, front,
/// front-right, right, back-left, back-right. Straight back is left out -- it is always the neck.
pub const EGO_RAYS: [(i32, i32); 7] = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, -1), (-1, 1)];
/// 3 values per ray, then food (2), tail (2), apples eaten, hunger.
pub const EGO_SIZE: usize = EGO_RAYS.len() * 3 + 2 + 2 + 1 + 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Label {
    Body,
    Head,
    Food,
}

impl Label {
    pub fn as_str(self) -> &'static str {
        match self {
            Label::Body => "body",
            Label::Head => "head",
            Label::Food => "food",
        }
    }
}

/// How a model sees the board. Ids are the versioned observer ids of docs/design/0007 -- an
/// observer's encoding never changes in place; a new encoding is a new variant here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observer {
    /// `features.v1`, level 3: danger straight/left/right, heading one-hot (RIGHT/DOWN/LEFT/UP),
    /// food left/right/up/down of the head. 11 values, all 0/1.
    Features,
    /// `grid-flat.v1`, level 1: every cell, row-major, empty 0 / body 1 / head 2 / food 3.
    GridFlat,
    /// `egocentric.v1`, level 2: line-of-sight rays in the head's own frame (`EGO_RAYS`), each giving wall / body /
    /// food proximity (1 / distance, 0 = not seen), then the food and tail as (ahead, right) offsets, apples eaten
    /// and the hunger clock. 27 values; no absolute heading, because everything is relative to it.
    Egocentric,
}

impl Observer {
    pub fn from_id(id: &str) -> Option<Observer> {
        match id {
            "features.v1" => Some(Observer::Features),
            "grid-flat.v1" => Some(Observer::GridFlat),
            "egocentric.v1" => Some(Observer::Egocentric),
            _ => None,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Observer::Features => "features.v1",
            Observer::GridFlat => "grid-flat.v1",
            Observer::Egocentric => "egocentric.v1",
        }
    }
}

/// `relative3.v1`: 3 outputs, argmax (first maximum wins, like Python's `max`) -> -1 / 0 / +1.
pub fn decode_relative3(outputs: &[f64]) -> i32 {
    let mut best = 0;
    for (i, &v) in outputs.iter().enumerate() {
        if v > outputs[best] {
            best = i;
        }
    }
    best as i32 - 1
}

#[derive(Clone, Debug)]
pub struct Snake {
    pub width: i32,
    pub height: i32,
    pub seed: u64,
    pub max_steps_without_food: u32,
    rng: Pcg32,
    /// Head first.
    pub body: VecDeque<Cell>,
    pub direction: usize,
    /// None only once the snake fills the board (there's nowhere left to put food).
    pub food: Option<Cell>,
    pub score: u32,
    pub alive: bool,
    pub steps_without_food: u32,
}

impl Snake {
    pub fn new(width: i32, height: i32, seed: u64, max_steps_without_food: Option<u32>) -> Snake {
        assert!(Snake::fits(width, height), "board too small for the starting snake");
        let mut snake = Snake {
            width,
            height,
            seed,
            max_steps_without_food: max_steps_without_food.unwrap_or(((width + height) * 4) as u32),
            rng: Pcg32::new(seed),
            body: VecDeque::new(),
            direction: 0,
            food: None,
            score: 0,
            alive: true,
            steps_without_food: 0,
        };
        snake.reset();
        snake
    }

    /// The starting snake is 3 cells long, tail two cells left of the middle column.
    pub fn fits(width: i32, height: i32) -> bool {
        width >= 4 && height >= 1
    }

    pub fn reset(&mut self) {
        self.rng = Pcg32::new(self.seed);
        let (mx, my) = (self.width / 2, self.height / 2);
        self.body = VecDeque::from([(mx, my), (mx - 1, my), (mx - 2, my)]);
        self.direction = 0;
        self.score = 0;
        self.alive = true;
        self.steps_without_food = 0;
        self.food = self.place_food();
    }

    fn in_bounds(&self, (x, y): Cell) -> bool {
        0 <= x && x < self.width && 0 <= y && y < self.height
    }

    /// Is `cell` part of the body, ignoring the last `skip_tail` segments?
    fn hits_body(&self, cell: Cell, skip_tail: usize) -> bool {
        let n = self.body.len().saturating_sub(skip_tail);
        self.body.iter().take(n).any(|&c| c == cell)
    }

    /// Every empty cell, x-major (`for x ... for y ...`, the Python original's order), and one of
    /// them chosen by the PRNG.
    fn place_food(&mut self) -> Option<Cell> {
        let mut empty = Vec::with_capacity((self.width * self.height) as usize);
        for x in 0..self.width {
            for y in 0..self.height {
                if !self.body.contains(&(x, y)) {
                    empty.push((x, y));
                }
            }
        }
        if empty.is_empty() {
            return None;
        }
        let i = self.rng.bounded(empty.len() as u32) as usize;
        Some(empty[i])
    }

    /// Apply a relative action (-1 left, 0 straight, +1 right; any integer, taken mod 4).
    /// Returns (reward, done).
    pub fn step(&mut self, action: i64) -> (f64, bool) {
        if !self.alive {
            return (0.0, true);
        }
        self.direction = (self.direction as i64 + action).rem_euclid(4) as usize;
        let (dx, dy) = DIRECTIONS[self.direction];
        let (hx, hy) = self.body[0];
        let new_head = (hx + dx, hy + dy);
        let will_eat = Some(new_head) == self.food;

        // The tail cell vacates this move unless the snake is growing, so moving into it is legal.
        if !self.in_bounds(new_head) || self.hits_body(new_head, if will_eat { 0 } else { 1 }) {
            self.alive = false;
            return (-1.0, true);
        }

        let food = self.food.expect("a live snake always has food to seek");
        let old_distance = (hx - food.0).abs() + (hy - food.1).abs();
        self.body.push_front(new_head);
        let reward;
        if will_eat {
            self.score += 1;
            self.steps_without_food = 0;
            self.food = self.place_food();
            reward = 1.0;
            if self.food.is_none() {
                // Filled the board: nothing left to eat, the game is won and over.
                self.alive = false;
                return (reward, true);
            }
        } else {
            self.body.pop_back();
            self.steps_without_food += 1;
            // Asymmetric shaping (see snake.py): closer +0.01, farther -0.02.
            let new_distance = (new_head.0 - food.0).abs() + (new_head.1 - food.1).abs();
            reward = if new_distance < old_distance { 0.01 } else { -0.02 };
        }
        if self.steps_without_food >= self.max_steps_without_food {
            self.alive = false;
            return (-1.0, true);
        }
        (reward, false)
    }

    /// Would this relative action hit a wall or the body (tail excluded) next step?
    fn danger(&self, relative: i64) -> f64 {
        let direction = (self.direction as i64 + relative).rem_euclid(4) as usize;
        let (dx, dy) = DIRECTIONS[direction];
        let next = (self.body[0].0 + dx, self.body[0].1 + dy);
        if !self.in_bounds(next) || self.hits_body(next, 1) {
            1.0
        } else {
            0.0
        }
    }

    /// What one ray sees from the head: `[wall, body, food]` proximity, each `1 / distance` in steps along the ray
    /// (0 when not seen). The wall is always found; food is only seen before the first body segment (line of
    /// sight). A body segment counts only if it will still be there when the head arrives -- the tail vacates as the
    /// snake moves, so segment `i` (head = 0) is gone after `len - i` moves, and a step along a diagonal ray takes
    /// two moves. At distance 1 that is exactly `danger()`.
    fn ego_ray(&self, (ahead, right): (i32, i32)) -> [f64; 3] {
        let (hx, hy) = self.body[0];
        let forward = DIRECTIONS[self.direction];
        let side = DIRECTIONS[(self.direction + 1) % 4];
        let (dx, dy) = (ahead * forward.0 + right * side.0, ahead * forward.1 + right * side.1);
        let moves_per_step = (ahead.abs() + right.abs()) as usize;
        let len = self.body.len();
        let (mut body, mut food) = (0.0, 0.0);
        let mut k = 1;
        loop {
            let cell = (hx + k * dx, hy + k * dy);
            if !self.in_bounds(cell) {
                return [1.0 / k as f64, body, food];
            }
            if body == 0.0 {
                if let Some(i) = self.body.iter().position(|&c| c == cell) {
                    if len - i > k as usize * moves_per_step {
                        body = 1.0 / k as f64;
                    }
                } else if food == 0.0 && Some(cell) == self.food {
                    food = 1.0 / k as f64;
                }
            }
            k += 1;
        }
    }

    /// `cell` relative to the head as (ahead, right), scaled by the board's longer side so it stays in [-1, 1].
    fn ego_offset(&self, cell: Cell) -> [f64; 2] {
        let (hx, hy) = self.body[0];
        let forward = DIRECTIONS[self.direction];
        let side = DIRECTIONS[(self.direction + 1) % 4];
        let (vx, vy) = (cell.0 - hx, cell.1 - hy);
        let scale = (self.width.max(self.height) - 1).max(1) as f64;
        [(vx * forward.0 + vy * forward.1) as f64 / scale, (vx * side.0 + vy * side.1) as f64 / scale]
    }

    pub fn encode(&self, observer: Observer) -> Vec<f64> {
        match observer {
            Observer::Features => {
                let (hx, hy) = self.body[0];
                let mut out = vec![self.danger(0), self.danger(-1), self.danger(1)];
                out.extend((0..4).map(|i| if i == self.direction { 1.0 } else { 0.0 }));
                let flag = |b: bool| if b { 1.0 } else { 0.0 };
                match self.food {
                    Some((fx, fy)) => out.extend([flag(fx < hx), flag(fx > hx), flag(fy < hy), flag(fy > hy)]),
                    None => out.extend([0.0; 4]),
                }
                out
            }
            Observer::Egocentric => {
                let mut out = Vec::with_capacity(EGO_SIZE);
                for ray in EGO_RAYS {
                    out.extend(self.ego_ray(ray));
                }
                out.extend(self.food.map_or([0.0, 0.0], |f| self.ego_offset(f)));
                out.extend(self.ego_offset(*self.body.back().expect("a snake has a tail")));
                out.push(self.score as f64 / (self.width * self.height) as f64);
                out.push(self.steps_without_food as f64 / self.max_steps_without_food as f64);
                out
            }
            Observer::GridFlat => {
                let mut grid = vec![0.0; (self.width * self.height) as usize];
                for (x, y, label) in self.cells() {
                    grid[(y * self.width + x) as usize] = match label {
                        Label::Body => 1.0,
                        Label::Head => 2.0,
                        Label::Food => 3.0,
                    };
                }
                grid
            }
        }
    }

    pub fn observation_size(&self, observer: Observer) -> usize {
        match observer {
            Observer::Features => 11,
            Observer::Egocentric => EGO_SIZE,
            Observer::GridFlat => (self.width * self.height) as usize,
        }
    }

    /// Labelled cells in the order renderers rely on: body behind the head (nearest the head first,
    /// tail last), then the head, then the food.
    pub fn cells(&self) -> Vec<(i32, i32, Label)> {
        let mut out: Vec<(i32, i32, Label)> = self.body.iter().skip(1).map(|&(x, y)| (x, y, Label::Body)).collect();
        out.push((self.body[0].0, self.body[0].1, Label::Head));
        if let Some((fx, fy)) = self.food {
            out.push((fx, fy, Label::Food));
        }
        out
    }

    /// Overwrite the position (tests and scenario setup): body head-first, heading, food.
    pub fn set_state(&mut self, body: Vec<Cell>, direction: usize, food: Option<Cell>) {
        assert!(!body.is_empty(), "a snake needs a head");
        self.body = body.into();
        self.direction = direction % 4;
        self.food = food;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_heading_right_in_the_middle() {
        let s = Snake::new(10, 10, 0, None);
        assert_eq!(s.body, VecDeque::from([(5, 5), (4, 5), (3, 5)]));
        let obs = s.encode(Observer::Features);
        assert_eq!(&obs[3..7], &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(s.encode(Observer::GridFlat).len(), 100);
    }

    #[test]
    fn moving_into_the_vacating_tail_is_legal() {
        let mut s = Snake::new(5, 5, 0, None);
        s.set_state(vec![(2, 2), (2, 1), (1, 1), (1, 2)], 2, Some((4, 4)));
        let (_, done) = s.step(0);
        assert!(!done);
    }

    #[test]
    fn wall_is_fatal() {
        let mut s = Snake::new(5, 5, 0, None);
        s.set_state(vec![(4, 2), (3, 2), (2, 2)], 0, Some((0, 0)));
        assert_eq!(s.step(0), (-1.0, true));
        assert!(!s.alive);
    }

    #[test]
    fn same_seed_same_food() {
        let a = Snake::new(10, 10, 123, None);
        let b = Snake::new(10, 10, 123, None);
        assert_eq!(a.food, b.food);
    }

    #[test]
    fn filling_the_board_ends_the_game_as_a_win() {
        // 4x1: the snake (3) plus one food fills the board -- nowhere left to place the next food.
        let mut s = Snake::new(4, 1, 0, None);
        assert_eq!(s.food, Some((3, 0)));
        assert_eq!(s.step(0), (1.0, true));
        assert_eq!((s.score, s.alive, s.food), (1, false, None));
        assert_eq!(s.encode(Observer::Features)[7..], [0.0; 4]);
    }

    #[test]
    fn egocentric_rays_see_walls_body_and_food_in_the_heads_frame() {
        let mut s = Snake::new(10, 10, 0, None);
        // Heading right at (5, 5), body trailing left; food two cells ahead.
        s.set_state(vec![(5, 5), (4, 5), (3, 5)], 0, Some((7, 5)));
        let obs = s.encode(Observer::Egocentric);
        assert_eq!(obs.len(), EGO_SIZE);
        let ray = |i: usize| obs[i * 3..i * 3 + 3].to_vec();
        // front: wall 5 cells away (x = 10), food at distance 2, no body ahead.
        assert_eq!(ray(2), vec![1.0 / 5.0, 0.0, 0.5]);
        // left is "up" on the board (y - 1): wall at distance 6 (y = -1), nothing else.
        assert_eq!(ray(0), vec![1.0 / 6.0, 0.0, 0.0]);
        // right is "down": wall at distance 5.
        assert_eq!(ray(4), vec![1.0 / 5.0, 0.0, 0.0]);
        // Food is 2 ahead, 0 to the right; the tail is 2 behind.
        assert_eq!(obs[21..25].to_vec(), vec![2.0 / 9.0, 0.0, -2.0 / 9.0, 0.0]);
    }

    #[test]
    fn egocentric_ignores_a_tail_that_will_have_vacated() {
        let mut s = Snake::new(5, 5, 0, None);
        // A U-shape: the tail sits directly in front of the head, one cell away -- it vacates on the first move.
        s.set_state(vec![(2, 2), (2, 1), (1, 1), (1, 2)], 2, Some((4, 4)));
        let obs = s.encode(Observer::Egocentric);
        assert_eq!(obs[2 * 3 + 1], 0.0); // front body proximity: the tail cell is free by the time we arrive
        assert_eq!(s.danger(0), 0.0); // and that is what features.v1 says too
    }

    #[test]
    fn egocentric_offsets_are_the_same_whatever_the_heading() {
        // The same local picture rotated a quarter turn about the head reads the same ahead/right offsets.
        let mut a = Snake::new(10, 10, 0, None);
        a.set_state(vec![(5, 5), (4, 5), (3, 5), (3, 4)], 0, Some((7, 6)));
        let mut b = Snake::new(10, 10, 0, None);
        b.set_state(vec![(5, 5), (5, 4), (5, 3), (6, 3)], 1, Some((4, 7)));
        assert_eq!(a.encode(Observer::Egocentric)[21..], b.encode(Observer::Egocentric)[21..]);
    }

    #[test]
    fn argmax_takes_the_first_maximum() {
        assert_eq!(decode_relative3(&[0.5, 0.5, 0.1]), -1);
        assert_eq!(decode_relative3(&[0.1, 0.2, 0.9]), 1);
    }
}
