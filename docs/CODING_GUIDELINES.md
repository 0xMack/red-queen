# Coding Guidelines

Standards for this repo, plus accumulated lessons (see [AGENTS.md](../AGENTS.md) for the bar on
what belongs here and how to add to it). Read before writing code, not after.

## General

- Keep abstractions pluggable only at proven extension points (`Program`, `FitnessEvaluator`,
  `SelectionStrategy`, `VariationStrategy` per docs/design/0001) — don't invent new ones
  speculatively.
- Dependency direction: algorithm/lib packages never import `telemetry`; only integration glue
  (usually in `jobs/`) imports both. See docs/design/0002.

## Python

- **Data models: pydantic, not bare dataclasses/TypedDicts**, for anything crossing a boundary
  (API payloads, stored records, config). Every field gets `Field(..., description=...)` and real
  validation (`ge=`/`gt=`, `min_length=`, `Literal`/`Enum`, or a `field_validator`) — not just a
  type hint. A bare type hint doesn't self-document and doesn't catch bad data at the boundary; a
  `Field` does both. Example: `libs/telemetry/src/telemetry/types.py`.
- Pluggable interfaces: `typing.Protocol`, not ABCs — matches the structural style already used in
  `libs/telemetry`.
- Tests: pytest, `tmp_path` fixture for anything touching the filesystem — no shared mutable state
  between tests.

## C++

- Never construct a `random_device`/`mt19937` (or any RNG) per call — reuse a `static` engine.
  Real perf bug we hit: it was happening 4x per instruction, 16 instructions per individual.
- A constructor parameter shadowing a member (`mutationRate = mutationRate;`) silently discards
  the argument — use `this->member = param` or distinct names. The compiler won't warn you.
- Don't `new` a value type just to `push_back` a copy into a container — construct in place
  (`emplace_back()`). The `new` here was an unfreed leak, one per program instruction.
- If a buffer is sized once and then only ever appended to via `back_inserter`/`push_back`, the
  initial size is dead weight and misleading — either size it correctly and write in place, or
  don't preallocate.
- Any per-item state that persists across items in a loop (e.g. registers across dataset rows)
  must be explicitly reset at the start of each item unless carrying over is intended — it was
  silently carrying over here and produced wrong predictions for every row after the first.

## Simulations / RL environments

- An environment's observation must contain enough information to determine the correct action —
  two episodes that differ only in something not exposed in the observation (e.g. the target, in
  `libs/games/src/games/reach1d.py`) are indistinguishable to any policy, no matter how it's
  trained/evolved. Found by running neuroevolution against `reach1d` and noticing two symmetric
  benchmark scenarios could never both improve; fixed by making the observation relative to the
  goal instead of absolute. Check this before spending compute tuning an algorithm against a new
  environment.
- A symmetric shaping reward (`+x` for progress, `-x` for regress) can be reward-hacked into
  oscillating in place forever for ~0 net reward — a real, lower-risk local optimum than continuing
  to pursue the sparse objective. Found in `games/snake.py`: an evolved policy learned to bounce
  between two cells instead of continuing toward food. Fixed by making the penalty for regress
  larger than the reward for progress, so standing still is strictly worse than seeking the goal,
  not merely no-better. Check what a policy can gain by doing nothing before trusting a shaped
  reward is safe.

## Lessons

Freeform notes that don't fit a section above yet. Once a pattern shows up twice, fold it into the
relevant section instead of leaving it here.

- The C++ bugs above were all found by actually building and running the code, not by reading it —
  verify an implementation runs before trusting that it looks correct.
