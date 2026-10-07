"""The best Checkers self-play recipe as one command, for leaderboard entrants (docs/design/0016).

Two ordinary self-play runs (jobs/checkers_selfplay_run.py), untagged, so both join the versus leaderboard:

1. plain TD(λ) with the opponent pool for `--td-games` games (the recipe's long, cheap phase), then
2. a TD-Leaf(λ) fine-tune of that network, searching `--leaf-depth` plies, for `--leaf-games` games.

`--td-run RUN_ID` skips phase 1 and fine-tunes an existing run instead. Training is deterministic, so entering the same
network at another search depth needs only phase 2 again: `--td-run <phase 1's id> --depth 4`.

  uv run python -u jobs/checkers_recipe_run.py [--hidden 128] [--layers 2] [--td-games 1000000] [--leaf-games 50000]
                                               [--leaf-depth 3] [--depth 3] [--rng-seed 0] [--td-run RUN_ID]

The defaults are the 0016 candidate: 2 x 128, 1M games, then 50k games of 3-ply TD-Leaf, seed 0 fixed in advance.
Phase 1 measured about 9,300 active seconds in 0016. Phase 2 hasn't been measured at this width: 2 x 64 took ~40 min
alone (0010's 4c), so expect roughly 1.5-3 hours. A 4-ply entrant of the same network adds only phase 2.
"""

from __future__ import annotations

import argparse

import checkers_selfplay_run

POOL = {"pool_every": 5000, "pool_size": 10, "pool_fraction": 0.5}


def main(
    hidden: int = 128,
    layers: int = 2,
    td_games: int = 1_000_000,
    leaf_games: int = 50_000,
    leaf_depth: int = 3,
    depth: int = 3,
    rng_seed: int = 0,
    td_run: str | None = None,
    games_per_iteration: int = 1000,
) -> tuple[str, str]:
    """Runs the recipe; returns (TD run id, fine-tuned run id)."""
    network = {**POOL, "hidden": hidden, "hidden_layers": layers}
    if td_run is None:
        iterations = max(1, td_games // games_per_iteration)
        td_run = checkers_selfplay_run.main(
            iterations=iterations,
            games_per_iteration=games_per_iteration,
            depth=depth,
            params=dict(network),
            rng_seed=rng_seed,
            held_out_every=max(1, iterations // 20),
        )
    iterations = max(1, leaf_games // games_per_iteration)
    leaf_run = checkers_selfplay_run.main(
        iterations=iterations,
        games_per_iteration=games_per_iteration,
        depth=depth,
        params={**network, "search_depth": leaf_depth},
        rng_seed=rng_seed,
        held_out_every=max(1, iterations // 10),
        init_run=td_run,
    )
    print(f"recipe done: TD run {td_run}, TD-Leaf run {leaf_run}")
    return td_run, leaf_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plain TD, then a TD-Leaf fine-tune: the best Checkers recipe.")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--td-games", type=int, default=1_000_000)
    parser.add_argument("--leaf-games", type=int, default=50_000)
    parser.add_argument("--leaf-depth", type=int, default=3)
    parser.add_argument("--depth", type=int, default=3, help="search depth for monitoring and the leaderboard")
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--td-run", default=None, help="fine-tune this existing run instead of training phase 1")
    args = parser.parse_args()
    main(
        hidden=args.hidden,
        layers=args.layers,
        td_games=args.td_games,
        leaf_games=args.leaf_games,
        leaf_depth=args.leaf_depth,
        depth=args.depth,
        rng_seed=args.rng_seed,
        td_run=args.td_run,
    )
