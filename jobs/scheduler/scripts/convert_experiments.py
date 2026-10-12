"""One-off (docs/design/0018 stage 6b): every historical experiment as an ExperimentSpec YAML under experiments/.

What ran is read from the runs themselves -- each tagged run's recorded config, mapped back to the TrainSpec that
reproduces it -- because the scripts' arm tables drifted after some experiments ran (rl-dqn-v1's long arms used
50k-step iterations that today's constants wouldn't give). Baselines and `init_from` come from the scripts' tables,
the only place they were recorded. Every (arm, seed) of every experiment is then checked: its expanded spec must map
to exactly what the recorded run's config says, or the conversion stops.

    uv run python jobs/scheduler/scripts/convert_experiments.py [--check]   (from the repo root)
"""

from __future__ import annotations

import dataclasses
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from jobcore.specs import ExperimentSpec, TrainSpec
from telemetry import RunInfo, open_stores

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments"
SKIP = {"rl-phase0-pipeline", "selfplay-pilot"}  # pilots: one untagged-arm run each, nothing to compare

# What each family's report is (docs/design/0008, 0010, 0013: the scripts' docstrings say how each was reported).
REPORTS = {
    "neat-vs-neuro-v1": {"protocol": "experiment.snake-report"},
    "snake-long-v1": {"protocol": "experiment.snake-report"},
    "snake-ego-v1": {"protocol": "experiment.snake-report"},
    "rl-tabular-v1": {"protocol": "experiment.rl-report"},
    "rl-dqn-v1": {"protocol": "experiment.rl-report"},
    "rl-pg-v1": {"protocol": "experiment.rl-report"},
    "rl-selfplay-v1": {"protocol": "experiment.checkers-field"},
    "selfplay-v2": {"protocol": "experiment.checkers-field", "params": {"games": 60}},
    "selfplay-v3": {"protocol": "experiment.checkers-h2h", "params": {"full": True}},
    "selfplay-v4": {"protocol": "experiment.checkers-h2h", "params": {"full": True}},
    "selfplay-v5": {"protocol": "experiment.checkers-h2h", "params": {"full": True}},
    "selfplay-v6": {"protocol": "experiment.checkers-h2h", "params": {"full": True}},
}


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def spec_of(config: dict[str, Any], runs_by_id: dict[str, RunInfo], neat_defaults: dict[str, Any]) -> dict[str, Any]:
    """The TrainSpec fields (no seed, tags or init_from) that reproduce a recorded run's config."""
    rep, game = config["representation"], config["game"]
    if game == "snake" and rep in ("neuroevolution", "neat"):
        params: dict[str, Any] = {
            "population_size": config["population_size"],
            "max_steps": config["max_steps"],
            "seeds": config["seed_strategy"],
        }
        if rep == "neuroevolution":
            tournament = re.fullmatch(r"tournament\(k=(\d+)\)", config["selection"])
            params |= {"hidden": config["layer_sizes"][1], "selection": "tournament" if tournament else "lexicase"}
            if tournament:
                params["tournament_k"] = int(tournament.group(1))
        else:
            changed = {k: v for k, v in config["neat"].items() if neat_defaults["snake"].get(k) != v}
            if changed:
                params["neat"] = changed
        return {
            "game": "snake",
            "algorithm": rep,
            "interface": config["interface"],
            "budget": {"generations": config["generations"]},
            "held_out_every": config["held_out_every"],
            "params": params,
        }
    if config.get("paradigm") == "reinforcement_learning" and game in ("snake", "reach1d"):
        return {
            "game": game,
            "algorithm": f"rl.{rep}",
            "interface": config["interface"],
            "budget": {"iterations": config["iterations"]},
            "held_out_every": config["held_out_every"],
            "params": {
                "steps_per_iteration": config["steps_per_iteration"],
                "reward": config["reward"],
                "snapshot_every": config["snapshot_every"],
                "agent": config["params"],
            },
        }
    if game == "checkers" and "population" in config:  # population-based training
        return {
            "game": "checkers",
            "algorithm": "pbt",
            "budget": {"games": config["games_per_member"]},
            "params": {
                "members": config["population"],
                "interval": config["interval"],
                "depth": config["search_depth"],
                "exploit": config["exploit"],
                "base": config["params"],
            },
        }
    if game == "checkers" and rep in ("td_lambda", "alphazero"):
        return {
            "game": "checkers",
            "algorithm": rep,
            "budget": {"iterations": config["iterations"]},
            "held_out_every": config["held_out_every"],
            "params": {
                "games_per_iteration": config["games_per_iteration"],
                "depth": config["search_depth"],
                "agent": config["params"],
            },
        }
    if game == "checkers" and rep == "neuroevolution" and "pool_positions" in config:  # search distillation
        sigma, rate = map(float, re.findall(r"[\d.]+", config["variation"]))
        return {
            "game": "checkers",
            "algorithm": "distill",
            "budget": {"generations": config["generations"]},
            "held_out_every": config["held_out_every"],
            "params": {
                "population_size": config["population_size"],
                "hidden": config["layer_sizes"][1],
                "depth": config["search_depth"],
                "label_depth": config["label_depth"],
                "label_kind": config.get("label_kind", "search"),
                "pool_games": distill_pool_games(config),
                "sample": config["sample_per_generation"],
                "sigma": sigma,
                "mutation_rate": rate,
                "opponents": config["opponents"],
            },
        }
    if game == "checkers" and rep == "neuroevolution":
        sigma, rate = map(float, re.findall(r"[\d.]+", config["variation"]))
        return {
            "game": "checkers",
            "algorithm": "neuroevolution",
            "budget": {"generations": config["generations"]},
            "held_out_every": config["held_out_every"],
            "params": _strip_none(
                {
                    "population_size": config["population_size"],
                    "hidden": config["layer_sizes"][1],
                    "opponents": config["opponents"],
                    "depth": config["search_depth"],
                    "hall": config["hall_of_fame"],
                    "seed_material": config["seeded_fraction"],
                    "margin": "margin" in config["fitness"],
                    "resample": config["resampled_opponents"],
                    "games_per_opponent": config["games_per_opponent"],
                    "sigma": sigma,
                    "mutation_rate": rate,
                    "opening_plies": config.get("opening_plies"),
                }
            ),
        }
    raise ValueError(f"no mapping for {rep} on {game}")


def distill_pool_games(config: dict[str, Any]) -> int:
    """The pool a distillation run fitted, from the cache it left (`pool-<kind>-k<depth>-g<games>-s0.json`)."""
    from telemetry import data_dir

    pattern = f"pool-{config.get('label_kind', 'search')}-k{config['label_depth']}-g*-s0.json"
    found = sorted((data_dir() / "distill").glob(pattern))
    if len(found) != 1:
        raise ValueError(f"can't tell the pool size of {config.get('experiment')}/{config.get('arm')}: {found}")
    return int(re.search(r"-g(\d+)-", found[0].name).group(1))


def script_arms() -> dict[str, dict[str, Any]]:
    """Each experiment family's arm table: baseline and init per arm (the only place they were recorded)."""
    sys.path.insert(0, str(ROOT / "jobs"))
    import checkers_selfplay_experiment as cx
    import rl_experiment as rx

    out: dict[str, dict[str, Any]] = {}
    for name, arm in rx.ARMS.items():
        out[name] = {"baseline": arm.baseline}
    for name, arm in cx.ARMS.items():
        baseline = "/".join(arm.baseline) if isinstance(arm.baseline, tuple) else arm.baseline
        out[name] = {"baseline": baseline, "init_from": "/".join(arm.init) if arm.init else None}
    return out


def deep_common(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """What every dict has in common, key by key (recursing into dicts)."""
    common = {}
    for key in dicts[0]:
        values = [d.get(key, _MISSING) for d in dicts]
        if any(v is _MISSING for v in values):
            continue
        if all(isinstance(v, dict) for v in values):
            nested = deep_common(values)
            if nested:
                common[key] = nested
        elif all(v == values[0] for v in values):
            common[key] = values[0]
    return common


def deep_minus(d: dict[str, Any], common: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in d.items():
        if key in common and isinstance(value, dict) and isinstance(common[key], dict):
            rest = deep_minus(value, common[key])
            if rest:
                out[key] = rest
        elif key not in common or common[key] != value:
            out[key] = value
    return out


_MISSING = object()


class _Dumper(yaml.SafeDumper):
    """Lists of numbers on one line (`seeds: [0, 1, 2]`), everything else block style."""


_Dumper.add_representer(
    list,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=all(isinstance(x, (int, float, str)) for x in data)
    ),
)


def convert(check_only: bool) -> None:
    from evolve import NeatConfig

    neat_defaults = {"snake": dataclasses.asdict(NeatConfig(target_species=6))}
    runs = open_stores().registry.list_runs()
    runs_by_id = {r.run_id: r for r in runs}
    tagged: dict[str, dict[str, dict[int, RunInfo]]] = defaultdict(lambda: defaultdict(dict))
    for run in sorted(runs, key=lambda r: r.created_at, reverse=True):  # oldest last: the oldest wins a duplicate
        c = run.config
        if c.get("experiment") and c.get("experiment") not in SKIP and run.status == "completed":
            tagged[c["experiment"]][c["arm"]][c["rng_seed"]] = run
    table = script_arms()
    OUT.mkdir(exist_ok=True)
    for name in sorted(tagged):
        arms_runs = tagged[name]
        arm_specs: dict[str, dict[str, Any]] = {}
        for arm, by_seed in arms_runs.items():
            mapped = {seed: spec_of(run.config, runs_by_id, neat_defaults) for seed, run in by_seed.items()}
            first = next(iter(mapped.values()))
            if any(m != first for m in mapped.values()):
                raise ValueError(f"{name}/{arm}: seeds were trained differently")
            arm_specs[arm] = first
        defaults = deep_common(list(arm_specs.values()))
        arm_seeds = {arm: sorted(by_seed) for arm, by_seed in arms_runs.items()}
        seed_sets = [tuple(s) for s in arm_seeds.values()]
        seeds = list(max(set(seed_sets), key=lambda s: (seed_sets.count(s), len(s))))  # what most arms ran
        arms = {}
        for arm in sorted(arm_specs, key=lambda a: min(r.created_at for r in arms_runs[a].values())):
            entry: dict[str, Any] = {}
            rest = deep_minus(arm_specs[arm], defaults)
            if rest:
                entry["train"] = rest
            info = table.get(arm, {})
            baseline = info.get("baseline")
            if baseline and ("/" in baseline or baseline in arms_runs):  # an arm this experiment never ran compared
                entry["baseline"] = baseline  # with nothing (the scripts skipped it), so it is left out
            if info.get("init_from"):
                entry["init_from"] = info["init_from"]
            if arm_seeds[arm] != seeds:
                entry["seeds"] = arm_seeds[arm]
            arms[arm] = entry
        spec = {"kind": "experiment", "name": name, "seeds": seeds, "defaults": defaults, "arms": arms}
        if name in REPORTS:
            spec["report"] = REPORTS[name]
        experiment = ExperimentSpec.model_validate(spec)
        verify(experiment, arms_runs, runs_by_id, neat_defaults)
        path = OUT / f"{name}.yaml"
        text = f"# {name}: converted from its recorded runs (docs/design/0018 stage 6b).\n" + yaml.dump(
            spec, Dumper=_Dumper, sort_keys=False, width=120
        )
        if check_only:
            if path.read_text(encoding="utf-8") != text:
                raise SystemExit(f"{path} is out of date")
        else:
            path.write_text(text, encoding="utf-8")
        print(f"{name}: {len(arms)} arms x {len(seeds)} seeds -- every recorded run matches")


def verify(experiment: ExperimentSpec, arms_runs: dict, runs_by_id: dict, neat_defaults: dict) -> None:
    """Each (arm, seed) the experiment expands to must map to what that seed's recorded run says, and a continuing
    arm's run must have continued the run its `init_from` names."""
    for arm, seed, train in experiment.expand():
        run = arms_runs.get(arm, {}).get(seed)
        if run is None:  # the scheduler would train it: the YAML would claim a cell that never ran
            raise ValueError(f"{experiment.name}/{arm}/{seed} has no recorded run")
        expected = spec_of(run.config, runs_by_id, neat_defaults)
        actual = train.model_dump(mode="json", exclude={"seed", "tags", "init_from", "kind"}, exclude_none=True)
        expected = TrainSpec.model_validate(expected).model_dump(
            mode="json", exclude={"seed", "tags", "init_from", "kind"}, exclude_none=True
        )
        if actual != expected:
            raise ValueError(f"{experiment.name}/{arm}/{seed}: {actual} != {expected}")
        if train.init_from is not None:
            parent = runs_by_id[run.config["init_run"]].config
            if (parent.get("experiment"), parent.get("arm"), parent.get("rng_seed")) != (
                train.init_from.experiment,
                train.init_from.arm,
                seed,
            ):
                raise ValueError(
                    f"{experiment.name}/{arm}/{seed} continued {run.config['init_run']}, not its init_from"
                )


if __name__ == "__main__":
    convert("--check" in sys.argv[1:])
