"""Run the scheduler: `uv run python -m scheduler [--workers N] [--once]`.

It claims queued jobs (POST /jobs) and runs each as a worker process, at most `--workers` at a time, and fans stored
experiments (POST /experiments) out into jobs. Ctrl-C stops it and, cleanly, the workers it started. `--once` runs a
single tick (recover, fan out, reap, claim) and returns -- for scripts and tests; workers it starts keep running.
`--submit FILE` stores (or resubmits) an experiment first, as POST /experiments would, or submits a pipeline (or any
job spec) as POST /jobs would -- for working without the backend.
"""

import argparse

from jobcore.specs import ExperimentSpec, load_spec
from telemetry import open_job_store, open_stores

from scheduler import Scheduler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m scheduler", description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=1, help="jobs running at once")
    parser.add_argument("--poll", type=float, default=1.0, help="seconds between ticks")
    parser.add_argument("--once", action="store_true", help="one tick, then exit")
    parser.add_argument(
        "--submit", metavar="FILE", action="append", default=[], help="submit this experiment or job spec first"
    )
    args = parser.parse_args(argv)
    jobs = open_job_store()
    for path in args.submit:
        spec = load_spec(path)
        spec.resolve()
        if isinstance(spec, ExperimentSpec):
            jobs.put_experiment(spec.name, spec.model_dump(mode="json"))
            print(f"scheduler: submitted experiment {spec.name}", flush=True)
        else:
            job = jobs.submit(spec.kind, spec.model_dump(mode="json"))
            print(f"scheduler: submitted {spec.kind} job {job.job_id}", flush=True)
    scheduler = Scheduler(jobs, open_stores().registry, max_workers=args.workers)
    if args.once:
        scheduler.recover()
        scheduler.tick()
        return
    print(f"scheduler: up to {args.workers} worker(s); Ctrl-C to stop", flush=True)
    try:
        scheduler.run_forever(args.poll)
    except KeyboardInterrupt:
        print("scheduler: stopped", flush=True)


if __name__ == "__main__":
    main()
