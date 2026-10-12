"""`rq` is a convenience alias nothing depends on (docs/design/0018, decision 8). Inside the venv it must be ours: a
workspace dependency that also shipped an `rq` script (RQ, Redis Queue, is a plausible pick for a job queue) would
overwrite it without a warning."""

from importlib.metadata import distributions


def _scripts(name: str) -> list[tuple[str, str]]:
    return [
        (dist.metadata["Name"], ep.value)
        for dist in distributions()
        for ep in dist.entry_points
        if ep.group == "console_scripts" and ep.name == name
    ]


def test_only_this_cli_ships_an_rq_script():
    assert _scripts("rq") == [("redqueen-cli", "redqueen_cli.main:run")]


def test_redqueen_and_rq_are_the_same_entry_point():
    assert _scripts("redqueen") == _scripts("rq")
