"""Downloads and builds Copy Cat's artwork sets where `copycat.report` looks for them, under the data dir. A report does this
itself on first use; this is for doing it ahead of time (`uv run redqueen dev fetch-sets`)."""

from arena.copycat_artwork import held_out, sources
from jobcore import experiments_dir


def main() -> None:
    root = experiments_dir() / "copycat-sets"
    for set_name in sorted(sources()):
        print(f"{set_name}: {held_out(set_name, root)}")


if __name__ == "__main__":
    main()
