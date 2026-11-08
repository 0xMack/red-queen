"""No control characters in source text.

A shell heredoc or `sed` on Windows has more than once turned a written backslash-b into a backspace character (U+0008).
In a regex that is not a word boundary but a literal backspace, so the pattern silently matches nothing -- it once left
Copy Cat's live layers empty with no error anywhere. Tabs, newlines, carriage returns and form feeds are fine.
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROL = re.compile("[\x00-\x08\x0b\x0e-\x1f\x7f]")


def _source_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return [ROOT / name for name in listed]


def test_no_control_characters_in_source_text():
    found = []
    for path in _source_files():
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue  # binary (wasm, images)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if CONTROL.search(line):
                found.append(f"{path.relative_to(ROOT).as_posix()}:{number}: {line.strip()[:80]!r}")
    assert not found, "control characters (a backslash escape that became a real character?):\n" + "\n".join(found)
