"""Where a claimed job's worker runs (docs/design/0018, decision 2): a Runner starts it and can stop it.

`LocalProcessRunner` starts `python -m <workload> <spec file> --result <file>` as a child process in its own process
group, its output appended to `<data dir>/jobs/<job_id>.log`. Stopping never relies on Ctrl-C (a console-less child
often never receives CTRL_C_EVENT -- the uvicorn `--reload` lesson in AGENTS.md): CTRL_BREAK_EVENT on Windows, SIGTERM
elsewhere, which the workload turns into KeyboardInterrupt so its run is marked `failed` (jobcore.cli); then a kill
after a grace period. A `KubernetesRunner` (a Job per workload image) would implement the same three methods.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from telemetry import Job, data_dir, workspace_root

WORKLOADS = {"train": "trainer", "evaluate": "evaluator", "publish": "publisher"}


@dataclass
class Handle:
    job_id: str
    pid: int
    log: str  # relative to the data directory
    result_path: Path
    process: Any = None


class Runner(Protocol):
    def start(self, job: Job) -> Handle: ...
    def poll(self, handle: Handle) -> int | None: ...
    def stop(self, handle: Handle, grace: float) -> None: ...


class LocalProcessRunner:
    def __init__(self, python: str = sys.executable):
        self._python = python

    def start(self, job: Job) -> Handle:
        root = data_dir()
        jobs_dir = root / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        spec_path, result_path = jobs_dir / f"{job.job_id}.spec.json", jobs_dir / f"{job.job_id}.result.json"
        log_path = jobs_dir / f"{job.job_id}.log"
        spec_path.write_text(json.dumps(job.spec), encoding="utf-8")
        result_path.unlink(missing_ok=True)
        env = {
            **os.environ,
            "REDQUEEN_DATA_DIR": str(root),  # the worker writes where this scheduler reads
            "PYTHONUNBUFFERED": "1",  # the log is followed live (GET /jobs/{id}/logs)
            "PYTHONIOENCODING": "utf-8",  # labels have arrows; a Windows console encoding can't print them
        }
        command = [self._python, "-m", WORKLOADS[job.kind], str(spec_path), "--result", str(result_path)]
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        with open(log_path, "ab") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=workspace_root(),
                env=env,
                creationflags=flags,
                start_new_session=sys.platform != "win32",
            )
        return Handle(job.job_id, process.pid, str(log_path.relative_to(root).as_posix()), result_path, process)

    def poll(self, handle: Handle) -> int | None:
        return handle.process.poll()

    def stop(self, handle: Handle, grace: float = 10.0) -> None:
        process = handle.process
        if process.poll() is not None:
            return
        process.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.1)
        process.kill()
        process.wait()


def read_result(handle: Handle) -> dict[str, Any] | None:
    try:
        return json.loads(handle.result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def log_tail(log: str, lines: int = 20) -> str:
    path = data_dir() / log
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def pid_alive(pid: int) -> bool:
    """Whether a process with this id exists (after a scheduler restart: is a `running` job's worker still there?)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
