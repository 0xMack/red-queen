"""Training/inference cost measurement shared by jobs (docs/design/0007, "an entry is a vector of
measurements, not one score").

Counters first, clocks second: episodes/steps/evaluations are exact and comparable across machines;
every clock or memory figure is recorded next to `hardware_fingerprint()` so timings are only ever
compared within a hardware class.
"""

from __future__ import annotations

import contextlib
import os
import platform
import sys
import time
from collections.abc import Callable
from typing import Any


def _cpu_name() -> str:
    # Best effort: whatever fails here, the fallback below is still informative.
    with contextlib.suppress(Exception):
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            import subprocess

            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown"


def hardware_fingerprint() -> dict[str, Any]:
    cpu = _cpu_name()
    impl = platform.python_implementation()
    version = platform.python_version()
    return {
        "cpu": cpu,
        "logical_cores": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": f"{impl} {version}",
        # Everything in libs/evolve + libs/games is pure Python today; RedQueenCbind runs would say
        # "c++". Timings across these are not comparable even on the same CPU.
        "engine": "pure-python",
        # What timings are grouped by in UIs.
        "hardware_class": f"{cpu} · {impl} {'.'.join(version.split('.')[:2])}",
    }


def peak_rss_bytes() -> int | None:
    """Peak resident memory of this process so far, or None where it can't be read."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            psapi = ctypes.WinDLL("psapi")
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
            return None
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)  # bytes on macOS, KiB on Linux
    except Exception:  # noqa: BLE001
        return None


class TrainingCostMeter:
    """Measures one training run's cost. Use `on_generation` as an evolve() callback, and wrap the
    pause/resume control callback with `excluding_pauses()` so time spent paused (docs/design/0005
    step 7) isn't counted as training time.

    `fitness` is the run's fitness evaluator; if it exposes `episodes`/`steps` (as
    SimulationFitnessEvaluator does), those exact counters are included.
    """

    def __init__(self, population_size: int, fitness: Any = None):
        self._population_size = population_size
        self._fitness = fitness
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()
        self._paused_s = 0.0
        self.generations = 0

    def on_generation(self, _summary: Any) -> None:
        self.generations += 1

    def excluding_pauses(self, callback: Callable[[Any], None]) -> Callable[[Any], None]:
        def wrapped(summary: Any) -> None:
            started = time.perf_counter()
            try:
                callback(summary)
            finally:
                self._paused_s += time.perf_counter() - started

        return wrapped

    def summary(self) -> dict[str, Any]:
        wall_s = time.perf_counter() - self._wall_start
        episodes = getattr(self._fitness, "episodes", None)
        steps = getattr(self._fitness, "steps", None)
        return {
            "measured": True,  # vs. "estimated" for pre-0007 runs (see jobs/evaluate.py)
            "generations": self.generations,
            "fitness_evaluations": self.generations * self._population_size,
            "episodes": episodes,
            "env_steps": steps,
            "wall_s": round(wall_s, 3),
            "paused_s": round(self._paused_s, 3),
            "active_s": round(wall_s - self._paused_s, 3),
            "cpu_s": round(time.process_time() - self._cpu_start, 3),
            "peak_rss_bytes": peak_rss_bytes(),
            "hardware": hardware_fingerprint(),
        }
