"""Process-memory tracking utilities for AFM simulation levels."""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

try:
    import psutil
except ImportError:  # pragma: no cover - fallback for minimal environments
    psutil = None


class MemoryTracker:
    """Sample process RSS in a background thread and retain the peak usage.

    Parameters
    ----------
    interval : float, optional
        Sampling interval in seconds. Defaults to 0.1 s.
    """

    def __init__(self, interval: float = 0.1):
        self.interval = max(float(interval), 0.01)
        self.peak_bytes = 0
        self.start_bytes = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process(os.getpid()) if psutil is not None else None

    def _rss(self) -> int:
        if self._process is None:
            return 0
        try:
            return int(self._process.memory_info().rss)
        except (OSError, RuntimeError):
            return 0

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._rss())
            self._stop.wait(self.interval)

    def start(self) -> "MemoryTracker":
        """Start sampling process RSS."""
        self.start_bytes = self._rss()
        self.peak_bytes = self.start_bytes
        if self._process is not None:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> float:
        """Stop sampling and return peak resident memory in GB."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.interval))
        self.peak_bytes = max(self.peak_bytes, self._rss())
        return self.peak_bytes / (1024 ** 3)


class track_memory:
    """Context manager returning peak process RSS in GB for one simulation level."""

    def __init__(self, interval: float = 0.1):
        self.tracker = MemoryTracker(interval=interval)
        self.peak_gb = 0.0

    def __enter__(self) -> "track_memory":
        self.tracker.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.peak_gb = self.tracker.stop()


def log_memory_usage(level_resolution: str, memory_gb: float,
                     logfile: str = "memory_usage_log.csv", output_dir: str = ".") -> None:
    """Append peak process memory for one simulation level to a CSV log.

    This function belongs to the optional memory-tracking component and is
    intentionally imported lazily by the simulation code only when memory
    tracking is enabled.
    """
    import csv
    import os

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, logfile)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["level resolution", "memory cost(in GB)"])
        writer.writerow([level_resolution, f"{float(memory_gb):.6f}"])


__all__ = ["MemoryTracker", "track_memory", "log_memory_usage"]
