"""Process-memory tracking utilities for AFM simulation levels."""

from __future__ import annotations

import os
import threading
import time
import csv
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

    def __init__(self, interval: float = 0.1, *, live_log_path: str | None = None,
                 stage: str = "unspecified"):
        self.interval = max(float(interval), 0.01)
        self.peak_bytes = 0
        self.start_bytes = 0
        self.live_log_path = live_log_path
        self.stage = stage
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process(os.getpid()) if psutil is not None else None

    def _rss(self) -> int:
        if self._process is None:
            return 0
        try:
            return int(self._process.memory_info().rss)
        except (OSError, RuntimeError):
            return 0

    def set_stage(self, stage: str) -> None:
        """Label future live samples with the active solver/preparation stage."""
        with self._lock:
            self.stage = str(stage)

    def _write_live_sample(self, rss_bytes: int) -> None:
        if not self.live_log_path:
            return
        try:
            parent = os.path.dirname(self.live_log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            file_exists = os.path.isfile(self.live_log_path)
            with open(self.live_log_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if not file_exists:
                    writer.writerow([
                        "timestamp_utc",
                        "stage",
                        "rss_gb",
                        "peak_rss_gb",
                    ])
                with self._lock:
                    stage = self.stage
                writer.writerow([
                    f"{time.time():.6f}",
                    stage,
                    f"{rss_bytes / (1024 ** 3):.6f}",
                    f"{self.peak_bytes / (1024 ** 3):.6f}",
                ])
                handle.flush()
        except OSError:
            # The solver must not fail merely because an optional diagnostic
            # file cannot be appended.
            return

    def _sample(self) -> None:
        while not self._stop.is_set():
            rss_bytes = self._rss()
            self.peak_bytes = max(self.peak_bytes, rss_bytes)
            self._write_live_sample(rss_bytes)
            self._stop.wait(self.interval)

    def start(self) -> "MemoryTracker":
        """Start sampling process RSS."""
        self.start_bytes = self._rss()
        self.peak_bytes = self.start_bytes
        self._write_live_sample(self.start_bytes)
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
        self._write_live_sample(self.peak_bytes)
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
