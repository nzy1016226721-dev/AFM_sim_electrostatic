"""Runtime-environment helpers for local IDE and Alliance batch execution.

The helpers in this module deliberately keep the legacy JSON defaults intact:
- plotting remains enabled in Spyder-like IDE kernels;
- normal command-line/Slurm execution suppresses interactive plot windows when
  disable_in_non_ide is true;
- output_dir_mode defaults to "default".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_spyder_like_ide() -> bool:
    """Return True when executing inside a Spyder-like IDE/IPython kernel."""
    if os.environ.get("SPYDER_KERNEL_ID"):
        return True
    if "spyder_kernels" in sys.modules:
        return True
    # Conservative fallback for IDE-managed IPython kernels.
    if os.environ.get("JPY_PARENT_PID") and os.environ.get("IPYTHONENABLE"):
        return True
    return False


def resolve_plotting_enabled(cfg: dict, *, cli_override: bool | None = None) -> bool:
    """Resolve whether interactive plotting should be performed.

    JSON:
        "plotting": {
            "enabled": true,
            "disable_in_non_ide": true
        }

    ``enabled`` is the master switch. ``disable_in_non_ide`` only affects
    non-IDE execution. A CLI override has highest priority.
    """
    plotting_cfg = cfg.get("plotting", {})
    if not isinstance(plotting_cfg, dict):
        plotting_cfg = {}

    enabled = bool(plotting_cfg.get("enabled", True))
    if not enabled:
        return False

    if cli_override is not None:
        return bool(cli_override)

    disable_in_non_ide = bool(plotting_cfg.get("disable_in_non_ide", True))
    if disable_in_non_ide and not is_spyder_like_ide():
        return False

    return True


def resolve_output_dir(base_output_dir: str, config_path: str, cfg: dict) -> str:
    """Resolve output_dir while preserving the legacy default behavior.

    Supported modes:
      default   -> use output_dir exactly as before
      config    -> output_dir/<config-file-stem>
      job_config -> output_dir/job_<SLURM_JOB_ID>/<config-file-stem> under Slurm;
                    outside Slurm, falls back to config mode.
    """
    mode = str(cfg.get("output_dir_mode", "default") or "default").strip().lower()

    if mode == "default":
        return base_output_dir

    config_stem = Path(config_path).stem

    if mode == "config":
        return os.path.join(base_output_dir, config_stem)

    if mode == "job_config":
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id:
            return os.path.join(base_output_dir, f"job_{job_id}", config_stem)
        print(
            "[output] output_dir_mode='job_config' requested, but SLURM_JOB_ID "
            "is not set; falling back to config-specific output."
        )
        return os.path.join(base_output_dir, config_stem)

    raise ValueError(
        f"Invalid output_dir_mode={mode!r}. Expected 'default', 'config', or 'job_config'."
    )
