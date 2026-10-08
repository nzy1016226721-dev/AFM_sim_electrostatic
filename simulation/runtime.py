"""Runtime-environment helpers for local IDE and Alliance batch execution."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path


def is_batch_execution() -> bool:
    """Return True when execution is explicitly batch/Slurm/headless."""
    flag = os.environ.get("AFM_NONINTERACTIVE", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    return bool(os.environ.get("SLURM_JOB_ID"))


def is_spyder_like_ide() -> bool:
    """Return True for Spyder-like interactive IDE kernels unless batch is forced."""
    if is_batch_execution():
        return False
    if os.environ.get("SPYDER_KERNEL_ID"):
        return True
    if "spyder_kernels" in sys.modules:
        return True
    if os.environ.get("JPY_PARENT_PID") and os.environ.get("IPYTHONENABLE"):
        return True
    return False




def is_afm_config(path: str | os.PathLike) -> bool:
    """Return True when *path* looks like a simulator AFM JSON config."""
    candidate = Path(path)
    if candidate.suffix.lower() != ".json" or not candidate.is_file():
        return False
    try:
        from .mpi_config import load_afm_config
        _, cfg = load_afm_config(candidate, normalize=False)
    except (OSError, ValueError, TypeError, FileNotFoundError):
        return False
    return (
        isinstance(cfg, dict)
        and isinstance(cfg.get("blocks", cfg.get("blocks_nm")), list)
        and "v_start" in cfg
        and "v_stop" in cfg
        and "v_step" in cfg
    )


def discover_afm_configs(directory: str | os.PathLike = ".") -> list[str]:
    """Return all AFM JSON configs directly in *directory*, newest first."""
    root = Path(directory).expanduser().resolve()
    configs = [p for p in root.iterdir() if p.is_file() and is_afm_config(p)]
    configs.sort(key=lambda p: (-p.stat().st_mtime_ns, p.name.lower()))
    return [str(p) for p in configs]


def resolve_config_path(config_path: str | os.PathLike | None = None,
                        directory: str | os.PathLike = ".") -> str:
    """Resolve one explicit JSON path, or the newest AFM JSON when omitted.

    No base-name/suffix matching is performed. Relative names are resolved
    against *directory* (normally the current working directory).
    """
    root = Path(directory).expanduser().resolve()
    if config_path is None or not str(config_path).strip():
        configs = discover_afm_configs(root)
        if not configs:
            raise FileNotFoundError(f"No AFM JSON configurations found in {root}")
        return configs[0]

    candidate = Path(os.path.expanduser(os.fspath(config_path)))
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if candidate.suffix.lower() != ".json":
        raise ValueError(f"Configuration must be a JSON file: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(f"JSON configuration not found: {candidate}")
    if not is_afm_config(candidate):
        raise ValueError(f"JSON file does not look like an AFM configuration: {candidate}")
    return str(candidate)


def resolve_plotting_enabled(cfg: dict, *, cli_override: bool | None = None) -> bool:
    """Resolve whether interactive plotting should be performed."""
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
    """Resolve the effective output directory for one configuration.

    ``AFM_JOB_OUTPUT_ROOT`` is an authoritative batch override used by the
    Alliance selector scripts. It appends the config stem so each config gets
    an isolated directory. Local execution otherwise follows the JSON setting.
    """
    env_root = os.environ.get("AFM_JOB_OUTPUT_ROOT", "").strip()
    if env_root:
        root = Path(env_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        return str(root / Path(config_path).stem)

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
