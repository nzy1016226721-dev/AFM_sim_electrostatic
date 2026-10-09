"""Configuration loading for compact serial and MPI templates."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .config_defaults import apply_solver_defaults
from .coordinates import normalize_config


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_afm_config(
    path: str | Path, *, max_depth: int = 8, normalize: bool = True
) -> tuple[str, dict]:
    """Load one JSON, resolving an optional relative ``extends`` chain.

    ``extends`` keeps comparison and large-grid templates small while reusing
    an established physical geometry.  Both serial ``run_all.py`` and the MPI
    entry point resolve the same chain so a parity pair cannot silently load
    different physics.
    """
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".json" or not path.is_file():
        raise FileNotFoundError(f"MPI JSON configuration not found: {path}")

    visited = []

    def load_one(current: Path, depth: int) -> dict:
        if depth > max_depth:
            raise ValueError(f"AFM config extends chain exceeds {max_depth} levels")
        resolved = current.resolve()
        if resolved in visited:
            chain = " -> ".join(str(item) for item in (*visited, resolved))
            raise ValueError(f"cyclic MPI config extends chain: {chain}")
        visited.append(resolved)
        with resolved.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"AFM config must contain a JSON object: {resolved}")
        parent_name = value.get("extends")
        if parent_name:
            parent = Path(parent_name)
            if not parent.is_absolute():
                parent = resolved.parent / parent
            merged = _deep_merge(load_one(parent, depth + 1), value)
        else:
            merged = deepcopy(value)
        visited.pop()
        return merged

    merged = apply_solver_defaults(load_one(path, 0))
    return str(path), normalize_config(merged) if normalize else merged


def load_mpi_config(path: str | Path, *, max_depth: int = 8) -> tuple[str, dict]:
    """Backward-compatible alias for :func:`load_afm_config`."""
    return load_afm_config(path, max_depth=max_depth)


__all__ = ["load_afm_config", "load_mpi_config"]
