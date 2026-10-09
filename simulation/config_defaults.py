"""Configuration-level RAM policy, independent of numerical/JIT imports.

Direct numerical APIs retain their explicit legacy defaults. JSON entry points
and physical-config normalization use the same lossless, float32 RAM policy.
"""
from copy import deepcopy


DEFAULT_MEMORY_MODE = "ram_compact"
RAM_MODES = ("ram_first", "ram_compact")


def apply_solver_defaults(cfg):
    """Return an independent config with effective storage/plotting defaults.

    Call after resolving inheritance so explicit child overrides win. This
    function does not change geometry, field precision or convergence controls.
    ``standard`` is the explicit legacy/float64-diagnostic opt-out.
    """
    out = deepcopy(cfg)
    mode = out.get("memory_mode")
    mode = DEFAULT_MEMORY_MODE if mode is None else str(mode).lower()
    if mode not in ("standard", *RAM_MODES):
        raise ValueError("memory_mode must be standard, ram_first or ram_compact")
    out["memory_mode"] = mode
    if mode in RAM_MODES:
        if out.get("phi_update_mode") is None:
            out["phi_update_mode"] = "in_place"
        if out.get("residual_accumulation") is None:
            out["residual_accumulation"] = "scalar" if mode == "ram_compact" else "row_array"
        plotting = out.get("plotting")
        if plotting is None:
            plotting = {}
            out["plotting"] = plotting
        if not isinstance(plotting, dict):
            raise ValueError("plotting must be an object")
        plotting.setdefault("enabled", False)
    return out
