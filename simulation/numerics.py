"""Shared numerical-policy helpers for serial and MPI AFM solvers.

Production defaults remain the historical float32, absolute-residual path.
Alternative policies are deliberately explicit in JSON so a diagnostic or a
voltage-normalized validation run cannot silently alter an ordinary result.
"""

from __future__ import annotations

from copy import copy
from typing import Mapping, Sequence

import numpy as np


ABSOLUTE_RESIDUAL = "absolute"
RELATIVE_DRIVE_RESIDUAL = "relative_drive"
FLOAT64_MEMORY_SMOKE = "float64_memory_smoke"

TIP_SHAPE_PYRAMID = "pyramid"
TIP_SHAPE_CONE = "cone"


def resolve_tip_shape(value: object = TIP_SHAPE_PYRAMID) -> str:
    """Validate the AFM tip cross-section shape requested by a config."""
    shape = str(value or TIP_SHAPE_PYRAMID).strip().lower()
    if shape not in (TIP_SHAPE_PYRAMID, TIP_SHAPE_CONE):
        raise ValueError("tip_shape must be 'pyramid' or 'cone'")
    return shape


def pyramid_tip_half_side(cone_radius: float, base_half_side: float) -> float:
    """Return the pyramid square half-side for one horizontal tip slice.

    Each side plane is tangent to the original cone along its centre
    generator, so the square circumscribes the cone disk: the half-side is
    the cone radius, capped at the configured base half-side.
    """
    return min(float(cone_radius), float(base_half_side))


def pyramid_tip_corner_radius(half_side: float, apex_radius: float) -> float:
    """Return the vertical-edge fillet radius for one tip slice.

    The fillet uses the apex curvature radius.  Where the half-side is
    smaller than that radius (within one apex radius of the tip apex) the
    fillet consumes the flats and the cross-section stays circular.
    """
    return min(max(float(apex_radius), 0.0), max(float(half_side), 0.0))


def pyramid_tip_inside(
    xc: np.ndarray, yc: np.ndarray, half_side: float, corner_radius: float
) -> np.ndarray:
    """Return a boolean mask for a rounded-square pyramid tip cross-section.

    Parameters
    ----------
    xc, yc : np.ndarray
        Tip-centred coordinates broadcastable to a common ``(..., ...)`` shape.
    half_side : float
        Square half-side ``s`` for this slice.
    corner_radius : float
        Vertical-edge fillet radius ``r`` for this slice.

    A point is inside when it lies within the ``[-s, s]`` square and either
    inside the flat side bands or within ``r`` of a fillet centre at
    ``(+(s - r), +(s - r))``.  For ``r == s`` this degenerates exactly to the
    disk of radius ``s``.  Operation order is fixed so serial and distributed
    builders produce bit-identical masks from identical coordinates.
    """
    s = float(half_side)
    r = float(corner_radius)
    ax = np.abs(xc)
    ay = np.abs(yc)
    in_square = (ax <= s) & (ay <= s)
    qx = ax - (s - r)
    qy = ay - (s - r)
    in_fillet = (qx <= 0.0) | (qy <= 0.0) | ((qx * qx + qy * qy) <= r * r)
    return in_square & in_fillet


def resolve_solver_dtype(value: object = "float32") -> np.dtype:
    """Return an explicitly supported potential/coefficient dtype."""
    dtype = np.dtype("float32" if value is None else value)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("solver_dtype must be 'float32' or 'float64'")
    return dtype


def resolve_residual_tolerance_mode(value: object = ABSOLUTE_RESIDUAL) -> str:
    """Validate the residual-tolerance interpretation requested by a config."""
    mode = str(value or ABSOLUTE_RESIDUAL).strip().lower()
    if mode not in (ABSOLUTE_RESIDUAL, RELATIVE_DRIVE_RESIDUAL):
        raise ValueError(
            "residual_tolerance_mode must be 'absolute' or 'relative_drive'"
        )
    return mode


def _gate_sequence(gates):
    if isinstance(gates, dict):
        return [gates]
    if gates is None:
        return []
    return list(gates)


def prepare_drive_voltages(
    tip_voltage: float,
    gates: Sequence[dict] | dict | None,
    mode: str,
) -> tuple[float, Sequence[dict] | dict | None, float]:
    """Return solver voltages and the scale back to physical volts.

    ``relative_drive`` is valid for the package's homogeneous equation
    ``div(epsilon*grad(phi)) = 0``.  Every nonzero Dirichlet voltage is divided
    by the same maximum absolute drive.  The converged field can therefore be
    multiplied by that scale without changing the physical solution.
    """
    mode = resolve_residual_tolerance_mode(mode)
    physical_tip = float(tip_voltage)
    if mode == ABSOLUTE_RESIDUAL:
        return physical_tip, gates, 1.0

    gate_items = _gate_sequence(gates)
    drive_scale = max(
        [abs(physical_tip)]
        + [abs(float(item.get("Vgate_val", 0.0))) for item in gate_items]
        + [0.0]
    )
    if not np.isfinite(drive_scale):
        raise ValueError("all tip and gate voltages must be finite")
    if drive_scale == 0.0:
        drive_scale = 1.0

    scaled_items = []
    for item in gate_items:
        scaled = copy(item)
        scaled["Vgate_val"] = float(item.get("Vgate_val", 0.0)) / drive_scale
        scaled_items.append(scaled)
    if isinstance(gates, dict):
        scaled_gates = scaled_items[0]
    elif gates is None:
        scaled_gates = None
    else:
        scaled_gates = scaled_items
    return physical_tip / drive_scale, scaled_gates, float(drive_scale)


def diagnostic_iteration_limit(
    diagnostic: Mapping[str, object] | None,
    shape: Sequence[int],
    *,
    solver_dtype: np.dtype,
) -> int | None:
    """Resolve a fixed iteration count only for the named memory smoke mode."""
    if not diagnostic:
        return None
    mode = str(diagnostic.get("mode", "")).strip().lower()
    if mode != FLOAT64_MEMORY_SMOKE:
        raise ValueError(
            f"unsupported diagnostic.mode={mode!r}; fixed iterations are allowed "
            f"only for {FLOAT64_MEMORY_SMOKE!r}"
        )
    if np.dtype(solver_dtype) != np.dtype(np.float64):
        raise ValueError("float64_memory_smoke requires solver_dtype='float64'")
    if len(set(int(value) for value in shape)) != 1:
        raise ValueError("float64_memory_smoke currently requires an isotropic grid")
    mapping = diagnostic.get("iterations_by_level")
    if not isinstance(mapping, Mapping):
        raise ValueError("diagnostic.iterations_by_level must be a JSON object")
    level = str(int(shape[0]))
    if level not in mapping:
        raise ValueError(f"diagnostic iteration count is missing for level {level}")
    limit = int(mapping[level])
    if limit < 1:
        raise ValueError("diagnostic iteration counts must be positive")
    return limit


__all__ = [
    "ABSOLUTE_RESIDUAL",
    "FLOAT64_MEMORY_SMOKE",
    "RELATIVE_DRIVE_RESIDUAL",
    "TIP_SHAPE_CONE",
    "TIP_SHAPE_PYRAMID",
    "diagnostic_iteration_limit",
    "prepare_drive_voltages",
    "pyramid_tip_corner_radius",
    "pyramid_tip_half_side",
    "pyramid_tip_inside",
    "resolve_residual_tolerance_mode",
    "resolve_solver_dtype",
    "resolve_tip_shape",
]
