"""
Physical-coordinate helpers for AFM simulation configuration.

The public configuration uses nanometres relative to a configurable origin in
the main simulation domain.  The solver itself continues to operate on
fractional coordinates [0, 1].
"""
from copy import deepcopy

from .numerics import resolve_tip_shape


def _grid_resolution(cfg, axis):
    """Return the main-grid sample/voxel count for one axis."""
    grid = cfg.get("grid_resolution", {})
    key = {"x": "nx", "y": "ny", "z": "nz"}[axis]
    value = grid.get(key)
    if value is None:
        raise ValueError(f"grid_resolution.{key} is required when using voxel_nm3")
    value = int(value)
    if value < 2:
        raise ValueError(f"grid_resolution.{key} must be at least 2")
    return value


def voxel_nm(cfg):
    """Return the main-grid voxel edge length in nanometres."""
    value = cfg.get("voxel_nm3")
    if value is None:
        raise ValueError("Configuration must define voxel_nm3 (nm per main-grid voxel edge)")
    value = float(value)
    if value <= 0:
        raise ValueError("voxel_nm3 must be positive")
    return value


def _axis_length_nm(cfg, axis):
    """Return the physical main-domain size as voxel count times voxel edge."""
    return _grid_resolution(cfg, axis) * voxel_nm(cfg)


def physical_domain_nm(cfg):
    """Return ``(Lx, Ly, Lz)`` derived from grid resolution and voxel_nm3."""
    return tuple(_axis_length_nm(cfg, axis) for axis in "xyz")


def origin_fraction(cfg):
    """Return the configured physical origin as fractional main-grid coordinates."""
    value = cfg.get("coordinate_system", {}).get("origin_fraction", [0.5, 0.5, 0.0])
    if len(value) != 3:
        raise ValueError("coordinate_system.origin_fraction must contain three values")
    return tuple(float(v) for v in value)


def nm_to_fraction(value_nm, axis, cfg):
    """Convert an origin-relative nanometre coordinate to a main-grid fraction."""
    i = "xyz".index(axis)
    return origin_fraction(cfg)[i] + float(value_nm) / _axis_length_nm(cfg, axis)


def nm_range_to_fraction(range_nm, axis, cfg):
    """Convert a two-element origin-relative nanometre range to fractions."""
    if range_nm is None:
        return [0.0, 1.0]
    if len(range_nm) != 2:
        raise ValueError(f"{axis}_range_nm must contain two values")
    return [nm_to_fraction(range_nm[0], axis, cfg),
            nm_to_fraction(range_nm[1], axis, cfg)]


def _fractional_range_or_full(range_value):
    """Return a fractional axis range, defaulting an omitted value to all space."""
    if range_value is None:
        return [0.0, 1.0]
    return deepcopy(range_value)


def _spatial_items(value, field_name):
    """Normalize one spatial config field to a list of mapping entries."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError(f"{field_name} must be an object or a list of objects")


def _convert_range_dict(item, cfg, *, default_gate_voltage=False):
    """Convert a spatial item to complete fractional x/y/z ranges.

    Every spatial object deliberately has the same omission rule: an absent or
    null axis range covers the complete simulation axis.  This makes, for
    example, a gate with only ``z_range_nm`` a full-width, full-depth gate.
    """
    out = {k: deepcopy(v) for k, v in item.items() if not k.endswith("_nm")}
    for axis in "xyz":
        key = f"{axis}_range_nm"
        if key in item:
            out[f"{axis}_range"] = nm_range_to_fraction(item[key], axis, cfg)
        elif f"{axis}_range" in item:
            out[f"{axis}_range"] = _fractional_range_or_full(
                item[f"{axis}_range"]
            )
        else:
            # Missing physical range means the full main-domain axis.
            out[f"{axis}_range"] = [0.0, 1.0]
    if default_gate_voltage:
        # A zero-volt gate is a useful geometry-only/default boundary; callers
        # can still provide any explicit voltage they need.
        out.setdefault("Vgate_val", 0.0)
    return out


def ordered_config_for_json(cfg, *, ensure_cpu_threads=False):
    """Return a non-mutating, human-oriented top-level JSON key ordering.

    CPU parallelism is placed first so it is immediately visible in generated
    configs.  Dielectric blocks are placed last because they can be very large
    and otherwise hide the run settings below them.  Both physical
    ``blocks_nm`` and legacy fractional ``blocks`` are handled.
    """
    if not isinstance(cfg, dict):
        raise TypeError("AFM JSON configuration must be a dictionary")

    ordered = {}
    if "cpu_threads" in cfg:
        ordered["cpu_threads"] = deepcopy(cfg["cpu_threads"])
    elif ensure_cpu_threads:
        ordered["cpu_threads"] = 1

    block_keys = ("blocks_nm", "blocks")
    for key, value in cfg.items():
        if key != "cpu_threads" and key not in block_keys:
            ordered[key] = deepcopy(value)
    for key in block_keys:
        if key in cfg:
            ordered[key] = deepcopy(cfg[key])
    return ordered


def normalize_config(cfg):
    """Return a solver-ready copy of a physical-coordinate configuration.

    ``blocks_nm`` and ``Vgate_nm`` are converted to fractional ``blocks`` and
    ``Vgate``. Tip dimensions and movement positions are likewise converted
    from nanometres. Existing fractional fields are accepted for compatibility,
    but new configurations should use only the ``*_nm`` form.
    """
    out = deepcopy(cfg)

    # Canonical physical scale: one voxel edge length plus main-grid resolution.
    # The old Lx_nm/Ly_nm/Lz_nm entries are derived internally for legacy
    # postprocessing APIs; they are not required or emitted in the JSON config.
    Lx_nm, Ly_nm, Lz_nm = physical_domain_nm(out)
    out["Lx_nm"] = Lx_nm
    out["Ly_nm"] = Ly_nm
    out["Lz_nm"] = Lz_nm

    # Preserve the physical geometry parameters for numerical routines.  The
    # solver-facing fractional representation remains available for masks,
    # while tip geometry is constructed from physical lengths so it stays
    # isotropic when nx, ny and nz are different.
    out["_physical"] = {
        "domain_nm": (Lx_nm, Ly_nm, Lz_nm),
        "origin_fraction": origin_fraction(out),
    }
    if "tip_z_nm" in out:
        out["_physical"]["tip_z_nm"] = float(out["tip_z_nm"])
    if "R_nm" in out:
        out["_physical"]["R_nm"] = float(out["R_nm"])
    if "r_tip_nm" in out:
        out["_physical"]["r_tip_nm"] = float(out["r_tip_nm"])

    # Tip cross-section shape: "pyramid" (square with rounded vertical edges,
    # default) or "cone" (legacy circular disk).  Normalizing here keeps every
    # downstream builder on the same validated value.
    out["tip_shape"] = resolve_tip_shape(out.get("tip_shape", "pyramid"))

    # Physical geometry is the preferred and canonical representation.
    if "blocks_nm" in out:
        out["blocks"] = [
            _convert_range_dict(b, out)
            for b in _spatial_items(out.pop("blocks_nm"), "blocks_nm")
        ]
    elif "blocks" in out:
        out["blocks"] = [
            _convert_range_dict(b, out)
            for b in _spatial_items(out["blocks"], "blocks")
        ]

    if "Vgate_nm" in out:
        out["Vgate"] = [
            _convert_range_dict(g, out, default_gate_voltage=True)
            for g in _spatial_items(out.pop("Vgate_nm"), "Vgate_nm")
        ]
    elif "Vgate" in out:
        out["Vgate"] = [
            _convert_range_dict(g, out, default_gate_voltage=True)
            for g in _spatial_items(out["Vgate"], "Vgate")
        ]

    zoom_cfg = out.get("zoom_simulation")
    if zoom_cfg is not None:
        if not isinstance(zoom_cfg, dict):
            raise ValueError("zoom_simulation must be an object when provided")
        cut = zoom_cfg.get("cut")
        if cut is None:
            cut = {}
        if not isinstance(cut, dict):
            raise ValueError("zoom_simulation.cut must be an object when provided")
        # Zoom cuts use the same range convention as blocks and gates.  The
        # conversion also accepts optional *_range_nm values for consistency.
        zoom_cfg["cut"] = _convert_range_dict(cut, out)

    for key, axis in (("tip_z_nm", "z"), ("R_nm", "x"), ("r_tip_nm", "x")):
        if key in out:
            solver_key = {"tip_z_nm": "tip_z", "R_nm": "R", "r_tip_nm": "r_tip"}[key]
            # Keep the legacy normalized fields for callers that still use
            # them.  Physical-config simulations use ``out["_physical"]`` for
            # tip construction, which is axis-aware and therefore valid for
            # non-cubic grids.
            if key == "tip_z_nm":
                out[solver_key] = (
                    origin_fraction(out)[2]
                    + float(out[key]) / _axis_length_nm(out, "z")
                )
            else:
                # A scalar legacy radius is only a compatibility value.  The
                # physical path uses the original nm radius directly.
                out[solver_key] = float(out[key]) / _axis_length_nm(out, axis)
            out.pop(key)

    mov = out.get("movement")
    if isinstance(mov, dict):
        physical_mov = out["_physical"].setdefault("movement", {})
        if "start_nm" in mov:
            physical_mov["start_nm"] = [float(v) for v in mov["start_nm"]]
            mov["start"] = [nm_to_fraction(v, a, out) for v, a in zip(mov["start_nm"], "xyz")]
            mov.pop("start_nm")
        if "end_nm" in mov:
            physical_mov["end_nm"] = [float(v) for v in mov["end_nm"]]
            mov["end"] = [nm_to_fraction(v, a, out) for v, a in zip(mov["end_nm"], "xyz")]
            mov.pop("end_nm")
        if "spacing_nm" in mov:
            physical_mov["spacing_nm"] = float(mov["spacing_nm"])
            mov["spacing"] = float(mov.pop("spacing_nm")) / _axis_length_nm(out, "x")
        if "spacing_nm_xyz" in mov:
            physical_mov["spacing_nm_xyz"] = [float(v) for v in mov["spacing_nm_xyz"]]
            mov["spacing_xyz"] = [
                float(v) / _axis_length_nm(out, a)
                for v, a in zip(mov.pop("spacing_nm_xyz"), "xyz")
            ]

    return out
