"""Configuration presimulation utilities.

Generates one solver-ready JSON per requested tip-z offset.  Geometry remains
in physical nanometres relative to the configured main-grid origin; the solver
performs the nm-to-grid transformation when loading each generated JSON.
"""
import json
import os
import re
from copy import deepcopy


def _format_offset_nm(value):
    """Format a tip-z offset for a generated filename."""
    value = float(value)
    if abs(value) < 1e-12:
        return "0nm"
    sign = "+" if value > 0 else "-"
    mag = abs(value)
    text = f"{mag:g}"
    return f"{sign}{text}nm"


def generate_tip_offset_configs(config_path, offsets_nm=None, overwrite=True):
    """Generate one JSON configuration for each tip-z offset.

    Parameters
    ----------
    config_path : str
        Base JSON configuration containing ``tip_z_nm`` and a
        ``presimulation.tip_z_offsets_nm`` list.
    offsets_nm : sequence of float, optional
        Explicit offsets. If omitted, the list in the base configuration is
        used.
    overwrite : bool, optional
        Replace existing generated files when True (default).

    Returns
    -------
    list of str
        Generated JSON paths in increasing offset order.
    """
    config_path = os.fspath(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if offsets_nm is None:
        offsets_nm = cfg.get("presimulation", {}).get("tip_z_offsets_nm", [0.0])

    offsets_nm = sorted({float(v) for v in offsets_nm})
    if not offsets_nm:
        raise ValueError("No tip-z offsets were supplied for presimulation.")

    base_dir = os.path.dirname(os.path.abspath(config_path))
    stem = os.path.splitext(os.path.basename(config_path))[0]
    generated = []

    for offset in offsets_nm:
        out_cfg = deepcopy(cfg)
        out_cfg.pop("presimulation", None)
        out_cfg["tip_z_nm"] = float(cfg["tip_z_nm"]) + offset

        suffix = _format_offset_nm(offset)
        out_path = os.path.join(base_dir, f"{stem}_{suffix}.json")
        if os.path.exists(out_path) and not overwrite:
            generated.append(out_path)
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_cfg, f, indent=2)
            f.write("\n")
        generated.append(out_path)

    return generated


def run_presimulation(config_path, offsets_nm=None, overwrite=True):
    """Generate and report tip-offset configurations."""
    paths = generate_tip_offset_configs(
        config_path, offsets_nm=offsets_nm, overwrite=overwrite
    )
    print(f"Generated {len(paths)} tip-z configurations from {os.path.basename(config_path)}:")
    for path in paths:
        print(f"  {os.path.basename(path)}")
    return paths
