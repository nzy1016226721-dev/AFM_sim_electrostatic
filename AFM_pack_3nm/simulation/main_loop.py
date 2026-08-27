import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import time
import os
import csv
import json
import sys
import gc
from scipy.ndimage import zoom

from .solver import build_downward_pointing_tip, mg_3d_masked, MG_TIME
from .materials import (generate_eps_level, build_eps_reference_memmap,
                        release_eps_reference)
from .io_utils import (make_gate_mask, get_sorted_config_files,
                        save_potential_full, save_potential_physical_cut)
from .plotting import plot_phi_plane, plot_residual_plane
from .zoom import run_zoom_simulation
from .runtime import resolve_output_dir, resolve_plotting_enabled, is_spyder_like_ide
from .coordinates import normalize_config


def run_afm_simulation(Vtip=5, nx=32, ny=32, nz=32,
                       tip_z=0.2, R=0.05, r_tip=0.15,
                       damping=0.8, nu1=2, nu2=2,
                       max_iter=1000, tol=1e-4, aspect_ratio=2.0,
                       verbose=True, eps_r=None, eps=True,
                        mg_max_runtime=None, blocks=None, Vgate=None,
                        output_dir=".", save_all_levels=False,
                        level_name_prefix=None, plotting_enabled=True, memory_tracking=False,
                        physical_params=None, eps_reference_resolution=512):
    """Run a multiresolution AFM electrostatic simulation.

    Starts from an 8x8x8 grid, solves Poisson's equation at each
    refinement level (doubling grid), and upscales the solution
    using scipy.ndimage.zoom until the target grid is reached.

    Parameters
    ----------
    Vtip : float, optional
        Tip voltage in V (default: 5).
    nx, ny, nz : int, optional
        Target grid dimensions (default: 32).
    tip_z : float, optional
        Tip apex fractional z (default: 0.2).
    R : float, optional
        Tip curvature radius (fractional, default: 0.05).
    r_tip : float, optional
        Tip truncation radius (fractional, default: 0.15).
    damping : float, optional
        SOR damping factor (default: 0.8).
    nu1, nu2 : int, optional
        Pre/post-smoothing steps (unused, default: 2).
    max_iter : int, optional
        Max total solver iterations (default: 1000).
    tol : float, optional
        Convergence tolerance (default: 1e-4).
    aspect_ratio : float, optional
        Tip aspect ratio (default: 2.0).
    verbose : bool, optional
        If True, print progress (default: True).
    eps_r : np.ndarray or None, optional
        Pre-built epsilon cell array (default: None).
    eps : bool, optional
        If True, use dielectric solver (default: True).
    mg_max_runtime : float or None, optional
        Max wall-clock time for MG solver (default: None).
    blocks : list of dict or None, optional
        Dielectric blocks (default: None).
    Vgate : list of dict or dict or None, optional
        Gate definitions with 'x_range', 'y_range', 'z_range', 'Vgate_val'.
    output_dir : str, optional
        Output directory for logs (default: ".").

    Returns
    -------
    dict
        Results containing 'phi', 'residual', 'tip_mask',
        'boundary_mask', and 'parameters'.
    """
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print("Starting multiresolution AFM simulation...")

    nx_target, ny_target, nz_target = int(nx), int(ny), int(nz)
    # Very large production grids do not need the legacy 8^3 warm-up levels.
    # If any final axis exceeds 512, start the multigrid hierarchy at 64^3;
    # otherwise retain the historical 8^3 start.  Each axis then advances
    # independently toward its requested target.
    initial_level = 64 if max(nx_target, ny_target, nz_target) > 512 else 8
    nx = min(initial_level, nx_target)
    ny = min(initial_level, ny_target)
    nz = min(initial_level, nz_target)
    phi = np.full((nx, ny, nz), 0.001, dtype=np.float32)
    level = 1

    # Build one temporary file-backed high-resolution material reference from the
    # current (already movement-adjusted) JSON block distribution.  It is used to
    # volume-average epsilon onto all coarse levels and never remains resident as
    # a second full in-RAM simulation array.
    eps_reference_path = None
    eps_reference_mmap = None
    if eps_r is None:
        ref_value = eps_reference_resolution
        ref_shape = (int(ref_value),) * 3 if not isinstance(ref_value, (list, tuple)) else tuple(int(v) for v in ref_value)
        eps_reference_path, eps_reference_mmap = build_eps_reference_memmap(
            ref_shape, blocks=blocks
        )

    while True:
        if verbose:
            print(f"\n[Level {level}] Solving on {nx}x{ny}x{nz} grid...")

        physical = physical_params
        if physical:
            domain_nm = physical["domain_nm"]
            origin = physical["origin_fraction"]
            tip_mask, tip_pos, base_pos = build_downward_pointing_tip(
                nx, ny, nz, tip_z, R, r_tip, aspect_ratio, verbose=False,
                tip_z_nm=physical.get("tip_z_nm"),
                R_nm=physical.get("R_nm"),
                r_tip_nm=physical.get("r_tip_nm"),
                domain_nm=domain_nm,
                center_fraction=origin,
            )
        else:
            tip_mask, tip_pos, base_pos = build_downward_pointing_tip(
                nx, ny, nz, tip_z, R, r_tip, aspect_ratio, verbose=False
            )

        boundary_mask = np.zeros((nx, ny, nz), dtype=bool)

        gate_masks = []
        gate_values = []
        if Vgate is not None and isinstance(Vgate, (list, tuple)) and len(Vgate) > 0:
            for g in Vgate:
                mask_g = make_gate_mask(nx, ny, nz, g)
                val_g  = float(g.get("Vgate_val", 0.0))
                phi[mask_g] = val_g
                boundary_mask[mask_g] = True
                gate_masks.append(mask_g)
                gate_values.append(val_g)
        else:
            if isinstance(Vgate, dict):
                mask_g = make_gate_mask(nx, ny, nz, Vgate)
                val_g  = float(Vgate.get("Vgate_val", 0.0))
                phi[mask_g] = val_g
                boundary_mask[mask_g] = True
                gate_masks.append(mask_g)
                gate_values.append(val_g)

        phi[tip_mask] = Vtip
        boundary_mask[tip_mask] = True

        if eps_r is None:
            eps_reference = eps_reference_resolution
            eps_reference_shape = (int(eps_reference), int(eps_reference), int(eps_reference)) if not isinstance(eps_reference, (list, tuple)) else tuple(int(v) for v in eps_reference)
            eps_cell = generate_eps_level(
                phi.shape, blocks, reference_shape=eps_reference_shape,
                reference=eps_reference_mmap
            )
        else:
            eps_cell = np.asarray(eps_r, dtype=np.float32)

        if memory_tracking:
            from .memory import track_memory, log_memory_usage
            memory_context = track_memory()
        else:
            from contextlib import nullcontext
            memory_context = nullcontext(None)

        with memory_context as mem_tracker:
            phi_solution, res_m = mg_3d_masked(Vtip, phi.copy(), boundary_mask,
                                        damping=damping, nu1=nu1, nu2=nu2,
                                        max_iter=max_iter, tol=tol,
                                        verbose=verbose, eps_r=eps_cell, eps=eps,
                                        mg_max_runtime=mg_max_runtime,
                                        output_dir=output_dir,
                                        plotting_enabled=plotting_enabled)

        if memory_tracking:
            log_memory_usage(f"main {nx}x{ny}x{nz}", mem_tracker.peak_gb, output_dir=output_dir)

        logfile = os.path.join(output_dir, "mg_timing_log.csv")
        os.makedirs(output_dir, exist_ok=True)
        file_exists = os.path.isfile(logfile)
        with open(logfile, "a", newline="") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["level", "Nx", "Ny", "Nz", "time_sec"])
            level_elapsed = MG_TIME.get("elapsed", 0.0)
            w.writerow([level, nx, ny, nz, f"{level_elapsed:.6f}"])

        is_final = (nx == nx_target and ny == ny_target and nz == nz_target)
        if save_all_levels and level_name_prefix and nx >= 32 and not is_final:
            level_base = os.path.splitext(level_name_prefix)[0]
            level_name = f"{level_base}_level{nx}x{ny}x{nz}.npy"
            level_path = os.path.join(output_dir, level_name)
            n = 0
            while os.path.exists(level_path):
                n += 1
                level_path = os.path.join(output_dir,
                                          f"{os.path.splitext(level_name)[0]} ({n}).npy")
            np.save(level_path, phi_solution)
            print(f"  Saved level: {os.path.basename(level_path)}")

        if is_final:
            if verbose:
                print(f"[Level {level}] Target grid size reached ({nx}x{ny}x{nz}). Simulation complete.")
            # Keep only the final potential and residual for the returned result.
            del eps_cell
            if "tip_mask" in locals():
                # tip_mask is returned, so do not delete it.
                pass
            if "gate_masks" in locals():
                del gate_masks
            if "gate_values" in locals():
                del gate_values
            gc.collect()
            break

        old_nx, old_ny, old_nz = nx, ny, nz
        nx = min(nx * 2, nx_target)
        ny = min(ny * 2, ny_target)
        nz = min(nz * 2, nz_target)

        scale = (
            nx / old_nx,
            ny / old_ny,
            nz / old_nz,
        )
        phi = zoom(phi_solution, scale, order=1)
        # scipy's zoom can round by one node for non-integer scale factors.
        # Enforce the exact target shape so rectangular targets such as
        # 256x256x100 are represented without silently changing nz.
        phi = phi[:nx, :ny, :nz]
        if phi.shape != (nx, ny, nz):
            pad = np.full((nx, ny, nz), float(phi[-1, -1, -1]), dtype=phi.dtype)
            pad[:phi.shape[0], :phi.shape[1], :phi.shape[2]] = phi
            phi = pad
        # Release all per-level arrays that are no longer needed before the next level.
        # Only the interpolated potential is retained for the next solve.
        del phi_solution
        del res_m
        del eps_cell
        del boundary_mask
        if "tip_mask" in locals():
            del tip_mask
        if "gate_masks" in locals():
            del gate_masks
        if "gate_values" in locals():
            del gate_values
        gc.collect()

        if verbose:
            print(f"[Level {level}] Converged. Upscaling to {nx}x{ny}x{nz} using factors {scale}...")

        level += 1

    if eps_reference_path is not None:
        release_eps_reference(eps_reference_path, eps_reference_mmap)
        eps_reference_path = None
        eps_reference_mmap = None
        gc.collect()

    results = {
        'phi': phi_solution,
        'residual': res_m,
        'tip_mask': tip_mask, 'boundary_mask': boundary_mask,
        'parameters': {'nx': nx, 'ny': ny, 'nz': nz,
                       'tip_pos': tip_pos, 'base_pos': base_pos,
                       'levels': level}
    }
    return results


def move_voltage_gate(Vgate, gate_index, center,
                      xrange, yrange, zrange, Vgate_val=None):
    """Reposition a voltage gate to a new centre with given half-extents.

    Parameters
    ----------
    Vgate : list of dict
        Gate list (will be modified in-place).
    gate_index : int
        Index of the gate to move.
    center : tuple of float
        (cx, cy, cz) new centre in fractional coordinates.
    xrange : tuple of float
        (xneg, xpos) half-extents in x.
    yrange : tuple of float
        (yneg, ypos) half-extents in y.
    zrange : tuple of float
        (zneg, zpos) half-extents in z.
    Vgate_val : float or None, optional
        New gate voltage value (default: None = keep existing).

    Returns
    -------
    list of dict
        Modified Vgate list.
    """
    cx, cy, cz = center
    xneg, xpos = xrange
    yneg, ypos = yrange
    zneg, zpos = zrange

    x1 = max(0.0, cx + xneg)
    x2 = min(1.0, cx + xpos)
    y1 = max(0.0, cy + yneg)
    y2 = min(1.0, cy + ypos)
    z1 = max(0.0, cz + zneg)
    z2 = min(1.0, cz + zpos)

    gate = Vgate[gate_index]
    gate["x_range"] = [x1, x2]
    gate["y_range"] = [y1, y2]
    gate["z_range"] = [z1, z2]

    if Vgate_val is not None:
        gate["Vgate_val"] = Vgate_val
    return Vgate


def move_dielectric_block(cfg, block_index, center,
                          xrange, yrange, zrange, eps_val=None):
    """Reposition a dielectric block to a new centre.

    Parameters
    ----------
    cfg : dict
        Config dict containing 'blocks' (modified in-place).
    block_index : int
        Index of the block to move.
    center : tuple of float
        (cx, cy, cz) new centre.
    xrange : tuple of float
        (xneg, xpos) half-extents in x.
    yrange : tuple of float
        (yneg, ypos) half-extents in y.
    zrange : tuple of float
        (zneg, zpos) half-extents in z.
    eps_val : float or None, optional
        New epsilon value (default: None = keep existing).

    Returns
    -------
    dict
        Modified config.
    """
    cx, cy, cz = center
    xneg, xpos = xrange
    yneg, ypos = yrange
    zneg, zpos = zrange

    x1 = max(0.0, cx + xneg)
    x2 = min(1.0, cx + xpos)
    y1 = max(0.0, cy + yneg)
    y2 = min(1.0, cy + ypos)
    z1 = max(0.0, cz + zneg)
    z2 = min(1.0, cz + zpos)

    blk = cfg["blocks"][block_index]
    blk["x_range"] = [x1, x2]
    blk["y_range"] = [y1, y2]
    blk["z_range"] = [z1, z2]

    if eps_val is not None:
        blk["eps_val"] = eps_val
    return cfg


def compute_block_positions(start_center, end_center, spacing, domain_nm=None):
    """Linearly interpolate centres at a physical or fractional spacing.

    If ``domain_nm=(Lx, Ly, Lz)`` is supplied, the spacing is interpreted in
    nanometres and the interpolation step count is based on the physical
    Euclidean distance.  This is required for rectangular grids where the
    fractional x/y/z axes have different scale factors.
    """
    start = np.array(start_center, float)
    end = np.array(end_center, float)
    dist = end - start

    if domain_nm is not None:
        lengths = np.asarray(domain_nm, dtype=float)
        physical_dist = dist * lengths
        max_dist = float(np.linalg.norm(physical_dist))
        spacing_value = float(spacing)
    else:
        max_dist = float(np.linalg.norm(dist))
        spacing_value = float(spacing)

    if spacing_value <= 0:
        raise ValueError("Movement spacing must be positive")

    nsteps = max(1, int(np.ceil(max_dist / spacing_value - 1e-12)))
    tvals = np.linspace(0.0, 1.0, nsteps + 1)
    return [tuple(start + t * dist) for t in tvals]


def apply_block_motion(cfg, block_motion_list, center):
    """Apply a list of block motions relative to a centre.

    Parameters
    ----------
    cfg : dict
        Config dict with 'blocks'.
    block_motion_list : list of dict
        Each dict has 'index', 'extent' (6-element list), optional 'eps_val'.
    center : tuple of float
        Centre position.

    Returns
    -------
    dict
        Modified config.
    """
    if not block_motion_list:
        return cfg

    cx, cy, cz = center

    for blk in block_motion_list:
        idx = blk["index"]
        ex = blk["extent"]
        xrange = (ex[0], ex[1])
        yrange = (ex[2], ex[3])
        zrange = (ex[4], ex[5])
        eps_val = blk.get("eps_val", None)

        cfg = move_dielectric_block(cfg, idx, center,
                                    xrange=xrange, yrange=yrange, zrange=zrange,
                                    eps_val=eps_val)
    return cfg


def apply_vgate_motion(Vgate_list, vgate_motion_list, center):
    """Apply a list of gate motions relative to a centre.

    Parameters
    ----------
    Vgate_list : list of dict
        Gate list (modified in-place).
    vgate_motion_list : list of dict
        Each dict has 'index', 'extent' (6-element), optional 'Vgate_val'.
    center : tuple of float
        Centre position.

    Returns
    -------
    list of dict
        Modified Vgate list.
    """
    if not vgate_motion_list:
        return Vgate_list

    cx, cy, cz = center

    for g in vgate_motion_list:
        idx = g["index"]
        ex = g["extent"]
        xrange = (ex[0], ex[1])
        yrange = (ex[2], ex[3])
        zrange = (ex[4], ex[5])

        val = g.get("Vgate_val", None)

        Vgate_list = move_voltage_gate(
            Vgate_list, idx, center,
            xrange=xrange, yrange=yrange, zrange=zrange,
            Vgate_val=val
        )
    return Vgate_list


def _clip01(a, b):
    """Clamp a pair of values to the [0, 1] range, returning (lo, hi)."""
    lo = max(0.0, min(a, b))
    hi = min(1.0, max(a, b))
    return lo, hi


def precompute_relative_offsets_for_blocks(blocks, indices, center0):
    """Compute relative (centre-relative) offsets for a set of blocks.

    Parameters
    ----------
    blocks : list of dict
        Dielectric blocks.
    indices : list of int
        Indices of mobile blocks.
    center0 : tuple of float
        Reference centre position.

    Returns
    -------
    dict
        Mapping from index to relative offsets dict with 'x', 'y', 'z', 'eps'.
    """
    cx0, cy0, cz0 = center0
    rel = {}
    for idx in indices:
        b = blocks[idx]
        x1, x2 = b["x_range"]; y1, y2 = b["y_range"]; z1, z2 = b["z_range"]
        rel[idx] = {
            "x": [x1 - cx0, x2 - cx0],
            "y": [y1 - cy0, y2 - cy0],
            "z": [z1 - cz0, z2 - cz0],
            "eps": b.get("eps_val", None),
        }
    return rel


def precompute_relative_offsets_for_vgates(vgates, indices, center0):
    """Compute relative offsets for a set of voltage gates.

    Parameters
    ----------
    vgates : list of dict
        Gate definitions.
    indices : list of int
        Mobile gate indices.
    center0 : tuple of float
        Reference centre.

    Returns
    -------
    dict
        Mapping from index to relative offsets dict with 'x', 'y', 'z', 'V'.
    """
    cx0, cy0, cz0 = center0
    rel = {}
    for idx in indices:
        g = vgates[idx]
        x1, x2 = g["x_range"]; y1, y2 = g["y_range"]; z1, z2 = g["z_range"]
        rel[idx] = {
            "x": [x1 - cx0, x2 - cx0],
            "y": [y1 - cy0, y2 - cy0],
            "z": [z1 - cz0, z2 - cz0],
            "V": g.get("Vgate_val", None),
        }
    return rel


def apply_relative_blocks(blocks, rel, center):
    """Apply precomputed relative offsets to position blocks at a new centre.

    Applies clipping and edge-fix entries (x_fix, y_fix, z_fix) if present.

    Parameters
    ----------
    blocks : list of dict
        Block list (modified in-place).
    rel : dict
        Relative offsets from precompute_relative_offsets_for_blocks.
    center : tuple of float
        New centre (cx, cy, cz).

    Returns
    -------
    list of dict
        Modified block list.
    """
    cx, cy, cz = center

    for idx, off in rel.items():
        blk = blocks[idx]

        x1, x2 = cx + off["x"][0], cx + off["x"][1]
        y1, y2 = cy + off["y"][0], cy + off["y"][1]
        z1, z2 = cz + off["z"][0], cz + off["z"][1]

        x1, x2 = _clip01(x1, x2)
        y1, y2 = _clip01(y1, y2)
        z1, z2 = _clip01(z1, z2)

        xf = blk.get("x_fix", ["", "", "", ""])
        yf = blk.get("y_fix", ["", "", "", ""])
        zf = blk.get("z_fix", ["", "", "", ""])

        xf0 = _parse_fix_entry(xf[0]); xf1 = _parse_fix_entry(xf[1])
        xf2 = _parse_fix_entry(xf[2]); xf3 = _parse_fix_entry(xf[3])

        yf0 = _parse_fix_entry(yf[0]); yf1 = _parse_fix_entry(yf[1])
        yf2 = _parse_fix_entry(yf[2]); yf3 = _parse_fix_entry(yf[3])

        zf0 = _parse_fix_entry(zf[0]); zf1 = _parse_fix_entry(zf[1])
        zf2 = _parse_fix_entry(zf[2]); zf3 = _parse_fix_entry(zf[3])

        x1 = _apply_edge_fix(x1, xf0, xf1)
        x2 = _apply_edge_fix(x2, xf2, xf3)
        y1 = _apply_edge_fix(y1, yf0, yf1)
        y2 = _apply_edge_fix(y2, yf2, yf3)
        z1 = _apply_edge_fix(z1, zf0, zf1)
        z2 = _apply_edge_fix(z2, zf2, zf3)

        blk["x_range"] = [x1, x2]
        blk["y_range"] = [y1, y2]
        blk["z_range"] = [z1, z2]

    return blocks


def apply_relative_vgates(vgates, rel, center):
    """Apply precomputed relative offsets to position gates at a new centre.

    Parameters
    ----------
    vgates : list of dict
        Gate list (modified in-place).
    rel : dict
        Relative offsets from precompute_relative_offsets_for_vgates.
    center : tuple of float
        New centre (cx, cy, cz).

    Returns
    -------
    list of dict
        Modified gate list.
    """
    cx, cy, cz = center

    for idx, off in rel.items():
        g = vgates[idx]

        x1, x2 = cx + off["x"][0], cx + off["x"][1]
        y1, y2 = cy + off["y"][0], cy + off["y"][1]
        z1, z2 = cz + off["z"][0], cz + off["z"][1]

        x1, x2 = _clip01(x1, x2)
        y1, y2 = _clip01(y1, y2)
        z1, z2 = _clip01(z1, z2)

        xf = g.get("x_fix", ["", "", "", ""])
        yf = g.get("y_fix", ["", "", "", ""])
        zf = g.get("z_fix", ["", "", "", ""])

        xf0 = _parse_fix_entry(xf[0]); xf1 = _parse_fix_entry(xf[1])
        xf2 = _parse_fix_entry(xf[2]); xf3 = _parse_fix_entry(xf[3])

        yf0 = _parse_fix_entry(yf[0]); yf1 = _parse_fix_entry(yf[1])
        yf2 = _parse_fix_entry(yf[2]); yf3 = _parse_fix_entry(yf[3])

        zf0 = _parse_fix_entry(zf[0]); zf1 = _parse_fix_entry(zf[1])
        zf2 = _parse_fix_entry(zf[2]); zf3 = _parse_fix_entry(zf[3])

        x1 = _apply_edge_fix(x1, xf0, xf1)
        x2 = _apply_edge_fix(x2, xf2, xf3)
        y1 = _apply_edge_fix(y1, yf0, yf1)
        y2 = _apply_edge_fix(y2, yf2, yf3)
        z1 = _apply_edge_fix(z1, zf0, zf1)
        z2 = _apply_edge_fix(z2, zf2, zf3)

        g["x_range"] = [x1, x2]
        g["y_range"] = [y1, y2]
        g["z_range"] = [z1, z2]

    return vgates


def _parse_fix_entry(v):
    """Parse a fix entry: None/empty/":" returns None, otherwise float."""
    if v is None:
        return None
    if v == "" or v == ":":
        return None
    return float(v)


def _apply_edge_fix(edge_value, lo_fix, hi_fix):
    """Constrain an edge value within optional lo/hi fix bounds, clamped [0,1]."""
    edge = edge_value
    if lo_fix is not None:
        edge = max(edge, lo_fix)
    if hi_fix is not None:
        edge = min(edge, hi_fix)
    return max(0.0, min(1.0, edge))


def preview_before_run(cfg, plotting_enabled=True):
    """Display an interactive 2D preview of dielectric blocks and gates.

    Shows projection slices with dual colorbars; prompts user to
    proceed with or abort the simulation.

    Parameters
    ----------
    cfg : dict
        Configuration dict with 'blocks', 'Vgate', 'movement'.

    Returns
    -------
    bool
        True if user confirms, calls sys.exit(0) otherwise.
    """
    if not plotting_enabled:
        return True

    blocks = cfg.get("blocks", [])
    vgates = cfg.get("Vgate", [])
    movement = cfg.get("movement", {})
    cx, cy, cz = movement.get("start", [0.5, 0.5, 0.5])

    nx = ny = nz = 200
    block_mask = np.zeros((nx, ny, nz), dtype=np.float32)
    gate_mask  = np.zeros((nx, ny, nz), dtype=int)

    eps_values = []
    for blk in blocks:
        xr, yr, zr = blk["x_range"], blk["y_range"], blk["z_range"]
        eps = blk["eps_val"]
        eps_values.append(eps)

        ix0 = int(xr[0]*(nx-1)); ix1 = int(xr[1]*(nx-1))
        iy0 = int(yr[0]*(ny-1)); iy1 = int(yr[1]*(ny-1))
        iz0 = int(zr[0]*(nz-1)); iz1 = int(zr[1]*(nz-1))
        block_mask[ix0:ix1+1, iy0:iy1+1, iz0:iz1+1] = eps

    eps_max = max(max(eps_values), 25)
    CUT = 25.0

    for g in vgates:
        xr, yr, zr = g["x_range"], g["y_range"], g["z_range"]
        ix0 = int(xr[0]*(nx-1)); ix1 = int(xr[1]*(nx-1))
        iy0 = int(yr[0]*(ny-1)); iy1 = int(yr[1]*(ny-1))
        iz0 = int(zr[0]*(nz-1)); iz1 = int(zr[1]*(nz-1))
        gate_mask[ix0:ix1+1, iy0:iy1+1, iz0:iz1+1] = 1

    proj_xy = np.max(block_mask, axis=2)
    proj_xz = np.max(block_mask, axis=1)
    proj_yz = np.max(block_mask, axis=0)

    gate_xy = np.max(gate_mask, axis=2)
    gate_xz = np.max(gate_mask, axis=1)
    gate_yz = np.max(gate_mask, axis=0)

    cmap_lo = matplotlib.colormaps["viridis"]
    cmap_hi_full = matplotlib.colormaps["inferno_r"]
    cmap_hi = matplotlib.colors.LinearSegmentedColormap.from_list(
        "inferno_red_only",
        [cmap_hi_full(0.00), cmap_hi_full(0.25), cmap_hi_full(0.5)]
    )

    def colorize(arr2d):
        H, W = arr2d.shape
        rgb = np.zeros((H, W, 4))
        zero_mask = arr2d <= 0
        rgb[zero_mask] = cmap_lo(0.0)
        lo_mask = (arr2d > 0) & (arr2d < CUT)
        if np.any(lo_mask):
            t = arr2d[lo_mask] / CUT
            rgb[lo_mask] = cmap_lo(t)
        hi_mask = arr2d >= CUT
        if np.any(hi_mask):
            t = (arr2d[hi_mask] - CUT) / (eps_max - CUT + 1e-12)
            rgb[hi_mask] = cmap_hi(t)
        return rgb

    def draw_dual_colorbars(fig):
        ax1 = fig.add_axes([0.02, 0.15, 0.02, 0.7])
        vals1 = np.linspace(0, CUT, 200).reshape(200, 1)
        ax1.imshow(cmap_lo(vals1 / CUT), origin="lower", aspect="auto")
        ax1.set_yticks([0, 199])
        ax1.set_yticklabels(["0", "25"])
        ax1.set_xticks([])
        ax1.set_title("eps < 25", fontsize=9)

        ax2 = fig.add_axes([0.95, 0.15, 0.02, 0.7])
        vals2 = np.linspace(CUT, eps_max, 200).reshape(200, 1)
        t = (vals2 - CUT) / (eps_max - CUT + 1e-12)
        ax2.imshow(cmap_hi(t), origin="lower", aspect="auto")
        ax2.set_yticks([0, 199])
        ax2.set_yticklabels(["25", f"{int(eps_max)}"])
        ax2.set_xticks([])
        ax2.set_title("eps >= 25", fontsize=9)

    def plot_panels(title, show_gates=False, show_colorbars=True):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        panels = [
            ("XY (compress Z)", proj_xy, gate_xy),
            ("XZ (compress Y)", proj_xz, gate_xz),
            ("YZ (compress X)", proj_yz, gate_yz),
        ]

        for ax, (ttl, arr, gm) in zip(axes, panels):
            img = colorize(arr).swapaxes(0,1)
            ax.imshow(img, origin="lower")
            ax.set_title(ttl)

            if show_gates:
                ax.imshow(gm.T, cmap="gray_r", alpha=1.0)

        if show_colorbars:
            draw_dual_colorbars(fig)

        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0.05, 0, 0.93, 1])
        plt.show()

    plot_panels(
        "PREVIEW - Dielectrics (viridis for eps<25, inferno_r for eps>=25)",
        show_gates=False, show_colorbars=True
    )

    if len(vgates) > 0:
        plot_panels(
            "PREVIEW - Gates (white overlay, black conductor gates)",
            show_gates=True, show_colorbars=False
        )

    while True:
        ans = input("\nProceed with AFM simulation? (y/n): ").strip().lower()
        if ans == "y":
            return True
        elif ans == "n":
            print("Simulation aborted.")
            sys.exit(0)
        else:
            print("Enter 'y' or 'n'.")


def preview_tip_only(cfg, plotting_enabled=True):
    """Display a 3-projection scatter plot of the AFM tip voxels.

    Parameters
    ----------
    cfg : dict
        Config dict with 'tip_z', 'R', 'r_tip', 'aspect_ratio'.

    Returns
    -------
    None
    """
    if not plotting_enabled:
        return

    tip_z = cfg.get("tip_z", 0.7)
    R = cfg.get("R", 0.5)
    r_tip = cfg.get("r_tip", 0.15)
    aspect = cfg.get("aspect_ratio", 1.0)
    physical = cfg.get("_physical")

    nx = ny = nz = 120

    if physical and all(k in physical for k in ("tip_z_nm", "R_nm", "r_tip_nm")):
        tip_mask, _, _ = build_downward_pointing_tip(
            nx, ny, nz, tip_z=tip_z, R=R, r_tip=r_tip,
            aspect_ratio=aspect, verbose=False,
            tip_z_nm=physical["tip_z_nm"],
            R_nm=physical["R_nm"],
            r_tip_nm=physical["r_tip_nm"],
            domain_nm=physical["domain_nm"],
            center_fraction=physical["origin_fraction"],
        )
    else:
        tip_mask, _, _ = build_downward_pointing_tip(
            nx, ny, nz, tip_z=tip_z, R=R, r_tip=r_tip,
            aspect_ratio=aspect, verbose=False
        )

    vox = np.array(np.where(tip_mask)).T
    if vox.size == 0:
        print("Tip preview: no voxels found; check tip parameters.")
        return

    xs = vox[:, 0] / (nx - 1)
    ys = vox[:, 1] / (ny - 1)
    zs = vox[:, 2] / (nz - 1)

    keep = zs >= tip_z
    xs, ys, zs = xs[keep], ys[keep], zs[keep]

    MAX_POINTS = 120_000
    if xs.size > MAX_POINTS:
        idx = np.linspace(0, xs.size - 1, MAX_POINTS).astype(int)
        xs, ys, zs = xs[idx], ys[idx], zs[idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(xs, ys, s=6, c="orange", edgecolors="none", alpha=0.85)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", "box")
    ax.set_title(f"AFM Tip - Top (XY) Projection  (z >= {tip_z:.2f})")
    ax.set_xlabel("x (fraction)"); ax.set_ylabel("y (fraction)")

    ax = axes[1]
    ax.scatter(xs, zs, s=6, c="orange", edgecolors="none", alpha=0.85)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", "box")
    ax.set_title("AFM Tip - Side (XZ) Projection")
    ax.set_xlabel("x (fraction)"); ax.set_ylabel("z (fraction)")

    ax = axes[2]
    ax.scatter(ys, zs, s=6, c="orange", edgecolors="none", alpha=0.85)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", "box")
    ax.set_title("AFM Tip - Side (YZ) Projection")
    ax.set_xlabel("y (fraction)"); ax.set_ylabel("z (fraction)")

    plt.tight_layout()
    plt.show()

    print("\nTip preview complete (voxel projections, consistent XY/XZ/YZ).")


def batch_main(CONFIG_BASE_NAME="afm_config", config_dir=".", plotting_override=None):
    """Main loop: iterate over config files, run simulation + zoom for each.

    Handles movement (block/gate repositioning), voltage sweeps, saving
    results, plotting, and cleanup.

    Parameters
    ----------
    CONFIG_BASE_NAME : str, optional
        Config base name (default: "afm_config").

    Returns
    -------
    None
    """
    while True:
        source = os.fspath(CONFIG_BASE_NAME)
        if os.path.isfile(source):
            config_files = [source]
        elif os.path.isdir(source):
            config_files = get_sorted_config_files("afm_config", source)
        else:
            config_files = get_sorted_config_files(source, directory=config_dir)

        if not config_files:
            print(f"\nNo JSON configuration files found for source '{CONFIG_BASE_NAME}'.")
            # Preserve legacy interactive behavior when no explicit JSON path was supplied.
            if os.path.isfile(source) or os.path.splitext(source)[1].lower() == ".json":
                raise FileNotFoundError(f"JSON configuration not found: {source}")
            ans = input("Enter a new base name (e.g. 'myconfig') or press Enter to exit: ").strip()
            if ans == "":
                print("Exiting.")
                break
            CONFIG_BASE_NAME = ans
            continue

        with open(config_files[0], 'r') as f:
            first_cfg = normalize_config(json.load(f))

        first_plotting_enabled = resolve_plotting_enabled(
            first_cfg, cli_override=plotting_override
        )

        print(f"\nFound {len(config_files)} config files:")
        for path in config_files:
            print(f"  {os.path.basename(path)}")

        preview_tip_only(first_cfg, plotting_enabled=first_plotting_enabled)
        preview_before_run(first_cfg, plotting_enabled=first_plotting_enabled)

        if is_spyder_like_ide():
            ans = input(f"\nRun simulation for all {len(config_files)} configurations? (y/n): ").strip().lower()
            if ans != 'y':
                ans = input("Enter a new base name (or press Enter to exit): ").strip()
                if ans == "":
                    print("Exiting.")
                    break
                CONFIG_BASE_NAME = ans
                continue
        else:
            print("\nNon-interactive execution detected; starting simulation without confirmation.")

        for config_idx, config_path in enumerate(config_files, start=1):
            print(f"\n{'='*60}")
            print(f"Processing configuration {config_idx}/{len(config_files)}: "
                  f"{os.path.basename(config_path)}")
            print(f"{'='*60}")

            with open(config_path, 'r') as f:
                cfg = normalize_config(json.load(f))

            plotting_enabled = resolve_plotting_enabled(
                cfg, cli_override=plotting_override
            )
            output_dir = resolve_output_dir(
                cfg.get("output_dir", "."),
                config_path,
                cfg,
            )
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            mg_max_runtime = cfg.get("mg_max_runtime", None)
            blocks = cfg["blocks"]

            mov = cfg.get("movement", {})
            start_center = tuple(mov.get("start", [0.5, 0.5, 0.5]))
            end_center   = tuple(mov.get("end",   [0.5, 0.5, 0.5]))
            physical_cfg = cfg.get("_physical", {})
            physical_mov = physical_cfg.get("movement", {})
            if "spacing_nm" in physical_mov:
                spacing = physical_mov["spacing_nm"]
                centers = compute_block_positions(
                    start_center, end_center, spacing,
                    domain_nm=physical_cfg.get("domain_nm")
                )
            else:
                spacing = mov.get("spacing", 0.1)
                centers = compute_block_positions(start_center, end_center, spacing)

            v_start = float(cfg["v_start"])
            v_stop  = float(cfg["v_stop"])
            v_step  = abs(float(cfg["v_step"]))
            if v_start <= v_stop:
                V_values = np.arange(v_start, v_stop + 1e-12, v_step)
            else:
                V_values = np.arange(v_start, v_stop - 1e-12, -v_step)
            if len(V_values) == 0:
                print("  WARNING: no V values in sweep range - skipping config.")

            phi_results = []
            time_log = []

            stationary_upto = cfg.get("fixed_blocks", [2])[0]
            mobile_block_indices = list(range(stationary_upto, len(cfg["blocks"])))
            mobile_vgate_indices = list(range(len(cfg.get("Vgate", []))))
            center0 = centers[0]
            movement_active = len(centers) > 1

            rel_blocks = precompute_relative_offsets_for_blocks(
                cfg["blocks"], mobile_block_indices, center0)
            rel_vgates = precompute_relative_offsets_for_vgates(
                cfg.get("Vgate", []), mobile_vgate_indices, center0)

            for center in centers:
                cx, cy, cz = center
                print(f"\nMoving to center {center}")
                x_frac, y_frac, z_frac = cx, cy, cz

                cfg["blocks"] = apply_relative_blocks(cfg["blocks"], rel_blocks, center)
                if cfg.get("Vgate") is not None:
                    cfg["Vgate"] = apply_relative_vgates(cfg["Vgate"], rel_vgates, center)

                for V in V_values:
                    print(f"\n=== Running AFM simulation at Vtip = {V:.2f} V ===")
                    start_time = time.time()
                    
                    grid = cfg.get("grid_resolution", {})
                    nx_target = grid.get("nx", 256)
                    ny_target = grid.get("ny", 256)
                    nz_target = grid.get("nz", 256)
                    print(f"Using grid resolution: {nx_target}x{ny_target}x{nz_target}")

                    npy_name = f"afm_phi_{config_idx}_{V:.2f}V.npy"
                    if movement_active:
                        npy_name = (f"afm_phi_{config_idx}_cx{cx:.2f}_cy{cy:.2f}_cz{cz:.2f}"
                                    f"_{V:.2f}V.npy")

                    results = run_afm_simulation(
                        Vtip=V,
                        nx=nx_target, ny=ny_target, nz=nz_target,
                        tip_z=cfg["tip_z"],
                        R=cfg["R"],
                        r_tip=cfg["r_tip"],
                        damping=1,
                        nu1=2, nu2=2,
                        max_iter=60000,
                        tol=cfg.get("res_tol_main", 5e-5),
                        aspect_ratio=cfg["aspect_ratio"],
                        verbose=False,
                        eps=True,
                        mg_max_runtime=mg_max_runtime,
                        blocks=cfg["blocks"],
                        Vgate=cfg.get("Vgate", []),
                        output_dir=output_dir,
                        save_all_levels=cfg.get("save_all_levels", False),
                        level_name_prefix=npy_name,
                        plotting_enabled=plotting_enabled,
                        memory_tracking=cfg.get("memory_tracking", False),
                        physical_params=cfg.get("_physical"),
                        eps_reference_resolution=cfg.get("epsilon_material", {}).get("reference_resolution", 512)
                    )

                    phi = results["phi"]
                    nx, ny, nz = phi.shape
                    ix = int(x_frac * (nx - 1))
                    iy = int(y_frac * (ny - 1))
                    iz = int(z_frac * (nz - 1))
                    phi_val = phi[ix, iy, iz]
                    phi_results.append((cx, cy, cz, V, phi_val))

                    print(f"  phi({x_frac:.2f}, {y_frac:.2f}, {z_frac:.2f}) = {phi_val:.6f} V")
                    tag = f"cx{cx:.2f}_cy{cy:.2f}_cz{cz:.2f}_V{V:.2f}"

                    elapsed = time.time() - start_time
                    print(f"Runtime: {elapsed:.2f} s")
                    time_log.append(elapsed)

                    save_full = bool(cfg.get("save_full", False))
                    save_cut = bool(cfg.get("save_cut", False))
                    cut_offsets_nm = cfg.get("save_cut_box_nm", [-32.0, 32.0, -32.0, 32.0, -32.0, 32.0])
                    physical_cfg = cfg.get("_physical", {})
                    domain_nm = tuple(float(v) for v in physical_cfg.get("domain_nm", (
                        nx_target * float(cfg.get("voxel_nm3", 1.0)),
                        ny_target * float(cfg.get("voxel_nm3", 1.0)),
                        nz_target * float(cfg.get("voxel_nm3", 1.0)),
                    )))
                    origin = tuple(float(v) for v in physical_cfg.get("origin_fraction", (0.5, 0.5, 0.0)))
                    center_nm = tuple((center[i] - origin[i]) * domain_nm[i] for i in range(3))
                    main_field_bounds_nm = tuple(
                        v for i in range(3)
                        for v in (-origin[i] * domain_nm[i], (1.0 - origin[i]) * domain_nm[i])
                    )
                    if save_full:
                        save_potential_full(phi, npy_name, output_dir=output_dir)
                    if save_cut:
                        cut_name = os.path.splitext(npy_name)[0] + "_cut.npy"
                        save_potential_physical_cut(
                            phi, center_nm, cut_offsets_nm, main_field_bounds_nm,
                            filename=cut_name, output_dir=output_dir
                        )

                    residual = results["residual"]

                    if plotting_enabled:
                        fig = plot_phi_plane(phi, results['boundary_mask'], plane=(True, True, z_frac), tip_mask=results['tip_mask'], apex=(0.5, 0.5, results['parameters']['tip_pos']))
                        ax = fig.axes[0] if fig.axes else None
                        if ax is not None:
                            ax.set_title(ax.get_title() + f"\n{tag}")
                        plt.show()
                        plt.close(fig)

                        fig = plot_phi_plane(phi, results['boundary_mask'], plane=(x_frac, True, True), tip_mask=results['tip_mask'], apex=(0.5, 0.5, results['parameters']['tip_pos']))
                        ax = fig.axes[0] if fig.axes else None
                        if ax is not None:
                            ax.set_title(ax.get_title() + f"\n{tag}")
                        plt.show()
                        plt.close(fig)

                        fig = plot_residual_plane(residual, results["boundary_mask"],
                                                  plane=(True, True, z_frac), tip_mask=results['tip_mask'], apex=(0.5, 0.5, results['parameters']['tip_pos']))
                        ax = fig.axes[0] if fig.axes else None
                        if ax is not None:
                            ax.set_title(ax.get_title() + f"\n{tag}")
                        plt.show()
                        plt.close(fig)

                        fig = plot_residual_plane(residual, results["boundary_mask"],
                                                  plane=(x_frac, True, True), tip_mask=results['tip_mask'], apex=(0.5, 0.5, results['parameters']['tip_pos']))
                        ax = fig.axes[0] if fig.axes else None
                        if ax is not None:
                            ax.set_title(ax.get_title() + f"\n{tag}")
                        plt.show()
                        plt.close(fig)

                    del residual
                    if 'residual' in results:
                        del results['residual']

                    run_zoom_simulation(cfg, results, V, config_idx, time_log, output_dir=output_dir,
                                        movement_active=movement_active, center=center, center0=center0,
                                        plotting_enabled=plotting_enabled)

                    del results
                    gc.collect()
                    plt.close('all')

        print("All JSON files processed.")
        if os.path.isfile(os.fspath(CONFIG_BASE_NAME)):
            print("Single-config run complete.")
            break
        ans = input("Enter a new base name to process another set (or press Enter to finish): ").strip()
        if ans == "":
            print("Finished.")
            break
        CONFIG_BASE_NAME = ans
