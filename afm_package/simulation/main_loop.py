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
from .materials import generate_eps_cell
from .io_utils import make_gate_mask, get_sorted_config_files, save_for_qtcad
from .plotting import plot_phi_plane, plot_residual_plane
from .zoom import run_zoom_simulation


def run_afm_simulation(Vtip=5, nx=32, ny=32, nz=32,
                       tip_z=0.2, R=0.05, r_tip=0.15,
                       damping=0.8, nu1=2, nu2=2,
                       max_iter=1000, tol=1e-4, aspect_ratio=2.0,
                       verbose=True, eps_r=None, eps=True,
                       mg_max_runtime=None, blocks=None, Vgate=None,
                       output_dir="."):
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
        Results containing 'phi', 'Ex', 'Ey', 'Ez', 'residual',
        'tip_mask', 'boundary_mask', 'tip_potential', 'tip_field',
        'surface_potential', 'surface_field', and 'parameters'.
    """
    if verbose:
        print("Starting multiresolution AFM simulation...")

    nx_target, ny_target, nz_target = nx, ny, nz
    nx, ny, nz = 8, 8, 8
    phi = np.full((nx, ny, nz), 0.001, dtype=np.float32)
    total_iter = 0
    level = 1

    while True:
        if verbose:
            print(f"\n[Level {level}] Solving on {nx}x{ny}x{nz} grid...")

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
            eps_cell = generate_eps_cell(phi, blocks)
        else:
            eps_cell = eps_r.copy()

        remaining_iter = max_iter - total_iter
        if remaining_iter <= 0:
            if verbose:
                print(f"[Level {level}] Reached max_iter={max_iter}, stopping.")
            break

        if verbose:
            print(f"[Level {level}] Running solver (remaining_iter={remaining_iter})...")

        phi_solution, res_m = mg_3d_masked(Vtip, phi.copy(), boundary_mask,
                                    damping=damping, nu1=nu1, nu2=nu2,
                                    max_iter=remaining_iter, tol=tol,
                                    verbose=verbose, eps_r=eps_cell, eps=eps,
                                    mg_max_runtime=mg_max_runtime,
                                    output_dir=output_dir)

        total_iter += remaining_iter // 2

        logfile = os.path.join(output_dir, "mg_timing_log.csv")
        os.makedirs(output_dir, exist_ok=True)
        file_exists = os.path.isfile(logfile)
        with open(logfile, "a", newline="") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["level", "Nx", "Ny", "Nz", "time_sec"])
            level_elapsed = MG_TIME.get("elapsed", 0.0)
            w.writerow([level, nx, ny, nz, f"{level_elapsed:.6f}"])

        if nx == nx_target and ny == ny_target and nz == nz_target:
            if verbose:
                print(f"[Level {level}] Target grid size reached ({nx}x{ny}x{nz}). Simulation complete.")
            break

        if total_iter >= max_iter:
            if verbose:
                print(f"[Level {level}] Max iterations {max_iter} reached before reaching target size.")
            break

        nx *= 2
        ny *= 2
        nz *= 2

        phi = zoom(phi_solution, 2.0, order=1)
        del phi_solution

        if verbose:
            print(f"[Level {level}] Converged. Upscaling to {nx}x{ny}x{nz} using zoom()...")

        level += 1

        if nx > nx_target: nx = nx_target
        if ny > ny_target: ny = ny_target
        if nz > nz_target: nz = nz_target

    Ex, Ey, Ez = np.gradient(-phi_solution)
    tip_potential = phi_solution[tip_mask]
    tip_field = np.sqrt(Ex[tip_mask]**2 + Ey[tip_mask]**2 + Ez[tip_mask]**2)
    surface_potential = phi_solution[:, :, 1]
    surface_field = np.sqrt(Ex[:, :, 1]**2 + Ey[:, :, 1]**2 + Ez[:, :, 1]**2)

    if verbose:
        print("Simulation results:")
        print(f" tip potential mean/std: {np.mean(tip_potential):.4f} +/- {np.std(tip_potential):.4f}")
        print(f" tip field mean/std:     {np.mean(tip_field):.4f} +/- {np.std(tip_field):.4f}")

    results = {
        'phi': phi_solution, 'Ex': Ex, 'Ey': Ey, 'Ez': Ez,
        'residual': res_m,
        'tip_mask': tip_mask, 'boundary_mask': boundary_mask,
        'tip_potential': tip_potential, 'tip_field': tip_field,
        'surface_potential': surface_potential, 'surface_field': surface_field,
        'parameters': {'nx': nx, 'ny': ny, 'nz': nz,
                       'tip_pos': tip_pos, 'base_pos': base_pos,
                       'total_iter': total_iter, 'levels': level}
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


def compute_block_positions(start_center, end_center, spacing):
    """Linearly interpolate positions between two centres at a given spacing.

    Parameters
    ----------
    start_center : tuple of float
        (x, y, z) starting centre.
    end_center : tuple of float
        (x, y, z) ending centre.
    spacing : float
        Step size in fractional units.

    Returns
    -------
    list of tuple
        Interpolated centre positions.
    """
    start = np.array(start_center, float)
    end   = np.array(end_center, float)

    low  = np.minimum(start, end)
    high = np.maximum(start, end)
    dist = high - low

    max_dist = np.linalg.norm(dist)
    nsteps = int(max_dist / spacing) + 1

    tvals = np.linspace(0, 1, nsteps)
    return [tuple(low + t * dist) for t in tvals]


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
    lo = edge_value
    hi = edge_value

    if lo_fix is not None:
        lo = max(lo, lo_fix)
    if hi_fix is not None:
        hi = min(hi, hi_fix)

    edge = max(0.0, min(1.0, lo))
    edge = max(0.0, min(1.0, hi))
    return edge


def preview_before_run(cfg):
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
    blocks = cfg.get("blocks", [])
    vgates = cfg.get("Vgate", [])
    movement = cfg.get("movement", {})
    cx, cy, cz = movement.get("start", [0.5, 0.5, 0.5])

    nx = ny = nz = 200
    block_mask = np.zeros((nx, ny, nz), dtype=float)
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


def preview_tip_only(cfg):
    """Display a 3-projection scatter plot of the AFM tip voxels.

    Parameters
    ----------
    cfg : dict
        Config dict with 'tip_z', 'R', 'r_tip', 'aspect_ratio'.

    Returns
    -------
    None
    """
    tip_z       = cfg.get("tip_z", 0.7)
    R           = cfg.get("R", 0.5)
    r_tip       = cfg.get("r_tip", 0.15)
    aspect      = cfg.get("aspect_ratio", 1.0)

    nx = ny = nz = 120

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


def batch_main(CONFIG_BASE_NAME="afm_config"):
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
        config_files = get_sorted_config_files(CONFIG_BASE_NAME)
        if not config_files:
            print(f"\nNo JSON files found matching '{CONFIG_BASE_NAME}_*.json'.")
            ans = input("Enter a new base name (e.g. 'myconfig') or press Enter to exit: ").strip()
            if ans == "":
                print("Exiting.")
                break
            CONFIG_BASE_NAME = ans
            continue

        with open(config_files[0], 'r') as f:
            first_cfg = json.load(f)

        print(f"\nFound {len(config_files)} config files:")
        for path in config_files:
            print(f"  {os.path.basename(path)}")

        preview_tip_only(first_cfg)
        preview_before_run(first_cfg)

        ans = input(f"\nRun simulation for all {len(config_files)} configurations? (y/n): ").strip().lower()
        if ans != 'y':
            ans = input("Enter a new base name (or press Enter to exit): ").strip()
            if ans == "":
                print("Exiting.")
                break
            CONFIG_BASE_NAME = ans
            continue

        for config_idx, config_path in enumerate(config_files, start=1):
            print(f"\n{'='*60}")
            print(f"Processing configuration {config_idx}/{len(config_files)}: "
                  f"{os.path.basename(config_path)}")
            print(f"{'='*60}")

            with open(config_path, 'r') as f:
                cfg = json.load(f)

            output_dir = cfg.get("output_dir", ".")
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            sigma_blocks_file = "sigma_blocks.json"
            if os.path.isfile(sigma_blocks_file):
                with open(sigma_blocks_file, 'r') as sf:
                    sigma_data = json.load(sf)
                cfg["sigma_blocks"] = sigma_data.get("sigma_blocks", [])
                print(f"  Loaded {len(cfg['sigma_blocks'])} sigma blocks from {sigma_blocks_file}")
            else:
                cfg["sigma_blocks"] = []
                print(f"  {sigma_blocks_file} not found - zoom Joule heating disabled.")

            mg_max_runtime = cfg.get("mg_max_runtime", None)
            blocks = cfg["blocks"]

            mov = cfg.get("movement", {})
            start_center = tuple(mov.get("start", [0.5, 0.5, 0.5]))
            end_center   = tuple(mov.get("end",   [0.5, 0.5, 0.5]))
            spacing      = mov.get("spacing", 0.1)
            centers = compute_block_positions(start_center, end_center, spacing)

            V_values = np.arange(cfg["v_start"], cfg["v_stop"] + 1e-12, cfg["v_step"])

            phi_results = []
            time_log = []

            stationary_upto = cfg.get("fixed_blocks", [2])[0]
            mobile_block_indices = list(range(stationary_upto, len(cfg["blocks"])))
            mobile_vgate_indices = list(range(len(cfg.get("Vgate", []))))
            center0 = centers[0]

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
                    print(f"📐 Using grid resolution: {nx_target}×{ny_target}×{nz_target}")


                    results = run_afm_simulation(
                        Vtip=V,
                        nx=nx_target, ny=ny_target, nz=nz_target,
                        tip_z=cfg["tip_z"],
                        R=cfg["R"],
                        r_tip=cfg["r_tip"],
                        damping=1,
                        nu1=2, nu2=2,
                        max_iter=100000,
                        tol=cfg.get("res_tol_main", 5e-5),
                        aspect_ratio=cfg["aspect_ratio"],
                        verbose=False,
                        eps=True,
                        mg_max_runtime=mg_max_runtime,
                        blocks=cfg["blocks"],
                        Vgate=cfg.get("Vgate", []),
                        output_dir=output_dir
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

                    npy_name = f"afm_phi_{config_idx}_{V:.2f}V.npy"
                    save_for_qtcad(results, npy_name, output_dir=output_dir)

                    residual = results["residual"]

                    fig = plot_phi_plane(phi, results['boundary_mask'], plane=(True, True, z_frac))
                    ax = fig.axes[0] if fig.axes else None
                    if ax is not None:
                        ax.set_title(ax.get_title() + f"\n{tag}")
                    plt.show()
                    plt.close(fig)

                    fig = plot_phi_plane(phi, results['boundary_mask'], plane=(x_frac, True, True))
                    ax = fig.axes[0] if fig.axes else None
                    if ax is not None:
                        ax.set_title(ax.get_title() + f"\n{tag}")
                    plt.show()
                    plt.close(fig)

                    fig = plot_residual_plane(residual, results["boundary_mask"],
                                              plane=(True, True, z_frac))
                    ax = fig.axes[0] if fig.axes else None
                    if ax is not None:
                        ax.set_title(ax.get_title() + f"\n{tag}")
                    plt.show()
                    plt.close(fig)

                    fig = plot_residual_plane(residual, results["boundary_mask"],
                                              plane=(x_frac, True, True))
                    ax = fig.axes[0] if fig.axes else None
                    if ax is not None:
                        ax.set_title(ax.get_title() + f"\n{tag}")
                    plt.show()
                    plt.close(fig)

                    run_zoom_simulation(cfg, results, V, config_idx, time_log, output_dir=output_dir)

                    del results
                    gc.collect()
                    plt.close('all')

        print("All JSON files processed.")
        ans = input("Enter a new base name to process another set (or press Enter to finish): ").strip()
        if ans == "":
            print("Finished.")
            break
        CONFIG_BASE_NAME = ans
