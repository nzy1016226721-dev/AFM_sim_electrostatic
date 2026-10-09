import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import time
import os
import csv
import sys
import gc
from scipy.ndimage import zoom

from .solver import build_downward_pointing_tip, mg_3d_masked, MG_TIME
from .materials import (generate_eps_level, build_eps_reference_memmap,
                        release_eps_reference)
from .io_utils import (gate_slices,
                        movement_suffix_nm,
                        resolution_tagged_cut_filename,
                        save_potential_full, save_potential_physical_cut)
from .plotting import plot_phi_plane, plot_residual_plane
from .zoom import run_zoom_simulation
from .runtime import (resolve_output_dir, resolve_plotting_enabled, is_spyder_like_ide,
                       resolve_config_path)
from .mpi_config import load_afm_config
from .numerics import (
    diagnostic_iteration_limit,
    prepare_drive_voltages,
    resolve_residual_tolerance_mode,
    resolve_solver_dtype,
)


LARGE_GRID_MIN_AXIS = 2048
LARGE_GRID_MIN_LEVELS = 6
LARGE_GRID_MAX_LEVELS = 8
LARGE_GRID_MIN_INITIAL_LEVEL = 64


def _hierarchy_level_count(largest_axis, initial_level):
    """Return the number of doubling levels needed along the largest axis."""
    current = min(int(initial_level), int(largest_axis))
    levels = 1
    while current < int(largest_axis):
        current = min(current * 2, int(largest_axis))
        levels += 1
    return levels


def default_initial_grid_level(*shape):
    """Return a dynamic isotropic first level for a target grid.

    Small targets retain the legacy 8^3/64^3 warm-up policy.  For a large
    target (largest axis at least 2048), the first level is selected from
    64, 128, 256, ... so the doubling hierarchy contains six to eight levels.
    Thus 2048^3, 4096^3, and 8192^3 all start at 64^3 (six, seven, and eight
    levels respectively), while 16384^3 starts at 128^3.  An explicit
    ``initial_grid_level`` remains available for diagnostics that deliberately
    need a nonstandard hierarchy.
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, np.ndarray)):
        shape = tuple(shape[0])
    if not shape:
        raise ValueError("target grid shape is required")
    largest = max(int(value) for value in shape)
    if largest < LARGE_GRID_MIN_AXIS:
        if largest > 512:
            return 64
        return 8

    initial = LARGE_GRID_MIN_INITIAL_LEVEL
    while _hierarchy_level_count(largest, initial) > LARGE_GRID_MAX_LEVELS:
        initial *= 2

    # With the 2048 threshold and a minimum 64^3 start, every large target
    # reaches at least six levels. Keep this check adjacent to the policy so a
    # future threshold change cannot silently violate the stated contract.
    if _hierarchy_level_count(largest, initial) < LARGE_GRID_MIN_LEVELS:
        raise RuntimeError("large-grid hierarchy policy selected fewer than six levels")
    return initial


def run_afm_simulation(Vtip=5, nx=32, ny=32, nz=32,
                       tip_z=0.2, R=0.05, r_tip=0.15,
                       damping=0.8, nu1=2, nu2=2,
                       max_iter=None, tol=1e-4, aspect_ratio=2.0,
                       verbose=True, eps_r=None, eps=True,
                       mg_max_runtime=None, blocks=None, Vgate=None,
                       output_dir=".", save_all_levels=False,
                       level_name_prefix=None, plotting_enabled=True, memory_tracking=False,
                       physical_params=None, eps_reference_resolution=512, cpu_threads=1,
                       return_residual=True, initial_grid_level=None,
                       live_memory_interval=None,
                       residual_tolerance_mode="absolute",
                       solver_dtype="float32",
                       tip_shape="pyramid",
                       diagnostic=None, memory_mode="standard",
                       phi_update_mode=None, residual_accumulation=None):
    """Run a multiresolution AFM electrostatic simulation.

    Starts from an 8x8x8 grid for targets at or below 512 cells per axis and
    64x64x64 through 2048 cells per axis. For targets at least 2048 cells on
    their largest axis, the start is dynamically chosen to make the doubling
    hierarchy contain six to eight levels. Each refinement level doubles
    toward the requested target and upscales the solution using
    scipy.ndimage.zoom.

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
        Tip truncation radius (fractional, default: 0.15).  For the default
        ``"pyramid"`` tip shape this is the distance from the tip axis to one
        side of the square base edge.
    tip_shape : str, optional
        ``"pyramid"`` (default) or ``"cone"`` tip cross-section.
    damping : float, optional
        SOR damping factor (default: 0.8).
    nu1, nu2 : int, optional
        Pre/post-smoothing steps (unused, default: 2).
    max_iter : int or None, optional
        Legacy compatibility argument. It does not terminate a solve; use
        ``mg_max_runtime`` to set the non-convergence wall-clock deadline.
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
    cpu_threads : int, optional
        Number of CPU threads for the 3-D numerical solver. Set to 1 to use
        the legacy NumPy reference path; values >1 use the Numba parallel path.
    return_residual : bool, optional
        If True (default), retain and return the full float32 residual field.
        Headless batch callers can set this to False when the diagnostic field
        will not be plotted, avoiding another full-grid allocation.
    initial_grid_level : int or None, optional
        First isotropic level in the hierarchy. None uses the normal
        8^3/64^3/dynamic-large-grid startup policy. A high-level diagnostic
        can start directly at 4096^3 and advance only to its larger target.
    live_memory_interval : float or None, optional
        When memory tracking is enabled, append process RSS samples at this
        interval to memory_live_rss.csv. The file is flushed per sample so it
        remains useful if a batch child is stopped before a level returns.
    residual_tolerance_mode : {"absolute", "relative_drive"}, optional
        ``relative_drive`` solves with every Dirichlet voltage divided by the
        largest absolute drive voltage, then scales the final field back to
        physical volts. The configured tolerance is consequently relative to
        the applied drive and remains attainable in float32 at large bias.
    solver_dtype : {"float32", "float64"}, optional
        Numerical dtype for potential and dielectric face fields. Float32 is
        the production default. Float64 is opt-in because it roughly doubles
        the solver's dominant memory footprint.
    diagnostic : dict or None, optional
        Fixed iteration counts are accepted only for the explicit
        ``float64_memory_smoke`` diagnostic mode.

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
    field_dtype = resolve_solver_dtype(solver_dtype)
    tolerance_mode = resolve_residual_tolerance_mode(residual_tolerance_mode)
    solver_vtip, solver_vgate, drive_scale = prepare_drive_voltages(
        Vtip, Vgate, tolerance_mode
    )
    if tolerance_mode == "relative_drive":
        print(
            "Voltage-normalized residual mode: "
            f"drive scale={drive_scale:.12g} V; solver tip={solver_vtip:.12g}; "
            "saved potential is rescaled to physical volts."
        )
    print(f"Solver dtype: {field_dtype.name}")
    # Very large production grids do not need the legacy tiny warm-up levels.
    # A diagnostic may explicitly begin at a later power-of-two level, for
    # example 4096^3 -> 8192^3.
    if initial_grid_level is None:
        initial_level = default_initial_grid_level(nx_target, ny_target, nz_target)
    else:
        initial_level = int(initial_grid_level)
        if initial_level < 2:
            raise ValueError("initial_grid_level must be at least 2")
    nx = min(initial_level, nx_target)
    ny = min(initial_level, ny_target)
    nz = min(initial_level, nz_target)

    live_memory_tracker = None
    if memory_tracking and live_memory_interval is not None:
        from .memory import MemoryTracker
        live_memory_tracker = MemoryTracker(
            interval=float(live_memory_interval),
            live_log_path=os.path.join(output_dir, "memory_live_rss.csv"),
            stage=f"initializing main {nx}x{ny}x{nz}",
        ).start()
    from .ram_first import resolve_memory_mode, resolve_storage_options
    memory_mode = resolve_memory_mode(memory_mode)
    resolve_storage_options(memory_mode, phi_update_mode, residual_accumulation)
    if memory_mode in ("ram_first", "ram_compact") and (field_dtype != np.float32 or plotting_enabled or return_residual):
        raise ValueError("RAM modes require float32 and headless solving without residual volumes")
    tip_builder = build_downward_pointing_tip
    reference_builder = build_eps_reference_memmap
    if memory_mode == "ram_compact":
        from .compact_storage import (PackedMask, build_tip_predicate,
                                      build_reference_planes, generate_eps_compact, PlaneEpsilon)
        tip_builder, reference_builder = build_tip_predicate, build_reference_planes
    phi = np.full((nx, ny, nz), 0.001, dtype=field_dtype)
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
        eps_reference_path, eps_reference_mmap = reference_builder(
            ref_shape, blocks=blocks,
            **({"directory":output_dir} if memory_mode == "ram_compact" else {})
        )
        if memory_mode == "ram_first":
            eps_reference_mmap._mmap.close()
            eps_reference_mmap = np.load(eps_reference_path, mmap_mode="r", allow_pickle=False)

    try:
        while True:
            if verbose:
                print(f"\n[Level {level}] Solving on {nx}x{ny}x{nz} grid...")
            if live_memory_tracker is not None:
                live_memory_tracker.set_stage(f"preparing main {nx}x{ny}x{nz}")

            physical = physical_params
            if physical:
                domain_nm = physical["domain_nm"]
                origin = physical["origin_fraction"]
                tip_mask, tip_pos, base_pos = tip_builder(
                    nx, ny, nz, tip_z, R, r_tip, aspect_ratio, verbose=False,
                    tip_z_nm=physical.get("tip_z_nm"),
                    R_nm=physical.get("R_nm"),
                    r_tip_nm=physical.get("r_tip_nm"),
                    domain_nm=domain_nm,
                    center_fraction=origin,
                    tip_shape=tip_shape,
                )
            else:
                tip_mask, tip_pos, base_pos = tip_builder(
                    nx, ny, nz, tip_z, R, r_tip, aspect_ratio, verbose=False,
                    tip_shape=tip_shape,
                )

            boundary_mask = PackedMask((nx,ny,nz)) if memory_mode == "ram_compact" else np.zeros((nx, ny, nz), dtype=bool)

            if solver_vgate is not None and isinstance(solver_vgate, (list, tuple)) and len(solver_vgate) > 0:
                for g in solver_vgate:
                    gate_region = gate_slices(nx, ny, nz, g)
                    val_g  = float(g.get("Vgate_val", 0.0))
                    # Apply rectangular gates directly.  Keeping one full
                    # boolean mask per gate would cost 1 GiB each at 1024^3
                    # despite those masks never being used after this point.
                    phi[gate_region] = val_g
                    if memory_mode == "ram_compact":
                        boundary_mask.mark_box(gate_region)
                    else:
                        boundary_mask[gate_region] = True
            else:
                if isinstance(solver_vgate, dict):
                    gate_region = gate_slices(nx, ny, nz, solver_vgate)
                    val_g  = float(solver_vgate.get("Vgate_val", 0.0))
                    phi[gate_region] = val_g
                    if memory_mode == "ram_compact":
                        boundary_mask.mark_box(gate_region)
                    else:
                        boundary_mask[gate_region] = True

            if memory_mode == "ram_compact":
                tip_mask.apply(phi,boundary_mask,solver_vtip)
            else:
                phi[tip_mask] = solver_vtip
                boundary_mask[tip_mask] = True

            if eps_r is None:
                eps_reference = eps_reference_resolution
                eps_reference_shape = (int(eps_reference), int(eps_reference), int(eps_reference)) if not isinstance(eps_reference, (list, tuple)) else tuple(int(v) for v in eps_reference)
                level_generator = generate_eps_level
                if memory_mode == "ram_first":
                    from .materials_bounded import generate_eps_level_bounded
                    level_generator = generate_eps_level_bounded
                elif memory_mode == "ram_compact":
                    level_generator = generate_eps_compact
                eps_cell = level_generator(
                    phi.shape, blocks, reference_shape=eps_reference_shape,
                    reference=eps_reference_mmap,
                    **({"directory":output_dir} if memory_mode == "ram_compact" else {})
                )
            else:
                eps_cell = (eps_r if isinstance(eps_r,PlaneEpsilon) else PlaneEpsilon.from_dense(np.asarray(eps_r,dtype=field_dtype),directory=output_dir)) if memory_mode == "ram_compact" else np.asarray(eps_r, dtype=field_dtype)
            if eps_cell.dtype != field_dtype:
                eps_cell = np.asarray(eps_cell, dtype=field_dtype)

            diagnostic_iterations = diagnostic_iteration_limit(
                diagnostic,
                (nx, ny, nz),
                solver_dtype=field_dtype,
            )

            if memory_tracking:
                from .memory import track_memory, log_memory_usage
                memory_context = track_memory()
            else:
                from contextlib import nullcontext
                memory_context = nullcontext(None)

            with memory_context as mem_tracker:
                if live_memory_tracker is not None:
                    live_memory_tracker.set_stage(f"solving main {nx}x{ny}x{nz}")
                phi_solution, res_m = mg_3d_masked(solver_vtip, phi, boundary_mask,
                                            damping=damping, nu1=nu1, nu2=nu2,
                                            max_iter=max_iter, tol=tol,
                                            verbose=verbose, eps_r=eps_cell, eps=eps,
                                            mg_max_runtime=mg_max_runtime,
                                            output_dir=output_dir,
                                            plotting_enabled=plotting_enabled,
                                            cpu_threads=cpu_threads,
                                            return_residual=return_residual,
                                            diagnostic_iterations=diagnostic_iterations,
                                            memory_mode=memory_mode,
                                            phi_update_mode=phi_update_mode,
                                            residual_accumulation=residual_accumulation)

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
                if drive_scale == 1.0:
                    np.save(level_path, phi_solution)
                else:
                    np.save(
                        level_path,
                        np.multiply(phi_solution, field_dtype.type(drive_scale)),
                    )
                print(f"  Saved level: {os.path.basename(level_path)}")

            if diagnostic_iterations is not None:
                diagnostic_log = os.path.join(output_dir, "float64_memory_smoke_levels.csv")
                diagnostic_exists = os.path.isfile(diagnostic_log)
                with open(diagnostic_log, "a", newline="") as handle:
                    writer = csv.writer(handle)
                    if not diagnostic_exists:
                        writer.writerow([
                            "level", "Nx", "Ny", "Nz", "dtype",
                            "requested_iterations", "completed_iterations",
                            "time_sec", "residual", "result",
                        ])
                    writer.writerow([
                        level, nx, ny, nz, field_dtype.name,
                        diagnostic_iterations,
                        MG_TIME.get("iterations", ""),
                        f"{MG_TIME.get('elapsed', 0.0):.6f}",
                        f"{MG_TIME.get('residual', float('nan')):.12e}",
                        MG_TIME.get("reason", "unknown"),
                    ])

            if is_final:
                if verbose:
                    print(f"[Level {level}] Target grid size reached ({nx}x{ny}x{nz}). Simulation complete.")
                # Keep only the final potential and residual for the returned result.
                if memory_mode == "ram_compact" and eps_cell is not eps_r:
                    eps_cell.close()
                del eps_cell
                if "tip_mask" in locals():
                    # tip_mask is returned, so do not delete it.
                    pass
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
            if live_memory_tracker is not None:
                live_memory_tracker.set_stage(
                    f"upscaling main {old_nx}x{old_ny}x{old_nz} to {nx}x{ny}x{nz}"
                )
            if memory_mode in ("ram_first", "ram_compact"):
                # These are not used by interpolation; free them BEFORE the
                # larger potential is allocated, not after the peak.
                if memory_mode == "ram_compact" and eps_cell is not eps_r:
                    eps_cell.close()
                del eps_cell, boundary_mask, tip_mask
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
            if memory_mode not in ("ram_first", "ram_compact"):
                del eps_cell
                del boundary_mask
            if "tip_mask" in locals():
                del tip_mask
            gc.collect()

            if verbose:
                print(f"[Level {level}] Converged. Upscaling to {nx}x{ny}x{nz} using factors {scale}...")

            level += 1
    finally:
        if memory_mode == "ram_compact" and "eps_cell" in locals() and isinstance(eps_cell,PlaneEpsilon) and eps_cell is not eps_r:
            eps_cell.close()
        if live_memory_tracker is not None:
            live_memory_tracker.set_stage("finalizing")
            live_memory_tracker.stop()
        # The reference is a per-simulation resource.  Always release it,
        # including when a solver, logger, or output operation raises.
        if eps_reference_path is not None:
            release_eps_reference(eps_reference_path, eps_reference_mmap)
            eps_reference_path = None
            eps_reference_mmap = None
            gc.collect()

    if drive_scale != 1.0:
        np.multiply(phi_solution, field_dtype.type(drive_scale), out=phi_solution)
        if res_m is not None:
            np.multiply(res_m, field_dtype.type(abs(drive_scale)), out=res_m)

    results = {
        'phi': phi_solution,
        'residual': res_m,
        'tip_mask': tip_mask, 'boundary_mask': boundary_mask,
        'parameters': {'nx': nx, 'ny': ny, 'nz': nz,
                       'tip_pos': tip_pos, 'base_pos': base_pos,
                       'levels': level,
                       'solver_dtype': field_dtype.name,
                       'residual_tolerance_mode': tolerance_mode,
                       'drive_scale_volts': drive_scale}
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
    # A stationary alignment run is one position, not two identical samples.
    # Returning a single centre also prevents duplicate output files when a
    # caller intentionally sets movement start and end to the same point.
    if max_dist <= 1e-12:
        return [tuple(start)]

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
            tip_shape=cfg.get("tip_shape", "pyramid"),
        )
    else:
        tip_mask, _, _ = build_downward_pointing_tip(
            nx, ny, nz, tip_z=tip_z, R=R, r_tip=r_tip,
            aspect_ratio=aspect, verbose=False,
            tip_shape=cfg.get("tip_shape", "pyramid"),
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


def batch_main(config_path=None, config_dir=".", plotting_override=None,
               interactive=False, output_dir_override=None):
    """Main loop: iterate over config files, run simulation + zoom for each.

    Handles movement (block/gate repositioning), voltage sweeps, saving
    results, plotting, and cleanup.

    Parameters
    ----------
    config_path : str or None, optional
        Explicit AFM JSON path. If omitted, the newest AFM JSON in
        ``config_dir`` is selected. No base-name/suffix matching is performed.
    config_dir : str, optional
        Directory used to resolve a relative config name.
    plotting_override : bool or None, optional
        Explicitly enable/disable plots.
    interactive : bool, optional
        Kept for API compatibility; the caller is responsible for choosing the
        configuration before invoking this function.
    output_dir_override : str or None, optional
        Optional output root selected by the launcher. It is preferred over
        the JSON output directory and is made config-specific.

    Returns
    -------
    None
    """
    config_path = resolve_config_path(config_path, directory=config_dir)
    config_files = [config_path]

    _, first_cfg = load_afm_config(config_files[0])

    first_plotting_enabled = resolve_plotting_enabled(
        first_cfg, cli_override=plotting_override
    )

    print(f"\nFound {len(config_files)} config files:")
    for path in config_files:
        print(f"  {os.path.basename(path)}")

    preview_tip_only(first_cfg, plotting_enabled=first_plotting_enabled)
    preview_before_run(first_cfg, plotting_enabled=first_plotting_enabled)

    print("\nStarting selected configuration without additional confirmation.")

    for config_idx, config_path in enumerate(config_files, start=1):
        print(f"\n{'='*60}")
        print(f"Processing configuration {config_idx}/{len(config_files)}: "
              f"{os.path.basename(config_path)}")
        print(f"{'='*60}")

        _, cfg = load_afm_config(config_path)
        if cfg.get("memory_mode") in ("ram_first", "ram_compact") and cfg.get("zoom_simulation",{}).get("enabled",False):
            raise ValueError("RAM-saving modes support the main grid only; legacy zoom requires standard mode")

        plotting_enabled = resolve_plotting_enabled(
            cfg, cli_override=plotting_override
        )
        if output_dir_override:
            override = os.path.abspath(os.path.expanduser(output_dir_override))
            output_dir = os.path.join(override, os.path.splitext(os.path.basename(config_path))[0])
        else:
            output_dir = resolve_output_dir(
                cfg.get("output_dir", "."),
                config_path,
                cfg,
            )
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        mg_max_runtime = cfg.get("mg_max_runtime", None)
        cpu_threads = int(cfg.get("cpu_threads", 1))
        if cpu_threads < 1:
            raise ValueError("cpu_threads must be >= 1 in the JSON configuration")
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
        movement_named = isinstance(cfg.get("movement"), dict)

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

                physical_cfg = cfg.get("_physical", {})
                domain_nm = tuple(float(v) for v in physical_cfg.get("domain_nm", (
                    nx_target * float(cfg.get("voxel_nm3", 1.0)),
                    ny_target * float(cfg.get("voxel_nm3", 1.0)),
                    nz_target * float(cfg.get("voxel_nm3", 1.0)),
                )))
                movement_suffix = movement_suffix_nm(center, center0, domain_nm)
                npy_name = f"afm_phi_{config_idx}_{V:.2f}V.npy"
                if movement_named:
                    npy_name = (
                        f"afm_phi_{config_idx}{movement_suffix}"
                        f"_{V:.2f}V.npy"
                    )

                results = run_afm_simulation(
                    Vtip=V,
                    nx=nx_target, ny=ny_target, nz=nz_target,
                    tip_z=cfg["tip_z"],
                    R=cfg["R"],
                    r_tip=cfg["r_tip"],
                    damping=1,
                    nu1=2, nu2=2,
                    # The solver is tolerance/time limited. Preserve a legacy
                    # JSON value only as an ignored compatibility argument.
                    max_iter=cfg.get("max_iter"),
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
                    eps_reference_resolution=cfg.get("epsilon_material", {}).get("reference_resolution", 512),
                    cpu_threads=cpu_threads,
                    return_residual=plotting_enabled,
                    initial_grid_level=cfg.get("initial_grid_level"),
                    live_memory_interval=cfg.get("memory_live_interval_seconds"),
                    residual_tolerance_mode=cfg.get(
                        "residual_tolerance_mode", "absolute"
                    ),
                    solver_dtype=cfg.get("solver_dtype", "float32"),
                    tip_shape=cfg.get("tip_shape", "pyramid"),
                    diagnostic=cfg.get("diagnostic"),
                    memory_mode=cfg.get("memory_mode", "standard"),
                    phi_update_mode=cfg.get("phi_update_mode"),
                    residual_accumulation=cfg.get("residual_accumulation"),
                )

                phi = results["phi"]
                nx, ny, nz = phi.shape
                ix = int(x_frac * (nx - 1))
                iy = int(y_frac * (ny - 1))
                iz = int(z_frac * (nz - 1))
                phi_val = phi[ix, iy, iz]
                phi_results.append((cx, cy, cz, V, phi_val))

                print(f"  phi({x_frac:.2f}, {y_frac:.2f}, {z_frac:.2f}) = {phi_val:.6f} V")
                tag = (
                    f"move{movement_suffix}_V{V:.2f}"
                    if movement_named else f"cx{cx:.2f}_cy{cy:.2f}_cz{cz:.2f}_V{V:.2f}"
                )

                elapsed = time.time() - start_time
                print(f"Runtime: {elapsed:.2f} s")
                time_log.append(elapsed)

                save_full = bool(cfg.get("save_full", False))
                save_cut = bool(cfg.get("save_cut", False))
                # Missing/empty physical cut ranges intentionally save the
                # full simulated field. Non-empty ranges are signed physical
                # offsets from the current movement center.
                cut_offsets_nm = cfg.get("save_cut_box_nm")
                origin = tuple(float(v) for v in physical_cfg.get("origin_fraction", (0.5, 0.5, 0.0)))
                center_nm = tuple((center[i] - origin[i]) * domain_nm[i] for i in range(3))
                main_field_bounds_nm = tuple(
                    v for i in range(3)
                    for v in (-origin[i] * domain_nm[i], (1.0 - origin[i]) * domain_nm[i])
                )
                if save_full:
                    save_potential_full(phi, npy_name, output_dir=output_dir)
                if save_cut:
                    cut_name = resolution_tagged_cut_filename(npy_name, phi.shape)
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

    print("Configuration processed successfully.")
