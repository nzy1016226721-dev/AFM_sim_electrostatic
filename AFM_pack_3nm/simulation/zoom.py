import numpy as np
import matplotlib.pyplot as plt
import time
import os
import gc
from scipy.ndimage import zoom

from .solver import mg_3d_masked, compute_residual_vec_unpadded, build_downward_pointing_tip
from .materials import generate_eps_level
from .io_utils import make_gate_mask, save_potential_full, save_potential_physical_cut
from .plotting import plot_phi_plane, plot_residual_plane
from .runtime import resolve_plotting_enabled


def _is_pow2(x):
    """True if x is an integer power of two (>= 4)."""
    return x >= 4 and (x & (x - 1)) == 0


def cut_indices(n, lo, hi):
    """Deterministic crop indices for a fractional cut range on an axis.

    When the cut maps onto exact integer grid units (i.e. both n*lo and n*hi
    are integers) and the resulting node count is a power of two, return the
    exact (lo, hi) based indices so the crop has precisely 2^k nodes.  This
    makes the crop identical for the same (n, lo, hi) on every voltage, zoom
    level and movement position.  Otherwise fall back to (n-1) truncation.

    Parameters
    ----------
    n : int
        Number of grid points along the axis.
    lo, hi : float
        Fractional cut bounds (automatically clamped and sorted).

    Returns
    -------
    (int, int)
        Start and end indices such that the slice [i0, i1] has the cut width.
    """
    lo = max(0.0, min(1.0, float(lo)))
    hi = max(0.0, min(1.0, float(hi)))
    if hi < lo:
        lo, hi = hi, lo
    a = n * lo
    b = n * hi
    ra, rb = round(a), round(b)
    s = int(rb - ra)
    if abs(a - ra) < 1e-9 and abs(b - rb) < 1e-9 and _is_pow2(s):
        return int(ra), int(rb) - 1
    i0 = int(lo * (n - 1))
    i1 = int(hi * (n - 1))
    return max(0, min(n - 1, i0)), max(0, min(n - 1, i1))


def shape_factor(s, n, zoom_factor):
    """Return a zoom factor for a crop while preserving base-2-compatible sizes.

    This helper is retained for compatibility with callers that use the older
    crop-relative scaling logic. The active zoom pipeline uses
    :func:`_exact_zoom_factor` so that each level reaches the requested
    magnification relative to the original main-grid spacing.
    """
    if s <= 1:
        raise ValueError("Zoom crop must contain at least two grid nodes")
    return float(zoom_factor)


def _target_axis_nodes(orig_n, global_lo, global_hi, magnification):
    """Return the exact node count required for a requested physical magnification.

    ``orig_n`` is the main-grid voxel/sample count. ``global_lo``/``global_hi``
    are the current physical-domain bounds expressed as fractions of the main
    domain. The target voxel width is the main-grid ``voxel_nm3`` divided by
    ``magnification``. This keeps 2x/4x/etc. levels compatible with the current
    fixed-array zoom pipeline even when the main grid is changed to 1024^3 or
    2048^3.
    """
    width = max(0.0, float(global_hi) - float(global_lo))
    if width <= 0.0:
        raise ValueError("Zoom physical-domain width must be positive")
    # ``orig_n`` is the number of main-grid voxels/samples represented by the
    # axis.  With voxel_nm3 as the physical scale, a magnification of M means
    # M times as many samples per physical distance.  Thus a cut occupying
    # ``width`` of the main domain contains ``width * orig_n * M`` samples.
    target = int(round(width * int(orig_n) * float(magnification)))
    return max(2, target)


def _exact_zoom_factor(crop_nodes, target_nodes):
    """Return a scipy zoom factor that produces ``target_nodes`` output nodes."""
    if crop_nodes < 2 or target_nodes < 2:
        raise ValueError("Zoom requires at least two nodes on every axis")
    return float(target_nodes) / float(crop_nodes)


def run_zoom_simulation(cfg, results, V, config_idx, time_log=None, output_dir=".",
                        movement_active=False, center=None, center0=None,
                        plotting_enabled=None):
    """Run a recursive central-crop zoom simulation on top of the main result.

    Crops the potential around the region of interest, upsamples, re-solves
    Poisson's equation, and saves the zoomed potential.

    Parameters
    ----------
    cfg : dict
        Configuration dict with 'zoom_simulation' sub-dict. The zoom configuration
        accepts ``clamp=True`` (default), which keeps the historical fixed
        Dirichlet values on the six outer faces of every zoomed domain, or
        ``clamp=False``, which uses natural homogeneous-Neumann outer
        boundaries. In either mode the potential is inherited from the prior
        level by linear interpolation and configured voltage masks are
        re-applied as fixed-voltage masks inside the new cut.
    results : dict
        Main simulation results (used for initial potential).
    V : float
        Tip voltage (V).
    config_idx : int
        Configuration index (for filenames).
    time_log : list or None, optional
        Timing log list (appended to).
    output_dir : str, optional
        Output directory.

    Returns
    -------
    dict
        The original results dict (unchanged).
    """

    if plotting_enabled is None:
        plotting_enabled = resolve_plotting_enabled(cfg)

    zoom_cfg = cfg.get("zoom_simulation", None)
    if zoom_cfg is None or not zoom_cfg.get("enabled", False):
        return results

    if time_log is None:
        time_log = []

    zoom_start_time = time.time()

    cascade = bool(cfg.get("movement", {}).get("cascade", False))

    print("\n==============================")
    mode_txt = "cascade cutting" if cascade else "recursive central crop"
    print(f"Starting layered AFM zoom simulation ({mode_txt})")
    print("==============================")

    zoom_factor   = zoom_cfg["zoom_factor"]
    zoom_limit    = zoom_cfg["zoom_limit"]
    # clamp=True preserves the historical Dirichlet outer-boundary behaviour.
    # clamp=False uses natural (homogeneous Neumann) outer boundaries while
    # retaining all configured voltage masks as fixed Dirichlet constraints.
    clamp = bool(zoom_cfg.get("clamp", True))
    print(f"Zoom boundary mode: {'fixed/Dirichlet outer boundary' if clamp else 'natural/Neumann outer boundary'}")

    Lx_nm = cfg.get("Lx_nm", 512.0)
    Ly_nm = cfg.get("Ly_nm", 512.0)
    Lz_nm = cfg.get("Lz_nm", 512.0)

    phi_full = results['phi'].astype(np.float32, copy=False)
    orig_shape = phi_full.shape

    x_min, x_max = zoom_cfg["cut"]["x_range"]
    y_min, y_max = zoom_cfg["cut"]["y_range"]
    z_min, z_max = zoom_cfg["cut"]["z_range"]

    if movement_active and center is not None and center0 is not None:
        orig_ranges = [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        new_ranges = []
        for (lo, hi), d in zip(orig_ranges, [center[i] - center0[i] for i in range(3)]):
            width = hi - lo
            lo2 = lo + d
            hi2 = hi + d
            if hi2 > 1.0:
                lo2 -= hi2 - 1.0
                hi2 = 1.0
            if lo2 < 0.0:
                hi2 -= lo2
                lo2 = 0.0
            lo2 = max(0.0, min(1.0, lo2))
            hi2 = max(0.0, min(1.0, hi2))
            if hi2 - lo2 <= 1e-9:
                print(f"  WARNING: shifted zoom cut degenerate on axis after clamping "
                      f"([{lo2:.6f}, {hi2:.6f}]) - keeping unshifted range [{lo:.6f}, {hi:.6f}].")
                new_ranges.append((lo, hi))
            else:
                new_ranges.append((lo2, hi2))
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = new_ranges
        print(f"  Movement-centered zoom cut: x=[{x_min:.6f}, {x_max:.6f}], "
              f"y=[{y_min:.6f}, {y_max:.6f}], z=[{z_min:.6f}, {z_max:.6f}]")

    x0, x1 = cut_indices(orig_shape[0], x_min, x_max)
    y0, y1 = cut_indices(orig_shape[1], y_min, y_max)
    z0, z1 = cut_indices(orig_shape[2], z_min, z_max)

    phi_crop = phi_full[x0:x1+1, y0:y1+1, z0:z1+1].astype(np.float32)
    print(f"Initial crop shape: {phi_crop.shape} (indices x:[{x0},{x1}] y:[{y0},{y1}] z:[{z0},{z1}])")

    zoom_initial_entry = zoom_cfg.get("zoom_initial", None)
    if zoom_initial_entry is None:
        sx, sy, sz = phi_crop.shape
        magnification = float(zoom_factor)
        tx = _target_axis_nodes(orig_shape[0], x_min, x_max, magnification)
        ty = _target_axis_nodes(orig_shape[1], y_min, y_max, magnification)
        tz = _target_axis_nodes(orig_shape[2], z_min, z_max, magnification)
        fx = _exact_zoom_factor(sx, tx)
        fy = _exact_zoom_factor(sy, ty)
        fz = _exact_zoom_factor(sz, tz)
        phi = zoom(phi_crop, (fx, fy, fz), order=1)
        del phi_crop
        print(f"No zoom_initial - crop magnified to nominal {zoom_factor}x "
              f"grid {phi.shape} (target {tx}x{ty}x{tz})")
    else:
        phi = zoom(phi_crop, zoom_initial_entry, order=1)
        del phi_crop
        magnification = float(zoom_initial_entry)
        print(f"After initial {zoom_initial_entry}x zoom, shape: {phi.shape}, "
              f"magnification: {magnification:.1f}x")
    del phi_full
    current_shape = phi.shape

    global_bounds = [x_min, x_max, y_min, y_max, z_min, z_max]
    level = 1

    while True:
        phys_dx = (global_bounds[1] - global_bounds[0]) * Lx_nm
        phys_dy = (global_bounds[3] - global_bounds[2]) * Ly_nm
        phys_dz = (global_bounds[5] - global_bounds[4]) * Lz_nm
        dx_nm = phys_dx / current_shape[0]
        dy_nm = phys_dy / current_shape[1]
        dz_nm = phys_dz / current_shape[2]
        print(f"\nLevel {level} (mag {magnification:.1f}x): shape {current_shape}, "
              f"res ({dx_nm:.4f}, {dy_nm:.4f}, {dz_nm:.4f}) nm")

        x_min_cur, x_max_cur = global_bounds[0], global_bounds[1]
        y_min_cur, y_max_cur = global_bounds[2], global_bounds[3]
        z_min_cur, z_max_cur = global_bounds[4], global_bounds[5]
        dx_frac = x_max_cur - x_min_cur
        dy_frac = y_max_cur - y_min_cur
        dz_frac = z_max_cur - z_min_cur

        zoom_blocks = []
        for b in cfg["blocks"]:
            bx0, bx1 = b["x_range"]
            by0, by1 = b["y_range"]
            bz0, bz1 = b["z_range"]
            ix0 = max(bx0, x_min_cur); ix1 = min(bx1, x_max_cur)
            iy0 = max(by0, y_min_cur); iy1 = min(by1, y_max_cur)
            iz0 = max(bz0, z_min_cur); iz1 = min(bz1, z_max_cur)
            if ix1 > ix0 and iy1 > iy0 and iz1 > iz0:
                zoom_blocks.append({
                    "eps_val": b["eps_val"],
                    "x_range": [(ix0 - x_min_cur)/dx_frac, (ix1 - x_min_cur)/dx_frac],
                    "y_range": [(iy0 - y_min_cur)/dy_frac, (iy1 - y_min_cur)/dy_frac],
                    "z_range": [(iz0 - z_min_cur)/dz_frac, (iz1 - z_min_cur)/dz_frac]
                })
        print(f"  Remapped {len(zoom_blocks)} dielectric blocks.")

        eps_reference = zoom_cfg.get("epsilon_reference_resolution", cfg.get("epsilon_material", {}).get("reference_resolution", 512))
        eps_reference_shape = (int(eps_reference),) * 3 if not isinstance(eps_reference, (list, tuple)) else tuple(int(v) for v in eps_reference)
        eps = generate_eps_level(phi.shape, zoom_blocks, reference_shape=eps_reference_shape)
        if zoom_blocks:
            qd_ids = [b for b in zoom_blocks if abs(b["eps_val"] - 15.0) < 1e-9]
            if qd_ids:
                z_mid = (min(b["z_range"][0] for b in qd_ids) + max(b["z_range"][1] for b in qd_ids)) / 2.0
                y_mid = (min(b["y_range"][0] for b in qd_ids) + max(b["y_range"][1] for b in qd_ids)) / 2.0
                x_mid = (min(b["x_range"][0] for b in qd_ids) + max(b["x_range"][1] for b in qd_ids)) / 2.0
            elif center is not None:
                z_mid = (center[2] - z_min_cur) / dz_frac
                y_mid = (center[1] - y_min_cur) / dy_frac
                x_mid = (center[0] - x_min_cur) / dx_frac
            else:
                z_mid = y_mid = x_mid = 0.5
            mid_z = int(np.clip(z_mid, 0.0, 1.0) * (eps.shape[2] - 1))
            mid_y = int(np.clip(y_mid, 0.0, 1.0) * (eps.shape[1] - 1))
            mid_x = int(np.clip(x_mid, 0.0, 1.0) * (eps.shape[0] - 1))
            mid_z_xy = min(mid_z + 1, eps.shape[2] - 1)
            eps_x_line = eps[:, mid_y, mid_z_xy]
            eps_z_line = eps[mid_x, mid_y, :]
            x_phys = __import__('numpy').linspace(0, phys_dx, len(eps_x_line))
            z_phys = __import__('numpy').linspace(0, phys_dz, len(eps_z_line))
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            im1 = axes[0,0].imshow(eps[:,:,mid_z_xy].T, origin='lower', cmap='viridis',
                                   extent=[0, phys_dx, 0, phys_dy])
            axes[0,0].set_title(f'epsilon(x,y) at central z (level {level})')
            axes[0,0].set_xlabel('x (nm)'); axes[0,0].set_ylabel('y (nm)')
            plt.colorbar(im1, ax=axes[0,0])
            im2 = axes[0,1].imshow(eps[:,mid_y,:].T, origin='lower', cmap='viridis',
                                   extent=[0, phys_dx, 0, phys_dz])
            axes[0,1].set_title(f'epsilon(x,z) at central y (level {level})')
            axes[0,1].set_xlabel('x (nm)'); axes[0,1].set_ylabel('z (nm)')
            plt.colorbar(im2, ax=axes[0,1])
            axes[1,0].plot(z_phys, eps_z_line, 'b-')
            axes[1,0].set_title(f'epsilon(z) at centre (level {level})')
            axes[1,0].set_xlabel('z (nm)'); axes[1,0].grid(True, alpha=0.3)
            axes[1,1].plot(x_phys, eps_x_line, 'r-')
            axes[1,1].set_title(f'epsilon(x) at centre (level {level})')
            axes[1,1].set_xlabel('x (nm)'); axes[1,1].grid(True, alpha=0.3)
            plt.suptitle('Dielectric preview - zoom level %d (slice at QD center)' % level)
            plt.tight_layout()
            plt.show()


        boundary_mask = np.zeros_like(phi, dtype=bool)
        nx, ny, nz = phi.shape

        physical = cfg.get("_physical", {})
        domain_main = physical.get("domain_nm", (Lx_nm, Ly_nm, Lz_nm))
        origin = physical.get("origin_fraction", (0.5, 0.5, 0.0))
        tip_z_nm = physical.get("tip_z_nm")
        R_nm = physical.get("R_nm")
        r_tip_nm = physical.get("r_tip_nm")
        aspect_ratio = float(cfg.get("aspect_ratio", 2.0))
        Vtip = V

        if tip_z_nm is not None and R_nm is not None and r_tip_nm is not None:
            # Convert the physical tip position into the current zoom domain.
            x_abs_min = x_min_cur * domain_main[0]
            x_abs_max = x_max_cur * domain_main[0]
            y_abs_min = y_min_cur * domain_main[1]
            y_abs_max = y_max_cur * domain_main[1]
            z_abs_min = z_min_cur * domain_main[2]
            z_abs_max = z_max_cur * domain_main[2]
            tip_x_abs = origin[0] * domain_main[0]
            tip_y_abs = origin[1] * domain_main[1]
            tip_z_abs = origin[2] * domain_main[2] + tip_z_nm

            local_L = (
                x_abs_max - x_abs_min,
                y_abs_max - y_abs_min,
                z_abs_max - z_abs_min,
            )
            tip_center_local = (
                (tip_x_abs - x_abs_min) / local_L[0],
                (tip_y_abs - y_abs_min) / local_L[1],
            )
            tip_z_local_phys = tip_z_abs - z_abs_min
            tip_in_domain = (
                0.0 <= tip_center_local[0] <= 1.0
                and 0.0 <= tip_center_local[1] <= 1.0
                and 0.0 <= tip_z_local_phys <= local_L[2]
            )
            zoom_apex = (
                tip_center_local[0],
                tip_center_local[1],
                tip_z_local_phys / local_L[2],
            )

            if tip_in_domain:
                tip_mask, _, _ = build_downward_pointing_tip(
                    nx, ny, nz,
                    aspect_ratio=aspect_ratio,
                    verbose=False,
                    tip_z_nm=tip_z_local_phys,
                    R_nm=R_nm,
                    r_tip_nm=r_tip_nm,
                    domain_nm=local_L,
                    center_fraction=tip_center_local,
                )
                phi[tip_mask] = Vtip
                boundary_mask[tip_mask] = True
                zoom_tip_mask = tip_mask
                print(
                    f"  Tip applied: local physical apex z={tip_z_local_phys:.4f} nm, "
                    f"centre=({tip_center_local[0]:.4f},{tip_center_local[1]:.4f})"
                )
        else:
            # Legacy normalized fallback for direct programmatic callers.
            tip_z_global = cfg["tip_z"]
            R_global = cfg["R"]
            r_tip_global = cfg["r_tip"]
            tip_z_local = (tip_z_global - z_min_cur) / dz_frac
            cx_local = (0.5 - x_min_cur) / dx_frac
            cy_local = (0.5 - y_min_cur) / dy_frac
            zoom_apex = (cx_local, cy_local, tip_z_local)
            if z_min_cur <= tip_z_global <= z_max_cur:
                if x_min_cur <= 0.5 <= x_max_cur and y_min_cur <= 0.5 <= y_max_cur:
                    R_local = R_global / dx_frac
                    r_tip_local = r_tip_global / dx_frac
                    tip_mask, _, _ = build_downward_pointing_tip(
                        nx, ny, nz, tip_z=tip_z_local, R=R_local,
                        r_tip=r_tip_local, aspect_ratio=aspect_ratio, verbose=False
                    )
                    phi[tip_mask] = Vtip
                    boundary_mask[tip_mask] = True
                    tip_in_domain = True
                    zoom_tip_mask = tip_mask
                    print(
                        f"  Tip applied: local apex z={tip_z_local:.4f}, "
                        f"centre=({cx_local:.4f},{cy_local:.4f})"
                    )

        if not tip_in_domain:
            print("  Tip not in current domain - skipped.")

        Vgate_list = cfg.get("Vgate", [])
        if Vgate_list:
            for g in Vgate_list:
                gx0, gx1 = g["x_range"]
                gy0, gy1 = g["y_range"]
                gz0, gz1 = g["z_range"]
                ix0 = max(gx0, x_min_cur); ix1 = min(gx1, x_max_cur)
                iy0 = max(gy0, y_min_cur); iy1 = min(gy1, y_max_cur)
                iz0 = max(gz0, z_min_cur); iz1 = min(gz1, z_max_cur)
                if ix1 >= ix0 and iy1 >= iy0 and iz1 >= iz0:
                    local_gate = {
                        "Vgate_val": g.get("Vgate_val", 0.0),
                        "x_range": [(ix0 - x_min_cur)/dx_frac, (ix1 - x_min_cur)/dx_frac],
                        "y_range": [(iy0 - y_min_cur)/dy_frac, (iy1 - y_min_cur)/dy_frac],
                        "z_range": [(iz0 - z_min_cur)/dz_frac, (iz1 - z_min_cur)/dz_frac]
                    }
                    gate_mask = make_gate_mask(nx, ny, nz, local_gate)
                    phi[gate_mask] = local_gate["Vgate_val"]
                    boundary_mask[gate_mask] = True
                    print(f"  Gate applied: V={local_gate['Vgate_val']} V")

        # Voltage masks are always retained.  The ``clamp`` switch controls
        # only the six outer faces of the zoomed domain:
        #   True  -> historical fixed/Dirichlet outer boundary.
        #   False -> natural homogeneous-Neumann outer boundary, implemented
        #            by the solver's edge-copy operation.
        # In both modes ``phi`` is already inherited from the previous level
        # through linear interpolation, so the Neumann mode starts from the
        # previous-level potential rather than from a zero/default field.
        if clamp:
            boundary_mask[0,:,:] = True
            boundary_mask[-1,:,:] = True
            boundary_mask[:,0,:] = True
            boundary_mask[:,-1,:] = True
            boundary_mask[:,:,0] = True
            boundary_mask[:,:,-1] = True

        if center is not None:
            zoom_cxl = max(0.0, min(1.0, (center[0] - x_min_cur) / dx_frac))
            zoom_cyl = max(0.0, min(1.0, (center[1] - y_min_cur) / dy_frac))
            zoom_czl = max(0.0, min(1.0, (center[2] - z_min_cur) / dz_frac))
        else:
            zoom_cxl = zoom_cyl = zoom_czl = 0.5

        if plotting_enabled:
            fig = plot_phi_plane(phi, boundary_mask, plane=(True, True, zoom_czl), tip_mask=zoom_tip_mask, apex=zoom_apex)
            ax = fig.axes[0] if fig.axes else None
            if ax is not None:
                ax.set_title(ax.get_title() + f"\nLevel {level}, mag {magnification:.1f}x (initial)")
            plt.show()
            plt.close(fig)

            fig = plot_phi_plane(phi, boundary_mask, plane=(zoom_cxl, True, True), tip_mask=zoom_tip_mask, apex=zoom_apex)
            ax = fig.axes[0] if fig.axes else None
            if ax is not None:
                ax.set_title(ax.get_title() + f"\nLevel {level}, mag {magnification:.1f}x (initial)")
            plt.show()
            plt.close(fig)

        memory_tracking = bool(cfg.get("memory_tracking", False))
        if memory_tracking:
            from .memory import track_memory, log_memory_usage
            memory_context = track_memory()
        else:
            from contextlib import nullcontext
            memory_context = nullcontext(None)

        with memory_context as mem_tracker:
            phi, residual = mg_3d_masked(
                Vtip, phi, boundary_mask, damping=0.8, max_iter=100000, tol=cfg.get("res_tol_zoom", 1e-5),
                eps_r=eps, eps=True,
                mg_max_runtime=cfg.get("mg_max_runtime", None), verbose=False,
                output_dir=output_dir,
                plotting_enabled=plotting_enabled
            )

        if memory_tracking:
            log_memory_usage(
                f"zoom {int(magnification)}x ({phi.shape[0]}x{phi.shape[1]}x{phi.shape[2]})",
                mem_tracker.peak_gb,
                output_dir=output_dir,
            )

        if plotting_enabled:
            fig = plot_phi_plane(phi, boundary_mask, plane=(True, True, zoom_czl), tip_mask=zoom_tip_mask, apex=zoom_apex)
            ax = fig.axes[0] if fig.axes else None
            if ax is not None:
                ax.set_title(ax.get_title() + f"\nLevel {level}, mag {magnification:.1f}x")
            plt.show()
            plt.close(fig)

            fig = plot_phi_plane(phi, boundary_mask, plane=(zoom_cxl, True, True), tip_mask=zoom_tip_mask, apex=zoom_apex)
            ax = fig.axes[0] if fig.axes else None
            if ax is not None:
                ax.set_title(ax.get_title() + f"\nLevel {level}, mag {magnification:.1f}x")
            plt.show()
            plt.close(fig)



        nx_s, ny_s, nz_s = phi.shape
        eff_nx = int(round(orig_shape[0] * magnification))
        eff_ny = int(round(orig_shape[1] * magnification))
        eff_nz = int(round(orig_shape[2] * magnification))
        zoom_lvl_base = f"afm_phi_zoom_{int(magnification)}x_{V}V_{config_idx}"
        if movement_active and center is not None:
            cx, cy, cz = center
            zoom_lvl_base += f"_cx{cx:.2f}_cy{cy:.2f}_cz{cz:.2f}"
        if cfg.get("save_all_levels", False):
            lvl_name = f"{zoom_lvl_base}_level{eff_nx}x{eff_ny}x{eff_nz}.npy"
            out_lvl = os.path.join(output_dir, lvl_name)
            n = 0
            while os.path.exists(out_lvl):
                n += 1
                out_lvl = os.path.join(output_dir, f"{os.path.splitext(lvl_name)[0]} ({n}).npy")
            np.save(out_lvl, phi)
            print(f"  Zoom level saved: {out_lvl} (actual grid {nx_s}x{ny_s}x{nz_s}, "
              f"effective {eff_nx}x{eff_ny}x{eff_nz})")


        if plotting_enabled and cfg.get("plot_zoom_residuals", True):
            ec = eps.astype(__import__('numpy').float32, copy=False)
            axp = 0.25*(ec[1:, :-1, :-1] + ec[1:, 1:, :-1] + ec[1:, :-1, 1:] + ec[1:, 1:, 1:])
            axm = 0.25*(ec[:-1, :-1, :-1] + ec[:-1, 1:, :-1] + ec[:-1, :-1, 1:] + ec[:-1, 1:, 1:])
            ayp = 0.25*(ec[:-1, 1:, :-1] + ec[1:, 1:, :-1] + ec[:-1, 1:, 1:] + ec[1:, 1:, 1:])
            aym = 0.25*(ec[:-1, :-1, :-1] + ec[1:, :-1, :-1] + ec[:-1, :-1, 1:] + ec[1:, :-1, 1:])
            azp = 0.25*(ec[:-1, :-1, 1:] + ec[1:, :-1, 1:] + ec[:-1, 1:, 1:] + ec[1:, 1:, 1:])
            azm = 0.25*(ec[:-1, :-1, :-1] + ec[1:, :-1, :-1] + ec[:-1, 1:, :-1] + ec[1:, 1:, :-1])
            a0 = axp + axm + ayp + aym + azp + azm
            _, _, lvl_res = compute_residual_vec_unpadded(phi, boundary_mask,
                                                          axp, axm, ayp, aym, azp, azm, a0)
            from .plotting import plot_residual_plane
            plot_residual_plane(lvl_res, boundary_mask, plane=(True, True, zoom_czl), tip_mask=zoom_tip_mask, apex=zoom_apex)
            plot_residual_plane(lvl_res, boundary_mask, plane=(zoom_cxl, True, True), tip_mask=zoom_tip_mask, apex=zoom_apex)
            plt.show()
            plt.close('all')
            del lvl_res
            del ec, axp, axm, ayp, aym, azp, azm, a0

        # The epsilon field and boundary masks are level-local. Release them
        # before constructing the next zoom level; only the solved potential is
        # intentionally retained.
        if "eps" in locals():
            del eps
        if "boundary_mask" in locals():
            del boundary_mask
        if "zoom_tip_mask" in locals():
            del zoom_tip_mask
        gc.collect()

        if "residual" in locals():
            del residual
        if magnification * zoom_factor > zoom_limit:
            break

        nx, ny, nz = phi.shape
        if cascade:
            xl, xu = zoom_cfg["cut"]["x_range"]
            yl, yu = zoom_cfg["cut"]["y_range"]
            zl, zu = zoom_cfg["cut"]["z_range"]
        else:
            xl, xu = 0.375, 0.625
            yl, yu = 0.375, 0.625
            zl, zu = zoom_cfg["cut"]["z_range"]
        i0, i1 = cut_indices(nx, xl, xu)
        j0, j1 = cut_indices(ny, yl, yu)
        k0, k1 = cut_indices(nz, zl, zu)
        if min(i1 - i0, j1 - j0, k1 - k0) < 1:
            print("  WARNING: cascade cut window below 2 nodes - no further refinement.")
            break
        clx, cux = xl, xu
        cly, cuy = yl, yu
        clz, cuz = zl, zu
        phi_crop_next = phi[i0:i1+1, j0:j1+1, k0:k1+1].copy()

        x_min_cur, x_max_cur = global_bounds[0], global_bounds[1]
        y_min_cur, y_max_cur = global_bounds[2], global_bounds[3]
        z_min_cur, z_max_cur = global_bounds[4], global_bounds[5]
        new_x_min = x_min_cur + clx * (x_max_cur - x_min_cur)
        new_x_max = x_min_cur + cux * (x_max_cur - x_min_cur)
        new_y_min = y_min_cur + cly * (y_max_cur - y_min_cur)
        new_y_max = y_min_cur + cuy * (y_max_cur - y_min_cur)
        new_z_min = z_min_cur + clz * (z_max_cur - z_min_cur)
        new_z_max = z_min_cur + cuz * (z_max_cur - z_min_cur)
        global_bounds = [new_x_min, new_x_max, new_y_min, new_y_max, new_z_min, new_z_max]

        next_magnification = magnification * zoom_factor
        tx = _target_axis_nodes(orig_shape[0], new_x_min, new_x_max, next_magnification)
        ty = _target_axis_nodes(orig_shape[1], new_y_min, new_y_max, next_magnification)
        tz = _target_axis_nodes(orig_shape[2], new_z_min, new_z_max, next_magnification)
        xr = _exact_zoom_factor(phi_crop_next.shape[0], tx)
        yr = _exact_zoom_factor(phi_crop_next.shape[1], ty)
        zr = _exact_zoom_factor(phi_crop_next.shape[2], tz)

        if cascade:
            print(f"  Cascade cut: window -> x=[{new_x_min:.6f}, {new_x_max:.6f}], "
                  f"y=[{new_y_min:.6f}, {new_y_max:.6f}], z=[{new_z_min:.6f}, {new_z_max:.6f}]")

        phi = zoom(phi_crop_next, (xr, yr, zr), order=1)
        del phi_crop_next
        gc.collect()
        magnification = next_magnification
        current_shape = phi.shape
        print(f"  Zoomed to nominal {magnification:.1f}x: grid {current_shape} "
              f"(target {tx}x{ty}x{tz})")
        level += 1

    save_full = bool(cfg.get("save_full", False))
    save_cut = bool(cfg.get("save_cut", False))
    cut_offsets_nm = cfg.get("save_cut_box_nm", [-32.0, 32.0, -32.0, 32.0, -32.0, 32.0])
    physical_cfg = cfg.get("_physical", {})
    domain_nm = tuple(float(v) for v in physical_cfg.get("domain_nm", (Lx_nm, Ly_nm, Lz_nm)))
    origin = tuple(float(v) for v in physical_cfg.get("origin_fraction", (0.5, 0.5, 0.0)))
    if center is None:
        center = tuple(float(v) for v in origin)
    center_nm = tuple((center[i] - origin[i]) * domain_nm[i] for i in range(3))

    zoom_base = f"afm_phi_zoom_{int(magnification)}x_{V}V_{config_idx}.npy"
    if movement_active and center is not None:
        cx, cy, cz = center
        zoom_base = (f"afm_phi_zoom_{int(magnification)}x_{V}V_{config_idx}"
                     f"_cx{cx:.2f}_cy{cy:.2f}_cz{cz:.2f}.npy")

    # ``global_bounds`` are fractions of the main domain. Convert the final
    # zoom field back to physical coordinates relative to the same origin so
    # the requested save box follows the physical movement center.
    zoom_field_bounds_nm = (
        (global_bounds[0] - origin[0]) * domain_nm[0],
        (global_bounds[1] - origin[0]) * domain_nm[0],
        (global_bounds[2] - origin[1]) * domain_nm[1],
        (global_bounds[3] - origin[1]) * domain_nm[1],
        (global_bounds[4] - origin[2]) * domain_nm[2],
        (global_bounds[5] - origin[2]) * domain_nm[2],
    )

    saved_paths = []
    if save_full:
        saved_paths.append(save_potential_full(phi, zoom_base, output_dir=output_dir))
    if save_cut:
        cut_name = os.path.splitext(zoom_base)[0] + "_cut.npy"
        path, actual_bounds = save_potential_physical_cut(
            phi, center_nm, cut_offsets_nm, zoom_field_bounds_nm,
            filename=cut_name, output_dir=output_dir
        )
        saved_paths.append(path)

    if saved_paths:
        print(f"Layered zoom finished. Final magnification: {magnification:.1f}x. Saved: {', '.join(saved_paths)}")
    else:
        print(f"Layered zoom finished. Final magnification: {magnification:.1f}x. No NPY saved (save_full/save_cut both false).")

    zoom_elapsed = time.time() - zoom_start_time
    time_log.append(zoom_elapsed)

    return results
