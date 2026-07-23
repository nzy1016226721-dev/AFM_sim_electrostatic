import numpy as np
import matplotlib.pyplot as plt
import time
import os
import csv
import gc
from scipy.ndimage import zoom

from .solver import mg_3d_masked, compute_residual_vec_unpadded
from .materials import generate_eps_cell, generate_sigma_cell
from .joule import compute_joule_heating, plot_scalar_plane
from .plotting import plot_phi_plane, plot_residual_plane
from .io_utils import make_gate_mask, log_joule_csv


def run_zoom_simulation(cfg, results, V, config_idx, time_log=None, output_dir="."):
    """Run a recursive central-crop zoom simulation on top of the main result.

    Crops the potential around the region of interest, upsamples, re-solves
    Poisson's equation, computes Joule heating if sigma blocks are available,
    and saves the zoomed potential and power density.

    Parameters
    ----------
    cfg : dict
        Configuration dict with 'zoom_simulation' sub-dict.
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

    zoom_cfg = cfg.get("zoom_simulation", None)
    if zoom_cfg is None or not zoom_cfg.get("enabled", False):
        return results

    if time_log is None:
        time_log = []

    zoom_start_time = time.time()
    show_zoom_residuals = True

    print("\n==============================")
    print("Starting layered AFM zoom simulation (recursive central crop)")
    print("==============================")

    zoom_factor   = zoom_cfg["zoom_factor"]
    zoom_limit    = zoom_cfg["zoom_limit"]
    zoom_initial  = zoom_cfg.get("zoom_initial", 2)

    Lx_nm = cfg.get("Lx_nm", 512.0)
    Ly_nm = cfg.get("Ly_nm", 512.0)
    Lz_nm = cfg.get("Lz_nm", 512.0)

    phi_full = results['phi'].astype(np.float32)
    orig_shape = phi_full.shape

    x_min, x_max = zoom_cfg["cut"]["x_range"]
    y_min, y_max = zoom_cfg["cut"]["y_range"]
    z_min, z_max = zoom_cfg["cut"]["z_range"]

    x0 = int(x_min * (orig_shape[0] - 1))
    x1 = int(x_max * (orig_shape[0] - 1))
    y0 = int(y_min * (orig_shape[1] - 1))
    y1 = int(y_max * (orig_shape[1] - 1))
    z0 = int(z_min * (orig_shape[2] - 1))
    z1 = int(z_max * (orig_shape[2] - 1))

    phi_crop = phi_full[x0:x1+1, y0:y1+1, z0:z1+1].astype(np.float32)
    print(f"Initial crop shape: {phi_crop.shape}")

    phi = zoom(phi_crop, zoom_initial, order=1)
    current_shape = phi.shape
    magnification = zoom_initial
    print(f"After initial {zoom_initial}x zoom, shape: {current_shape}, "
          f"magnification: {magnification:.1f}x")

    global_bounds = [x_min, x_max, y_min, y_max, z_min, z_max]
    level = 1

    sigma_blocks_global = cfg.get("sigma_blocks", [])

    while True:
        phys_dx = (global_bounds[1] - global_bounds[0]) * Lx_nm
        phys_dy = (global_bounds[3] - global_bounds[2]) * Ly_nm
        phys_dz = (global_bounds[5] - global_bounds[4]) * Lz_nm
        dx_nm = phys_dx / (current_shape[0] - 1)
        dy_nm = phys_dy / (current_shape[1] - 1)
        dz_nm = phys_dz / (current_shape[2] - 1)
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

        eps = generate_eps_cell(phi, zoom_blocks, use_precomputed=False)

        if zoom_blocks:
            mid_z = eps.shape[2] // 2
            mid_y = eps.shape[1] // 2
            eps_x_line = eps[:, mid_y, mid_z]
            eps_z_line = eps[eps.shape[0]//2, mid_y, :]
            x_phys = np.linspace(0, phys_dx, len(eps_x_line))
            z_phys = np.linspace(0, phys_dz, len(eps_z_line))
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            im1 = axes[0,0].imshow(eps[:,:,mid_z].T, origin='lower', cmap='viridis',
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
            plt.suptitle(f'Dielectric preview - zoom level {level}')
            plt.tight_layout()
            plt.show()

        boundary_mask = np.zeros_like(phi, dtype=bool)

        tip_z_global = cfg["tip_z"]
        R_global = cfg["R"]
        r_tip_global = cfg["r_tip"]
        aspect_ratio = cfg["aspect_ratio"]
        Vtip = V

        tip_in_domain = False
        if z_min_cur <= tip_z_global <= z_max_cur:
            if x_min_cur <= 0.5 <= x_max_cur and y_min_cur <= 0.5 <= y_max_cur:
                tip_z_local = (tip_z_global - z_min_cur) / dz_frac
                R_local = R_global / dx_frac
                r_tip_local = r_tip_global / dx_frac
                cx_local = (0.5 - x_min_cur) / dx_frac
                cy_local = (0.5 - y_min_cur) / dy_frac
                nx, ny, nz = phi.shape
                x = np.linspace(0, 1, nx)
                y = np.linspace(0, 1, ny)
                z = np.linspace(0, 1, nz)
                tip_mask = np.zeros((nx, ny, nz), dtype=bool)
                theta_asym = np.arctan(aspect_ratio)
                a = R_local * np.tan(theta_asym)
                b = R_local * np.tan(theta_asym)**2
                z0_tip = tip_z_local - b
                z_base = z0_tip + np.sqrt(b**2 * (1 + (r_tip_local**2 / a**2)))
                for k in range(nz):
                    zk = z[k]
                    if zk < tip_z_local or zk > z_base:
                        continue
                    dz_vert = zk - z0_tip
                    if dz_vert**2 < b**2:
                        continue
                    r_max = a * np.sqrt((dz_vert**2 / b**2) - 1)
                    r_max = min(r_max, r_tip_local)
                    X, Y = np.meshgrid(x - cx_local, y - cy_local, indexing='ij')
                    R_dist = np.sqrt(X**2 + Y**2)
                    tip_mask[:, :, k] = (R_dist <= r_max)
                phi[tip_mask] = Vtip
                boundary_mask[tip_mask] = True
                tip_in_domain = True
                print(f"  Tip applied: local apex z={tip_z_local:.4f}, "
                      f"centre=({cx_local:.4f},{cy_local:.4f})")
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
                if ix1 > ix0 and iy1 > iy0 and iz1 > iz0:
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

        boundary_mask[0,:,:] = True
        boundary_mask[-1,:,:] = True
        boundary_mask[:,0,:] = True
        boundary_mask[:,-1,:] = True
        boundary_mask[:,:,0] = True
        boundary_mask[:,:,-1] = True

        fig = plot_phi_plane(phi, boundary_mask, plane=(True, True, 0.5))
        ax = fig.axes[0] if fig.axes else None
        if ax is not None:
            ax.set_title(ax.get_title() + f"\nLevel {level}, mag {magnification:.1f}x (initial)")
        plt.show()
        plt.close(fig)

        fig = plot_phi_plane(phi, boundary_mask, plane=(0.5, True, True))
        ax = fig.axes[0] if fig.axes else None
        if ax is not None:
            ax.set_title(ax.get_title() + f"\nLevel {level}, mag {magnification:.1f}x (initial)")
        plt.show()
        plt.close(fig)

        phi, residual = mg_3d_masked(
            Vtip, phi, boundary_mask, damping=0.8, max_iter=5000, tol=cfg.get("res_tol_zoom", 1e-5),
            eps_r=eps, eps=True,
            mg_max_runtime=cfg.get("mg_max_runtime", None), verbose=False,
            output_dir=output_dir
        )

        fig = plot_phi_plane(phi, boundary_mask, plane=(True, True, 0.5))
        ax = fig.axes[0] if fig.axes else None
        if ax is not None:
            ax.set_title(ax.get_title() + f"\nLevel {level}, mag {magnification:.1f}x")
        plt.show()
        plt.close(fig)

        fig = plot_phi_plane(phi, boundary_mask, plane=(0.5, True, True))
        ax = fig.axes[0] if fig.axes else None
        if ax is not None:
            ax.set_title(ax.get_title() + f"\nLevel {level}, mag {magnification:.1f}x")
        plt.show()
        plt.close(fig)

        if show_zoom_residuals:
            ec = eps.astype(np.float32, copy=False)
            axp = 0.25*(ec[1:, :-1, :-1] + ec[1:, 1:, :-1] + ec[1:, :-1, 1:] + ec[1:, 1:, 1:])
            axm = 0.25*(ec[:-1, :-1, :-1] + ec[:-1, 1:, :-1] + ec[:-1, :-1, 1:] + ec[:-1, 1:, 1:])
            ayp = 0.25*(ec[:-1, 1:, :-1] + ec[1:, 1:, :-1] + ec[:-1, 1:, 1:] + ec[1:, 1:, 1:])
            aym = 0.25*(ec[:-1, :-1, :-1] + ec[1:, :-1, :-1] + ec[:-1, :-1, 1:] + ec[1:, :-1, 1:])
            azp = 0.25*(ec[:-1, :-1, 1:] + ec[1:, :-1, 1:] + ec[:-1, 1:, 1:] + ec[1:, 1:, 1:])
            azm = 0.25*(ec[:-1, :-1, :-1] + ec[1:, :-1, :-1] + ec[:-1, 1:, :-1] + ec[1:, 1:, :-1])
            a0 = axp + axm + ayp + aym + azp + azm
            _, _, lvl_res = compute_residual_vec_unpadded(phi, boundary_mask,
                                                          axp, axm, ayp, aym, azp, azm, a0)
            plot_residual_plane(lvl_res, boundary_mask, plane=(True, True, 0.5))
            plot_residual_plane(lvl_res, boundary_mask, plane=(0.5, True, True))
            plt.show()
            plt.close('all')

        if magnification >= zoom_limit:
            break

        nx, ny, nz = phi.shape
        i0 = int(0.375 * (nx - 1))
        i1 = int(0.625 * (nx - 1))
        j0 = int(0.375 * (ny - 1))
        j1 = int(0.625 * (ny - 1))
        k0 = int(0.375 * (nz - 1))
        k1 = int(0.625 * (nz - 1))
        phi_crop_next = phi[i0:i1+1, j0:j1+1, k0:k1+1]

        x_min_cur, x_max_cur = global_bounds[0], global_bounds[1]
        y_min_cur, y_max_cur = global_bounds[2], global_bounds[3]
        z_min_cur, z_max_cur = global_bounds[4], global_bounds[5]
        new_x_min = x_min_cur + 0.375 * (x_max_cur - x_min_cur)
        new_x_max = x_min_cur + 0.625 * (x_max_cur - x_min_cur)
        new_y_min = y_min_cur + 0.375 * (y_max_cur - y_min_cur)
        new_y_max = y_min_cur + 0.625 * (y_max_cur - y_min_cur)
        new_z_min = z_min_cur + 0.375 * (z_max_cur - z_min_cur)
        new_z_max = z_min_cur + 0.625 * (z_max_cur - z_min_cur)
        global_bounds = [new_x_min, new_x_max, new_y_min, new_y_max, new_z_min, new_z_max]

        phi = zoom(phi_crop_next, zoom_factor, order=1)
        magnification *= zoom_factor
        current_shape = phi.shape
        level += 1

    if sigma_blocks_global:
        x_min_cur, x_max_cur = global_bounds[0], global_bounds[1]
        y_min_cur, y_max_cur = global_bounds[2], global_bounds[3]
        z_min_cur, z_max_cur = global_bounds[4], global_bounds[5]
        dx_frac = x_max_cur - x_min_cur
        dy_frac = y_max_cur - y_min_cur
        dz_frac = z_max_cur - z_min_cur

        zoom_sigma_blocks = []
        for b in sigma_blocks_global:
            bx0, bx1 = b["x_range"]
            by0, by1 = b["y_range"]
            bz0, bz1 = b["z_range"]
            ix0 = max(bx0, x_min_cur); ix1 = min(bx1, x_max_cur)
            iy0 = max(by0, y_min_cur); iy1 = min(by1, y_max_cur)
            iz0 = max(bz0, z_min_cur); iz1 = min(bz1, z_max_cur)
            if ix1 > ix0 and iy1 > iy0 and iz1 > iz0:
                zoom_sigma_blocks.append({
                    "sigma_val": b["sigma_val"],
                    "x_range": [(ix0 - x_min_cur)/dx_frac, (ix1 - x_min_cur)/dx_frac],
                    "y_range": [(iy0 - y_min_cur)/dy_frac, (iy1 - y_min_cur)/dy_frac],
                    "z_range": [(iz0 - z_min_cur)/dz_frac, (iz1 - z_min_cur)/dz_frac]
                })

        sigma_cell_zoom = generate_sigma_cell(phi, zoom_sigma_blocks, use_precomputed=False)

        Lx_zoom = dx_frac * Lx_nm
        Ly_zoom = dy_frac * Ly_nm
        Lz_zoom = dz_frac * Lz_nm
        p_dens_zoom, P_total_zoom, Jx, Jy, Jz, Ex, Ey, Ez = \
            compute_joule_heating(phi, sigma_cell_zoom, Lx_zoom, Ly_zoom, Lz_zoom)

        print(f"  Zoomed Joule power: {P_total_zoom:.6e} W")

        os.makedirs(output_dir, exist_ok=True)
        zoom_p_name = os.path.join(output_dir, f"power_density_zoom_{int(magnification)}x_{V}V_{config_idx}.npy")
        np.save(zoom_p_name, p_dens_zoom)

        log_joule_csv(config_idx, V, P_total_zoom, csv_file="joule_power_zoom.csv", output_dir=output_dir)

        Jmag = np.sqrt(Jx**2 + Jy**2 + Jz**2)

        fig = plot_scalar_plane(p_dens_zoom, None, plane=(True, True, 0.5),
                                cmap='hot', label='Power density (W/m3)')
        plt.show(); plt.close(fig)
        fig = plot_scalar_plane(Jmag, None, plane=(True, True, 0.5),
                                cmap='plasma', label='|J| (A/m2)')
        plt.show(); plt.close(fig)

        fig = plot_scalar_plane(p_dens_zoom, None, plane=(True, 0.5, True),
                                cmap='hot', label='Power density (W/m3)')
        plt.show(); plt.close(fig)
        fig = plot_scalar_plane(Jmag, None, plane=(True, 0.5, True),
                                cmap='plasma', label='|J| (A/m2)')
        plt.show(); plt.close(fig)
    else:
        print("  No sigma blocks available - zoom Joule heating skipped.")

    out_name = os.path.join(output_dir, f"afm_phi_zoom_{int(magnification)}x_{V}V_{config_idx}.npy")
    n = 0
    while os.path.exists(out_name):
        n += 1
        out_name = os.path.join(output_dir, f"afm_phi_zoom_{int(magnification)}x_{V}V_{config_idx} ({n}).npy")
    np.save(out_name, phi)
    print(f"Layered zoom finished. Final magnification: {magnification:.1f}x. "
          f"Saved: {out_name}")

    zoom_elapsed = time.time() - zoom_start_time
    time_log.append(zoom_elapsed)

    return results
