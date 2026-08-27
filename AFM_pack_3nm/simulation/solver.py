import numpy as np
import time
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

MG_TIME = {"elapsed": 0.0}
RESIDUAL_CSV = "residual_history.csv"


def build_downward_pointing_tip(nx, ny, nz, tip_z=0.2, R=0.05, r_tip=0.15,
                                aspect_ratio=2.0, verbose=True,
                                tip_z_nm=None, R_nm=None, r_tip_nm=None,
                                domain_nm=None, center_fraction=(0.5, 0.5)):
    """Generate a boolean mask for a downward-pointing hyperbolic AFM tip.

    The legacy ``tip_z``, ``R`` and ``r_tip`` arguments are fractional-domain
    quantities.  When the physical arguments ``tip_z_nm``, ``R_nm``,
    ``r_tip_nm`` and ``domain_nm`` are supplied, the geometry is constructed
    directly in nanometres.  This physical mode is the canonical path for
    JSON configurations and remains correct for rectangular grids such as
    ``256 x 256 x 100`` because each axis uses its own physical domain length.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.
    tip_z : float, optional
        Legacy fractional z-coordinate of the tip apex.
    R, r_tip : float, optional
        Legacy fractional tip curvature and truncation radii.
    aspect_ratio : float, optional
        Tip height/radius aspect ratio.
    verbose : bool, optional
        Print geometry information.
    tip_z_nm, R_nm, r_tip_nm : float, optional
        Physical tip parameters relative to the configured origin.
    domain_nm : tuple(float, float, float), optional
        Physical main-domain lengths ``(Lx, Ly, Lz)`` in nm.
    center_fraction : tuple(float, float), optional
        XY location of the physical origin as fractions of the domain.

    Returns
    -------
    mask : np.ndarray
        Boolean tip mask with shape ``(nx, ny, nz)``.
    z_tip : float
        Fractional z-coordinate of the tip apex.
    z_base : float
        Fractional z-coordinate of the tip base.
    """
    physical_mode = (
        tip_z_nm is not None and R_nm is not None and r_tip_nm is not None
        and domain_nm is not None
    )

    if physical_mode:
        Lx, Ly, Lz = (float(v) for v in domain_nm)
        ox, oy = (float(v) for v in center_fraction[:2])
        x = np.linspace(0.0, Lx, nx)
        y = np.linspace(0.0, Ly, ny)
        z = np.linspace(0.0, Lz, nz)
        cx = ox * Lx
        cy = oy * Ly
        tip_z_abs = float(tip_z_nm) + 0.0  # physical origin is z=0 by default
        # Allow a nonzero z-origin fraction if supplied as a 3-vector by caller.
        if len(center_fraction) >= 3:
            oz = float(center_fraction[2])
            tip_z_abs += oz * Lz
        R_phys = float(R_nm)
        r_tip_phys = float(r_tip_nm)

        theta_asym = np.arctan(float(aspect_ratio))
        a = R_phys * np.tan(theta_asym)
        b = R_phys * np.tan(theta_asym) ** 2
        z0 = tip_z_abs - b
        z_base_abs = z0 + np.sqrt(b**2 * (1 + (r_tip_phys**2 / a**2)))

        z_tip_frac = tip_z_abs / Lz
        z_base_frac = z_base_abs / Lz
    else:
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        z = np.linspace(0, 1, nz)
        cx, cy = 0.5, 0.5
        tip_z_idx = int(np.clip(tip_z, 0, 1) * (nz - 1))
        z_tip_frac = z[tip_z_idx]
        z_base_frac = 0.0
        theta_asym = np.arctan(aspect_ratio)
        a = R * np.tan(theta_asym)
        b = R * np.tan(theta_asym) ** 2
        z0 = z_tip_frac - b
        z_base_frac = z0 + np.sqrt(b**2 * (1 + (r_tip**2 / a**2)))
        tip_z_abs = z_tip_frac
        z_base_abs = z_base_frac
        R_phys = R
        r_tip_phys = r_tip

    mask = np.zeros((nx, ny, nz), dtype=bool)

    if verbose:
        print("\n[Hyperbolic AFM Tip]")
        print(f"  Aspect ratio (tanθ) = {aspect_ratio:.3f}, θ_asym = {np.degrees(theta_asym):.2f}°")
        if physical_mode:
            print(f"  Physical R = {R_phys:.4f} nm, r_tip = {r_tip_phys:.4f} nm")
            print(f"  Physical tip_z = {tip_z_nm:.4f} nm")
        else:
            print(f"  Fractional R = {R:.6f}, r_tip = {r_tip:.6f}")
        print(f"  Tip_z = {z_tip_frac:.6f}, Base_z = {z_base_frac:.6f}")

    for k, zk in enumerate(z):
        if physical_mode:
            if zk < tip_z_abs or zk > z_base_abs:
                continue
            dz = zk - z0
            if dz**2 < b**2:
                continue
            r_max = a * np.sqrt((dz**2 / b**2) - 1)
            r_max = min(r_max, r_tip_phys)
            X, Y = np.meshgrid(x - cx, y - cy, indexing="ij")
            mask[:, :, k] = np.sqrt(X**2 + Y**2) <= r_max
        else:
            if zk < z_tip_frac or zk > z_base_frac:
                continue
            dz = zk - z0
            if dz**2 < b**2:
                continue
            r_max = a * np.sqrt((dz**2 / b**2) - 1)
            r_max = min(r_max, r_tip)
            X, Y = np.meshgrid(x - cx, y - cy, indexing="ij")
            mask[:, :, k] = np.sqrt(X**2 + Y**2) <= r_max

    if verbose:
        print(f"  Total points: {int(np.sum(mask))}")

    return mask, z_tip_frac, z_base_frac

def compute_residual_vec_unpadded(V, mask, axp, axm, ayp, aym, azp, azm, a0):
    """Compute residual of the discretized Poisson equation (full matrix output).

    Evaluates L(phi) = div(eps * grad(phi)) at interior points, masks
    boundary voxels to NaN, and returns the full residual matrix alongside
    summary statistics.

    Parameters
    ----------
    V : np.ndarray
        3D potential array (Nx x Ny x Nz).
    mask : np.ndarray (bool)
        Boundary mask (True = Dirichlet nodes).
    axp, axm, ayp, aym, azp, azm : np.ndarray
        Harmonic-mean dielectric coefficients at cell faces.
    a0 : np.ndarray
        Sum of the six neighbour coefficients.

    Returns
    -------
    res_mean : float
        RMS residual over interior nodes.
    res_max : float
        Maximum absolute residual over interior nodes.
    res_matrix : np.ndarray
        3D array of |residual| with NaN at boundaries.
    """
    Nx, Ny, Nz = V.shape

    num = (
        axp * V[2:  , 1:-1, 1:-1] +
        axm * V[ :-2, 1:-1, 1:-1] +
        ayp * V[1:-1, 2:  , 1:-1] +
        aym * V[1:-1,  :-2, 1:-1] +
        azp * V[1:-1, 1:-1, 2:  ] +
        azm * V[1:-1, 1:-1,  :-2]
    )
    center = a0 * V[1:-1, 1:-1, 1:-1]
    resid_core = num - center

    interior_mask = ~mask[1:-1, 1:-1, 1:-1]
    resid_core_masked = np.zeros_like(resid_core)
    resid_core_masked[~interior_mask] = np.nan
    resid_core_masked[interior_mask] = resid_core[interior_mask]

    if np.any(interior_mask):
        res_mean = np.sqrt(np.mean(resid_core[interior_mask]**2))
        res_max  = np.sqrt(np.max(np.abs(resid_core[interior_mask])**2))
    else:
        res_mean, res_max = 0.0, 0.0

    res_matrix = np.full_like(V, np.nan, dtype=np.float32)
    res_matrix[1:-1, 1:-1, 1:-1] = np.abs(resid_core_masked)

    return res_mean, res_max, res_matrix


def compute_residual_scalars(V, mask, axp, axm, ayp, aym, azp, azm, a0):
    """Compute scalar residual norms (no full matrix allocation).

    Parameters
    ----------
    V : np.ndarray
        3D potential array.
    mask : np.ndarray (bool)
        Boundary mask.
    axp, axm, ayp, aym, azp, azm : np.ndarray
        Dielectric face coefficients.
    a0 : np.ndarray
        Sum of coefficients.

    Returns
    -------
    res_L2 : float
        L2-norm residual.
    res_max : float
        Maximum residual.
    """
    num = (
        axp * V[2:  , 1:-1, 1:-1] +
        axm * V[ :-2, 1:-1, 1:-1] +
        ayp * V[1:-1, 2:  , 1:-1] +
        aym * V[1:-1,  :-2, 1:-1] +
        azp * V[1:-1, 1:-1, 2:  ] +
        azm * V[1:-1, 1:-1,  :-2]
    )
    resid = num - a0 * V[1:-1, 1:-1, 1:-1]

    free = ~mask[1:-1, 1:-1, 1:-1]
    if not free.any():
        return 0.0, 0.0
    res_L2  = np.sqrt(np.mean(resid[free]**2))
    res_max = np.sqrt(np.max(np.abs(resid[free])**2))
    return res_L2, res_max


def log_residual_csv(iteration, res_avg, res_max, csv_file=None, output_dir="."):
    """Append one residual measurement to the simulation residual CSV log."""
    if csv_file is None:
        csv_file = RESIDUAL_CSV
    csv_path = os.path.join(output_dir, csv_file)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["iteration", "residual_avg", "residual_max"])
        writer.writerow([iteration, res_avg, res_max])


def plot_convergence(csv_file=None, output_dir=".", show=True):
    """
    Plot convergence history from residual CSV.
    If csv_file is None, use RESIDUAL_CSV.
    """
    if csv_file is None:
        csv_file = RESIDUAL_CSV

    # Prepend output directory if path is relative
    if not os.path.isabs(csv_file):
        csv_file = os.path.join(output_dir, csv_file)

    if not os.path.isfile(csv_file):
        print(f"Warning: residual file not found: {csv_file}")
        return

    # Try to read with different encodings
    encodings = ['utf-8-sig', 'latin-1', 'cp1252']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_file, encoding=enc)
            break
        except (UnicodeDecodeError, pd.errors.EmptyDataError):
            continue
    if df is None:
        print(f"Warning: could not read {csv_file} with any encoding. Skipping plot.")
        return

    if df.empty:
        print("Warning: residual file is empty. Skipping plot.")
        return

    plt.figure(figsize=(7, 5))
    plt.semilogy(df["iteration"], df["residual_avg"],
                 marker="o", markersize=2, linestyle="None", color="tab:blue",
                 label="Average (L2) residual")
    plt.semilogy(df["iteration"], df["residual_max"],
                 marker="x", markersize=2, linestyle="None", color="tab:red",
                 label="Max residual")
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title("Convergence History (Average vs Max)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


def mg_3d_masked(Vtip, phi, boundary_mask, damping=0.8, nu1=2, nu2=2,
                 max_iter=50, tol=1e-6, verbose=True, eps_r=None, eps=True,
                 mg_max_runtime=None, output_dir=".", plotting_enabled=True):
    """Multigrid solver for the 3D Poisson equation with dielectric variation."""
    nx, ny, nz = phi.shape

    def neumann(a):
        """Apply homogeneous Neumann BC by copying the nearest interior plane."""
        a[0,:,:] = a[1,:,:]; a[-1,:,:] = a[-2,:,:]
        a[:,0,:] = a[:,1,:]; a[:,-1,:] = a[:,-2,:]
        a[:,:,0] = a[:,:,1]; a[:,:,-1] = a[:,:,-2]
        return a

    print(f"   Starting 3D MG solver: {nx}x{ny}x{nz} grid")

    def solve_varying_dielectric_3d_zero_rhs(V, eps_cell, omega=1.5,
                                              tol=1e-10, max_iter=20000,
                                              mask=None, verbose=False,
                                              output_dir="."):
        """Optimised SOR solver for div(eps * grad(phi)) = 0."""
        Nx, Ny, Nz = V.shape
        if eps_cell.shape != (Nx-1, Ny-1, Nz-1):
            raise ValueError("eps_cell must have shape (Nx-1,Ny-1,Nz-1).")

        if mask is None:
            mask = np.zeros_like(V, dtype=bool)

        Vfix = V.copy()

        ec = eps_cell.astype(np.float32, copy=False)
        axp = 0.25 * (ec[1:, :-1, :-1] + ec[1:, 1:, :-1] + ec[1:, :-1, 1:] + ec[1:, 1:, 1:])
        axm = 0.25 * (ec[:-1, :-1, :-1] + ec[:-1, 1:, :-1] + ec[:-1, :-1, 1:] + ec[:-1, 1:, 1:])
        ayp = 0.25 * (ec[:-1, 1:, :-1] + ec[1:, 1:, :-1] + ec[:-1, 1:, 1:] + ec[1:, 1:, 1:])
        aym = 0.25 * (ec[:-1, :-1, :-1] + ec[1:, :-1, :-1] + ec[:-1, :-1, 1:] + ec[1:, :-1, 1:])
        azp = 0.25 * (ec[:-1, :-1, 1:] + ec[1:, :-1, 1:] + ec[:-1, 1:, 1:] + ec[1:, 1:, 1:])
        azm = 0.25 * (ec[:-1, :-1, :-1] + ec[1:, :-1, :-1] + ec[:-1, 1:, :-1] + ec[1:, 1:, :-1])

        a0 = axp + axm + ayp + aym + azp + azm

        # ---------- Ensure residual CSV exists with header ----------
        csv_path = os.path.join(output_dir, RESIDUAL_CSV)
        if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["iteration", "residual_avg", "residual_max"])
        # -----------------------------------------------------------

        Ni, Nj, Nk = Nx-2, Ny-2, Nz-2

        num       = np.empty((Ni, Nj, Nk), dtype=np.float32)
        Vnew_core = np.empty((Ni, Nj, Nk), dtype=np.float32)
        R_arr     = np.empty((Ni, Nj, Nk), dtype=np.float32)

        V_int = V[1:-1, 1:-1, 1:-1]
        V_xp = V[2:  , 1:-1, 1:-1]
        V_xm = V[ :-2, 1:-1, 1:-1]
        V_yp = V[1:-1, 2:  , 1:-1]
        V_ym = V[1:-1,  :-2, 1:-1]
        V_zp = V[1:-1, 1:-1, 2:  ]
        V_zm = V[1:-1, 1:-1,  :-2]

        level_start_time = time.time()

        for it in range(1, max_iter + 1):
            np.multiply(axp, V_xp, out=num)
            np.add(num, axm * V_xm, out=num)
            np.add(num, ayp * V_yp, out=num)
            np.add(num, aym * V_ym, out=num)
            np.add(num, azp * V_zp, out=num)
            np.add(num, azm * V_zm, out=num)

            np.divide(num, a0, out=Vnew_core)
            np.subtract(Vnew_core, V_int, out=R_arr)
            np.add(V_int, omega * R_arr, out=V_int)

            neumann(V)
            V[mask] = Vfix[mask]

            elapsed = time.time() - level_start_time
            MG_TIME["elapsed"] = elapsed

            if it % 10 == 0 or it <= 5:
                res, res_max = compute_residual_scalars(
                    V, mask, axp, axm, ayp, aym, azp, azm, a0
                )
                if verbose:
                    print(f"iter {it:6d}: res_avg={res:.5e}, res_max={res_max:.5e}")
                log_residual_csv(it, res, res_max, output_dir=output_dir)

                if res < tol:
                    print(f"Converged in {it} iterations in {elapsed:.2f} s; residual={res:.5e}")
                    plot_convergence(output_dir=output_dir, show=plotting_enabled)
                    full_res = np.full_like(V, np.nan, dtype=np.float32)
                    full_res[1:-1, 1:-1, 1:-1] = np.abs(R_arr)
                    return V, full_res

            if mg_max_runtime is not None and elapsed > mg_max_runtime:
                print(f"MG aborted early: exceeded max runtime ({mg_max_runtime} s). "
                      f"Completed {it} iterations in {elapsed:.2f} s; last residual={res:.5e}")
                plot_convergence(output_dir=output_dir, show=plotting_enabled)
                full_res = np.full_like(V, np.nan, dtype=np.float32)
                full_res[1:-1, 1:-1, 1:-1] = np.abs(R_arr)
                return V, full_res

        # If we exit the loop without converging or hitting runtime limit
        print(f"NOT converged after {max_iter} iterations in {elapsed:.2f} s; last residual={res:.5e}")
        plot_convergence(output_dir=output_dir, show=plotting_enabled)
        full_res = np.full_like(V, np.nan, dtype=np.float32)
        full_res[1:-1, 1:-1, 1:-1] = np.abs(R_arr)
        return V, full_res

    # ---- End of inner solver definition ----

    if eps:
        return solve_varying_dielectric_3d_zero_rhs(
            phi, eps_r, damping, tol, max_iter, boundary_mask, verbose,
            output_dir=output_dir
        )
