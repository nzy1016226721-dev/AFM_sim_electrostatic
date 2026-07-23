import numpy as np
import time
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

MG_TIME = {"elapsed": 0.0}
RESIDUAL_CSV = "residual_history.csv"


def build_downward_pointing_tip(nx, ny, nz, tip_z=0.2, R=0.05, r_tip=0.15,
                                aspect_ratio=2.0, verbose=True):
    """Generate a boolean mask for a hyperbolic AFM tip geometry.

    The tip is modelled as a hyperboloid pointing downward along z,
    centred at (0.5, 0.5) in xy, with curvature radius R, truncation
    radius r_tip, and vertical aspect ratio.

    Parameters
    ----------
    nx : int
        Grid points in x.
    ny : int
        Grid points in y.
    nz : int
        Grid points in z.
    tip_z : float, optional
        Fractional z-coordinate of the tip apex (default: 0.2).
    R : float, optional
        Curvature radius (fractional, default: 0.05).
    r_tip : float, optional
        Tip truncation radius (fractional, default: 0.15).
    aspect_ratio : float, optional
        Height/radius aspect ratio (default: 2.0).
    verbose : bool, optional
        If True, print geometry info (default: True).

    Returns
    -------
    mask : np.ndarray (bool)
        3D boolean mask where True indicates tip voxels.
    z_tip : float
        Fractional z-coordinate of the tip apex.
    z_base : float
        Fractional z-coordinate of the tip base.
    """

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    cx, cy = 0.5, 0.5
    tip_z_idx = int(tip_z * (nz - 1))
    z_tip = z[tip_z_idx]
    mask = np.zeros((nx, ny, nz), dtype=bool)

    theta_asym = np.arctan(aspect_ratio)
    a = R * np.tan(theta_asym)
    b = R * np.tan(theta_asym) ** 2

    z0 = z_tip - b
    z_base = z0 + np.sqrt(b**2 * (1 + (r_tip**2 / a**2)))

    if verbose:
        print(f"\n[Hyperbolic AFM Tip]")
        print(f"  Aspect ratio (tan{chr(952)}) = {aspect_ratio:.3f}, {chr(952)}_asym = {np.degrees(theta_asym):.2f}{chr(176)}")
        print(f"  a = {a:.6f}, b = {b:.6f}")
        print(f"  Curvature radius at tip R = {R:.4f}")
        print(f"  Tip_z = {z_tip:.3f}, Base_z = {z_base:.3f}, r_tip = {r_tip:.3f}")

    for k in range(nz):
        zk = z[k]
        if zk < z_tip or zk > z_base:
            continue
        dz = zk - z0
        if dz**2 < b**2:
            continue
        r_max = a * np.sqrt((dz**2 / b**2) - 1)
        r_max = min(r_max, r_tip)
        for j in range(ny):
            for i in range(nx):
                r = np.sqrt((x[i] - cx)**2 + (y[j] - cy)**2)
                if r <= r_max:
                    mask[i, j, k] = True

    if verbose:
        print(f"  Tip_z = {z_tip:.3f}, Base_z = {z_base:.3f}, Total points: {np.sum(mask)}")

    return mask, z_tip, z_base


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

    res_matrix = np.full_like(V, np.nan, dtype=float)
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


def log_residual_csv(iteration, res_avg, res_max, csv_file=RESIDUAL_CSV, output_dir="."):
    """Append one row to the residual convergence CSV log.

    Parameters
    ----------
    iteration : int
        Solver iteration number.
    res_avg : float
        Average residual value.
    res_max : float
        Maximum residual value.
    csv_file : str, optional
        CSV filename (default: RESIDUAL_CSV).
    output_dir : str, optional
        Output directory (default: ".").

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, csv_file)
    file_exists = os.path.isfile(path)
    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["iteration", "residual_avg", "residual_max"])
        writer.writerow([iteration, res_avg, res_max])


def plot_convergence(csv_file=RESIDUAL_CSV, output_dir="."):
    """Plot the convergence history from a residual CSV log.

    Parameters
    ----------
    csv_file : str, optional
        Residual CSV filename (default: RESIDUAL_CSV).
    output_dir : str, optional
        Directory containing the CSV (default: ".").

    Returns
    -------
    None
    """
    csv_file = os.path.join(output_dir, csv_file) if output_dir else csv_file
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"No residual history file found: {csv_file}")

    df = pd.read_csv(csv_file)

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
    plt.show()


def mg_3d_masked(Vtip, phi, boundary_mask, damping=0.8, nu1=2, nu2=2,
                 max_iter=50, tol=1e-6, verbose=True, eps_r=None, eps=True,
                 mg_max_runtime=None, output_dir="."):
    """Multigrid solver for the 3D Poisson equation with dielectric variation.

    Uses successive over-relaxation (SOR) with Neumann boundary conditions
    on free faces and Dirichlet on tip/gate boundaries. Implements two
    inner solver variants (legacy and optimised).

    Parameters
    ----------
    Vtip : float
        Tip voltage (V).
    phi : np.ndarray
        Initial potential array (Nx x Ny x Nz).
    boundary_mask : np.ndarray (bool)
        True at Dirichlet boundary nodes.
    damping : float, optional
        SOR damping factor / omega (default: 0.8).
    nu1 : int, optional
        Pre-smoothing steps (unused in current code, default: 2).
    nu2 : int, optional
        Post-smoothing steps (unused, default: 2).
    max_iter : int, optional
        Maximum solver iterations (default: 50).
    tol : float, optional
        Convergence tolerance on L2 residual (default: 1e-6).
    verbose : bool, optional
        If True, print iteration info (default: True).
    eps_r : np.ndarray or None, optional
        Dielectric constant cell array (Nx-1 x Ny-1 x Nz-1).
    eps : bool, optional
        If True, solve with dielectric (default: True).
    mg_max_runtime : float or None, optional
        Max wall-clock time in seconds (default: None).
    output_dir : str, optional
        Directory for log files (default: ".").

    Returns
    -------
    phi : np.ndarray
        Solved potential array.
    residual : np.ndarray
        Final residual matrix (NaN at boundaries).
    """
    nx, ny, nz = phi.shape

    def neumann(a):
        """Apply homogeneous Neumann BC by copying the nearest interior plane."""
        a[0,:,:] = a[1,:,:]; a[-1,:,:] = a[-2,:,:]
        a[:,0,:] = a[:,1,:]; a[:,-1,:] = a[:,-2,:]
        a[:,:,0] = a[:,:,1]; a[:,:,-1] = a[:,:,-2]
        return a

    print(f"   Starting 3D MG solver: {nx}x{ny}x{nz} grid")

    def solve_varying_dielectric_3d_zero_rhs_old(V, eps_cell, omega=1.5,
                                                  tol=1e-10, max_iter=20000,
                                                  mask=None, verbose=False,
                                                  output_dir="."):
        """(Legacy) SOR solver for div(eps * grad(phi)) = 0.

        Uses per-iteration residual logging and safety checks. Replaced by
        the optimised version below but retained for reference.

        Parameters
        ----------
        V : np.ndarray
            Potential array (will be modified in-place).
        eps_cell : np.ndarray
            Dielectric on cells (Nx-1 x Ny-1 x Nz-1).
        omega : float, optional
            SOR over-relaxation factor (default: 1.5).
        tol : float, optional
            Convergence tolerance (default: 1e-10).
        max_iter : int, optional
            Maximum iterations (default: 20000).
        mask : np.ndarray (bool) or None, optional
            Dirichlet mask (default: None = all free).
        verbose : bool, optional
            If True, print progress (default: False).
        output_dir : str, optional
            Log directory (default: ".").

        Returns
        -------
        V : np.ndarray
            Converged potential.
        res_matrix : np.ndarray
            Final residual per node.
        """
        Nx, Ny, Nz = V.shape
        if eps_cell.shape != (Nx-1, Ny-1, Nz-1):
            raise ValueError("eps_cell must have shape (Nx-1,Ny-1,Nz-1).")

        if mask is None:
            mask = np.zeros_like(V, dtype=bool)

        Vfix = V.copy()

        ec = eps_cell
        axp = 0.25*(ec[1:, :-1, :-1] + ec[1:, 1:, :-1] + ec[1:, :-1, 1:] + ec[1:, 1:, 1:])
        axm = 0.25*(ec[:-1, :-1, :-1] + ec[:-1, 1:, :-1] + ec[:-1, :-1, 1:] + ec[:-1, 1:, 1:])
        ayp = 0.25*(ec[:-1, 1:, :-1] + ec[1:, 1:, :-1] + ec[:-1, 1:, 1:] + ec[1:, 1:, 1:])
        aym = 0.25*(ec[:-1, :-1, :-1] + ec[1:, :-1, :-1] + ec[:-1, :-1, 1:] + ec[1:, :-1, 1:])
        azp = 0.25*(ec[:-1, :-1, 1:] + ec[1:, :-1, 1:] + ec[:-1, 1:, 1:] + ec[1:, 1:, 1:])
        azm = 0.25*(ec[:-1, :-1, :-1] + ec[1:, :-1, :-1] + ec[:-1, 1:, :-1] + ec[1:, 1:, :-1])

        def pad3(arr):
            out = np.zeros((Nx,Ny,Nz))
            out[1:-1,1:-1,1:-1] = arr
            return out

        axp, axm, ayp, aym, azp, azm = map(pad3, (axp, axm, ayp, aym, azp, azm))
        a0 = axp + axm + ayp + aym + azp + azm

        for it in range(1, max_iter+1):
            if it == 1:
                level_start_time = time.time()

            num = (
                axp[1:-1,1:-1,1:-1] * V[2:  ,1:-1,1:-1] +
                axm[1:-1,1:-1,1:-1] * V[ :-2,1:-1,1:-1] +
                ayp[1:-1,1:-1,1:-1] * V[1:-1,2:  ,1:-1] +
                aym[1:-1,1:-1,1:-1] * V[1:-1, :-2,1:-1] +
                azp[1:-1,1:-1,1:-1] * V[1:-1,1:-1,2:  ] +
                azm[1:-1,1:-1,1:-1] * V[1:-1,1:-1, :-2]
            )

            denom = a0[1:-1,1:-1,1:-1]

            Vnew_core = num / denom
            R = Vnew_core - V[1:-1,1:-1,1:-1]
            V[1:-1,1:-1,1:-1] += omega*R
            V = neumann(V)
            V[mask] = Vfix[mask]

            if (it % 10 == 0 or it <= 5):
                res, res_max, res_matrix = compute_residual_vec_unpadded(V, mask, axp[1:-1,1:-1,1:-1], axm[1:-1,1:-1,1:-1], ayp[1:-1,1:-1,1:-1], aym[1:-1,1:-1,1:-1], azp[1:-1,1:-1,1:-1], azm[1:-1,1:-1,1:-1], a0[1:-1,1:-1,1:-1])
                print(f"iter {it:6d}: res_avg={res:.5e}, res_max={res_max:.5e}")
                log_residual_csv(it, res, res_max, output_dir=output_dir)

            elapsed = time.time() - level_start_time
            MG_TIME["elapsed"] = elapsed

            if res < tol:
                if verbose:
                    print(f"Converged in {it} iterations; residual={res:.5e}")
                plot_convergence(output_dir=output_dir)
                return V, res_matrix

            if (mg_max_runtime is not None) and (elapsed > mg_max_runtime):
                print(f"MG aborted early: exceeded max runtime ({mg_max_runtime} s). "
                      f"Completed {it} iterations.")
                plot_convergence(output_dir=output_dir)
                return V, res_matrix

        print(f"NOT converged after {max_iter} iterations; last residual={res:.5e}")
        plot_convergence(output_dir=output_dir)
        return V, res_matrix

    def solve_varying_dielectric_3d_zero_rhs(V, eps_cell, omega=1.5,
                                              tol=1e-10, max_iter=20000,
                                              mask=None, verbose=False,
                                              output_dir="."):
        """(Optimised) SOR solver for div(eps * grad(phi)) = 0.

        Uses pre-allocated buffers and in-place operations for performance.
        Same interface as the legacy version.

        Parameters
        ----------
        V : np.ndarray
            Potential array (modified in-place).
        eps_cell : np.ndarray
            Dielectric on cells (Nx-1 x Ny-1 x Nz-1).
        omega : float, optional
            Over-relaxation factor (default: 1.5).
        tol : float, optional
            Convergence tolerance (default: 1e-10).
        max_iter : int, optional
            Maximum iterations (default: 20000).
        mask : np.ndarray (bool) or None, optional
            Dirichlet mask (default: None).
        verbose : bool, optional
            If True, print progress (default: False).
        output_dir : str, optional
            Log directory (default: ".").

        Returns
        -------
        V : np.ndarray
            Converged potential.
        res_matrix : np.ndarray
            Final residual per node.
        """
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

        for it in range(1, max_iter+1):
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
                res, res_max, res_matrix = compute_residual_vec_unpadded(
                    V, mask, axp, axm, ayp, aym, azp, azm, a0
                )
                if verbose:
                    print(f"iter {it:6d}: res_avg={res:.5e}, res_max={res_max:.5e}")
                    log_residual_csv(it, res, res_max, output_dir=output_dir)

                if res < tol:
                    if verbose:
                        print(f"Converged in {it} iterations; residual={res:.5e}")
                    plot_convergence(output_dir=output_dir)
                    return V, res_matrix

            if mg_max_runtime is not None and elapsed > mg_max_runtime:
                print(f"MG aborted early: exceeded max runtime ({mg_max_runtime} s). "
                      f"Completed {it} iterations.")
                plot_convergence(output_dir=output_dir)
                full_res = np.full_like(V, np.nan, dtype=np.float32)
                full_res[1:-1,1:-1,1:-1] = np.abs(R_arr)
                return V, full_res

        print(f"NOT converged after {max_iter} iterations; last residual={res:.5e}")
        plot_convergence(output_dir=output_dir)
        full_res = np.full_like(V, np.nan, dtype=np.float32)
        full_res[1:-1,1:-1,1:-1] = np.abs(R_arr)
        return V, full_res

    if eps:
        return solve_varying_dielectric_3d_zero_rhs(
            phi, eps_r, damping, tol, max_iter, boundary_mask, verbose,
            output_dir=output_dir
        )
