import numpy as np
import time
import os
import csv
import matplotlib.pyplot as plt

from .parallel import (
    weighted_jacobi_residual_parallel,
    weighted_jacobi_update_faces_parallel,
    weighted_jacobi_update_faces_parallel_float64,
    residual_scalars_parallel,
    residual_scalars_faces_parallel,
    residual_scalars_faces_parallel_float64,
    configure_cpu_threads,
)
from .numerics import (
    pyramid_tip_corner_radius,
    pyramid_tip_half_side,
    pyramid_tip_inside,
    resolve_tip_shape,
)

MG_TIME = {"elapsed": 0.0}
RESIDUAL_CSV = "residual_history.csv"


def _average_four_float32(a, b, c, d):
    """Return the float32 mean of four same-shaped material-cell views.

    Chained NumPy expressions leave several full-core temporaries alive while
    building every dielectric face coefficient.  Accumulating into the final
    array uses the same left-to-right float32 arithmetic and materially lowers
    the coefficient-construction peak memory.
    """
    out = np.empty(a.shape, dtype=np.float32)
    np.add(a, b, out=out)
    np.add(out, c, out=out)
    np.add(out, d, out=out)
    np.multiply(out, np.float32(0.25), out=out)
    return out


def _average_four_float64(a, b, c, d):
    """Return the float64 mean of four material-cell views in bounded memory."""
    out = np.empty(a.shape, dtype=np.float64)
    np.add(a, b, out=out)
    np.add(out, c, out=out)
    np.add(out, d, out=out)
    np.multiply(out, np.float64(0.25), out=out)
    return out


def _build_dielectric_face_fields(eps_cell, dtype=np.float32):
    """Build one face field for each axis from cell-centered epsilon.

    The positive and negative coefficients for neighboring interior nodes are
    shifted views of the same oriented face field.  Retaining those three
    fields instead of six duplicated core arrays cuts stencil-material storage
    nearly in half without changing any epsilon values or arithmetic order.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dielectric face fields require float32 or float64")
    ec = np.asarray(eps_cell, dtype=dtype)
    if ec.ndim != 3 or min(ec.shape) < 2:
        raise ValueError("eps_cell must be a 3-D array with at least two cells per axis")

    average_four = (
        _average_four_float32 if dtype == np.dtype(np.float32)
        else _average_four_float64
    )

    # x faces: shape (Nx-1, Ny-2, Nz-2)
    x_faces = average_four(
        ec[:, :-1, :-1], ec[:, 1:, :-1],
        ec[:, :-1, 1:], ec[:, 1:, 1:],
    )
    # y faces: shape (Nx-2, Ny-1, Nz-2)
    y_faces = average_four(
        ec[:-1, :, :-1], ec[1:, :, :-1],
        ec[:-1, :, 1:], ec[1:, :, 1:],
    )
    # z faces: shape (Nx-2, Ny-2, Nz-1)
    z_faces = average_four(
        ec[:-1, :-1, :], ec[1:, :-1, :],
        ec[:-1, 1:, :], ec[1:, 1:, :],
    )
    return x_faces, y_faces, z_faces


def build_downward_pointing_tip(nx, ny, nz, tip_z=0.2, R=0.05, r_tip=0.15,
                                aspect_ratio=2.0, verbose=True,
                                tip_z_nm=None, R_nm=None, r_tip_nm=None,
                                domain_nm=None, center_fraction=(0.5, 0.5),
                                tip_shape="pyramid"):
    """Generate a boolean mask for a downward-pointing AFM tip geometry.

    The default ``"pyramid"`` cross-section is a square with rounded vertical
    edges: each side plane is tangent to the legacy cone along its centre
    generator, so the square circumscribes the cone disk and the configured
    aspect ratio still measures the opening angle at every side-plane centre.
    The ``"cone"`` shape preserves the legacy circular cross-section exactly.

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
    tip_shape : str, optional
        ``"pyramid"`` (default) builds a square cross-section with vertical
        edges filleted at the apex curvature radius ``R``; near the apex the
        section stays circular.  ``"cone"`` keeps the legacy circular disk.
        Here ``r_tip``/``r_tip_nm`` is the distance from the tip axis to one
        side of the square base edge.

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
    shape = resolve_tip_shape(tip_shape)

    if verbose:
        print("\n[Hyperbolic AFM Tip]")
        print(f"  Tip cross-section shape = {shape}")
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
            X, Y = np.meshgrid(x - cx, y - cy, indexing="ij")
            if shape == "cone":
                mask[:, :, k] = np.sqrt(X**2 + Y**2) <= min(r_max, r_tip_phys)
            else:
                half = pyramid_tip_half_side(r_max, r_tip_phys)
                mask[:, :, k] = pyramid_tip_inside(
                    X, Y, half, pyramid_tip_corner_radius(half, R_phys)
                )
        else:
            if zk < z_tip_frac or zk > z_base_frac:
                continue
            dz = zk - z0
            if dz**2 < b**2:
                continue
            r_max = a * np.sqrt((dz**2 / b**2) - 1)
            X, Y = np.meshgrid(x - cx, y - cy, indexing="ij")
            if shape == "cone":
                mask[:, :, k] = np.sqrt(X**2 + Y**2) <= min(r_max, r_tip)
            else:
                half = pyramid_tip_half_side(r_max, r_tip)
                mask[:, :, k] = pyramid_tip_inside(
                    X, Y, half, pyramid_tip_corner_radius(half, R)
                )

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
    if not show:
        # Headless/batch runs still call this helper so the solver can share
        # its convergence-exit paths. Do not allocate a Matplotlib figure
        # when plotting is disabled; callers may invoke the solver directly
        # without a surrounding ``plt.close('all')`` cleanup.
        return

    if csv_file is None:
        csv_file = RESIDUAL_CSV

    # Prepend output directory if path is relative
    if not os.path.isabs(csv_file):
        csv_file = os.path.join(output_dir, csv_file)

    if not os.path.isfile(csv_file):
        print(f"Warning: residual file not found: {csv_file}")
        return

    # Read with the standard library so headless solver jobs do not require
    # pandas merely to render an optional diagnostic plot.
    encodings = ['utf-8-sig', 'latin-1', 'cp1252']
    rows = None
    for enc in encodings:
        try:
            with open(csv_file, newline='', encoding=enc) as handle:
                rows = list(csv.DictReader(handle))
            break
        except UnicodeDecodeError:
            continue
    if rows is None:
        print(f"Warning: could not read {csv_file} with any encoding. Skipping plot.")
        return
    if not rows:
        print("Warning: residual file is empty. Skipping plot.")
        return

    iterations = []
    residual_avg = []
    residual_max = []
    for row in rows:
        try:
            iterations.append(int(row["iteration"]))
            residual_avg.append(float(row["residual_avg"]))
            residual_max.append(float(row["residual_max"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not iterations:
        print("Warning: residual file has no valid rows. Skipping plot.")
        return

    plt.figure(figsize=(7, 5))
    plt.semilogy(iterations, residual_avg,
                 marker="o", markersize=2, linestyle="None", color="tab:blue",
                 label="Average (L2) residual")
    plt.semilogy(iterations, residual_max,
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
                 max_iter=None, tol=1e-6, verbose=True, eps_r=None, eps=True,
                 mg_max_runtime=None, output_dir=".", plotting_enabled=True,
                 cpu_threads=1, return_residual=True,
                 diagnostic_iterations=None):
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
                                               tol=1e-10, max_iter=None,
                                              mask=None, verbose=False,
                                              output_dir=".", cpu_threads_requested=1,
                                              return_residual=True,
                                              diagnostic_iteration_limit=None):
        """Optimised SOR solver for div(eps * grad(phi)) = 0."""
        Nx, Ny, Nz = V.shape
        if eps_cell.shape != (Nx-1, Ny-1, Nz-1):
            raise ValueError("eps_cell must have shape (Nx-1,Ny-1,Nz-1).")

        if mask is None:
            mask = np.zeros_like(V, dtype=bool)

        # Only constrained values need to survive an iteration.  A full V
        # snapshot is 4 GiB at 1024^3, whereas normal gates/tips constrain a
        # small subset of nodes.  Boolean indexing order is stable because the
        # mask is immutable throughout this solve.
        fixed_values = V[mask].copy()

        x_faces, y_faces, z_faces = _build_dielectric_face_fields(
            eps_cell, dtype=V.dtype
        )
        # +/- coefficients are shifted views, not separate arrays.
        axp, axm = x_faces[1:, :, :], x_faces[:-1, :, :]
        ayp, aym = y_faces[:, 1:, :], y_faces[:, :-1, :]
        azp, azm = z_faces[:, :, 1:], z_faces[:, :, :-1]

        # ---------- Ensure residual CSV exists with header ----------
        csv_path = os.path.join(output_dir, RESIDUAL_CSV)
        if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["iteration", "residual_avg", "residual_max"])
        # -----------------------------------------------------------

        Ni, Nj, Nk = Nx-2, Ny-2, Nz-2

        # The config controls CPU parallelism. On Slurm, never exceed the CPU
        # allocation actually requested by the job. cpu_threads=1 uses the
        # original NumPy update path for a bit-for-bit reference mode; values
        # greater than 1 use the fused Numba CPU-parallel stencil.
        cpu_threads = configure_cpu_threads(cpu_threads_requested)
        low_memory_parallel = cpu_threads > 1 and not return_residual

        # The strict NumPy reference path and the legacy residual-volume path
        # retain a0.  Normal non-plotting parallel runs recompute it in the
        # fused kernel, saving one full float32 core array at the largest level.
        a0 = None
        if not low_memory_parallel:
            a0 = np.empty_like(axp)
            np.add(axp, axm, out=a0)
            np.add(a0, ayp, out=a0)
            np.add(a0, aym, out=a0)
            np.add(a0, azp, out=a0)
            np.add(a0, azm, out=a0)

        # The serial path uses Vnew_core for its vectorized numerator and then
        # reuses it as its damped-update scratch.  The parallel path writes
        # residuals directly, so it needs no equivalent full core workspace.
        Vnew_core = (
            np.empty((Ni, Nj, Nk), dtype=V.dtype)
            if cpu_threads == 1 else None
        )
        R_arr = (
            np.empty((Ni, Nj, Nk), dtype=V.dtype)
            if not low_memory_parallel else None
        )
        # The parallel Jacobi path needs a full-field snapshot. Keep it out of
        # the serial reference path so cpu_threads=1 does not pay an avoidable
        # extra allocation at 512^3 (about 0.5 GiB for float32).
        V_old = np.empty_like(V) if cpu_threads > 1 else None
        if verbose:
            print(f"   CPU solver threads: {cpu_threads}")

        V_int = V[1:-1, 1:-1, 1:-1]
        if cpu_threads == 1:
            V_xp = V[2:  , 1:-1, 1:-1]
            V_xm = V[ :-2, 1:-1, 1:-1]
            V_yp = V[1:-1, 2:  , 1:-1]
            V_ym = V[1:-1,  :-2, 1:-1]
            V_zp = V[1:-1, 1:-1, 2:  ]
            V_zm = V[1:-1, 1:-1,  :-2]
        elif not low_memory_parallel:
            # V_old is fully refreshed before the next stencil pass, so after
            # that pass its interior is safe temporary storage for omega * R.
            V_old_int = V_old[1:-1, 1:-1, 1:-1]

        level_start_time = time.time()
        if max_iter is not None and verbose:
            print("   Ignoring legacy max_iter: solver stopping is tolerance/time based.")
        if diagnostic_iteration_limit is not None:
            diagnostic_iteration_limit = int(diagnostic_iteration_limit)
            if diagnostic_iteration_limit < 1:
                raise ValueError("diagnostic_iterations must be positive")
            print(
                "   Diagnostic-only fixed iteration count: "
                f"{diagnostic_iteration_limit}"
            )

        # A fixed iteration count makes runtime and convergence depend on the
        # process decomposition rather than the physical stopping criterion.
        # Retain ``max_iter`` as a compatibility argument, but intentionally do
        # not use it as a termination condition. A configured mg_max_runtime
        # (or the Slurm wall time) is the only non-convergence cap.
        it = 0
        while True:
            it += 1
            if cpu_threads == 1:
                # Original operation order retained intentionally for reference
                # and compatibility testing.
                np.multiply(axp, V_xp, out=Vnew_core)
                np.add(Vnew_core, axm * V_xm, out=Vnew_core)
                np.add(Vnew_core, ayp * V_yp, out=Vnew_core)
                np.add(Vnew_core, aym * V_ym, out=Vnew_core)
                np.add(Vnew_core, azp * V_zp, out=Vnew_core)
                np.add(Vnew_core, azm * V_zm, out=Vnew_core)
                np.divide(Vnew_core, a0, out=Vnew_core)
                np.subtract(Vnew_core, V_int, out=R_arr)
                np.multiply(R_arr, V.dtype.type(omega), out=Vnew_core)
                np.add(V_int, Vnew_core, out=V_int)
            elif not low_memory_parallel:
                # The parallel stencil is Jacobi, so every point in this
                # iteration must read the same pre-update solution. Copy the
                # full field once, compute into work arrays, then relax V in
                # a separate phase after the kernel returns. Updating V from
                # inside the prange stencil would race with neighboring reads.
                np.copyto(V_old, V)
                weighted_jacobi_residual_parallel(
                    V_old, axp, axm, ayp, aym, azp, azm, a0, R_arr
                )
                np.multiply(R_arr, V.dtype.type(omega), out=V_old_int)
                np.add(V_int, V_old_int, out=V_int)
            else:
                # Two-buffer Jacobi: every output point reads only ``V`` and
                # is written once into ``V_old``.  Swapping the buffers removes
                # both the full-field copy and the full residual workspace.
                if V.dtype == np.float64:
                    weighted_jacobi_update_faces_parallel_float64(
                        V,
                        V_old,
                        axp, axm, ayp, aym, azp, azm,
                        np.float64(omega),
                    )
                else:
                    weighted_jacobi_update_faces_parallel(
                        V,
                        V_old,
                        axp, axm, ayp, aym, azp, azm,
                        np.float32(omega),
                    )
                neumann(V_old)
                V_old[mask] = fixed_values
                V, V_old = V_old, V

            if not low_memory_parallel:
                neumann(V)
                V[mask] = fixed_values

            elapsed = time.time() - level_start_time
            MG_TIME["elapsed"] = elapsed

            if it % 10 == 0 or it <= 5:
                if cpu_threads == 1:
                    res, res_max = compute_residual_scalars(
                        V, mask, axp, axm, ayp, aym, azp, azm, a0
                    )
                elif not low_memory_parallel:
                    res, res_max = residual_scalars_parallel(
                        V, mask, axp, axm, ayp, aym, azp, azm, a0
                    )
                else:
                    if V.dtype == np.float64:
                        res, res_max = residual_scalars_faces_parallel_float64(
                            V, mask, axp, axm, ayp, aym, azp, azm
                        )
                    else:
                        res, res_max = residual_scalars_faces_parallel(
                            V, mask, axp, axm, ayp, aym, azp, azm
                        )
                if verbose:
                    print(f"iter {it:6d}: res_avg={res:.5e}, res_max={res_max:.5e}")
                log_residual_csv(it, res, res_max, output_dir=output_dir)

                MG_TIME.update(
                    iterations=it,
                    residual=float(res),
                    residual_max=float(res_max),
                    reason="running",
                )

                if diagnostic_iteration_limit is None and res < tol:
                    print(f"Converged in {it} iterations in {elapsed:.2f} s; residual={res:.5e}")
                    MG_TIME["reason"] = "converged"
                    plot_convergence(output_dir=output_dir, show=plotting_enabled)
                    if not return_residual:
                        return V, None
                    full_res = np.full_like(V, np.nan, dtype=V.dtype)
                    np.abs(R_arr, out=full_res[1:-1, 1:-1, 1:-1])
                    return V, full_res

            if (
                diagnostic_iteration_limit is not None
                and it >= diagnostic_iteration_limit
            ):
                if not (it % 10 == 0 or it <= 5):
                    if cpu_threads == 1:
                        res, res_max = compute_residual_scalars(
                            V, mask, axp, axm, ayp, aym, azp, azm, a0
                        )
                    elif not low_memory_parallel:
                        res, res_max = residual_scalars_parallel(
                            V, mask, axp, axm, ayp, aym, azp, azm, a0
                        )
                    elif V.dtype == np.float64:
                        res, res_max = residual_scalars_faces_parallel_float64(
                            V, mask, axp, axm, ayp, aym, azp, azm
                        )
                    else:
                        res, res_max = residual_scalars_faces_parallel(
                            V, mask, axp, axm, ayp, aym, azp, azm
                        )
                    log_residual_csv(it, res, res_max, output_dir=output_dir)
                MG_TIME.update(
                    elapsed=elapsed,
                    iterations=it,
                    residual=float(res),
                    residual_max=float(res_max),
                    reason="diagnostic_iteration_limit",
                )
                print(
                    "Diagnostic iteration limit reached: "
                    f"{it} iterations in {elapsed:.2f} s; residual={res:.5e}"
                )
                return V, None

            if mg_max_runtime is not None and elapsed > mg_max_runtime:
                print(f"MG aborted early: exceeded max runtime ({mg_max_runtime} s). "
                      f"Completed {it} iterations in {elapsed:.2f} s; last residual={res:.5e}")
                plot_convergence(output_dir=output_dir, show=plotting_enabled)
                MG_TIME.update(
                    elapsed=elapsed,
                    iterations=it,
                    residual=float(res),
                    residual_max=float(res_max),
                    reason="time_limit",
                )
                if not return_residual:
                    return V, None
                full_res = np.full_like(V, np.nan, dtype=V.dtype)
                np.abs(R_arr, out=full_res[1:-1, 1:-1, 1:-1])
                return V, full_res

    # ---- End of inner solver definition ----

    if eps:
        return solve_varying_dielectric_3d_zero_rhs(
            phi, eps_r, damping, tol, max_iter, boundary_mask, verbose,
            output_dir=output_dir, cpu_threads_requested=cpu_threads,
            return_residual=return_residual,
            diagnostic_iteration_limit=diagnostic_iterations,
        )
