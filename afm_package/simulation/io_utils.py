import numpy as np
import csv
import os
import re


RESIDUAL_CSV = "residual_history.csv"


def save_for_qtcad(results, filename="afm_potential.npy", output_dir="."):
    """Save the potential array to a .npy file for QTCAD compatibility.

    Parameters
    ----------
    results : dict
        Simulation results dict containing 'phi'.
    filename : str, optional
        Output filename (default: "afm_potential.npy").
    output_dir : str, optional
        Output directory (default: ".").

    Returns
    -------
    None
    """
    phi = results['phi']
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    np.save(path, phi)
    nx, ny, nz = phi.shape
    print(f"Saved potential to {path}")
    print(f"Grid dimensions: nx={nx}, ny={ny}, nz={nz}")


def save_phi_3d(phi, results, tag="", output_dir="."):
    """Save the full 3D potential plus fields as a compressed NPZ.

    Parameters
    ----------
    phi : np.ndarray
        3D potential array.
    results : dict
        Simulation results containing 'Ex', 'Ey', 'Ez', 'parameters'.
    tag : str, optional
        Tag appended to filename (default: "").
    output_dir : str, optional
        Output directory (default: ".").

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"phi_3d_{tag}.npz")
    np.savez_compressed(
        filename,
        phi=phi,
        Ex=results.get("Ex"),
        Ey=results.get("Ey"),
        Ez=results.get("Ez"),
        params=results.get("parameters", {}),
    )
    print(f"Saved 3D phi to {filename}")


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


def log_timing(level, nx, ny, nz, elapsed, logfile="mg_timing_log.csv", output_dir="."):
    """Log the elapsed time for one MG level to a CSV file.

    Parameters
    ----------
    level : int
        Multigrid refinement level.
    nx, ny, nz : int
        Grid dimensions at this level.
    elapsed : float
        Wall-clock time in seconds.
    logfile : str, optional
        Log filename (default: "mg_timing_log.csv").
    output_dir : str, optional
        Output directory (default: ".").

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, logfile)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["level", "Nx", "Ny", "Nz", "time_sec"])
        w.writerow([level, nx, ny, nz, f"{elapsed:.6f}"])


def log_joule_csv(config_idx, V, P_total, is_zoom=False, csv_file="joule_power_zoom.csv", output_dir="."):
    """Log Joule heating results to a CSV file.

    Parameters
    ----------
    config_idx : int
        Configuration index.
    V : float
        Tip voltage (V).
    P_total : float
        Total Joule power (W).
    is_zoom : bool, optional
        Whether this is a zoom simulation result (default: False).
    csv_file : str, optional
        CSV filename (default: "joule_power_zoom.csv").
    output_dir : str, optional
        Output directory (default: ".").

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, csv_file)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["config", "Vtip", "P_total_zoom (W)"])
        writer.writerow([config_idx, f"{V:.2f}", f"{P_total:.6e}"])


def get_sorted_config_files(base_name, directory="."):
    """Get a numerically sorted list of config files matching base_name_*.json.

    Parameters
    ----------
    base_name : str
        Config base name (e.g. "afm_config").
    directory : str, optional
        Directory to search (default: ".").

    Returns
    -------
    list of str
        Sorted file paths.
    """
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)\.json$")
    files = []
    for fname in os.listdir(directory):
        match = pattern.match(fname)
        if match:
            num = int(match.group(1))
            files.append((num, os.path.join(directory, fname)))
    files.sort(key=lambda x: x[0])
    return [f[1] for f in files]


def _clamp01(a):
    """Clamp a value to the [0, 1] range."""
    return max(0.0, min(1.0, float(a)))


def _range_to_indices(lo, hi, N):
    """Convert a fractional range to integer grid indices.

    Parameters
    ----------
    lo : float
        Lower bound (fractional 0-1).
    hi : float
        Upper bound (fractional 0-1).
    N : int
        Number of grid points along this axis.

    Returns
    -------
    (int, int)
        Start and end indices (clamped to [0, N-1]).
    """
    lo = _clamp01(lo); hi = _clamp01(hi)
    if hi < lo: lo, hi = hi, lo
    i0 = int(np.floor(lo * (N - 1)))
    i1 = int(np.ceil (hi * (N - 1)))
    i0 = max(0, min(N-1, i0))
    i1 = max(0, min(N-1, i1))
    return i0, i1


def make_gate_mask(nx, ny, nz, gate):
    """Create a boolean 3D mask for a gate region.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.
    gate : dict
        Gate dict with 'x_range', 'y_range', 'z_range' (fractional).

    Returns
    -------
    np.ndarray (bool)
        Mask of shape (nx, ny, nz), True inside the gate.
    """
    x0, x1 = gate["x_range"]
    y0, y1 = gate["y_range"]
    z0, z1 = gate["z_range"]
    ix0, ix1 = _range_to_indices(x0, x1, nx)
    iy0, iy1 = _range_to_indices(y0, y1, ny)
    iz0, iz1 = _range_to_indices(z0, z1, nz)
    mask = np.zeros((nx, ny, nz), dtype=bool)
    mask[ix0:ix1+1, iy0:iy1+1, iz0:iz1+1] = True
    return mask
