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


def _unique_output_path(output_dir, filename):
    """Return a non-colliding output path in ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(filename)
    n = 1
    while True:
        candidate = os.path.join(output_dir, f"{stem} ({n}){ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1


def save_potential_full(phi, filename="afm_potential.npy", output_dir="."):
    """Save one complete 3-D potential array as a NumPy ``.npy`` file."""
    path = _unique_output_path(output_dir, filename)
    np.save(path, np.asarray(phi))
    print(f"Saved full potential to {path}")
    return path


def _normalize_cut_offsets_nm(offsets_nm):
    """Normalize a six-value relative cut specification to three axis pairs.

    Accepted forms are ``[xmin, xmax, ymin, ymax, zmin, zmax]`` or
    ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]``.  Values are physical
    offsets in nm relative to the current movement centre.
    """
    try:
        vals = list(offsets_nm)
    except TypeError as exc:
        raise ValueError("save_cut_box_nm must contain six numeric values") from exc

    if len(vals) == 6 and not any(isinstance(v, (list, tuple, np.ndarray)) for v in vals):
        pairs = [(vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5])]
    elif len(vals) == 3:
        try:
            pairs = [(v[0], v[1]) for v in vals]
        except (TypeError, IndexError) as exc:
            raise ValueError(
                "save_cut_box_nm must be [xmin,xmax,ymin,ymax,zmin,zmax] "
                "or [[xmin,xmax],[ymin,ymax],[zmin,zmax]]"
            ) from exc
    else:
        raise ValueError(
            "save_cut_box_nm must contain six values: "
            "[xmin,xmax,ymin,ymax,zmin,zmax]"
        )

    out = []
    for axis, pair in enumerate(pairs):
        try:
            lo, hi = float(pair[0]), float(pair[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"save_cut_box_nm axis {axis} must contain numeric offsets") from exc
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            raise ValueError(
                f"save_cut_box_nm axis {axis} requires finite offsets with min < max; got {pair}"
            )
        out.append((lo, hi))
    return tuple(out)


def save_potential_physical_cut(phi, center_nm, box_offsets_nm, field_bounds_nm,
                                filename="afm_potential_cut.npy", output_dir="."):
    """Save a physical-nanometre box relative to the current movement centre.

    ``box_offsets_nm`` defines six signed offsets from ``center_nm`` in the
    order ``[xmin, xmax, ymin, ymax, zmin, zmax]``.  For example,
    ``[-1, 20, -5, 5, 10, 15]`` saves x in ``center_x-1`` through
    ``center_x+20`` nm, y in ``center_y-5`` through ``center_y+5`` nm, and
    z in ``center_z+10`` through ``center_z+15`` nm.

    The requested physical box is intersected with the available field on
    each axis.  Consequently a box that extends beyond the simulation or
    zoom boundary is clipped safely rather than producing an out-of-range
    array access.  If the requested box has no intersection with the field,
    no file is written and ``(None, None)`` is returned.

    Parameters
    ----------
    phi : np.ndarray
        3-D potential array.
    center_nm : sequence of float
        Current physical movement centre relative to the configured origin.
    box_offsets_nm : sequence
        Six signed physical offsets from ``center_nm``.
    field_bounds_nm : sequence of float
        Six values ``(xmin, xmax, ymin, ymax, zmin, zmax)`` describing the
        physical extent represented by ``phi`` relative to the same origin.
    filename : str, optional
        Output filename.
    output_dir : str, optional
        Output directory.

    Returns
    -------
    tuple
        ``(path, actual_bounds_nm)``.  Returns ``(None, None)`` when there is
        no intersection with the available field.
    """
    arr = np.asarray(phi)
    if arr.ndim != 3:
        raise ValueError("phi must be a 3-D array")
    center = tuple(float(v) for v in center_nm)
    if len(center) != 3 or not all(np.isfinite(v) for v in center):
        raise ValueError("center_nm must contain three finite values")
    pairs = _normalize_cut_offsets_nm(box_offsets_nm)

    bounds = tuple(float(v) for v in field_bounds_nm)
    if len(bounds) != 6 or not all(np.isfinite(v) for v in bounds):
        raise ValueError("field_bounds_nm must contain six finite values")

    actual = []
    slices = []
    for axis, n in enumerate(arr.shape):
        flo, fhi = bounds[2 * axis], bounds[2 * axis + 1]
        if fhi < flo:
            flo, fhi = fhi, flo
        req_lo = center[axis] + pairs[axis][0]
        req_hi = center[axis] + pairs[axis][1]
        lo = max(flo, req_lo)
        hi = min(fhi, req_hi)
        if hi <= lo:
            print(
                f"Physical cut does not intersect field on axis {axis}: "
                f"field=({flo},{fhi}), requested=({req_lo},{req_hi}); skipping cut."
            )
            return None, None

        span = fhi - flo
        if span <= 0 or n <= 0:
            raise ValueError(f"Invalid field extent on axis {axis}: ({flo}, {fhi})")

        # Each array entry represents one physical voxel/bin.  Floor/ceil the
        # requested interval and clip indices so partially out-of-domain boxes
        # remain valid and deterministic.
        i0 = max(0, min(n - 1, int(np.floor((lo - flo) / span * n))))
        i1 = max(i0, min(n - 1, int(np.ceil((hi - flo) / span * n)) - 1))
        slices.append(slice(i0, i1 + 1))
        actual.extend([flo + span * i0 / n, flo + span * (i1 + 1) / n])

    cut = np.asarray(arr[tuple(slices)]).copy()
    path = _unique_output_path(output_dir, filename)
    np.save(path, cut)
    actual_bounds = tuple(actual)
    print(
        f"Saved physical cut to {path}: shape={cut.shape}, "
        f"bounds_nm={actual_bounds}"
    )
    del cut
    return path, actual_bounds


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



def get_sorted_config_files(base_name, directory="."):
    """Get generated configuration files for a common base name.

    Matches both legacy numeric suffixes (``base_1.json``) and physical
    tip-offset suffixes generated by presimulation
    (``base_+5nm.json``, ``base_0nm.json``, ``base_-5nm.json``).  Files are
    returned in increasing numeric order, with non-offset files first.
    """
    base = os.path.splitext(os.path.basename(base_name))[0]
    pattern = re.compile(
        rf"^{re.escape(base)}_(?P<offset>[+-]?(?:\d+(?:\.\d+)?))nm\.json$"
    )
    legacy = re.compile(rf"^{re.escape(base)}_(\d+)\.json$")
    files = []
    for fname in os.listdir(directory):
        m = pattern.match(fname)
        if m:
            files.append((0, float(m.group("offset")), fname))
            continue
        m = legacy.match(fname)
        if m:
            files.append((1, int(m.group(1)), fname))
    files.sort(key=lambda x: (x[0], x[1], x[2]))
    return [os.path.join(directory, x[2]) for x in files]

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
