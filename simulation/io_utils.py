import numpy as np
import csv
import os
import re


RESIDUAL_CSV = "residual_history.csv"


def _format_nm_filename_value(value):
    """Format one finite distance for a stable, compact filename token.

    ``numpy.format_float_positional`` deliberately avoids scientific notation,
    which keeps a physical displacement readable in a filename.  Rounding to
    twelve *significant* digits removes harmless binary interpolation tails
    (for example, ``7.071000000000001``) without reverting to the old
    two-decimal convention that could alias distinct movement positions.
    """
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("movement offset must be finite")
    if value == 0.0:
        return "0"
    coarse_value = round(value, 12)
    if abs(value - coarse_value) <= abs(value) * 1e-11:
        value = coarse_value
    decimal_places = max(0, 11 - int(np.floor(np.log10(abs(value)))))
    value = round(value, decimal_places)
    return np.format_float_positional(value, unique=True, trim="-")


def movement_offset_nm(center, center0, domain_nm, *, zero_tolerance_nm=1e-12):
    """Return the signed scalar displacement from a reference movement centre.

    The movement engine produces points on a straight physical path.  A
    single signed path distance is therefore sufficient for a compact and
    collision-free output name: the reference point is ``0 nm`` and later
    positions are measured by their Euclidean displacement.  The sign is
    taken from the first changing physical axis, so a path towards negative
    x/y/z produces negative suffixes while a path towards positive x/y/z
    produces positive suffixes.

    Parameters are fractional ``center`` coordinates plus the physical
    ``domain_nm`` lengths of the three main-grid axes.
    """
    try:
        current = np.asarray(center, dtype=float)
        reference = np.asarray(center0, dtype=float)
        lengths = np.asarray(domain_nm, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("movement centres and domain_nm must contain three finite values") from exc

    if current.shape != (3,) or reference.shape != (3,) or lengths.shape != (3,):
        raise ValueError("movement centres and domain_nm must each contain three values")
    if not (np.isfinite(current).all() and np.isfinite(reference).all() and np.isfinite(lengths).all()):
        raise ValueError("movement centres and domain_nm must contain finite values")
    if np.any(lengths <= 0.0):
        raise ValueError("domain_nm lengths must be positive")

    delta_nm = (current - reference) * lengths
    magnitude = float(np.linalg.norm(delta_nm))
    if magnitude <= float(zero_tolerance_nm):
        return 0.0

    for component in delta_nm:
        if abs(component) > float(zero_tolerance_nm):
            return magnitude if component > 0.0 else -magnitude
    return 0.0


def movement_suffix_nm(center, center0, domain_nm):
    """Build the canonical physical movement suffix for AFM NPY filenames.

    The centre/reference position is ``_0nm``.  A positive offset begins with
    ``_`` (for example ``_0.001nm``), while a negative offset begins with
    ``-`` (for example ``-0.001nm``).  This replaces rounded fractional
    ``cx/cy/cz`` tokens and keeps sub-centinanometre movement positions
    distinct.
    """
    offset_nm = movement_offset_nm(center, center0, domain_nm)
    if offset_nm == 0.0:
        return "_0nm"
    magnitude = _format_nm_filename_value(abs(offset_nm))
    return f"_{magnitude}nm" if offset_nm > 0.0 else f"-{magnitude}nm"


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


def resolution_tagged_cut_filename(filename, source_shape):
    """Return a cut filename tagged with the original uncut grid shape.

    The existing stem is preserved verbatim so configuration index, movement
    position, voltage, zoom magnification, and other caller-provided metadata
    remain in the saved filename.  For example, a 512-cubed source named
    ``afm_phi_1_-1.00V.npy`` becomes
    ``afm_phi_1_-1.00V_cut_from_grid512x512x512.npy``.
    """
    try:
        shape = tuple(int(v) for v in source_shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("source_shape must contain three positive integers") from exc
    if len(shape) != 3 or any(v <= 0 for v in shape):
        raise ValueError("source_shape must contain three positive integers")
    stem, ext = os.path.splitext(filename)
    if not ext:
        ext = ".npy"
    return f"{stem}_cut_from_grid{shape[0]}x{shape[1]}x{shape[2]}{ext}"


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
    box_offsets_nm : sequence or None
        Six signed physical offsets from ``center_nm``.  Omit the setting or
        pass an empty sequence to save the whole available field.
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

    bounds = tuple(float(v) for v in field_bounds_nm)
    if len(bounds) != 6 or not all(np.isfinite(v) for v in bounds):
        raise ValueError("field_bounds_nm must contain six finite values")

    # A missing/empty saved-cut definition is intentionally a full-field cut.
    # Deriving relative offsets from the actual field also works for movement
    # positions and zoom domains whose extents are not symmetric about center.
    if box_offsets_nm is None:
        offsets = []
    else:
        try:
            offsets = list(box_offsets_nm)
        except TypeError as exc:
            raise ValueError("save_cut_box_nm must contain six numeric values") from exc
    if offsets:
        pairs = _normalize_cut_offsets_nm(offsets)
    else:
        pairs = tuple(
            (
                min(bounds[2 * axis], bounds[2 * axis + 1]) - center[axis],
                max(bounds[2 * axis], bounds[2 * axis + 1]) - center[axis],
            )
            for axis in range(3)
        )

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

    # ``np.save`` writes non-contiguous views correctly.  Avoid duplicating a
    # full float32 potential when an omitted box intentionally selects the
    # whole field.
    cut = np.asarray(arr[tuple(slices)])
    path = _unique_output_path(output_dir, filename)
    np.save(path, cut)
    from .output_coordinates import write_coordinate_receipt
    write_coordinate_receipt(path, arr.shape, bounds, slices)
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


def gate_slices(nx, ny, nz, gate):
    """Return three slices covering a gate region.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.
    gate : dict
        Gate dict with optional fractional 'x_range', 'y_range', and
        'z_range'.  Any omitted or null axis range spans the whole axis.

    Returns
    -------
    tuple(slice, slice, slice)
        Inclusive grid-index extent expressed as NumPy slices.
    """
    x_range = gate.get("x_range")
    y_range = gate.get("y_range")
    z_range = gate.get("z_range")
    x0, x1 = (0.0, 1.0) if x_range is None else x_range
    y0, y1 = (0.0, 1.0) if y_range is None else y_range
    z0, z1 = (0.0, 1.0) if z_range is None else z_range
    ix0, ix1 = _range_to_indices(x0, x1, nx)
    iy0, iy1 = _range_to_indices(y0, y1, ny)
    iz0, iz1 = _range_to_indices(z0, z1, nz)
    return slice(ix0, ix1 + 1), slice(iy0, iy1 + 1), slice(iz0, iz1 + 1)


def make_gate_mask(nx, ny, nz, gate):
    """Create a boolean 3D mask for a gate region.

    This compatibility helper uses :func:`gate_slices`; simulation paths that
    only need to apply a rectangular gate use those slices directly and avoid
    retaining a full-grid boolean mask.
    """
    mask = np.zeros((nx, ny, nz), dtype=bool)
    mask[gate_slices(nx, ny, nz, gate)] = True
    return mask
