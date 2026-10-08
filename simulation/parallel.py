"""CPU-parallel numerical kernels for the AFM solver."""

from __future__ import annotations

import os

import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads


@njit(parallel=True, cache=True, fastmath=False)
def weighted_jacobi_step_parallel(
    V_old,
    axp, axm, ayp, aym, azp, azm, a0,
    omega,
    Vnew_core,
    R_arr,
):
    """Perform one weighted-Jacobi iteration over all interior nodes.

    ``V_old`` is a read-only snapshot of the current solution. The arithmetic
    order intentionally mirrors the existing NumPy implementation:
    six weighted neighbor additions, division by a0, subtraction from the old
    center value, then damping.  ``prange`` parallelizes only independent grid
    points. The caller applies the resulting relaxation update to ``V`` only
    after this kernel has completed, preserving Jacobi semantics without a
    read/write race between neighboring points.
    """
    ni, nj, nk = Vnew_core.shape
    for i in prange(ni):
        ii = i + 1
        for j in range(nj):
            jj = j + 1
            for k in range(nk):
                kk = k + 1

                # Match the legacy NumPy operation sequence as closely as possible.
                num = axp[i, j, k] * V_old[ii + 1, jj, kk]
                num = num + axm[i, j, k] * V_old[ii - 1, jj, kk]
                num = num + ayp[i, j, k] * V_old[ii, jj + 1, kk]
                num = num + aym[i, j, k] * V_old[ii, jj - 1, kk]
                num = num + azp[i, j, k] * V_old[ii, jj, kk + 1]
                num = num + azm[i, j, k] * V_old[ii, jj, kk - 1]

                Vnew = num / a0[i, j, k]
                Vnew_core[i, j, k] = Vnew
                residual = Vnew - V_old[ii, jj, kk]
                R_arr[i, j, k] = residual


@njit(parallel=True, cache=True, fastmath=False)
def weighted_jacobi_residual_parallel(
    V_old,
    axp, axm, ayp, aym, azp, azm, a0,
    R_arr,
):
    """Write weighted-Jacobi residuals without a dead new-value workspace.

    The production solver only consumes the raw residual after each parallel
    stencil pass.  Keeping this lean variant separate preserves the original
    ``weighted_jacobi_step_parallel`` helper for callers that also need the
    candidate new value, while avoiding one full interior float32 array in
    every real parallel solve.
    """
    ni, nj, nk = R_arr.shape
    for i in prange(ni):
        ii = i + 1
        for j in range(nj):
            jj = j + 1
            for k in range(nk):
                kk = k + 1

                num = axp[i, j, k] * V_old[ii + 1, jj, kk]
                num = num + axm[i, j, k] * V_old[ii - 1, jj, kk]
                num = num + ayp[i, j, k] * V_old[ii, jj + 1, kk]
                num = num + aym[i, j, k] * V_old[ii, jj - 1, kk]
                num = num + azp[i, j, k] * V_old[ii, jj, kk + 1]
                num = num + azm[i, j, k] * V_old[ii, jj, kk - 1]

                Vnew = num / a0[i, j, k]
                R_arr[i, j, k] = Vnew - V_old[ii, jj, kk]


@njit(parallel=True, cache=True, fastmath=False)
def weighted_jacobi_update_faces_parallel(
    current,
    target,
    axp, axm, ayp, aym, azp, azm,
    omega,
):
    """Write one Jacobi update directly into the alternate solution buffer.

    This is the low-memory production path used when no residual volume is
    requested.  It deliberately recomputes the six-face denominator instead
    of retaining a grid-sized ``a0`` array and writes the relaxed value into a
    second solution field instead of allocating a residual array.  Every
    float32 operation follows the serial NumPy and MPI stencil order exactly.
    """
    ni, nj, nk = current.shape[0] - 2, current.shape[1] - 2, current.shape[2] - 2
    for i in prange(ni):
        ii = i + 1
        for j in range(nj):
            jj = j + 1
            for k in range(nk):
                kk = k + 1
                cxp = axp[i, j, k]
                cxm = axm[i, j, k]
                cyp = ayp[i, j, k]
                cym = aym[i, j, k]
                czp = azp[i, j, k]
                czm = azm[i, j, k]

                denom = np.float32(cxp + cxm)
                denom = np.float32(denom + cyp)
                denom = np.float32(denom + cym)
                denom = np.float32(denom + czp)
                denom = np.float32(denom + czm)

                numerator = np.float32(cxp * current[ii + 1, jj, kk])
                numerator = np.float32(
                    numerator + cxm * current[ii - 1, jj, kk]
                )
                numerator = np.float32(
                    numerator + cyp * current[ii, jj + 1, kk]
                )
                numerator = np.float32(
                    numerator + cym * current[ii, jj - 1, kk]
                )
                numerator = np.float32(
                    numerator + czp * current[ii, jj, kk + 1]
                )
                numerator = np.float32(
                    numerator + czm * current[ii, jj, kk - 1]
                )

                old = current[ii, jj, kk]
                candidate = np.float32(numerator / denom)
                residual = np.float32(candidate - old)
                update = np.float32(omega * residual)
                target[ii, jj, kk] = np.float32(old + update)


@njit(parallel=True, cache=True, fastmath=False)
def weighted_jacobi_update_faces_parallel_float64(
    current,
    target,
    axp, axm, ayp, aym, azp, azm,
    omega,
):
    """Float64 counterpart used only by explicit double-precision runs."""
    ni, nj, nk = current.shape[0] - 2, current.shape[1] - 2, current.shape[2] - 2
    for i in prange(ni):
        ii = i + 1
        for j in range(nj):
            jj = j + 1
            for k in range(nk):
                kk = k + 1
                cxp = axp[i, j, k]
                cxm = axm[i, j, k]
                cyp = ayp[i, j, k]
                cym = aym[i, j, k]
                czp = azp[i, j, k]
                czm = azm[i, j, k]

                denom = cxp + cxm
                denom = denom + cyp
                denom = denom + cym
                denom = denom + czp
                denom = denom + czm

                numerator = cxp * current[ii + 1, jj, kk]
                numerator = numerator + cxm * current[ii - 1, jj, kk]
                numerator = numerator + cyp * current[ii, jj + 1, kk]
                numerator = numerator + cym * current[ii, jj - 1, kk]
                numerator = numerator + czp * current[ii, jj, kk + 1]
                numerator = numerator + czm * current[ii, jj, kk - 1]

                old = current[ii, jj, kk]
                candidate = numerator / denom
                residual = candidate - old
                target[ii, jj, kk] = old + omega * residual


@njit(parallel=True, cache=True, fastmath=False)
def residual_scalars_faces_parallel(
    field,
    mask,
    axp, axm, ayp, aym, azp, azm,
):
    """Compute residual scalars without a persistent denominator volume."""
    ni, nj, nk = field.shape[0] - 2, field.shape[1] - 2, field.shape[2] - 2
    row_sums = np.zeros(ni, dtype=np.float64)
    row_max = np.zeros(ni, dtype=np.float64)
    row_counts = np.zeros(ni, dtype=np.int64)

    for i in prange(ni):
        row_total = 0.0
        row_max_abs = 0.0
        row_count = 0
        for j in range(nj):
            for k in range(nk):
                ii, jj, kk = i + 1, j + 1, k + 1
                if not mask[ii, jj, kk]:
                    cxp = axp[i, j, k]
                    cxm = axm[i, j, k]
                    cyp = ayp[i, j, k]
                    cym = aym[i, j, k]
                    czp = azp[i, j, k]
                    czm = azm[i, j, k]

                    denom = np.float32(cxp + cxm)
                    denom = np.float32(denom + cyp)
                    denom = np.float32(denom + cym)
                    denom = np.float32(denom + czp)
                    denom = np.float32(denom + czm)

                    numerator = np.float32(cxp * field[ii + 1, jj, kk])
                    numerator = np.float32(
                        numerator + cxm * field[ii - 1, jj, kk]
                    )
                    numerator = np.float32(
                        numerator + cyp * field[ii, jj + 1, kk]
                    )
                    numerator = np.float32(
                        numerator + cym * field[ii, jj - 1, kk]
                    )
                    numerator = np.float32(
                        numerator + czp * field[ii, jj, kk + 1]
                    )
                    numerator = np.float32(
                        numerator + czm * field[ii, jj, kk - 1]
                    )
                    residual = np.float32(
                        numerator - np.float32(denom * field[ii, jj, kk])
                    )
                    row_total += residual * residual
                    absolute = abs(residual)
                    if absolute > row_max_abs:
                        row_max_abs = absolute
                    row_count += 1

        row_sums[i] = row_total
        row_max[i] = row_max_abs
        row_counts[i] = row_count

    total = 0.0
    maximum = 0.0
    count = 0
    for i in range(ni):
        total += row_sums[i]
        if row_max[i] > maximum:
            maximum = row_max[i]
        count += row_counts[i]
    if count == 0:
        return 0.0, 0.0
    return (total / count) ** 0.5, maximum


@njit(parallel=True, cache=True, fastmath=False)
def residual_scalars_faces_parallel_float64(
    field,
    mask,
    axp, axm, ayp, aym, azp, azm,
):
    """Float64 residual reduction without a persistent denominator volume."""
    ni, nj, nk = field.shape[0] - 2, field.shape[1] - 2, field.shape[2] - 2
    row_sums = np.zeros(ni, dtype=np.float64)
    row_max = np.zeros(ni, dtype=np.float64)
    row_counts = np.zeros(ni, dtype=np.int64)

    for i in prange(ni):
        row_total = 0.0
        row_max_abs = 0.0
        row_count = 0
        for j in range(nj):
            for k in range(nk):
                ii, jj, kk = i + 1, j + 1, k + 1
                if not mask[ii, jj, kk]:
                    cxp = axp[i, j, k]
                    cxm = axm[i, j, k]
                    cyp = ayp[i, j, k]
                    cym = aym[i, j, k]
                    czp = azp[i, j, k]
                    czm = azm[i, j, k]

                    denom = cxp + cxm
                    denom = denom + cyp
                    denom = denom + cym
                    denom = denom + czp
                    denom = denom + czm

                    numerator = cxp * field[ii + 1, jj, kk]
                    numerator = numerator + cxm * field[ii - 1, jj, kk]
                    numerator = numerator + cyp * field[ii, jj + 1, kk]
                    numerator = numerator + cym * field[ii, jj - 1, kk]
                    numerator = numerator + czp * field[ii, jj, kk + 1]
                    numerator = numerator + czm * field[ii, jj, kk - 1]
                    residual = numerator - denom * field[ii, jj, kk]
                    row_total += residual * residual
                    absolute = abs(residual)
                    if absolute > row_max_abs:
                        row_max_abs = absolute
                    row_count += 1

        row_sums[i] = row_total
        row_max[i] = row_max_abs
        row_counts[i] = row_count

    total = 0.0
    maximum = 0.0
    count = 0
    for i in range(ni):
        total += row_sums[i]
        if row_max[i] > maximum:
            maximum = row_max[i]
        count += row_counts[i]
    if count == 0:
        return 0.0, 0.0
    return (total / count) ** 0.5, maximum


@njit(parallel=True, cache=True, fastmath=False)
def residual_scalars_parallel(
    V,
    mask,
    axp, axm, ayp, aym, azp, azm, a0,
):
    """Compute L2 and max residuals without allocating full temporary arrays."""
    ni, nj, nk = V.shape[0] - 2, V.shape[1] - 2, V.shape[2] - 2
    # Reduce through one accumulator per outer row. Numba's prange reduction
    # support is reliable for sums, but an in-place ``if value > max`` update
    # does not safely combine maxima across workers. The row buffers keep the
    # diagnostic reduction race-free while remaining tiny compared with V.
    row_sums = np.zeros(ni, dtype=np.float64)
    row_max = np.zeros(ni, dtype=np.float64)
    row_counts = np.zeros(ni, dtype=np.int64)

    for i in prange(ni):
        row_total = 0.0
        row_max_abs = 0.0
        row_count = 0
        for j in range(nj):
            for k in range(nk):
                if not mask[i + 1, j + 1, k + 1]:
                    num = axp[i, j, k] * V[i + 2, j + 1, k + 1]
                    num = num + axm[i, j, k] * V[i, j + 1, k + 1]
                    num = num + ayp[i, j, k] * V[i + 1, j + 2, k + 1]
                    num = num + aym[i, j, k] * V[i + 1, j, k + 1]
                    num = num + azp[i, j, k] * V[i + 1, j + 1, k + 2]
                    num = num + azm[i, j, k] * V[i + 1, j + 1, k]
                    resid = num - a0[i, j, k] * V[i + 1, j + 1, k + 1]
                    row_total += resid * resid
                    ares = abs(resid)
                    if ares > row_max_abs:
                        row_max_abs = ares
                    row_count += 1

        row_sums[i] = row_total
        row_max[i] = row_max_abs
        row_counts[i] = row_count

    total = 0.0
    max_abs = 0.0
    count = 0
    for i in range(ni):
        total += row_sums[i]
        if row_max[i] > max_abs:
            max_abs = row_max[i]
        count += row_counts[i]
    if count == 0:
        return 0.0, 0.0
    return (total / count) ** 0.5, max_abs


def configure_cpu_threads(requested, *, slurm_cap=True):
    """Resolve and apply the number of Numba CPU threads for one solver."""
    cpu_count = os.cpu_count() or 1
    if requested is None:
        requested = 1
    requested = int(requested)
    if requested < 1:
        raise ValueError("cpu_threads must be >= 1")

    # Never use more than the machine/allocation makes available.
    threads = min(requested, cpu_count)
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if slurm_cap and slurm_cpus:
        try:
            threads = min(threads, max(1, int(slurm_cpus)))
        except ValueError:
            pass

    set_num_threads(threads)
    return threads


def active_cpu_threads():
    """Return the currently configured Numba CPU thread count."""
    return get_num_threads()
