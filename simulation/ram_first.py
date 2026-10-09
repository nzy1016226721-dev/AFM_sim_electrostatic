"""Lossless, RAM-first snapshot Jacobi for standalone serial and MPI runs.

Only one full potential and one float32 cell-epsilon volume are retained.
Three old y-z planes per worker preserve old-iterate reads, including both
sides of every worker boundary. Recomputing face coefficients costs time;
it does NOT change the stencil, precision, update ordering, or MPI halos.
"""
from __future__ import annotations

import math
import time
import numpy as np
from numba import njit, prange

from .parallel import configure_cpu_threads


def resolve_memory_mode(mode):
    mode = str(mode).lower()
    if mode not in ("standard", "ram_first", "ram_compact"):
        raise ValueError("memory_mode must be standard, ram_first or ram_compact")
    return mode


def resolve_storage_options(memory_mode, phi_update_mode=None, residual_accumulation=None):
    """Resolve explicit storage selectors without changing stencil arithmetic."""
    memory_mode = resolve_memory_mode(memory_mode)
    if memory_mode == "standard" and (phi_update_mode is not None or residual_accumulation is not None):
        raise ValueError("phi_update_mode/residual_accumulation require memory_mode=ram_first or ram_compact")
    update = "in_place" if phi_update_mode is None else str(phi_update_mode).lower()
    residual = ("scalar" if memory_mode == "ram_compact" else "row_array") if residual_accumulation is None else str(residual_accumulation).lower()
    if update not in ("in_place", "buffered"):
        raise ValueError("phi_update_mode must be in_place or buffered")
    if residual not in ("scalar", "row_array", "array"):
        raise ValueError("residual_accumulation must be scalar, row_array or array")
    return update, residual


@njit(inline="always", fastmath=False)
def _mean4(a, b, c, d):
    value = np.float32(a + b)
    value = np.float32(value + c)
    value = np.float32(value + d)
    return np.float32(value * np.float32(0.25))


@njit(inline="always", fastmath=False)
def cell_faces(e, i, j, k):
    xp = _mean4(e[i+1,j,k], e[i+1,j+1,k], e[i+1,j,k+1], e[i+1,j+1,k+1])
    xm = _mean4(e[i,j,k], e[i,j+1,k], e[i,j,k+1], e[i,j+1,k+1])
    yp = _mean4(e[i,j+1,k], e[i+1,j+1,k], e[i,j+1,k+1], e[i+1,j+1,k+1])
    ym = _mean4(e[i,j,k], e[i+1,j,k], e[i,j,k+1], e[i+1,j,k+1])
    zp = _mean4(e[i,j,k+1], e[i+1,j,k+1], e[i,j+1,k+1], e[i+1,j+1,k+1])
    zm = _mean4(e[i,j,k], e[i+1,j,k], e[i,j+1,k], e[i+1,j+1,k])
    return xp, xm, yp, ym, zp, zm


@njit(inline="always", fastmath=False)
def _denominator(xp, xm, yp, ym, zp, zm):
    value = np.float32(xp + xm)
    value = np.float32(value + yp)
    value = np.float32(value + ym)
    value = np.float32(value + zp)
    return np.float32(value + zm)


def allocate_stream_scratch(field, owned_mask, threads):
    if field.dtype != np.float32 or owned_mask.dtype != np.bool_:
        raise ValueError("ram_first requires float32 potentials and a bool mask")
    if tuple(v-2 for v in field.shape) != owned_mask.shape:
        raise ValueError("owned mask must cover the field excluding its one-node halo")
    workers = min(int(threads), owned_mask.shape[0])
    if workers < 1:
        raise ValueError("ram_first needs at least one worker and owned x node")
    return np.empty((workers, 3, field.shape[1], field.shape[2]), dtype=np.float32)


@njit(parallel=True, cache=True, fastmath=False)
def jacobi_cells_in_place(field, epsilon, mask, starts, global_shape, omega, scratch,
                          update_fixed=False):
    """One exact old-iterate update of a halo-padded block, without a full copy.

    The caller exchanges MPI halos only AFTER this complete pass. All worker
    boundary planes are copied BEFORE any worker writes. Updating an ordinary
    stencil in place without these snapshots would instead be Gauss-Seidel.
    """
    lx, ly, lz = mask.shape
    workers = scratch.shape[0]
    for worker in range(workers):
        first = worker * lx // workers
        stop = (worker + 1) * lx // workers
        scratch[worker, 0, :, :] = field[first, :, :]
        scratch[worker, 2, :, :] = field[stop+1, :, :]
    for worker in prange(workers):
        first = worker * lx // workers
        stop = (worker + 1) * lx // workers
        previous = scratch[worker, 0]
        old = scratch[worker, 1]
        right = scratch[worker, 2]
        for i in range(first, stop):
            ii = i + 1
            old[:, :] = field[ii, :, :]
            future = right if i == stop-1 else field[ii+1, :, :]
            gi = starts[0] + i
            for j in range(ly):
                jj = j + 1
                gj = starts[1] + j
                for k in range(lz):
                    kk = k + 1
                    gk = starts[2] + k
                    if ((mask[i,j,k] and not update_fixed) or gi == 0 or gi == global_shape[0]-1
                            or gj == 0 or gj == global_shape[1]-1
                            or gk == 0 or gk == global_shape[2]-1):
                        continue
                    xp, xm, yp, ym, zp, zm = cell_faces(epsilon, i, j, k)
                    denom = _denominator(xp, xm, yp, ym, zp, zm)
                    num = np.float32(xp * future[jj,kk])
                    num = np.float32(num + xm * previous[jj,kk])
                    num = np.float32(num + yp * old[jj+1,kk])
                    num = np.float32(num + ym * old[jj-1,kk])
                    num = np.float32(num + zp * old[jj,kk+1])
                    num = np.float32(num + zm * old[jj,kk-1])
                    candidate = np.float32(num / denom)
                    residual = np.float32(candidate - old[jj,kk])
                    update = np.float32(omega * residual)
                    field[ii,jj,kk] = np.float32(old[jj,kk] + update)
            previous, old = old, previous


@njit(parallel=True, cache=True, fastmath=False)
def residual_cells_rows(field, epsilon, mask, starts, global_shape):
    lx, ly, lz = mask.shape
    sums = np.zeros(lx, dtype=np.float64)
    maxima = np.zeros(lx, dtype=np.float64)
    counts = np.zeros(lx, dtype=np.int64)
    for i in prange(lx):
        total, maximum, count = 0.0, 0.0, 0
        gi = starts[0] + i
        for j in range(ly):
            gj = starts[1] + j
            for k in range(lz):
                gk = starts[2] + k
                if (mask[i,j,k] or gi == 0 or gi == global_shape[0]-1
                        or gj == 0 or gj == global_shape[1]-1
                        or gk == 0 or gk == global_shape[2]-1):
                    continue
                ii, jj, kk = i+1, j+1, k+1
                xp, xm, yp, ym, zp, zm = cell_faces(epsilon, i, j, k)
                denom = _denominator(xp, xm, yp, ym, zp, zm)
                num = np.float32(xp * field[ii+1,jj,kk])
                num = np.float32(num + xm * field[ii-1,jj,kk])
                num = np.float32(num + yp * field[ii,jj+1,kk])
                num = np.float32(num + ym * field[ii,jj-1,kk])
                num = np.float32(num + zp * field[ii,jj,kk+1])
                num = np.float32(num + zm * field[ii,jj,kk-1])
                residual = np.float32(num - np.float32(denom * field[ii,jj,kk]))
                total += residual * residual
                maximum = max(maximum, abs(residual))
                count += 1
        sums[i], maxima[i], counts[i] = total, maximum, count
    return sums, maxima, counts


@njit(inline="always", fastmath=False)
def _point_residual(field, epsilon, i, j, k):
    ii, jj, kk = i+1, j+1, k+1
    xp, xm, yp, ym, zp, zm = cell_faces(epsilon, i, j, k)
    denom = _denominator(xp, xm, yp, ym, zp, zm)
    num = np.float32(xp * field[ii+1,jj,kk])
    num = np.float32(num + xm * field[ii-1,jj,kk])
    num = np.float32(num + yp * field[ii,jj+1,kk])
    num = np.float32(num + ym * field[ii,jj-1,kk])
    num = np.float32(num + zp * field[ii,jj,kk+1])
    num = np.float32(num + zm * field[ii,jj,kk-1])
    return np.float32(num - np.float32(denom * field[ii,jj,kk]))


@njit(inline="always", fastmath=False)
def _included(mask, starts, global_shape, i, j, k):
    gi, gj, gk = starts[0]+i, starts[1]+j, starts[2]+k
    return not (mask[i,j,k] or gi == 0 or gi == global_shape[0]-1
                or gj == 0 or gj == global_shape[1]-1
                or gk == 0 or gk == global_shape[2]-1)


@njit(cache=True, fastmath=False)
def _residual_row_scalar(field, epsilon, mask, starts, global_shape, i, volume):
    total, maximum, count = 0.0, 0.0, 0
    for j in range(mask.shape[1]):
        for k in range(mask.shape[2]):
            if _included(mask, starts, global_shape, i, j, k):
                value = (_point_residual(field, epsilon, i, j, k)
                         if volume is None else volume[i,j,k])
                # Keep the original float32 square and float64 scalar sum.
                total += value * value
                maximum = max(maximum, abs(value))
                count += 1
    return total, maximum, count


@njit(cache=True, fastmath=False)
def residual_cells_scalar(field, epsilon, mask, starts, global_shape, volume=None):
    """No residual/row arrays. Match the shared solver's x-row reduction order."""
    total, maximum, count = 0.0, 0.0, 0
    for i in range(mask.shape[0]):
        row_sum, row_max, row_count = _residual_row_scalar(
            field, epsilon, mask, starts, global_shape, i, volume)
        total += row_sum
        maximum = max(maximum, row_max)
        count += row_count
    return total, maximum, count


@njit(parallel=True, cache=True, fastmath=False)
def fill_residual_volume(field, epsilon, mask, starts, global_shape, volume):
    for i in prange(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                volume[i,j,k] = (_point_residual(field, epsilon, i, j, k)
                    if _included(mask, starts, global_shape, i, j, k) else np.float32(0))


def _pairwise_scalar_stats(field, epsilon, mask, starts, global_shape, volume, first, n):
    """NumPy 2.3's eight-accumulator/block-128 sum order, without row arrays.

    MPI previously used np.sum on its row sums; the shared solver did not.
    Only O(log(nx)) scalar stack state is retained, never a row/volume copy.
    """
    def row(i):
        return _residual_row_scalar(field, epsilon, mask, starts, global_shape, i, volume)
    maximum, count = 0.0, 0
    if n < 8:
        total = -0.0
        for i in range(first, first+n):
            value, peak, points = row(i)
            total += value
            maximum = max(maximum, peak)
            count += points
        return total, maximum, count
    if n > 128:
        split = n//2
        split -= split % 8
        left = _pairwise_scalar_stats(field, epsilon, mask, starts, global_shape, volume, first, split)
        right = _pairwise_scalar_stats(field, epsilon, mask, starts, global_shape, volume, first+split, n-split)
        return left[0]+right[0], max(left[1], right[1]), left[2]+right[2]
    # Explicit scalar variables are intentional: no list/array of row totals.
    r0, peak, points = row(first); maximum = max(maximum, peak); count += points
    r1, peak, points = row(first+1); maximum = max(maximum, peak); count += points
    r2, peak, points = row(first+2); maximum = max(maximum, peak); count += points
    r3, peak, points = row(first+3); maximum = max(maximum, peak); count += points
    r4, peak, points = row(first+4); maximum = max(maximum, peak); count += points
    r5, peak, points = row(first+5); maximum = max(maximum, peak); count += points
    r6, peak, points = row(first+6); maximum = max(maximum, peak); count += points
    r7, peak, points = row(first+7); maximum = max(maximum, peak); count += points
    for i in range(8, n-n%8, 8):
        value, peak, points = row(first+i); r0 += value; maximum = max(maximum, peak); count += points
        value, peak, points = row(first+i+1); r1 += value; maximum = max(maximum, peak); count += points
        value, peak, points = row(first+i+2); r2 += value; maximum = max(maximum, peak); count += points
        value, peak, points = row(first+i+3); r3 += value; maximum = max(maximum, peak); count += points
        value, peak, points = row(first+i+4); r4 += value; maximum = max(maximum, peak); count += points
        value, peak, points = row(first+i+5); r5 += value; maximum = max(maximum, peak); count += points
        value, peak, points = row(first+i+6); r6 += value; maximum = max(maximum, peak); count += points
        value, peak, points = row(first+i+7); r7 += value; maximum = max(maximum, peak); count += points
    total = ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))
    for i in range(n-n%8, n):
        value, peak, points = row(first+i)
        total += value; maximum = max(maximum, peak); count += points
    return total, maximum, count


def residual_cells_stats(field, epsilon, mask, starts, global_shape,
                         accumulation="row_array", volume=None, pairwise=False):
    if accumulation == "row_array":
        rows, maxima, counts = residual_cells_rows(field, epsilon, mask, starts, global_shape)
        if pairwise:
            return float(np.sum(rows, dtype=np.float64)), float(np.max(maxima, initial=0)), int(np.sum(counts))
        total, maximum, count = 0.0, 0.0, 0
        for i in range(rows.size):
            total += float(rows[i]); maximum = max(maximum, float(maxima[i])); count += int(counts[i])
        return total, maximum, count
    if accumulation == "array":
        if volume is None or volume.shape != mask.shape or volume.dtype != np.float32:
            raise ValueError("array residual mode needs an owned float32 residual workspace")
        fill_residual_volume(field, epsilon, mask, starts, global_shape, volume)
    elif accumulation != "scalar":
        raise ValueError("unknown residual accumulation mode")
    else:
        volume = None
    if pairwise:
        return _pairwise_scalar_stats(field, epsilon, mask, starts, global_shape, volume, 0, mask.shape[0])
    return residual_cells_scalar(field, epsilon, mask, starts, global_shape, volume)


def residual_cells_shared(field, epsilon, mask, accumulation="row_array", volume=None):
    total, maximum, count = residual_cells_stats(field, epsilon, mask[1:-1,1:-1,1:-1],
                                               (1,1,1), field.shape, accumulation, volume)
    return (math.sqrt(total/count) if count else 0.0), maximum


def solve_ram_first(phi, epsilon, mask, *, damping, tol, cpu_threads,
                    max_runtime, output_dir, verbose=False,
                    diagnostic_iterations=None, phi_update_mode=None,
                    residual_accumulation=None):
    from .solver import MG_TIME, log_residual_csv

    if phi.dtype != np.float32 or epsilon.dtype != np.float32:
        raise ValueError("ram_first never changes precision: use float32 inputs")
    if epsilon.shape != tuple(v-1 for v in phi.shape):
        raise ValueError("epsilon must contain the original solver cell grid")
    threads = configure_cpu_threads(cpu_threads)
    update_mode, accumulation = resolve_storage_options("ram_first", phi_update_mode, residual_accumulation)
    owned_mask = mask[1:-1,1:-1,1:-1]
    scratch = allocate_stream_scratch(phi, owned_mask, threads)
    target = np.empty_like(phi) if update_mode == "buffered" else None
    residual_volume = np.empty(owned_mask.shape, np.float32) if accumulation == "array" else None
    fixed = phi[mask]  # Already a compact copy; do not make a second copy.
    if diagnostic_iterations is not None and int(diagnostic_iterations) < 1:
        raise ValueError("diagnostic_iterations must be positive")
    started = time.monotonic()
    iteration = 0
    while True:
        iteration += 1
        destination = phi
        if target is not None:
            np.copyto(target, phi)
            destination = target
        jacobi_cells_in_place(destination, epsilon, owned_mask, (1,1,1), phi.shape,
                             np.float32(damping), scratch, True)
        destination[0,:,:] = destination[1,:,:]; destination[-1,:,:] = destination[-2,:,:]
        destination[:,0,:] = destination[:,1,:]; destination[:,-1,:] = destination[:,-2,:]
        destination[:,:,0] = destination[:,:,1]; destination[:,:,-1] = destination[:,:,-2]
        destination[mask] = fixed
        if target is not None:
            phi, target = destination, phi
        elapsed = time.monotonic() - started
        check = iteration <= 5 or iteration % 10 == 0
        diagnostic_end = diagnostic_iterations is not None and iteration >= int(diagnostic_iterations)
        timeout = max_runtime is not None and elapsed > float(max_runtime)
        if check or diagnostic_end or timeout:
            rms, maximum = residual_cells_shared(phi, epsilon, mask, accumulation, residual_volume)
            log_residual_csv(iteration, rms, maximum, output_dir=output_dir)
            if verbose:
                print(f"iter {iteration:6d}: res_avg={rms:.5e}, res_max={maximum:.5e}")
        reason = "running"
        if diagnostic_iterations is None and check and rms < float(tol):
            reason = "converged"
        elif diagnostic_end:
            reason = "diagnostic_iteration_limit"
        elif timeout:
            reason = "time_limit"
        MG_TIME.update(elapsed=elapsed, iterations=iteration, residual=float(rms),
                       residual_max=float(maximum), reason=reason, memory_mode="ram_first",
                       scratch_bytes=int(scratch.nbytes))
        MG_TIME.update(phi_update_mode=update_mode, residual_accumulation=accumulation,
                       phi_buffer_bytes=0 if target is None else int(target.nbytes),
                       residual_workspace_bytes=0 if residual_volume is None else int(residual_volume.nbytes))
        if reason != "running":
            print(f"RAM-first {reason}: {iteration} iterations in {elapsed:.2f} s; residual={rms:.5e}")
            return phi, None
