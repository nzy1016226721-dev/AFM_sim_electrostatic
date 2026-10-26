"""Bounded regional/precision reductions; no global residual or old-phi arrays."""
import math
import time
import numpy as np
from numba import njit
from .ram_first import cell_faces, _denominator


@njit(inline="always", fastmath=False)
def _selected_qd(gi, gj, gk, shape, blocks):
    # One-cell adjacency: include a node touching at least one physical
    # last-block-wins eps15 cell centre, not averaged-epsilon == 15.
    for di in (-1, 0):
        ci = gi + di
        if ci < 0 or ci >= shape[0] - 1:
            continue
        x = (ci + .5) / (shape[0] - 1)
        for dj in (-1, 0):
            cj = gj + dj
            if cj < 0 or cj >= shape[1] - 1:
                continue
            y = (cj + .5) / (shape[1] - 1)
            for dk in (-1, 0):
                ck = gk + dk
                if ck < 0 or ck >= shape[2] - 1:
                    continue
                z = (ck + .5) / (shape[2] - 1)
                material = 1.0
                for row in blocks:
                    if row[0] <= x <= row[1] and row[2] <= y <= row[3] and row[4] <= z <= row[5]:
                        material = row[6]
                if material == 15.0:
                    return True
    return False


@njit(cache=True, fastmath=False)
def _measure(field, cells, mask, faces, starts, shape, bounds, blocks, qd, precision, weights, use_faces):
    total, maximum, count = 0.0, 0.0, 0
    precise_squares, precise_max, error_squares, error_max, envelope = 0.0, 0.0, 0.0, 0.0, 0.0
    lo = (max(0, bounds[0][0]-starts[0], 1-starts[0]),
          max(0, bounds[1][0]-starts[1], 1-starts[1]),
          max(0, bounds[2][0]-starts[2], 1-starts[2]))
    hi = (min(mask.shape[0], bounds[0][1]-starts[0], shape[0]-1-starts[0]),
          min(mask.shape[1], bounds[1][1]-starts[1], shape[1]-1-starts[1]),
          min(mask.shape[2], bounds[2][1]-starts[2], shape[2]-1-starts[2]))
    for i in range(lo[0], hi[0]):
        for j in range(lo[1], hi[1]):
            for k in range(lo[2], hi[2]):
                if mask[i, j, k]:
                    continue
                if qd and not _selected_qd(starts[0]+i, starts[1]+j, starts[2]+k, shape, blocks):
                    continue
                if use_faces:
                    xp, xm, yp, ym, zp, zm = (faces[0][i,j,k], faces[1][i,j,k], faces[2][i,j,k],
                                            faces[3][i,j,k], faces[4][i,j,k], faces[5][i,j,k])
                else:
                    xp, xm, yp, ym, zp, zm = cell_faces(cells, i, j, k)
                    if weights[0] != 1.0:
                        xp = np.float32(xp * np.float32(weights[0])); xm = np.float32(xm * np.float32(weights[0]))
                    if weights[1] != 1.0:
                        yp = np.float32(yp * np.float32(weights[1])); ym = np.float32(ym * np.float32(weights[1]))
                    if weights[2] != 1.0:
                        zp = np.float32(zp * np.float32(weights[2])); zm = np.float32(zm * np.float32(weights[2]))
                diagonal = _denominator(xp, xm, yp, ym, zp, zm)
                ii, jj, kk = i+1, j+1, k+1
                value = np.float32(xp * field[ii+1,jj,kk])
                value = np.float32(value + xm * field[ii-1,jj,kk])
                value = np.float32(value + yp * field[ii,jj+1,kk])
                value = np.float32(value + ym * field[ii,jj-1,kk])
                value = np.float32(value + zp * field[ii,jj,kk+1])
                value = np.float32(value + zm * field[ii,jj,kk-1])
                value = np.float32(value - np.float32(diagonal * field[ii,jj,kk]))
                total += value * value
                maximum = max(maximum, abs(value))
                count += 1
                if precision:
                    terms = (np.float64(xp)*np.float64(field[ii+1,jj,kk]), np.float64(xm)*np.float64(field[ii-1,jj,kk]),
                             np.float64(yp)*np.float64(field[ii,jj+1,kk]), np.float64(ym)*np.float64(field[ii,jj-1,kk]),
                             np.float64(zp)*np.float64(field[ii,jj,kk+1]), np.float64(zm)*np.float64(field[ii,jj,kk-1]),
                             -np.float64(diagonal)*np.float64(field[ii,jj,kk]))
                    exact = 0.0
                    scale = 0.0
                    for term in terms:
                        exact += term
                        scale += abs(term)
                    error = float(value) - exact
                    precise_squares += exact * exact
                    precise_max = max(precise_max, abs(exact))
                    error_squares += error * error
                    error_max = max(error_max, abs(error))
                    # A conservative evaluation envelope, NOT a forward-error certificate.
                    envelope = max(envelope, (12 * 2.0**-24 / (1 - 12 * 2.0**-24)) * scale)
    return total, maximum, count, precise_squares, precise_max, error_squares, error_max, envelope


def _plans(shape, controller):
    context = controller.context
    block_rows = []
    for block in context.get("blocks", []):
        values = []
        for axis in "xyz":
            pair = block.get(axis + "_range", [0.0, 1.0])
            pair = [0.0, 1.0] if pair is None else pair
            values.extend(sorted(float(v) for v in pair))
        values.append(float(block.get("eps_val", 1.0)))
        block_rows.append(values)
    blocks = np.asarray(block_rows, dtype=np.float64).reshape((-1, 7))
    plans = []
    if controller.needs_precision:
        plans.append(("global", tuple((1, n-1) for n in shape), False))
    for name in controller.spec["regions"]:
        if name == "cut":
            selected = context.get("cut_slices")
            if selected is None:
                if not all(key in context for key in ("center_nm", "field_bounds_nm")):
                    raise ValueError("Cut stopping requires physical cut context or explicit cut_slices")
                from .mpi_io import physical_cut_slices
                slices, _ = physical_cut_slices(shape, context["center_nm"], context.get("cut_offsets_nm"),
                                                context["field_bounds_nm"])
                selected = [(s.start, s.stop) for s in slices] if slices is not None else [(0, 0)] * 3
            plans.append((name, tuple(tuple(int(v) for v in pair) for pair in selected), False))
        else:
            selected = blocks[blocks[:, 6] == 15.0]
            if not selected.size:
                raise ValueError("QD stopping requires declared eps15 blocks")
            bounds = tuple((max(0, int(math.floor(float(selected[:,2*a].min()) * (shape[a]-1))) - 1),
                            min(shape[a], int(math.ceil(float(selected[:,2*a+1].max()) * (shape[a]-1))) + 2))
                           for a in range(3))
            plans.append((name, bounds, True))
    return blocks, plans


def extra_statistics(field, cells, mask, starts, shape, controller, faces=None,
                     weights=(1.0,1.0,1.0), comm=None):
    """Reduce selected scopes only; MPI owns each node once and excludes halos."""
    started = time.monotonic()
    blocks, plans = _plans(shape, controller)
    result = {}
    if not plans:
        return result
    precision = controller.needs_precision
    dummy = np.empty((1,1,1), dtype=np.float32)
    kernel_cells = cells if cells is not None else dummy
    if faces is not None and len(faces) == 3:
        # MPI retains3oriented fields; shared dense callers pass6shifted views.
        x,y,z=faces
        faces=(x[1:,:,:],x[:-1,:,:],y[:,1:,:],y[:,:-1,:],z[:,:,1:],z[:,:,:-1])
    kernel_faces = faces if faces is not None else (dummy,dummy,dummy,dummy,dummy,dummy)
    for name, bounds, qd in plans:
        values = _measure(field, kernel_cells, mask, kernel_faces, tuple(starts), tuple(shape), bounds, blocks,
                          qd, precision, tuple(weights), cells is None)
        total, maximum, count, exact_sum, exact_max, error_sum, error_max, envelope = values
        if comm is not None:
            from mpi4py import MPI
            sums = np.array([total, exact_sum, error_sum], dtype=np.float64)
            maxima = np.array([maximum, exact_max, error_max, envelope], dtype=np.float64)
            comm.Allreduce(MPI.IN_PLACE, sums, op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, maxima, op=MPI.MAX)
            count = comm.allreduce(count, op=MPI.SUM)
            total, exact_sum, error_sum = sums
            maximum, exact_max, error_max, envelope = maxima
        values = {"rms": math.sqrt(total/count) if count else 0.0,
                  "maximum": float(maximum), "count": int(count), "global_index_bounds": bounds}
        if precision:
            values["precision"] = {"rms": math.sqrt(exact_sum/count) if count else 0.0,
                                   "maximum": float(exact_max), "evaluation_error_rms": math.sqrt(error_sum/count) if count else 0.0,
                                   "evaluation_error_maximum": float(error_max), "rounding_envelope_maximum": float(envelope),
                                   "same_float32_field_and_coefficients": True}
        result[name] = values
    controller.extra_seconds += time.monotonic() - started
    return result

