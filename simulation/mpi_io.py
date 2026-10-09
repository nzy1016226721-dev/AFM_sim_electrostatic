"""MPI-safe output helpers for distributed AFM fields."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .io_utils import _normalize_cut_offsets_nm, _unique_output_path
from .mpi_domain import DistributedField, human_bytes


def _require_mpi():
    try:
        from mpi4py import MPI
    except ImportError as exc:  # pragma: no cover - exercised on cluster
        raise RuntimeError("mpi4py is required for distributed AFM output") from exc
    return MPI


def numpy_header_bytes(shape: Sequence[int], dtype=np.float32, *, version=2) -> bytes:
    """Return a complete NumPy v2.0 header without allocating the array."""
    header = {
        "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
        "fortran_order": False,
        "shape": tuple(int(v) for v in shape),
    }
    buffer = io.BytesIO()
    writer = np.lib.format.write_array_header_1_0 if version == 1 else np.lib.format.write_array_header_2_0
    writer(buffer, header)
    return buffer.getvalue()


def _save_streamed_subarray(field, filename, output_dir, global_slices, *, header_version=2,
                           buffer_bytes=8*1024**2):
    """Bounded MPI-IO: no full owned copy and no root gather of cut pieces.

    All ranks perform the same number of collectives; nonintersecting/finished
    ranks contribute zero elements. A file subarray view preserves xyz C-order.
    """
    MPI = _require_mpi()
    decomp = field.decomposition
    comm = decomp.cart
    cut_shape = tuple(s.stop-s.start for s in global_slices)
    if decomp.rank == 0:
        path = _unique_output_path(output_dir,filename)
    else:
        path = None
    path = comm.bcast(path,root=0)
    header = numpy_header_bytes(cut_shape,version=header_version)
    local_selection, destination, counts = [],[],[]
    for selection,start,count in zip(global_slices,decomp.starts,decomp.counts):
        lo,hi = max(selection.start,start),min(selection.stop,start+count)
        counts.append(max(0,hi-lo))
        local_selection.append(slice(lo-start+1,hi-start+1))
        destination.append(lo-selection.start)
    intersects = all(counts)
    view = field.data[tuple(local_selection)] if intersects else None
    rows = counts[0]*counts[1] if intersects else 0
    rows_per_chunk = max(1,int(buffer_bytes)//(max(1,counts[2])*4))
    calls = int(comm.allreduce((rows+rows_per_chunk-1)//rows_per_chunk,op=MPI.MAX))
    handle = MPI.File.Open(comm,path,MPI.MODE_WRONLY|MPI.MODE_CREATE)
    filetype = None
    try:
        handle.Set_size(len(header)+math_prod(cut_shape)*4)
        if decomp.rank == 0:
            handle.Write_at(0,np.frombuffer(header,dtype=np.uint8))
        comm.Barrier()
        if intersects:
            filetype = MPI.FLOAT.Create_subarray(cut_shape,counts,destination,order=MPI.ORDER_C)
            filetype.Commit()
        handle.Set_view(len(header),etype=MPI.FLOAT,filetype=filetype if intersects else MPI.FLOAT,datarep="native")
        for step in range(calls):
            first = step*rows_per_chunk
            stop = min(rows,first+rows_per_chunk)
            if first >= rows:
                buffer = np.empty(0,np.float32)
            else:
                buffer = np.empty((stop-first,counts[2]),np.float32)
                for row in range(first,stop):
                    i,j = divmod(row,counts[1])
                    buffer[row-first] = view[i,j,:]
            handle.Write_all(buffer)
            del buffer
    finally:
        if filetype is not None:
            filetype.Free()
        handle.Close()
    return str(path)


def save_distributed_npy(
    field: DistributedField,
    filename: str,
    *,
    output_dir: str,
    memory_mode="standard",
) -> str:
    """Collectively write a C-order float32 ``.npy`` through MPI-IO."""
    MPI = _require_mpi()
    decomp = field.decomposition
    comm = decomp.cart
    if field.data.dtype != np.float32:
        raise TypeError("distributed NPY output currently requires float32 fields")
    if memory_mode in ("ram_first", "ram_compact"):
        return _save_streamed_subarray(field,filename,output_dir,
                                      tuple(slice(0,n) for n in decomp.global_shape))

    if decomp.rank == 0:
        path = _unique_output_path(output_dir, filename)
        header = numpy_header_bytes(decomp.global_shape, np.float32)
    else:
        path = None
        header = None
    path = comm.bcast(path, root=0)
    header = comm.bcast(header, root=0)

    handle = MPI.File.Open(comm, path, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    try:
        total_bytes = len(header) + int(math_prod(decomp.global_shape)) * np.dtype(np.float32).itemsize
        handle.Set_size(total_bytes)
        if decomp.rank == 0:
            handle.Write_at(0, np.frombuffer(header, dtype=np.uint8))
        comm.Barrier()
        filetype = MPI.FLOAT.Create_subarray(
            decomp.global_shape,
            decomp.counts,
            decomp.starts,
            order=MPI.ORDER_C,
        )
        filetype.Commit()
        try:
            handle.Set_view(len(header), etype=MPI.FLOAT, filetype=filetype, datarep="native")
            handle.Write_all(np.ascontiguousarray(field.owned, dtype=np.float32))
        finally:
            filetype.Free()
    finally:
        handle.Close()
    if decomp.rank == 0:
        print(f"Saved distributed full potential to {path}")
    return str(path)


def math_prod(values: Sequence[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def physical_cut_slices(
    global_shape: Sequence[int],
    center_nm: Sequence[float],
    box_offsets_nm,
    field_bounds_nm: Sequence[float],
) -> tuple[tuple[slice, slice, slice] | None, tuple[float, ...] | None]:
    """Resolve the same physical cut contract as the serial NPY helper."""
    shape = tuple(int(v) for v in global_shape)
    center = tuple(float(v) for v in center_nm)
    bounds = tuple(float(v) for v in field_bounds_nm)
    if len(shape) != 3 or any(v < 1 for v in shape):
        raise ValueError("global_shape must contain three positive integers")
    if len(center) != 3 or not all(np.isfinite(v) for v in center):
        raise ValueError("center_nm must contain three finite values")
    if len(bounds) != 6 or not all(np.isfinite(v) for v in bounds):
        raise ValueError("field_bounds_nm must contain six finite values")

    offsets = [] if box_offsets_nm is None else list(box_offsets_nm)
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

    slices = []
    actual = []
    for axis, n in enumerate(shape):
        flo, fhi = bounds[2 * axis], bounds[2 * axis + 1]
        if fhi < flo:
            flo, fhi = fhi, flo
        lo = max(flo, center[axis] + pairs[axis][0])
        hi = min(fhi, center[axis] + pairs[axis][1])
        if hi <= lo:
            return None, None
        span = fhi - flo
        if span <= 0:
            raise ValueError(f"invalid field extent on axis {axis}: ({flo}, {fhi})")
        i0 = max(0, min(n - 1, int(np.floor((lo - flo) / span * n))))
        i1 = max(i0, min(n - 1, int(np.ceil((hi - flo) / span * n)) - 1))
        slices.append(slice(i0, i1 + 1))
        actual.extend((flo + span * i0 / n, flo + span * (i1 + 1) / n))
    return tuple(slices), tuple(actual)


def _local_cut(field: DistributedField, global_slices: Sequence[slice]):
    decomp = field.decomposition
    local_slices = []
    destination_starts = []
    for axis_slice, start, count in zip(global_slices, decomp.starts, decomp.counts):
        lo = max(int(axis_slice.start), start)
        hi = min(int(axis_slice.stop), start + count)
        if hi <= lo:
            return None
        local_slices.append(slice(lo - start, hi - start))
        destination_starts.append(lo - int(axis_slice.start))
    chunk = np.ascontiguousarray(field.owned[tuple(local_slices)], dtype=np.float32)
    return tuple(destination_starts), chunk


def save_distributed_physical_cut(
    field: DistributedField,
    center_nm: Sequence[float],
    box_offsets_nm,
    field_bounds_nm: Sequence[float],
    *,
    filename: str,
    output_dir: str,
    max_root_bytes: int = 8 * 1024**3,
    memory_mode="standard",
) -> tuple[str | None, tuple[float, ...] | None]:
    """Gather a bounded physical cut to rank zero and save it as ``.npy``."""
    decomp = field.decomposition
    comm = decomp.cart
    if decomp.rank == 0:
        global_slices, actual = physical_cut_slices(
            decomp.global_shape, center_nm, box_offsets_nm, field_bounds_nm
        )
        if global_slices is None:
            payload = (None, None, None)
        else:
            cut_shape = tuple(s.stop - s.start for s in global_slices)
            cut_bytes = math_prod(cut_shape) * np.dtype(np.float32).itemsize
            error = None
            if memory_mode not in ("ram_first", "ram_compact") and cut_bytes > int(max_root_bytes):
                error = (
                    f"MPI cut would gather {human_bytes(cut_bytes)} on rank zero, above "
                    f"mpi.max_cut_gather_gib={max_root_bytes / 1024**3:.2f} GiB. "
                    "Use a smaller save_cut_box_nm or save_full with MPI-IO."
                )
            payload = (global_slices, actual, error)
    else:
        payload = None
    global_slices, actual, error = comm.bcast(payload, root=0)
    if error:
        raise MemoryError(error)
    if global_slices is None:
        if decomp.rank == 0:
            print("Physical cut does not intersect the distributed field; skipping cut.")
        return None, None

    if memory_mode in ("ram_first", "ram_compact"):
        path = _save_streamed_subarray(field,filename,output_dir,global_slices,header_version=1)
        if decomp.rank == 0:
            from .output_coordinates import write_coordinate_receipt
            write_coordinate_receipt(path,decomp.global_shape,field_bounds_nm,global_slices)
        return path,actual
    local = _local_cut(field, global_slices)
    pieces = comm.gather(local, root=0)
    if decomp.rank == 0:
        cut_shape = tuple(s.stop - s.start for s in global_slices)
        cut = np.empty(cut_shape, dtype=np.float32)
        for piece in pieces:
            if piece is None:
                continue
            starts, chunk = piece
            destination = tuple(
                slice(start, start + length)
                for start, length in zip(starts, chunk.shape)
            )
            cut[destination] = chunk
        path = _unique_output_path(output_dir, filename)
        np.save(path, cut)
        from .output_coordinates import write_coordinate_receipt
        write_coordinate_receipt(path, decomp.global_shape, field_bounds_nm, global_slices)
        print(f"Saved distributed physical cut to {path}: shape={cut.shape}, bounds_nm={actual}")
        del cut
    else:
        path = None
    path = comm.bcast(path, root=0)
    return str(path), actual


def distributed_probe(field: DistributedField, global_index: Sequence[int]) -> float:
    """Return one global field value on every rank without gathering the field."""
    MPI = _require_mpi()
    decomp = field.decomposition
    index = tuple(int(v) for v in global_index)
    owns = all(start <= value < start + count for value, start, count in zip(
        index, decomp.starts, decomp.counts
    ))
    if owns:
        local_index = tuple(value - start for value, start in zip(index, decomp.starts))
        value = float(field.owned[local_index])
        present = 1
    else:
        value = 0.0
        present = 0
    total = float(decomp.cart.allreduce(value, op=MPI.SUM))
    count = int(decomp.cart.allreduce(present, op=MPI.SUM))
    if count != 1:
        raise RuntimeError(f"global probe {index} had {count} owners")
    return total


__all__ = [
    "distributed_probe",
    "numpy_header_bytes",
    "physical_cut_slices",
    "save_distributed_npy",
    "save_distributed_physical_cut",
]
