"""Distributed-memory AFM Poisson solver primitives.

The existing :mod:`simulation.solver` is a shared-memory implementation: every
process owns every voxel.  This module provides a separate MPI path in which a
single global grid is split over a 3-D Cartesian communicator.  Each rank owns
only one rectangular block plus a one-node halo on every face.

The numerical method deliberately remains the package's damped weighted-Jacobi
scheme for ``div(epsilon * grad(phi)) = 0``.  MPI changes data ownership and
global reductions; it does not farm independent voltages to different ranks.
Consequently, adding nodes reduces the peak memory of one 16384^3 solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import socket
import time
from typing import Iterable, Sequence

import numpy as np
from numba import njit, prange

from .materials import generate_eps_level
from .numerics import (
    pyramid_tip_corner_radius,
    pyramid_tip_half_side,
    pyramid_tip_inside,
    resolve_tip_shape,
)
from .parallel import configure_cpu_threads


BOOL_BYTES = np.dtype(np.bool_).itemsize


def _require_mpi():
    """Import and return ``mpi4py.MPI`` with an actionable error."""
    try:
        from mpi4py import MPI
    except ImportError as exc:  # pragma: no cover - exercised on cluster
        raise RuntimeError(
            "MPI execution requires mpi4py. On Alliance load the Python and "
            "MPI modules, then install mpi4py in AFM_VENV with "
            "jobs/setup_afm_mpi_env.sh."
        ) from exc
    return MPI


def partition_axis(global_n: int, parts: int, coordinate: int) -> tuple[int, int]:
    """Return ``(start, count)`` for one balanced contiguous axis partition."""
    global_n = int(global_n)
    parts = int(parts)
    coordinate = int(coordinate)
    if global_n < 2:
        raise ValueError("global grid axes must contain at least two nodes")
    if parts < 1 or not 0 <= coordinate < parts:
        raise ValueError("invalid process-grid coordinate")
    base, extra = divmod(global_n, parts)
    count = base + (1 if coordinate < extra else 0)
    start = coordinate * base + min(coordinate, extra)
    return start, count


def validate_process_grid(
    process_grid: Sequence[int], world_size: int, global_shape: Sequence[int]
) -> tuple[int, int, int]:
    """Validate a user/Slurm supplied Cartesian process grid."""
    dims = tuple(int(v) for v in process_grid)
    shape = tuple(int(v) for v in global_shape)
    if len(dims) != 3 or any(v < 1 for v in dims):
        raise ValueError("mpi.process_grid must contain three positive integers")
    if math.prod(dims) != int(world_size):
        raise ValueError(
            f"mpi.process_grid={dims} has {math.prod(dims)} ranks, but MPI launched "
            f"{world_size} ranks"
        )
    for axis, (n, p) in enumerate(zip(shape, dims)):
        if p > n // 2:
            raise ValueError(
                f"process-grid axis {axis} has {p} ranks for only {n} nodes; "
                "every rank must own at least two nodes"
            )
    return dims


def suggest_process_grid(world_size: int, global_shape: Sequence[int]) -> tuple[int, int, int]:
    """Choose a factorization that minimizes local-block surface area.

    This pure-Python helper is used by the offline resource planner.  Runtime
    MPI may still use ``MPI.Compute_dims`` when no explicit grid is supplied;
    production commands should pass the planner's result explicitly so the
    estimate and launched layout are identical.
    """
    size = int(world_size)
    shape = tuple(int(v) for v in global_shape)
    if size < 1 or len(shape) != 3:
        raise ValueError("world_size must be positive and global_shape must be 3-D")
    best = None
    for px in range(1, size + 1):
        if size % px:
            continue
        remaining = size // px
        for py in range(1, remaining + 1):
            if remaining % py:
                continue
            pz = remaining // py
            dims = (px, py, pz)
            if any(p > n // 2 for p, n in zip(dims, shape)):
                continue
            local = tuple(n / p for n, p in zip(shape, dims))
            surface = local[0] * local[1] + local[0] * local[2] + local[1] * local[2]
            aspect = max(local) / min(local)
            score = (surface, aspect, max(dims) - min(dims), dims)
            if best is None or score < best[0]:
                best = (score, dims)
    if best is None:
        raise ValueError(f"cannot decompose global shape {shape} over {size} ranks")
    return best[1]


@dataclass
class DomainDecomposition:
    """One rank's ownership inside a 3-D Cartesian MPI decomposition."""

    cart: object
    global_shape: tuple[int, int, int]
    dims: tuple[int, int, int]
    coords: tuple[int, int, int]
    starts: tuple[int, int, int]
    counts: tuple[int, int, int]
    neighbours: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    _halo_buffers: dict[str, tuple[np.ndarray, ...]] = field(
        default_factory=dict, init=False, repr=False
    )

    @classmethod
    def create(
        cls,
        comm,
        global_shape: Sequence[int],
        process_grid: Sequence[int] | None = None,
    ) -> "DomainDecomposition":
        MPI = _require_mpi()
        shape = tuple(int(v) for v in global_shape)
        size = int(comm.Get_size())
        if process_grid is None:
            dims = tuple(int(v) for v in MPI.Compute_dims(size, [0, 0, 0]))
        else:
            dims = tuple(int(v) for v in process_grid)
        dims = validate_process_grid(dims, size, shape)
        # Reordering is disabled so rank zero remains the sole filesystem/log
        # writer and batch stdout remains straightforward to interpret.
        cart = comm.Create_cart(dims, periods=(False, False, False), reorder=False)
        coords = tuple(int(v) for v in cart.Get_coords(cart.Get_rank()))
        neighbours = tuple(tuple(int(v) for v in cart.Shift(axis, 1)) for axis in range(3))
        starts_counts = [partition_axis(n, p, c) for n, p, c in zip(shape, dims, coords)]
        starts = tuple(item[0] for item in starts_counts)
        counts = tuple(item[1] for item in starts_counts)
        if any(v < 2 for v in counts):
            raise ValueError(
                f"rank {cart.Get_rank()} owns {counts}; every local axis must contain "
                "at least two nodes"
            )
        return cls(cart, shape, dims, coords, starts, counts, neighbours)

    def for_shape(self, global_shape: Sequence[int]) -> "DomainDecomposition":
        """Reuse the same Cartesian communicator for a refinement level."""
        shape = tuple(int(v) for v in global_shape)
        validate_process_grid(self.dims, self.cart.Get_size(), shape)
        starts_counts = [
            partition_axis(n, p, c)
            for n, p, c in zip(shape, self.dims, self.coords)
        ]
        return DomainDecomposition(
            self.cart,
            shape,
            self.dims,
            self.coords,
            tuple(item[0] for item in starts_counts),
            tuple(item[1] for item in starts_counts),
            self.neighbours,
        )

    @property
    def rank(self) -> int:
        return int(self.cart.Get_rank())

    @property
    def size(self) -> int:
        return int(self.cart.Get_size())

    @property
    def stops(self) -> tuple[int, int, int]:
        return tuple(s + c for s, c in zip(self.starts, self.counts))

    @property
    def owned_slices(self) -> tuple[slice, slice, slice]:
        return (slice(1, -1), slice(1, -1), slice(1, -1))

    def allocate(self, value: float = 0.0, dtype=np.float32) -> np.ndarray:
        """Allocate one supported local field including one halo per face."""
        shape = tuple(v + 2 for v in self.counts)
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("distributed fields require float32 or float64")
        return np.full(shape, dtype.type(value), dtype=dtype)

    def owned(self, array: np.ndarray) -> np.ndarray:
        return array[self.owned_slices]

    def _buffers_for(self, dtype: np.dtype) -> tuple[np.ndarray, ...]:
        key = np.dtype(dtype).str
        if key not in self._halo_buffers:
            lx, ly, lz = self.counts
            shapes = (
                (ly, lz), (ly, lz), (ly, lz), (ly, lz),
                (lx + 2, lz), (lx + 2, lz), (lx + 2, lz), (lx + 2, lz),
                (lx + 2, ly + 2), (lx + 2, ly + 2),
                (lx + 2, ly + 2), (lx + 2, ly + 2),
            )
            self._halo_buffers[key] = tuple(np.empty(s, dtype=dtype) for s in shapes)
        return self._halo_buffers[key]

    def exchange_halos(self, array: np.ndarray) -> None:
        """Exchange faces, edges, and corners with the six Cartesian neighbours."""
        MPI = _require_mpi()
        if array.shape != tuple(v + 2 for v in self.counts):
            raise ValueError(
                f"halo field shape {array.shape} does not match local counts {self.counts}"
            )
        b = self._buffers_for(array.dtype)
        proc_null = int(MPI.PROC_NULL)

        # X sends only owned Y/Z. Later Y and Z exchanges propagate the X
        # halos into edges and corners without communicating with 26 peers.
        xm, xp = self.neighbours[0]
        np.copyto(b[0], array[1, 1:-1, 1:-1])
        self.cart.Sendrecv(b[0], dest=xm, sendtag=10, recvbuf=b[1], source=xp, recvtag=10)
        if xp != proc_null:
            array[-1, 1:-1, 1:-1] = b[1]
        np.copyto(b[2], array[-2, 1:-1, 1:-1])
        self.cart.Sendrecv(b[2], dest=xp, sendtag=11, recvbuf=b[3], source=xm, recvtag=11)
        if xm != proc_null:
            array[0, 1:-1, 1:-1] = b[3]

        ym, yp = self.neighbours[1]
        np.copyto(b[4], array[:, 1, 1:-1])
        self.cart.Sendrecv(b[4], dest=ym, sendtag=20, recvbuf=b[5], source=yp, recvtag=20)
        if yp != proc_null:
            array[:, -1, 1:-1] = b[5]
        np.copyto(b[6], array[:, -2, 1:-1])
        self.cart.Sendrecv(b[6], dest=yp, sendtag=21, recvbuf=b[7], source=ym, recvtag=21)
        if ym != proc_null:
            array[:, 0, 1:-1] = b[7]

        zm, zp = self.neighbours[2]
        np.copyto(b[8], array[:, :, 1])
        self.cart.Sendrecv(b[8], dest=zm, sendtag=30, recvbuf=b[9], source=zp, recvtag=30)
        if zp != proc_null:
            array[:, :, -1] = b[9]
        np.copyto(b[10], array[:, :, -2])
        self.cart.Sendrecv(b[10], dest=zp, sendtag=31, recvbuf=b[11], source=zm, recvtag=31)
        if zm != proc_null:
            array[:, :, 0] = b[11]


@dataclass
class DistributedField:
    decomposition: DomainDecomposition
    data: np.ndarray

    @property
    def owned(self) -> np.ndarray:
        return self.decomposition.owned(self.data)


def estimate_rank_peak_bytes(counts: Sequence[int], dtype=np.float32, *,
                             memory_mode="standard", cpu_threads=1,
                             phi_update_mode=None, residual_accumulation=None) -> int:
    """Conservative peak for the distributed solver's persistent/scratch arrays."""
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("distributed peak estimates require float32 or float64")
    float_bytes = dtype.itemsize
    lx, ly, lz = (int(v) for v in counts)
    halo = (lx + 2) * (ly + 2) * (lz + 2)
    owned = lx * ly * lz
    eps_halo = (lx + 1) * (ly + 1) * (lz + 1)
    x_faces = (lx + 1) * ly * lz
    y_faces = lx * (ly + 1) * lz
    z_faces = lx * ly * (lz + 1)
    # Two solution fields, epsilon construction scratch, three face fields,
    # one fixed mask, worst-case fixed values, and communication surfaces.
    surfaces = 4 * (
        ly * lz + (lx + 2) * lz + (lx + 2) * (ly + 2)
    )
    if memory_mode in ("ram_first", "ram_compact"):
        from .ram_first import resolve_storage_options
        _, accumulation = resolve_storage_options(memory_mode, phi_update_mode, residual_accumulation)
        if dtype != np.dtype(np.float32):
            raise ValueError("ram_first estimates require float32")
        # Two fields conservatively cover interpolation (the solver keeps ONE).
        # Retain worst-case fixed values, both temporary geometry masks,
        # exact epsilon, three old planes/worker, reusable halos and bounded IO.
        planes = 3*min(int(cpu_threads),lx)*(ly+2)*(lz+2)
        if memory_mode == "ram_compact":
            # All-unique planes can occupy the original full cell volume.
            # Count this worst case, packed mask, fixed prefixes, per-plane IO,
            # row workspace and JIT; compression estimates are not a guarantee.
            return int(4*(2*halo+eps_halo+owned+surfaces+planes)
                       +lx*((ly*lz+7)//8)+8*(lx+1)+4*(lx+1)+24*lx
                       +16*1024**2+256*1024**2
                       +(4*owned if accumulation == "array" else 0))
        return int(4*(2*halo+eps_halo+owned+surfaces+planes)
                   +2*BOOL_BYTES*owned+8*1024**2+256*1024**2
                   +(4*owned if accumulation == "array" else 0))
    if memory_mode != "standard":
        raise ValueError("unknown memory mode")
    return int(
        float_bytes * (2 * halo + eps_halo + x_faces + y_faces + z_faces + owned)
        + BOOL_BYTES * owned
        + float_bytes * surfaces
    )


def human_bytes(value: int | float) -> str:
    value = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def verify_node_memory(
    decomposition: DomainDecomposition,
    safety_fraction: float = 0.80,
    *,
    dtype=np.float32,
    memory_mode="standard",
    cpu_threads=1,
    reference_shape=(512,512,512),
    phi_update_mode=None,
    residual_accumulation=None,
) -> dict:
    """Collectively reject a rank layout whose estimated peak exceeds node RAM."""
    if not 0.1 <= float(safety_fraction) <= 0.95:
        raise ValueError("mpi.memory_fraction must be between 0.1 and 0.95")
    try:
        import psutil

        node_total = int(psutil.virtual_memory().total)
    except Exception:
        node_total = 0
    estimated = estimate_rank_peak_bytes(decomposition.counts, dtype=dtype,
                                         memory_mode=memory_mode, cpu_threads=cpu_threads,
                                         phi_update_mode=phi_update_mode, residual_accumulation=residual_accumulation)
    if memory_mode in ("ram_first", "ram_compact") and decomposition.rank == 0:
        # Rank-zero coarse epsilon staging is still global up to the reference
        # mesh. Count reference residency, the largest staged coarse field,
        # and a generous slab allowance; do not conceal this serial MPI stage.
        coarse_cells = tuple(min(n-1,int(r)) for n,r in zip(decomposition.global_shape,reference_shape))
        estimated += 4*(math.prod(reference_shape)+math.prod(coarse_cells))+32*1024**2
    record = (
        socket.gethostname(),
        estimated,
        node_total,
        decomposition.rank,
    )
    records = decomposition.cart.allgather(record)
    by_host: dict[str, dict[str, object]] = {}
    for hostname, estimated, total, rank in records:
        item = by_host.setdefault(
            hostname,
            {"estimated_bytes": 0, "total_bytes": int(total), "ranks": []},
        )
        item["estimated_bytes"] = int(item["estimated_bytes"]) + int(estimated)
        item["ranks"].append(int(rank))
    failures = []
    for hostname, item in by_host.items():
        total = int(item["total_bytes"])
        estimated = int(item["estimated_bytes"])
        if total and estimated > total * float(safety_fraction):
            failures.append(
                f"{hostname}: estimated {human_bytes(estimated)} for ranks "
                f"{item['ranks']}, allowed {human_bytes(total * float(safety_fraction))} "
                f"({safety_fraction:.0%} of {human_bytes(total)})"
            )
    if failures:
        raise MemoryError(
            "Unsafe MPI layout before allocation. Reduce ranks per node or increase "
            "nodes/memory:\n" + "\n".join(failures)
        )
    return by_host


def _axis_block_indices(
    indices: np.ndarray, global_cells: int, bounds: Sequence[float]
) -> np.ndarray:
    lo, hi = (float(v) for v in bounds)
    if hi < lo:
        lo, hi = hi, lo
    valid = (indices >= 0) & (indices < global_cells)
    coords = (indices.astype(np.float64) + 0.5) / float(global_cells)
    return np.flatnonzero(valid & (coords >= lo) & (coords < hi))


def rasterize_epsilon_halo(
    global_shape: Sequence[int],
    starts: Sequence[int],
    counts: Sequence[int],
    blocks: Iterable[dict] | None,
) -> np.ndarray:
    """Rasterize level-cell epsilon needed by one rank, including one-cell reach."""
    shape = tuple(int(v) for v in global_shape)
    starts = tuple(int(v) for v in starts)
    counts = tuple(int(v) for v in counts)
    cell_shape = tuple(v - 1 for v in shape)
    cell_indices = tuple(
        np.arange(start - 1, start + count, dtype=np.int64)
        for start, count in zip(starts, counts)
    )
    eps = np.ones(tuple(v + 1 for v in counts), dtype=np.float32)
    for block in blocks or ():
        axis_ids = []
        for axis, indices, n_cells in zip("xyz", cell_indices, cell_shape):
            bounds = block.get(f"{axis}_range", (0.0, 1.0))
            if bounds is None:
                bounds = (0.0, 1.0)
            axis_ids.append(_axis_block_indices(indices, n_cells, bounds))
        if all(ids.size for ids in axis_ids):
            eps[np.ix_(*axis_ids)] = np.float32(block.get("eps_val", 1.0))
    return eps


def _epsilon_halo_from_global(
    eps_cell: np.ndarray,
    starts: Sequence[int],
    counts: Sequence[int],
) -> np.ndarray:
    """Extract one rank's cell halo from a global cell-centred epsilon field.

    The one-cell padding outside a physical boundary is left at one.  Those
    values are never used by an interior stencil, but retaining the historical
    padding makes this array have exactly the same layout as
    :func:`rasterize_epsilon_halo`.
    """
    source = np.asarray(eps_cell, dtype=np.float32)
    if source.ndim != 3:
        raise ValueError("eps_cell must be a 3-D array")
    starts = tuple(int(v) for v in starts)
    counts = tuple(int(v) for v in counts)
    if len(starts) != 3 or len(counts) != 3 or any(v < 1 for v in counts):
        raise ValueError("starts and counts must contain three valid axes")

    result = np.ones(tuple(v + 1 for v in counts), dtype=np.float32)
    source_slices = []
    destination_slices = []
    for axis, (start, count, available) in enumerate(
        zip(starts, counts, source.shape)
    ):
        requested_start = start - 1
        requested_stop = start + count
        source_start = max(0, requested_start)
        source_stop = min(int(available), requested_stop)
        if source_stop <= source_start:
            raise ValueError(
                f"rank epsilon halo does not intersect global cells on axis {axis}"
            )
        destination_start = source_start - requested_start
        source_slices.append(slice(source_start, source_stop))
        destination_slices.append(
            slice(destination_start, destination_start + source_stop - source_start)
        )
    result[tuple(destination_slices)] = source[tuple(source_slices)]
    return np.ascontiguousarray(result)


def build_distributed_epsilon_halo(
    decomposition: DomainDecomposition,
    blocks: Iterable[dict] | None,
    *,
    reference_shape: Sequence[int] = (512, 512, 512),
    dtype=np.float32,
    memory_mode="standard",
    work_directory=None,
) -> np.ndarray:
    """Build the serial-compatible epsilon halo for one distributed rank.

    Serial production runs rasterize one high-resolution material reference and
    volume-average it onto every solver level at or below that reference.  The
    original MPI path rasterized those coarse levels directly, changing
    dielectric interface coefficients and therefore the refined solution.

    For a coarse level, rank zero now performs the established deterministic
    volume average once and streams only the required halo to each rank.  The
    staging field is bounded by ``reference_shape`` (512^3 by default), so it
    never grows with a 4096^3--16384^3 production grid.  Finer levels continue
    to be rasterized independently and locally on every rank.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("distributed epsilon requires float32 or float64")
    ref_shape = tuple(int(v) for v in reference_shape)
    if len(ref_shape) != 3 or any(v < 2 for v in ref_shape):
        raise ValueError("epsilon reference_shape must contain three values >= 2")
    cell_shape = tuple(int(v) - 1 for v in decomposition.global_shape)
    if memory_mode == "ram_compact":
        if dtype != np.dtype(np.float32):
            raise ValueError("ram_compact requires float32 epsilon")
        from .compact_storage import build_distributed_compact
        return build_distributed_compact(decomposition,blocks,reference_shape=ref_shape,directory=work_directory)
    if not all(target <= reference for target, reference in zip(cell_shape, ref_shape)):
        return np.asarray(rasterize_epsilon_halo(
            decomposition.global_shape,
            decomposition.starts,
            decomposition.counts,
            blocks,
        ), dtype=dtype)

    comm = decomposition.cart
    rank = decomposition.rank
    full_level = None
    error = None
    if rank == 0:
        try:
            level_generator = generate_eps_level
            if memory_mode == "ram_first":
                from .materials_bounded import generate_eps_level_bounded
                def level_generator(*args, **kwargs):
                    return generate_eps_level_bounded(*args, directory=work_directory, **kwargs)
            full_level = level_generator(
                decomposition.global_shape,
                blocks,
                reference_shape=ref_shape,
            )
        except Exception as exc:  # broadcast before peers enter a blocking receive
            error = f"{type(exc).__name__}: {exc}"
    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(f"rank-zero epsilon volume averaging failed: {error}")

    layouts = comm.gather(
        (decomposition.starts, decomposition.counts), root=0
    )
    tag = 27182
    if rank == 0:
        local = None
        try:
            for destination, (starts, counts) in enumerate(layouts):
                chunk = np.asarray(
                    _epsilon_halo_from_global(full_level, starts, counts),
                    dtype=dtype,
                )
                if destination == 0:
                    local = chunk
                else:
                    comm.Send(chunk, dest=destination, tag=tag)
            if local is None:
                raise RuntimeError("rank zero was absent from the gathered MPI layout")
            return local
        finally:
            del full_level

    local = np.empty(tuple(v + 1 for v in decomposition.counts), dtype=dtype)
    comm.Recv(local, source=0, tag=tag)
    return local


def build_local_face_fields(eps_halo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the three oriented dielectric face fields for one local block."""
    dtype = np.dtype(eps_halo.dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("local epsilon halo must use float32 or float64")
    e = np.asarray(eps_halo, dtype=dtype)
    if e.ndim != 3 or min(e.shape) < 3:
        raise ValueError("local epsilon halo must be a 3-D array with >=3 cells per axis")

    x_faces = np.empty((e.shape[0], e.shape[1] - 1, e.shape[2] - 1), dtype=dtype)
    np.add(e[:, :-1, :-1], e[:, 1:, :-1], out=x_faces)
    np.add(x_faces, e[:, :-1, 1:], out=x_faces)
    np.add(x_faces, e[:, 1:, 1:], out=x_faces)
    np.multiply(x_faces, dtype.type(0.25), out=x_faces)

    y_faces = np.empty((e.shape[0] - 1, e.shape[1], e.shape[2] - 1), dtype=dtype)
    np.add(e[:-1, :, :-1], e[1:, :, :-1], out=y_faces)
    np.add(y_faces, e[:-1, :, 1:], out=y_faces)
    np.add(y_faces, e[1:, :, 1:], out=y_faces)
    np.multiply(y_faces, dtype.type(0.25), out=y_faces)

    z_faces = np.empty((e.shape[0] - 1, e.shape[1] - 1, e.shape[2]), dtype=dtype)
    np.add(e[:-1, :-1, :], e[1:, :-1, :], out=z_faces)
    np.add(z_faces, e[:-1, 1:, :], out=z_faces)
    np.add(z_faces, e[1:, 1:, :], out=z_faces)
    np.multiply(z_faces, dtype.type(0.25), out=z_faces)
    return x_faces, y_faces, z_faces


def build_local_tip_mask(
    decomposition: DomainDecomposition,
    *,
    tip_z: float,
    R: float,
    r_tip: float,
    aspect_ratio: float,
    physical_params: dict | None,
    tip_shape: str = "pyramid",
) -> tuple[np.ndarray, float, float]:
    """Construct the AFM tip only over this rank's owned global nodes.

    The ``tip_shape`` selects the pyramid cross-section (square with rounded
    vertical edges, default) or the legacy cone disk.  Both this function and
    :func:`simulation.solver.build_downward_pointing_tip` evaluate the same
    shared helper from :mod:`simulation.numerics`, so serial and distributed
    masks stay bit-identical.
    """
    shape = decomposition.global_shape
    starts = decomposition.starts
    counts = decomposition.counts
    theta = np.arctan(float(aspect_ratio))

    physical_mode = bool(
        physical_params
        and physical_params.get("tip_z_nm") is not None
        and physical_params.get("R_nm") is not None
        and physical_params.get("r_tip_nm") is not None
    )
    if physical_mode:
        Lx, Ly, Lz = (float(v) for v in physical_params["domain_nm"])
        origin = tuple(float(v) for v in physical_params["origin_fraction"])
        # Slice the same global linspace arrays used by the established solver.
        # Algebraically equivalent arange/(n-1) coordinates can differ in the
        # last bit and move points exactly on a tip boundary between masks.
        x = np.linspace(0.0, Lx, shape[0])[starts[0]:starts[0] + counts[0]]
        y = np.linspace(0.0, Ly, shape[1])[starts[1]:starts[1] + counts[1]]
        z = np.linspace(0.0, Lz, shape[2])[starts[2]:starts[2] + counts[2]]
        cx, cy = origin[0] * Lx, origin[1] * Ly
        tip_abs = float(physical_params["tip_z_nm"]) + origin[2] * Lz
        radius = float(physical_params["R_nm"])
        trunc = float(physical_params["r_tip_nm"])
        a = radius * np.tan(theta)
        b = radius * np.tan(theta) ** 2
        z0 = tip_abs - b
        base_abs = z0 + np.sqrt(b**2 * (1.0 + trunc**2 / a**2))
        tip_frac, base_frac = tip_abs / Lz, base_abs / Lz
    else:
        x = np.linspace(0.0, 1.0, shape[0])[starts[0]:starts[0] + counts[0]]
        y = np.linspace(0.0, 1.0, shape[1])[starts[1]:starts[1] + counts[1]]
        z_global = np.linspace(0.0, 1.0, shape[2])
        z = z_global[starts[2]:starts[2] + counts[2]]
        cx = cy = 0.5
        tip_index = int(np.clip(tip_z, 0.0, 1.0) * (shape[2] - 1))
        tip_abs = z_global[tip_index]
        a = float(R) * np.tan(theta)
        b = float(R) * np.tan(theta) ** 2
        z0 = tip_abs - b
        trunc = float(r_tip)
        base_abs = z0 + np.sqrt(b**2 * (1.0 + trunc**2 / a**2))
        tip_frac, base_frac = tip_abs, base_abs

    shape = resolve_tip_shape(tip_shape)
    apex_radius = radius if physical_mode else float(R)
    xc = (x - cx)[:, None]
    yc = (y - cy)[None, :]
    radial2 = xc ** 2 + yc ** 2
    mask = np.zeros(counts, dtype=bool)
    for local_k, z_value in enumerate(z):
        if z_value < tip_abs or z_value > base_abs:
            continue
        dz = z_value - z0
        if dz**2 < b**2:
            continue
        cone_radius = a * np.sqrt((dz**2 / b**2) - 1.0)
        if shape == "cone":
            mask[:, :, local_k] = np.sqrt(radial2) <= min(cone_radius, trunc)
        else:
            half = pyramid_tip_half_side(cone_radius, trunc)
            mask[:, :, local_k] = pyramid_tip_inside(
                xc, yc, half, pyramid_tip_corner_radius(half, apex_radius)
            )
    return mask, float(tip_frac), float(base_frac)


def _intersect_global_slice(global_slice: slice, start: int, count: int) -> slice | None:
    lo = max(int(global_slice.start), int(start))
    hi = min(int(global_slice.stop), int(start + count))
    if hi <= lo:
        return None
    return slice(lo - start, hi - start)


def apply_fixed_geometry(
    field: DistributedField,
    *,
    Vtip: float,
    Vgate: Sequence[dict] | dict | None,
    tip_z: float,
    R: float,
    r_tip: float,
    aspect_ratio: float,
    physical_params: dict | None,
    tip_shape: str = "pyramid",
    memory_mode="standard",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Apply local gate/tip Dirichlet values and return mask + fixed values."""
    from .io_utils import gate_slices

    decomp = field.decomposition
    owned = field.owned
    compact = memory_mode == "ram_compact"
    if compact:
        from .compact_storage import PackedMask,TipPredicate
    mask = PackedMask(decomp.counts) if compact else np.zeros(decomp.counts, dtype=bool)
    gates = []
    if isinstance(Vgate, dict):
        gates = [Vgate]
    elif Vgate is not None:
        gates = list(Vgate)
    for gate in gates:
        slices = gate_slices(*decomp.global_shape, gate)
        local = tuple(
            _intersect_global_slice(s, start, count)
            for s, start, count in zip(slices, decomp.starts, decomp.counts)
        )
        if all(s is not None for s in local):
            owned[local] = owned.dtype.type(gate.get("Vgate_val", 0.0))
            if compact:
                mask.mark_box(local)
            else:
                mask[local] = True

    if compact:
        tip=TipPredicate(decomp.global_shape,starts=decomp.starts,counts=decomp.counts,
            tip_z=tip_z,R=R,r_tip=r_tip,aspect_ratio=aspect_ratio,
            physical_params=physical_params,tip_shape=tip_shape)
        tip.apply(field.data,mask,owned.dtype.type(Vtip),offset=(1,1,1))
        return mask,mask.fixed_values(field.data,(1,1,1)),tip.tip_pos,tip.base_pos

    tip_mask, tip_pos, base_pos = build_local_tip_mask(
        decomp,
        tip_z=tip_z,
        R=R,
        r_tip=r_tip,
        aspect_ratio=aspect_ratio,
        physical_params=physical_params,
        tip_shape=tip_shape,
    )
    owned[tip_mask] = owned.dtype.type(Vtip)
    mask[tip_mask] = True
    fixed_values = owned[mask]
    return mask, fixed_values, tip_pos, base_pos


@njit(parallel=True, cache=True, fastmath=False)
def _jacobi_owned(
    current,
    target,
    mask,
    x_faces,
    y_faces,
    z_faces,
    start_x,
    start_y,
    start_z,
    global_x,
    global_y,
    global_z,
    omega,
):
    lx, ly, lz = mask.shape
    for i in prange(lx):
        gi = start_x + i
        for j in range(ly):
            gj = start_y + j
            for k in range(lz):
                gk = start_z + k
                old = current[i + 1, j + 1, k + 1]
                if (
                    mask[i, j, k]
                    or gi == 0 or gi == global_x - 1
                    or gj == 0 or gj == global_y - 1
                    or gk == 0 or gk == global_z - 1
                ):
                    target[i + 1, j + 1, k + 1] = old
                    continue
                axp = x_faces[i + 1, j, k]
                axm = x_faces[i, j, k]
                ayp = y_faces[i, j + 1, k]
                aym = y_faces[i, j, k]
                azp = z_faces[i, j, k + 1]
                azm = z_faces[i, j, k]

                # Match the established shared-memory solver operation for
                # operation.  In particular, its denominator is accumulated
                # in the order axp, axm, ayp, aym, azp, azm and rounded to
                # float32 after every addition.  The former MPI ordering
                # changed the last bits at dielectric interfaces, then
                # amplified those differences over thousands of iterations.
                denom = axp
                denom = denom + axm
                denom = denom + ayp
                denom = denom + aym
                denom = denom + azp
                denom = denom + azm

                numerator = axp * current[i + 2, j + 1, k + 1]
                numerator = numerator + axm * current[i, j + 1, k + 1]
                numerator = numerator + ayp * current[i + 1, j + 2, k + 1]
                numerator = numerator + aym * current[i + 1, j, k + 1]
                numerator = numerator + azp * current[i + 1, j + 1, k + 2]
                numerator = numerator + azm * current[i + 1, j + 1, k]

                value = np.float32(numerator / denom)
                residual = np.float32(value - old)
                update = np.float32(omega * residual)
                target[i + 1, j + 1, k + 1] = np.float32(old + update)


@njit(parallel=True, cache=True, fastmath=False)
def _jacobi_owned_float64(
    current,
    target,
    mask,
    x_faces,
    y_faces,
    z_faces,
    start_x,
    start_y,
    start_z,
    global_x,
    global_y,
    global_z,
    omega,
):
    """Double-precision distributed Jacobi update for opt-in diagnostics."""
    lx, ly, lz = mask.shape
    for i in prange(lx):
        gi = start_x + i
        for j in range(ly):
            gj = start_y + j
            for k in range(lz):
                gk = start_z + k
                old = current[i + 1, j + 1, k + 1]
                if (
                    mask[i, j, k]
                    or gi == 0 or gi == global_x - 1
                    or gj == 0 or gj == global_y - 1
                    or gk == 0 or gk == global_z - 1
                ):
                    target[i + 1, j + 1, k + 1] = old
                    continue
                axp = x_faces[i + 1, j, k]
                axm = x_faces[i, j, k]
                ayp = y_faces[i, j + 1, k]
                aym = y_faces[i, j, k]
                azp = z_faces[i, j, k + 1]
                azm = z_faces[i, j, k]

                denom = axp + axm
                denom = denom + ayp
                denom = denom + aym
                denom = denom + azp
                denom = denom + azm

                numerator = axp * current[i + 2, j + 1, k + 1]
                numerator = numerator + axm * current[i, j + 1, k + 1]
                numerator = numerator + ayp * current[i + 1, j + 2, k + 1]
                numerator = numerator + aym * current[i + 1, j, k + 1]
                numerator = numerator + azp * current[i + 1, j + 1, k + 2]
                numerator = numerator + azm * current[i + 1, j + 1, k]

                candidate = numerator / denom
                target[i + 1, j + 1, k + 1] = old + omega * (candidate - old)


@njit(parallel=True, cache=True, fastmath=False)
def _residual_rows(
    current,
    mask,
    x_faces,
    y_faces,
    z_faces,
    start_x,
    start_y,
    start_z,
    global_x,
    global_y,
    global_z,
    row_sums,
    row_max,
    row_counts,
):
    lx, ly, lz = mask.shape
    for i in prange(lx):
        gi = start_x + i
        total = 0.0
        maximum = 0.0
        count = 0
        for j in range(ly):
            gj = start_y + j
            for k in range(lz):
                gk = start_z + k
                if (
                    mask[i, j, k]
                    or gi == 0 or gi == global_x - 1
                    or gj == 0 or gj == global_y - 1
                    or gk == 0 or gk == global_z - 1
                ):
                    continue
                axp = x_faces[i + 1, j, k]
                axm = x_faces[i, j, k]
                ayp = y_faces[i, j + 1, k]
                aym = y_faces[i, j, k]
                azp = z_faces[i, j, k + 1]
                azm = z_faces[i, j, k]
                denom = axp
                denom = denom + axm
                denom = denom + ayp
                denom = denom + aym
                denom = denom + azp
                denom = denom + azm
                center = current[i + 1, j + 1, k + 1]
                numerator = axp * current[i + 2, j + 1, k + 1]
                numerator = numerator + axm * current[i, j + 1, k + 1]
                numerator = numerator + ayp * current[i + 1, j + 2, k + 1]
                numerator = numerator + aym * current[i + 1, j, k + 1]
                numerator = numerator + azp * current[i + 1, j + 1, k + 2]
                numerator = numerator + azm * current[i + 1, j + 1, k]
                residual = numerator - denom * center
                total += residual * residual
                absolute = abs(residual)
                if absolute > maximum:
                    maximum = absolute
                count += 1
        row_sums[i] = total
        row_max[i] = maximum
        row_counts[i] = count


def _apply_neumann_boundaries(array: np.ndarray, decomposition: DomainDecomposition) -> None:
    starts = decomposition.starts
    stops = decomposition.stops
    shape = decomposition.global_shape
    if starts[0] == 0:
        array[1, :, :] = array[2, :, :]
    if stops[0] == shape[0]:
        array[-2, :, :] = array[-3, :, :]
    if starts[1] == 0:
        array[:, 1, :] = array[:, 2, :]
    if stops[1] == shape[1]:
        array[:, -2, :] = array[:, -3, :]
    if starts[2] == 0:
        array[:, :, 1] = array[:, :, 2]
    if stops[2] == shape[2]:
        array[:, :, -2] = array[:, :, -3]


def distributed_residual(
    field: DistributedField,
    mask: np.ndarray,
    faces: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    epsilon_cells=None,
    residual_accumulation="row_array",
    residual_volume=None,
) -> tuple[float, float, int]:
    """Return global RMS/max residuals and the global free-node count."""
    MPI = _require_mpi()
    decomp = field.decomposition
    if epsilon_cells is not None:
        from .ram_first import residual_cells_stats
        compact = False
        if not isinstance(epsilon_cells, np.ndarray):
            from .compact_storage import PlaneEpsilon
            compact = isinstance(epsilon_cells, PlaneEpsilon)
        local_sum, local_max, local_count = residual_cells_stats(
            field.data, epsilon_cells.kernel if compact else epsilon_cells,
            mask.kernel_view() if compact else mask, decomp.starts, decomp.global_shape,
            residual_accumulation, residual_volume, pairwise=True)
    else:
        row_sums = np.empty(decomp.counts[0], dtype=np.float64)
        row_max = np.empty(decomp.counts[0], dtype=np.float64)
        row_counts = np.empty(decomp.counts[0], dtype=np.int64)
        _residual_rows(
            field.data,
            mask,
            *faces,
            *decomp.starts,
            *decomp.global_shape,
            row_sums,
            row_max,
            row_counts,
        )
        local_sum = float(np.sum(row_sums, dtype=np.float64))
        local_max = float(np.max(row_max)) if row_max.size else 0.0
        local_count = int(np.sum(row_counts, dtype=np.int64))
    global_sum = float(decomp.cart.allreduce(local_sum, op=MPI.SUM))
    global_max = float(decomp.cart.allreduce(local_max, op=MPI.MAX))
    global_count = int(decomp.cart.allreduce(local_count, op=MPI.SUM))
    rms = math.sqrt(global_sum / global_count) if global_count else 0.0
    return rms, global_max, global_count


def _append_root_csv(path: str, header: Sequence[str], row: Sequence[object]) -> None:
    import csv

    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def solve_distributed_level(
    field: DistributedField,
    mask: np.ndarray,
    fixed_values: np.ndarray,
    faces: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    omega: float,
    tol: float,
    max_iter: int | None = None,
    max_runtime: float | None,
    cpu_threads: int,
    output_dir: str,
    residual_check_interval: int = 10,
    diagnostic_iterations: int | None = None,
    epsilon_cells=None,
    phi_update_mode=None,
    residual_accumulation=None,
) -> tuple[DistributedField, dict]:
    """Solve one distributed refinement level with global convergence tests.

    ``max_iter`` is retained as an ignored compatibility argument for existing
    JSON files. A production solve stops only on convergence or its configured
    wall-clock deadline. ``diagnostic_iterations`` is reserved for the named
    float64 memory-smoke path.
    """
    decomp = field.decomposition
    rank = decomp.rank
    actual_threads = configure_cpu_threads(cpu_threads)
    current = field.data
    target = None
    scratch = None
    residual_volume = None
    update_mode, accumulation = "buffered", "row_array"
    compact = False
    if epsilon_cells is not None and not isinstance(epsilon_cells, np.ndarray):
        from .compact_storage import PlaneEpsilon, PackedMask, restore_fixed
        compact = isinstance(epsilon_cells, PlaneEpsilon)
    kernel_epsilon,kernel_mask,fixed_prefixes=epsilon_cells,mask,None
    if epsilon_cells is not None:
        from .ram_first import allocate_stream_scratch, jacobi_cells_in_place, resolve_storage_options
        update_mode, accumulation = resolve_storage_options("ram_compact" if compact else "ram_first", phi_update_mode, residual_accumulation)
        if current.dtype != np.float32 or epsilon_cells.dtype != np.float32:
            raise ValueError("ram_first MPI requires float32 fields and epsilon")
        if compact:
            if not isinstance(mask,PackedMask):
                mask=PackedMask.from_dense(mask)
            if isinstance(fixed_values,tuple):
                fixed_values,fixed_prefixes=fixed_values
            else:
                fixed_prefixes=mask.prefixes()
            kernel_epsilon,kernel_mask=epsilon_cells.kernel,mask.kernel_view()
        scratch = allocate_stream_scratch(current, mask, actual_threads)
        if update_mode == "buffered":
            target = np.empty_like(current)
        if accumulation == "array":
            residual_volume = np.empty(mask.shape, np.float32)
    else:
        from .ram_first import resolve_storage_options
        resolve_storage_options("standard", phi_update_mode, residual_accumulation)
        target = np.empty_like(current)
    decomp.exchange_halos(current)
    started = time.monotonic()
    residual = float("inf")
    residual_max = float("inf")
    reason = "time_limit"
    completed = 0

    if rank == 0:
        print(
            f"   Starting distributed 3D solver: "
            f"{decomp.global_shape[0]}x{decomp.global_shape[1]}x{decomp.global_shape[2]}, "
            f"ranks={decomp.size}, process_grid={decomp.dims}, threads/rank={cpu_threads}"
        )

    if max_iter is not None and rank == 0:
        print("   Ignoring legacy mpi.max_iter: solver stopping is tolerance/time based.")
    if diagnostic_iterations is not None:
        diagnostic_iterations = int(diagnostic_iterations)
        if diagnostic_iterations < 1:
            raise ValueError("diagnostic_iterations must be positive")
        if rank == 0:
            print(
                "   Diagnostic-only fixed iteration count: "
                f"{diagnostic_iterations}"
            )

    iteration = 0
    while True:
        iteration += 1
        if epsilon_cells is not None:
            destination = current
            if update_mode == "buffered":
                np.copyto(target, current)
                destination = target
            jacobi_cells_in_place(destination, kernel_epsilon, kernel_mask, decomp.starts,
                                 decomp.global_shape, np.float32(omega), scratch)
            target = destination
        elif current.dtype == np.float64:
            _jacobi_owned_float64(
                current,
                target,
                mask,
                *faces,
                *decomp.starts,
                *decomp.global_shape,
                np.float64(omega),
            )
        else:
            _jacobi_owned(
                current,
                target,
                mask,
                *faces,
                *decomp.starts,
                *decomp.global_shape,
                np.float32(omega),
            )
        _apply_neumann_boundaries(target, decomp)
        if compact:
            restore_fixed(target,mask.bits,mask.shape,fixed_values,fixed_prefixes,(1,1,1))
        else:
            decomp.owned(target)[mask] = fixed_values
        if epsilon_cells is None or update_mode == "buffered":
            current, target = target, current
        decomp.exchange_halos(current)
        completed = iteration
        elapsed = time.monotonic() - started

        check = iteration <= 5 or iteration % int(residual_check_interval) == 0
        if check:
            residual, residual_max, free_count = distributed_residual(
                DistributedField(decomp, current), mask, faces, epsilon_cells, accumulation, residual_volume
            )
            if rank == 0:
                _append_root_csv(
                    os.path.join(output_dir, "residual_history_mpi.csv"),
                    ("iteration", "residual_avg", "residual_max", "free_nodes"),
                    (iteration, f"{residual:.12e}", f"{residual_max:.12e}", free_count),
                )
            if diagnostic_iterations is None and residual < float(tol):
                reason = "converged"
                break
        if diagnostic_iterations is not None and iteration >= diagnostic_iterations:
            if not check:
                residual, residual_max, free_count = distributed_residual(
                    DistributedField(decomp, current), mask, faces, epsilon_cells, accumulation, residual_volume
                )
                if rank == 0:
                    _append_root_csv(
                        os.path.join(output_dir, "residual_history_mpi.csv"),
                        ("iteration", "residual_avg", "residual_max", "free_nodes"),
                        (
                            iteration,
                            f"{residual:.12e}",
                            f"{residual_max:.12e}",
                            free_count,
                        ),
                    )
            reason = "diagnostic_iteration_limit"
            break
        if max_runtime is not None and elapsed > float(max_runtime):
            if not check:
                residual, residual_max, _ = distributed_residual(
                    DistributedField(decomp, current), mask, faces, epsilon_cells, accumulation, residual_volume
                )
            reason = "time_limit"
            break

    elapsed = time.monotonic() - started
    if rank == 0:
        if reason == "converged":
            print(
                f"Converged in {completed} iterations in {elapsed:.2f} s; "
                f"residual={residual:.5e}"
            )
        elif reason == "diagnostic_iteration_limit":
            print(
                "Diagnostic iteration limit reached: "
                f"{completed} iterations in {elapsed:.2f} s; residual={residual:.5e}"
            )
        elif reason == "time_limit":
            print(
                f"MPI solver stopped at per-level runtime limit ({max_runtime} s): "
                f"{completed} iterations in {elapsed:.2f} s; residual={residual:.5e}"
            )
        else:
            print(
                f"MPI solver NOT converged after {completed} iterations in {elapsed:.2f} s; "
                f"residual={residual:.5e}"
            )
    return DistributedField(decomp, current), {
        "iterations": completed,
        "elapsed": elapsed,
        "residual": residual,
        "residual_max": residual_max,
        "reason": reason,
        "memory_mode": "ram_compact" if compact else ("ram_first" if epsilon_cells is not None else "standard"),
        "epsilon_storage_bytes": epsilon_cells.nbytes if compact else None,
        "mask_storage_bytes": mask.nbytes if compact else None,
        "stream_scratch_bytes": int(scratch.nbytes) if scratch is not None else 0,
        "phi_update_mode": update_mode,
        "residual_accumulation": accumulation,
        "phi_buffer_bytes": int(current.nbytes) if update_mode == "buffered" else 0,
        "residual_workspace_bytes": int(residual_volume.nbytes) if residual_volume is not None else 0,
    }


@njit(parallel=True, cache=True, fastmath=False)
def _trilinear_prolongate(
    coarse,
    fine_owned,
    x_lo,
    x_w,
    y_lo,
    y_w,
    z_lo,
    z_w,
):
    for i in prange(fine_owned.shape[0]):
        i0 = x_lo[i]
        i1 = i0 + 1
        wx = x_w[i]
        wx0 = 1.0 - wx
        for j in range(fine_owned.shape[1]):
            j0 = y_lo[j]
            j1 = j0 + 1
            wy = y_w[j]
            wy0 = 1.0 - wy
            for k in range(fine_owned.shape[2]):
                k0 = z_lo[k]
                k1 = k0 + 1
                wz = z_w[k]
                wz0 = 1.0 - wz

                # Match scipy.ndimage.zoom(order=1, grid_mode=False) rather
                # than using nested lerps.  SciPy accumulates the eight
                # footprint values in C-order (last axis fastest), carrying
                # the interpolation products and accumulator in float64 before
                # the single final cast to the float32 output.  Keeping the
                # same operation order removes refinement-only roundoff from
                # the otherwise identical serial and distributed solvers.
                value = 0.0
                coeff = float(coarse[i0, j0, k0])
                coeff *= wx0
                coeff *= wy0
                coeff *= wz0
                value += coeff
                coeff = float(coarse[i0, j0, k1])
                coeff *= wx0
                coeff *= wy0
                coeff *= wz
                value += coeff
                coeff = float(coarse[i0, j1, k0])
                coeff *= wx0
                coeff *= wy
                coeff *= wz0
                value += coeff
                coeff = float(coarse[i0, j1, k1])
                coeff *= wx0
                coeff *= wy
                coeff *= wz
                value += coeff
                coeff = float(coarse[i1, j0, k0])
                coeff *= wx
                coeff *= wy0
                coeff *= wz0
                value += coeff
                coeff = float(coarse[i1, j0, k1])
                coeff *= wx
                coeff *= wy0
                coeff *= wz
                value += coeff
                coeff = float(coarse[i1, j1, k0])
                coeff *= wx
                coeff *= wy
                coeff *= wz0
                value += coeff
                coeff = float(coarse[i1, j1, k1])
                coeff *= wx
                coeff *= wy
                coeff *= wz
                value += coeff
                # Assignment performs the one final cast for float32 fields
                # and retains the full accumulator for opt-in float64 fields.
                fine_owned[i, j, k] = value


def _prolongation_axis(
    old_n: int, new_n: int, fine_start: int, fine_count: int, coarse_start: int, coarse_count: int
) -> tuple[np.ndarray, np.ndarray]:
    fine_indices = np.arange(fine_start, fine_start + fine_count, dtype=np.float64)
    # SciPy computes this scale once in float64 and then multiplies each
    # output coordinate by it.  Division after the coordinate multiplication
    # can differ by a last bit and therefore changes the float32 refinement
    # seed even when the interpolation formula is otherwise equivalent.
    scale = float(old_n - 1) / float(new_n - 1)
    positions = fine_indices * scale
    low_global = np.floor(positions).astype(np.int64)
    high_global = np.minimum(low_global + 1, old_n - 1)
    # The local coarse field includes one halo on each side.  At the upper
    # physical endpoint low==high.  The interpolation kernel always reads
    # i0+1, so represent that endpoint as the preceding interval at weight 1.
    # The former weight-0 representation still read the uninitialised outer
    # physical ghost; NaN*0 can remain NaN and arbitrary finite ghost values
    # introduced nondeterministic refinement-boundary error.
    weights = positions - low_global
    low_local = low_global - int(coarse_start) + 1
    high_local = high_global - int(coarse_start) + 1
    if low_local.min() < 0 or high_local.max() >= coarse_count + 2:
        raise RuntimeError(
            "Refinement mapping reaches beyond the one-cell coarse halo. "
            "Use a balanced process grid whose dimensions divide the refinement hierarchy."
        )
    endpoint = high_global == low_global
    low_local[endpoint] -= 1
    weights[endpoint] = 1.0
    return low_local.astype(np.int64), weights


def prolongate_distributed(
    coarse: DistributedField, fine_decomposition: DomainDecomposition
) -> DistributedField:
    """Endpoint-aligned trilinear prolongation using only local coarse halos."""
    old = coarse.decomposition
    if old.dims != fine_decomposition.dims or old.coords != fine_decomposition.coords:
        raise ValueError("coarse and fine fields must use the same Cartesian rank layout")
    old.exchange_halos(coarse.data)
    axes = [
        _prolongation_axis(old_n, new_n, fs, fc, cs, cc)
        for old_n, new_n, fs, fc, cs, cc in zip(
            old.global_shape,
            fine_decomposition.global_shape,
            fine_decomposition.starts,
            fine_decomposition.counts,
            old.starts,
            old.counts,
        )
    ]
    fine_data = fine_decomposition.allocate(0.0, dtype=coarse.data.dtype)
    _trilinear_prolongate(
        coarse.data,
        fine_decomposition.owned(fine_data),
        axes[0][0], axes[0][1],
        axes[1][0], axes[1][1],
        axes[2][0], axes[2][1],
    )
    fine_decomposition.exchange_halos(fine_data)
    return DistributedField(fine_decomposition, fine_data)


def rank_layout_table(
    global_shape: Sequence[int], process_grid: Sequence[int], dtype=np.float32,
    *, memory_mode="standard", cpu_threads=1, reference_shape=(512,512,512),
    phi_update_mode=None, residual_accumulation=None,
) -> list[dict[str, object]]:
    """Pure-Python layout/memory table used by tests and offline preflight."""
    shape = tuple(int(v) for v in global_shape)
    dims = tuple(int(v) for v in process_grid)
    rows = []
    rank = 0
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                starts_counts = [
                    partition_axis(n, p, c)
                    for n, p, c in zip(shape, dims, (x, y, z))
                ]
                counts = tuple(item[1] for item in starts_counts)
                peak = estimate_rank_peak_bytes(
                    counts, dtype=dtype, memory_mode=memory_mode, cpu_threads=cpu_threads,
                    phi_update_mode=phi_update_mode, residual_accumulation=residual_accumulation,
                )
                if rank == 0 and memory_mode in ("ram_first", "ram_compact"):
                    coarse_cells = tuple(min(n-1,int(r)) for n,r in zip(shape,reference_shape))
                    peak += 4*(math.prod(reference_shape)+math.prod(coarse_cells))+32*1024**2
                rows.append(
                    {
                        "rank": rank,
                        "coords": (x, y, z),
                        "starts": tuple(item[0] for item in starts_counts),
                        "counts": counts,
                        "estimated_peak_bytes": peak,
                    }
                )
                rank += 1
    return rows


__all__ = [
    "DistributedField",
    "DomainDecomposition",
    "apply_fixed_geometry",
    "build_distributed_epsilon_halo",
    "build_local_face_fields",
    "build_local_tip_mask",
    "distributed_residual",
    "estimate_rank_peak_bytes",
    "human_bytes",
    "partition_axis",
    "prolongate_distributed",
    "rank_layout_table",
    "rasterize_epsilon_halo",
    "solve_distributed_level",
    "suggest_process_grid",
    "validate_process_grid",
    "verify_node_memory",
]
