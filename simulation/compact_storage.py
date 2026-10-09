"""Lossless plane material storage and exact packed/predicate fixed masks.

Only this module knows representation details. Kernel views satisfy the same
integer-index/shape interface as dense arrays, so the original ram_first
Jacobi/face/residual kernels can run unchanged. No implicit dense conversion.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import numpy as np
from numba import njit, prange, types
from numba.experimental import jitclass

from .materials import _as_blocks, _range
from .numerics import resolve_tip_shape, pyramid_tip_half_side, pyramid_tip_corner_radius


@jitclass([("bank", types.float32[:, :, ::1]), ("ids", types.int32[::1]),
           ("shape", types.UniTuple(types.int64, 3))])
class _CellView:
    def __init__(self, bank, ids, shape):
        self.bank, self.ids, self.shape = bank, ids, shape

    def __getitem__(self, index):
        return self.bank[self.ids[index[0]], index[1], index[2]]


@jitclass([("bits", types.uint8[:, ::1]), ("shape", types.UniTuple(types.int64, 3)),
           ("offset", types.UniTuple(types.int64, 3)), ("plane_nz", types.int64)])
class _MaskView:
    def __init__(self, bits, shape, offset, plane_nz):
        self.bits, self.shape, self.offset, self.plane_nz = bits, shape, offset, plane_nz

    def __getitem__(self, index):
        i, j, k = index
        linear = (j+self.offset[1])*self.plane_nz + k+self.offset[2]
        return ((self.bits[i+self.offset[0], linear >> 3] >> (linear & 7)) & 1) != 0


class PlaneEpsilon:
    dtype = np.dtype(np.float32)

    def __init__(self, shape, bank, ids, path):
        self.shape, self.bank, self.ids = tuple(shape), bank, ids
        self.path, self._kernel = Path(path), None

    @property
    def kernel(self):
        if self.bank is None:
            raise RuntimeError("Closed epsilon bank")
        if self._kernel is None:
            self._kernel = _CellView(self.bank, self.ids, self.shape)
        return self._kernel

    @property
    def nbytes(self):
        return int(self.bank.nbytes + self.ids.nbytes)

    def plane(self, i):
        return self.bank[self.ids[i]]

    def to_dense(self):
        """Explicit diagnostic allocation, never called by the compact solve."""
        result = np.empty(self.shape, np.float32)
        for i in range(self.shape[0]):
            result[i] = self.plane(i)
        return result

    def close(self):
        self._kernel = None
        if self.bank is not None:
            self.bank._mmap.close()
            self.bank = None
            self.path.unlink(missing_ok=True)

    @classmethod
    def from_dense(cls, array, *, directory=None):
        if array.dtype != np.float32 or array.ndim != 3:
            raise ValueError("Lossless epsilon input must be 3-D float32")
        with _PlaneWriter(array.shape, directory) as writer:
            for i in range(array.shape[0]):
                writer.add(array[i], i, i+1)
            return writer.finish()


class _PlaneWriter:
    def __init__(self, shape, directory=None):
        self.shape = tuple(int(v) for v in shape)
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError("Epsilon shape must contain three positive dimensions")
        fd, name = tempfile.mkstemp(prefix="afm_plane_bank_", suffix=".bin", dir=directory)
        self.path = Path(name)
        self.handle = os.fdopen(fd, "w+b")
        self.ids = np.full(self.shape[0], -1, np.int32)
        self.lookup, self.count, self.finished = {}, 0, False
        self.plane_bytes = self.shape[1]*self.shape[2]*4

    def add(self, plane, start, stop):
        if plane.dtype != np.float32 or plane.shape != self.shape[1:]:
            raise ValueError("Bank planes must retain their exact float32 shape")
        blob = np.ascontiguousarray(plane).tobytes()
        key = hashlib.sha256(blob).digest()
        identifier = None
        for candidate in self.lookup.get(key, ()):
            self.handle.seek(candidate*self.plane_bytes)
            if self.handle.read(self.plane_bytes) == blob:  # verify hash collisions
                identifier = candidate
                break
        if identifier is None:
            identifier = self.count
            self.handle.seek(0, os.SEEK_END)
            self.handle.write(blob)
            self.lookup.setdefault(key, []).append(identifier)
            self.count += 1
        self.ids[start:stop] = identifier

    def finish(self):
        if np.any(self.ids < 0):
            raise ValueError("Every x-plane must be supplied before sealing a bank")
        self.handle.close()
        bank = np.memmap(self.path, mode="r+", dtype=np.float32,
                         shape=(self.count, *self.shape[1:]))
        self.finished = True
        return PlaneEpsilon(self.shape, bank, self.ids, self.path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self.handle.closed:
            self.handle.close()
        if not self.finished:
            self.path.unlink(missing_ok=True)


def rasterize_planes(global_cells, blocks=None, *, starts=(0,0,0), counts=None, directory=None):
    """Exact cell-centre, last-block-wins raster, including MPI padding of1."""
    global_cells = tuple(int(v) for v in global_cells)
    counts = global_cells if counts is None else tuple(int(v) for v in counts)
    indices = [np.arange(s,s+c,dtype=np.int64) for s,c in zip(starts,counts)]
    coords = [(idx.astype(np.float64)+.5)/n for idx,n in zip(indices,global_cells)]
    selected = []
    breaks = {0, counts[0]}
    for block in _as_blocks(blocks):
        axes = []
        for axis, idx, coord, n in zip("xyz", indices, coords, global_cells):
            low, high = _range(block,axis)
            axes.append(np.flatnonzero((idx>=0)&(idx<n)&(coord>=low)&(coord<high)))
        if all(ids.size for ids in axes):
            first, stop = int(axes[0][0]), int(axes[0][-1])+1
            selected.append((first,stop,axes[1],axes[2],float(block.get("eps_val",1))))
            breaks.update((first,stop))
    breaks = sorted(breaks)
    with _PlaneWriter(counts, directory) as writer:
        for first, stop in zip(breaks,breaks[1:]):
            plane = np.ones(counts[1:],np.float32)
            for low, high, iy, iz, value in selected:
                if low <= first < high:
                    plane[np.ix_(iy,iz)] = value
            writer.add(plane,first,stop)
        return writer.finish()


def build_reference_planes(reference_shape, blocks=None, directory=None):
    """Create the original reference without touching a full resident mapping."""
    cells = rasterize_planes(reference_shape, blocks, directory=directory)
    fd, path = tempfile.mkstemp(prefix="afm_eps_reference_compact_", suffix=".npy", dir=directory)
    os.close(fd)
    try:
        initial = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=cells.shape)
        initial._mmap.close()
        for i in range(cells.shape[0]):
            writer = np.load(path,mmap_mode="r+",allow_pickle=False)
            try:
                writer[i] = cells.plane(i)
                writer.flush()
            finally:
                writer._mmap.close()
        return path, np.load(path,mmap_mode="r",allow_pickle=False)
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise
    finally:
        cells.close()


def generate_eps_compact(phi_shape, blocks=None, reference_shape=(512,512,512),
                         reference=None, *, directory=None):
    target = tuple(max(int(v)-1,1) for v in phi_shape)
    if not all(t<=int(r) for t,r in zip(target,reference_shape)):
        return rasterize_planes(target,blocks,directory=directory)
    from .materials import release_eps_reference
    from .materials_bounded import average_reference_bounded
    own_path = own_reference = final = None
    fd, final_name = tempfile.mkstemp(prefix="afm_compact_rebin_",suffix=".npy",dir=directory)
    os.close(fd)
    try:
        if reference is None:
            own_path, own_reference = build_reference_planes(reference_shape,blocks,directory)
            reference = own_reference
        final = average_reference_bounded(reference,target,directory=directory,final_path=final_name)
        # Reopen each plane so reading the large final NPY does not retain its
        # complete resident mapping alongside the compressed bank.
        final._mmap.close()
        final = None
        with _PlaneWriter(target,directory) as writer:
            for i in range(target[0]):
                reader = np.load(final_name,mmap_mode="r",allow_pickle=False)
                try:
                    plane = np.array(reader[i],copy=True)
                finally:
                    reader._mmap.close()
                writer.add(plane,i,i+1)
            return writer.finish()
    finally:
        if final is not None:
            final._mmap.close()
        Path(final_name).unlink(missing_ok=True)
        if own_path is not None:
            release_eps_reference(own_path,own_reference)


class PackedMask:
    dtype = np.dtype(np.bool_)

    def __init__(self, shape, bits=None):
        self.shape = tuple(int(v) for v in shape)
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError("Mask shape must have three positive dimensions")
        packed_shape = (self.shape[0], (self.shape[1]*self.shape[2]+7)//8)
        self.bits = np.zeros(packed_shape,np.uint8) if bits is None else bits
        if self.bits.shape != packed_shape or self.bits.dtype != np.uint8:
            raise ValueError("Invalid packed mask")

    @property
    def nbytes(self):
        return int(self.bits.nbytes)

    def plane(self, i):
        return np.unpackbits(self.bits[i],bitorder="little",count=self.shape[1]*self.shape[2]).view(np.bool_).reshape(self.shape[1:])

    def put_plane(self, i, plane, *, union=False):
        values = np.packbits(plane.reshape(-1),bitorder="little")
        if union:
            np.bitwise_or(self.bits[i],values,out=self.bits[i])
        else:
            self.bits[i] = values

    def mark_box(self, slices):
        for i in range(*slices[0].indices(self.shape[0])):
            plane = self.plane(i)
            plane[slices[1:]] = True
            self.put_plane(i,plane)

    def kernel_view(self, *, offset=(0,0,0), shape=None):
        offset = tuple(int(v) for v in offset)
        shape = self.shape if shape is None else tuple(int(v) for v in shape)
        if any(o<0 or o+n>s for o,n,s in zip(offset,shape,self.shape)):
            raise ValueError("Mask view exceeds packed storage")
        return _MaskView(self.bits,shape,offset,self.shape[2])

    def prefixes(self):
        prefixes = np.zeros(self.shape[0]+1,np.int64)
        for i in range(self.shape[0]):
            prefixes[i+1] = prefixes[i] + np.count_nonzero(self.plane(i))
        return prefixes

    def fixed_values(self, field, offset=(0,0,0)):
        prefixes = self.prefixes()
        values = np.empty(int(prefixes[-1]),np.float32)
        for i in range(self.shape[0]):
            plane = field[i+offset[0],offset[1]:offset[1]+self.shape[1],offset[2]:offset[2]+self.shape[2]]
            values[prefixes[i]:prefixes[i+1]] = plane[self.plane(i)]
        return values, prefixes

    def to_dense(self):
        result = np.empty(self.shape,np.bool_)
        for i in range(self.shape[0]):
            result[i] = self.plane(i)
        return result

    @classmethod
    def from_dense(cls, mask):
        if mask.dtype != np.bool_ or mask.ndim != 3:
            raise ValueError("Mask must retain bool dtype and three dimensions")
        result = cls(mask.shape)
        for i in range(mask.shape[0]):
            result.put_plane(i,mask[i])
        return result


@njit(parallel=True, cache=True, fastmath=False)
def restore_fixed(field, bits, shape, values, prefixes, offset):
    for i in prange(shape[0]):
        cursor = prefixes[i]
        for byte in range(bits.shape[1]):
            packed = bits[i,byte]
            if packed == 0:
                continue
            for bit in range(8):
                linear = byte*8+bit
                if linear < shape[1]*shape[2] and ((packed>>bit)&1):
                    j, k = linear//shape[2], linear%shape[2]
                    field[i+offset[0],j+offset[1],k+offset[2]] = values[cursor]
                    cursor += 1


class TipPredicate:
    """Exact old tip predicate, retaining coordinates not a second mask volume."""
    dtype = np.dtype(np.bool_)

    def __init__(self, global_shape, *, tip_z=.2, R=.05, r_tip=.15,
                 aspect_ratio=2., physical_params=None, tip_shape="pyramid",
                 starts=(0,0,0), counts=None):
        counts = tuple(global_shape) if counts is None else tuple(counts)
        self.shape = counts
        physical = bool(physical_params and all(physical_params.get(k) is not None
                          for k in ("tip_z_nm","R_nm","r_tip_nm","domain_nm")))
        if physical:
            lengths = tuple(float(v) for v in physical_params["domain_nm"])
            origin = physical_params["origin_fraction"]
            cx, cy = float(origin[0])*lengths[0], float(origin[1])*lengths[1]
            tip_abs = float(physical_params["tip_z_nm"])+0.
            if len(origin) >= 3:
                tip_abs += float(origin[2])*lengths[2]
            radius, trunc = float(physical_params["R_nm"]), float(physical_params["r_tip_nm"])
            theta = np.arctan(float(aspect_ratio))
        else:
            lengths, cx, cy = (1.,1.,1.), .5, .5
            radius, trunc = R, r_tip
            theta = np.arctan(aspect_ratio)
            zglobal = np.linspace(0,1,global_shape[2])
            tip_abs = zglobal[int(np.clip(tip_z,0,1)*(global_shape[2]-1))]
        xyz = [np.linspace(0.,L,n)[s:s+c] for L,n,s,c in zip(lengths,global_shape,starts,counts)]
        self.x, self.y = xyz[0]-cx, xyz[1]-cy
        a, b = radius*np.tan(theta), radius*np.tan(theta)**2
        z0 = tip_abs-b
        base_abs = z0 + np.sqrt(b**2*(1+(trunc**2/a**2)))
        self.tip_pos, self.base_pos = float(tip_abs/lengths[2]), float(base_abs/lengths[2])
        self.tip_shape = resolve_tip_shape(tip_shape)
        self.valid = np.zeros(counts[2],bool)
        self.half, self.corner = np.zeros(counts[2]), np.zeros(counts[2])
        for k, z in enumerate(xyz[2]):
            if z < tip_abs or z > base_abs:
                continue
            dz = z-z0
            if dz**2 < b**2:
                continue
            cone = a*np.sqrt(dz**2/b**2-1)
            self.valid[k] = True
            self.half[k] = min(cone,trunc) if self.tip_shape == "cone" else pyramid_tip_half_side(cone,trunc)
            self.corner[k] = pyramid_tip_corner_radius(self.half[k],radius)

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.x,self.y,self.half,self.corner,self.valid))

    def plane(self, i):
        s, r = self.half[None,:], self.corner[None,:]
        if self.tip_shape == "cone":
            result = np.sqrt(self.x[i]**2+self.y[:,None]**2) <= s
        else:
            ax, ay = np.abs(self.x[i]), np.abs(self.y[:,None])
            qx, qy = ax-(s-r), ay-(s-r)
            result = ((ax<=s)&(ay<=s)&((qx<=0)|(qy<=0)|((qx*qx+qy*qy)<=r*r)))
        return result & self.valid[None,:]

    def apply(self, field, mask, value, offset=(0,0,0)):
        for i in range(self.shape[0]):
            plane = self.plane(i)
            owned = field[i+offset[0],offset[1]:offset[1]+self.shape[1],offset[2]:offset[2]+self.shape[2]]
            owned[plane] = value
            mask.put_plane(i,plane,union=True)

    def to_dense(self):
        result = np.empty(self.shape,bool)
        for i in range(self.shape[0]):
            result[i] = self.plane(i)
        return result


def build_tip_predicate(nx,ny,nz,tip_z=.2,R=.05,r_tip=.15,aspect_ratio=2.,verbose=False,
                        tip_z_nm=None,R_nm=None,r_tip_nm=None,domain_nm=None,
                        center_fraction=(.5,.5),tip_shape="pyramid"):
    physical = None
    if all(v is not None for v in (tip_z_nm,R_nm,r_tip_nm,domain_nm)):
        physical = dict(tip_z_nm=tip_z_nm,R_nm=R_nm,r_tip_nm=r_tip_nm,
                        domain_nm=domain_nm,origin_fraction=center_fraction)
    tip = TipPredicate((nx,ny,nz),tip_z=tip_z,R=R,r_tip=r_tip,
                       aspect_ratio=aspect_ratio,physical_params=physical,tip_shape=tip_shape)
    return tip,tip.tip_pos,tip.base_pos


def build_distributed_compact(decomp,blocks,*,reference_shape=(512,512,512),directory=None):
    """Rank-local exact cells; coarse rank-zero staging sends only one plane."""
    global_cells = tuple(v-1 for v in decomp.global_shape)
    local_cells = tuple(v+1 for v in decomp.counts)
    if not all(n<=r for n,r in zip(global_cells,reference_shape)):
        return rasterize_planes(global_cells,blocks,starts=tuple(v-1 for v in decomp.starts),
                                counts=local_cells,directory=directory)
    comm, rank = decomp.cart, decomp.rank
    global_bank, error = None, None
    if rank == 0:
        try:
            global_bank = generate_eps_compact(decomp.global_shape,blocks,reference_shape,directory=directory)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = comm.bcast(error,root=0)
    if error:
        raise RuntimeError(f"Rank-zero compact epsilon volume averaging failed: {error}")
    layouts = comm.gather((decomp.starts,decomp.counts),root=0)
    tag=27183
    if rank != 0:
        with _PlaneWriter(local_cells,directory) as writer:
            plane=np.empty(local_cells[1:],np.float32)
            for i in range(local_cells[0]):
                comm.Recv(plane,source=0,tag=tag)
                writer.add(plane,i,i+1)
            return writer.finish()
    try:
        with _PlaneWriter(local_cells,directory) as writer:
            for destination,(starts,counts) in enumerate(layouts):
                halo_shape=tuple(v+1 for v in counts)
                plane=np.ones(halo_shape[1:],np.float32)
                sy,sz=starts[1]-1,starts[2]-1
                ylo,yhi=max(0,sy),min(global_cells[1],sy+halo_shape[1])
                zlo,zhi=max(0,sz),min(global_cells[2],sz+halo_shape[2])
                for i in range(halo_shape[0]):
                    plane.fill(1)
                    gx=starts[0]-1+i
                    if 0<=gx<global_cells[0]:
                        plane[ylo-sy:yhi-sy,zlo-sz:zhi-sz]=global_bank.plane(gx)[ylo:yhi,zlo:zhi]
                    if destination == 0:
                        writer.add(plane,i,i+1)
                    else:
                        comm.Send(plane,dest=destination,tag=tag)
            return writer.finish()
    finally:
        global_bank.close()
