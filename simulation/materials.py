import numpy as np
import os
import tempfile


def _as_blocks(blocks):
    if blocks is None:
        return []
    return [blocks] if isinstance(blocks, dict) else list(blocks)


def _range(item, axis):
    r = item.get(f"{axis}_range")
    if r is None:
        return 0.0, 1.0
    a, b = float(r[0]), float(r[1])
    return min(a,b), max(a,b)


def _fine_eps(reference_shape, blocks, dtype=np.float32):
    nx, ny, nz = reference_shape
    # Cell centers; the resulting reference is a true voxel/material field.
    x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    y = (np.arange(ny, dtype=np.float64) + 0.5) / ny
    z = (np.arange(nz, dtype=np.float64) + 0.5) / nz
    eps = np.ones((nx, ny, nz), dtype=dtype)
    # Ordered assignment reproduces JSON block precedence at the fine reference
    # resolution. Coarse levels are obtained by volume averaging this field.
    for blk in _as_blocks(blocks):
        eps_val = float(blk.get("eps_val", 1.0))
        x0,x1=_range(blk,'x'); y0,y1=_range(blk,'y'); z0,z1=_range(blk,'z')
        ix=np.flatnonzero((x>=x0)&(x<x1))
        iy=np.flatnonzero((y>=y0)&(y<y1))
        iz=np.flatnonzero((z>=z0)&(z<z1))
        if ix.size and iy.size and iz.size:
            eps[np.ix_(ix,iy,iz)] = eps_val
    return eps



def build_eps_reference_memmap(reference_shape, blocks=None, directory=None, prefix="afm_eps_reference_"):
    """Create a temporary high-resolution epsilon NPY/memmap from JSON-derived blocks.

    The file-backed reference mirrors the historical high-resolution NPY workflow
    without keeping the full reference field resident in RAM. The caller owns the
    returned ``(path, mmap)`` pair and must call :func:`release_eps_reference`.
    """
    if directory is None:
        directory = tempfile.gettempdir()
    os.makedirs(directory, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".npy", dir=directory)
    os.close(fd)
    shape = tuple(int(v) for v in reference_shape)
    mmap = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    mmap.fill(1.0)
    nx, ny, nz = shape
    x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    y = (np.arange(ny, dtype=np.float64) + 0.5) / ny
    z = (np.arange(nz, dtype=np.float64) + 0.5) / nz
    for blk in _as_blocks(blocks):
        eps_val = float(blk.get("eps_val", 1.0))
        x0,x1=_range(blk,'x'); y0,y1=_range(blk,'y'); z0,z1=_range(blk,'z')
        ix=np.flatnonzero((x>=x0)&(x<x1))
        iy=np.flatnonzero((y>=y0)&(y<y1))
        iz=np.flatnonzero((z>=z0)&(z<z1))
        if ix.size and iy.size and iz.size:
            mmap[np.ix_(ix,iy,iz)] = eps_val
    mmap.flush()
    return path, mmap


def release_eps_reference(path, mmap=None):
    """Flush and remove a temporary epsilon reference NPY file."""
    if mmap is not None:
        mmap.flush()
        try:
            mmap._mmap.close()
        except Exception:
            pass
        del mmap
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _rebin_axis(a, out_n, axis):
    """Exact box-average rebin of a piecewise-constant voxel field along one axis."""
    in_n = a.shape[axis]
    if out_n == in_n:
        return a
    edges = np.linspace(0.0, float(in_n), out_n + 1, dtype=np.float64)
    lo = np.floor(edges).astype(np.int64)
    hi = np.ceil(edges).astype(np.int64)
    t = edges - lo

    cshape = list(a.shape)
    cshape[axis] = in_n + 1
    c = np.empty(cshape, dtype=np.float32)
    first = [slice(None)] * a.ndim
    first[axis] = 0
    c[tuple(first)] = 0.0
    dst = [slice(None)] * a.ndim
    dst[axis] = slice(1, None)
    np.cumsum(a, axis=axis, dtype=np.float32, out=c[tuple(dst)])

    shape = [1] * a.ndim
    shape[axis] = edges.size
    lo_v = np.take(c, lo, axis=axis)
    hi_v = np.take(c, hi, axis=axis)
    integ = (1.0 - t).reshape(shape).astype(np.float32) * lo_v + t.reshape(shape).astype(np.float32) * hi_v
    del c, lo_v, hi_v

    widths = np.diff(edges).reshape([1] * axis + [out_n] + [1] * (a.ndim - axis - 1)).astype(np.float32)
    return (np.diff(integ, axis=axis) / widths).astype(np.float32, copy=False)


def average_reference_to_cells(reference, target_cells):
    """Volume-average a high-resolution material field to target voxel cells."""
    out=reference
    for axis,n in enumerate(target_cells):
        out=_rebin_axis(out,int(n),axis)
    return np.asarray(out,dtype=np.float32)


def generate_eps_level(phi_shape, blocks=None, reference_shape=(512,512,512), reference=None):
    """Build a deterministic, volume-averaged epsilon array for one solver level.

    The JSON block distribution is rasterized once on a high-resolution reference
    voxel grid and then box-averaged onto the requested solver-cell grid. This
    avoids selecting an arbitrary block value when a coarse cell contains more
    than one dielectric value. The reference is temporary and should be released
    after the returned level array is handed to the solver.
    """
    nx,ny,nz=map(int,phi_shape)
    target=(max(nx-1,1),max(ny-1,1),max(nz-1,1))
    ref=tuple(max(2, int(r)) if t <= int(r) else int(t) for t,r in zip(target,reference_shape))
    # Levels at or below the reference resolution are generated by volume-averaging
    # the same high-resolution material field. Larger/finer levels are rasterized
    # directly because no coarse downsampling is involved.
    if all(t <= int(r) for t,r in zip(target,reference_shape)):
        owned_reference = reference is None
        if owned_reference:
            reference = _fine_eps(ref,blocks)
        try:
            return average_reference_to_cells(reference,target)
        finally:
            if owned_reference:
                del reference
    return _fine_eps(target,blocks)


def generate_eps_cell(phi, blocks=None, reference_shape=(512,512,512), reference=None):
    """Backward-compatible wrapper for :func:`generate_eps_level`."""
    return generate_eps_level(phi.shape,blocks,reference_shape=reference_shape,reference=reference)
