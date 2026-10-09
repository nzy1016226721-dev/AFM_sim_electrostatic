"""Bounded-RAM implementation of the EXISTING float32 volume average.

Uses the original _rebin_axis on independent column slabs. No new geometry
approximation, reference mesh, averaging arithmetic, or lower precision.
Temporary NPY mappings are reopened/closed per slab, rather than accumulating
resident pages for the complete intermediate grids. Disk I/O is intentional.
"""
from pathlib import Path
import os
import tempfile
import numpy as np

from .materials import _rebin_axis, _fine_eps, build_eps_reference_memmap, release_eps_reference


def _close(array):
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def average_reference_bounded(reference, target_cells, *, directory=None,
                              scratch_budget_bytes=16*1024**2, final_path=None):
    target = tuple(int(v) for v in target_cells)
    if len(target) != 3 or min(target) < 1 or scratch_budget_bytes < 1:
        raise ValueError("invalid target or scratch budget")
    source_path = getattr(reference, "filename", None)
    current_shape = reference.shape
    current_array = reference if source_path is None else None
    temporary_paths = []
    final = None
    axes = [axis for axis in range(3) if target[axis] != current_shape[axis]]
    if not axes:
        axes = [2]  # Stream a copy without retaining a second full mapping.
    try:
        for index, axis in enumerate(axes):
            out_shape = list(current_shape)
            out_shape[axis] = target[axis]
            out_shape = tuple(out_shape)
            last = index == len(axes)-1
            if last and final_path is None:
                final = np.empty(out_shape, dtype=np.float32)
                destination_path = None
            else:
                if last:
                    name = str(final_path) # Caller owns this optional file.
                else:
                    descriptor, name = tempfile.mkstemp(prefix="afm_rebin_", suffix=".npy", dir=directory)
                    os.close(descriptor)
                    temporary_paths.append(Path(name))
                destination_path = name
                output = np.lib.format.open_memmap(name, mode="w+", shape=out_shape, dtype=np.float32)
                _close(output)
            slab_axis = 1 if axis == 0 else 0
            column_bytes = np.prod([current_shape[d] for d in range(3) if d != slab_axis])*4
            slab_size = max(1, int(scratch_budget_bytes // (8*column_bytes)))
            for start in range(0, current_shape[slab_axis], slab_size):
                selection = [slice(None)]*3
                selection[slab_axis] = slice(start, min(start+slab_size, current_shape[slab_axis]))
                selection = tuple(selection)
                if source_path is not None:
                    reader = np.load(source_path, mmap_mode="r", allow_pickle=False)
                    try:
                        slab = np.array(reader[selection], dtype=np.float32, copy=True)
                    finally:
                        _close(reader)
                else:
                    slab = np.array(current_array[selection], dtype=np.float32, copy=True)
                rebinned = _rebin_axis(slab, target[axis], axis)
                if last and final_path is None:
                    final[selection] = rebinned
                else:
                    writer = np.load(destination_path, mmap_mode="r+", allow_pickle=False)
                    try:
                        writer[selection] = rebinned
                        writer.flush()
                    finally:
                        _close(writer)
                del slab, rebinned
            source_path, current_array = destination_path, final if last else None
            current_shape = out_shape
        if final_path is not None:
            final = np.load(final_path,mmap_mode="r",allow_pickle=False)
        return final
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def generate_eps_level_bounded(phi_shape, blocks=None, reference_shape=(512,512,512),
                               reference=None, *, directory=None):
    target = tuple(max(int(v)-1, 1) for v in phi_shape)
    if not all(t <= int(r) for t, r in zip(target, reference_shape)):
        return _fine_eps(target, blocks)
    own_path = own_reference = None
    if reference is None:
        own_path, own_reference = build_eps_reference_memmap(reference_shape, blocks, directory)
        _close(own_reference)
        own_reference = np.load(own_path, mmap_mode="r", allow_pickle=False)
        reference = own_reference
    try:
        return average_reference_bounded(reference, target, directory=directory)
    finally:
        if own_path is not None:
            release_eps_reference(own_path, own_reference)
