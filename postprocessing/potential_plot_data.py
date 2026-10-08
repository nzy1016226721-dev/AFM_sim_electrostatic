"""Read AFM MPI/serial potential outputs without allocating a full 3-D copy.

The effective JSON is authoritative for physical coordinates. This module does
not modify the electrostatic solver, MPI communication, or deprecated zoom mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np


@dataclass(frozen=True)
class PotentialData:
    path: Path
    array: np.memmap
    # Exact original solver node coordinates along each axis (nm, origin relative).
    centres: tuple[np.ndarray, np.ndarray, np.ndarray]
    # MPI cut-bin bounds (nm); full fields use original domain bounds.
    bounds_nm: tuple[float, float, float, float, float, float]
    is_cut: bool
    source_grid: tuple[int, int, int]
    voltage: float | None

    def index(self, axis: int, coordinate_nm: float) -> int:
        coordinates = self.centres[axis]
        value = float(coordinate_nm)
        if not np.isfinite(value) or value < coordinates[0] or value > coordinates[-1]:
            raise ValueError(f"{('x','y','z')[axis]}={value:g} nm lies outside saved volume "
                             f"[{coordinates[0]:g}, {coordinates[-1]:g}] nm")
        return int(np.argmin(np.abs(coordinates - value)))

    def line(self, axis: str, at_nm: tuple[float, float]):
        """Return (positions_nm, potential) for a single 1-D line, never full cube."""
        ax = 'xyz'.index(axis.lower())
        fixed_axes = [a for a in range(3) if a != ax]
        indices = [slice(None)] * 3
        for i, fixed in zip(fixed_axes, at_nm):
            indices[i] = self.index(i, fixed)
        return self.centres[ax], np.asarray(self.array[tuple(indices)])

    def plane(self, plane: str, at_nm: float):
        """Return (2-D data, imshow extent, coordinate labels) for one plane."""
        plane = plane.lower()
        if plane not in ('xy', 'xz', 'yz'):
            raise ValueError('plane must be xy, xz or yz')
        a, b = ('xyz'.index(c) for c in plane)
        fixed = ({0, 1, 2} - {a, b}).pop()
        idx = [slice(None)] * 3
        idx[fixed] = self.index(fixed, at_nm)
        data = np.asarray(self.array[tuple(idx)]).T
        def edges(axis):
            values = self.centres[axis]
            step = (values[-1]-values[0])/(len(values)-1) if len(values)>1 else 0.0
            return values[0]-step/2, values[-1]+step/2
        extent = (*edges(a), *edges(b))
        return data, extent, ('xyz'[a], 'xyz'[b])


def _case_center_fraction(meta: dict, cfg: dict) -> tuple[float, float, float]:
    """Find the exact movement centre whose filename matches the MPI writer."""
    from simulation.io_utils import movement_suffix_nm
    from simulation.main_loop import compute_block_positions

    movement = cfg.get('movement', {})
    if not isinstance(movement, dict):
        movement = {}
    if meta.get('position') is not None:
        return tuple(float(v) for v in meta['position'])
    start = tuple(movement.get('start', (0.5, 0.5, 0.5)))
    end = tuple(movement.get('end', start))
    physical = cfg.get('_physical', {})
    domain = tuple(float(v) for v in physical['domain_nm'])
    physical_move = physical.get('movement', {})
    if 'spacing_nm' in physical_move:
        centres = compute_block_positions(start, end, physical_move['spacing_nm'], domain_nm=domain)
    else:
        centres = compute_block_positions(start, end, movement.get('spacing', 0.1))
    centres = [tuple(float(v) for v in c) for c in centres]
    offset = meta.get('movement_offset_nm')
    if offset is None:
        if len(centres) > 1:
            # Explicitly unnamed files belong to the first case when movement is configured.
            return centres[0]
        return start
    wanted = float(offset)
    matches = []
    for c in centres:
        suffix = movement_suffix_nm(c, centres[0], domain)
        # Compare with the exact serial/MPI naming convention, avoiding
        # floating-point roundoff assumptions about the filename precision.
        if (suffix.startswith('-') and abs(float(suffix[1:-2]) + wanted) < 1e-10) or (
            suffix.startswith('_') and abs(float(suffix[1:-2]) - wanted) < 1e-10):
            matches.append(c)
    if len(matches) != 1:
        raise ValueError(f"Cannot identify a unique movement centre for offset {wanted} nm; "
                         "provide an unambiguous output/config pair")
    return matches[0]


def load_potential(path: str | os.PathLike, *, config_path: str | os.PathLike,
                   large_grid_limit: int = 2048, allow_full: bool = False) -> PotentialData:
    """Open a potential NPY with mmap and reconstruct its true physical extent.

    Large-grid full-array inputs are rejected by default: use a saved cut.
    ``config_path`` may extend another JSON and uses the MPI normalizer.
    """
    from postprocessing.npy_utils import parse_phi_filename
    from simulation.mpi_config import load_afm_config
    from simulation.mpi_io import physical_cut_slices

    path = Path(path).expanduser().resolve()
    meta = parse_phi_filename(path.name)
    if meta is None:
        raise ValueError(f'Unrecognized AFM potential filename: {path.name}')
    _, cfg = load_afm_config(config_path)
    data = np.load(path, mmap_mode='r', allow_pickle=False)
    if data.ndim != 3:
        raise ValueError('Potential array must have three dimensions')
    is_cut = bool(meta.get('is_cut', False))
    expected = tuple(int(cfg['grid_resolution'][f'n{axis}']) for axis in 'xyz')
    source_grid = tuple(meta['source_grid']) if is_cut else tuple(data.shape)
    if source_grid != expected:
        raise ValueError(f'Source grid {source_grid} does not match config grid {expected}')
    if not is_cut and max(source_grid) >= large_grid_limit and not allow_full:
        raise ValueError(f'Full-grid potential {source_grid} is not accepted for large-grid plotting; '
                         'select its _cut_from_grid...npy file (or explicitly set allow_full=True).')

    physical = cfg['_physical']
    domain = tuple(float(v) for v in physical['domain_nm'])
    origin = tuple(float(v) for v in physical.get('origin_fraction', (0.5, 0.5, 0.0)))
    bounds = tuple(value for i in range(3)
                   for value in (-origin[i]*domain[i], (1.0-origin[i])*domain[i]))
    if is_cut:
        centre_frac = _case_center_fraction(meta, cfg)
        centre_nm = tuple((centre_frac[i]-origin[i])*domain[i] for i in range(3))
        slices, bounds_cut = physical_cut_slices(
            source_grid, centre_nm, cfg.get('save_cut_box_nm'), bounds)
        if slices is None:
            raise ValueError('Configured cut has no intersection with the grid')
        cut_shape = tuple(s.stop-s.start for s in slices)
        if cut_shape != data.shape:
            raise ValueError(f'Cut shape {data.shape} does not match configured expected {cut_shape}; '
                             'verify effective JSON, movement case, and output identity')
        bounds = tuple(float(v) for v in bounds_cut)
        # The saved cut was selected via voxel bins (n divisor), but the
        # potential itself lives on solver nodes (n-1 divisor). Recover exact
        # node positions from the global indices, not the cut-bin edge centres.
        full_bounds = tuple(value for i in range(3)
                            for value in (-origin[i]*domain[i], (1.0-origin[i])*domain[i]))
        centres = tuple(full_bounds[2*i] +
                        np.arange(slices[i].start, slices[i].stop, dtype=float) *
                        (full_bounds[2*i+1]-full_bounds[2*i])/(source_grid[i]-1)
                        if source_grid[i]>1 else np.array([full_bounds[2*i]])
                        for i in range(3))
    else:
        centres = tuple(np.linspace(bounds[2*i], bounds[2*i+1], data.shape[i]) for i in range(3))
    return PotentialData(path, data, centres, bounds, is_cut, source_grid, meta.get('Vtip'))


def discover_potentials(directory: str | os.PathLike, *, prefer_cuts: bool = True) -> list[Path]:
    """List recognized AFM NPY outputs, preferring cut NPYs over full files.

    Chooses cuts by filename stem including the `_cut_from_grid...` suffix;
    diagnostic/unknown NPYs are excluded.
    """
    from postprocessing.npy_utils import parse_phi_filename
    root = Path(directory).expanduser()
    files = [p for p in root.glob('*.npy') if parse_phi_filename(p.name) is not None]
    if prefer_cuts:
        cut_bases = {p.name.split('_cut_from_grid', 1)[0] for p in files
                     if '_cut_from_grid' in p.name}
        files = [p for p in files if '_cut_from_grid' in p.name or
                 p.stem not in cut_bases]
    return sorted(files, key=lambda p: p.name)
