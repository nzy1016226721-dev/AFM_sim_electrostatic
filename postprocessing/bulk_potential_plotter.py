"""MPI-compatible comparative potential plots, adapted from the QoL plotters.

Works with full (small grids) or saved physical-cut .npy outputs. All read
operations are memory-mapped. No dependence on conductivity/zoom code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .potential_plot_data import discover_potentials, load_potential


def _as_files(paths: Sequence[str | Path]) -> list[Path]:
    files = [Path(p).expanduser() for p in paths]
    if not files:
        raise ValueError('At least one AFM NPY file is required')
    return files


def plot_potential_lines(paths: Sequence[str | Path], config_path: str | Path,
                         *, axis: str, fixed_nm: tuple[float, float],
                         limits_nm: tuple[float, float] | None = None,
                         allow_full: bool = False, show: bool = False):
    """Compare 1-D potential lines at fixed physical coordinates.

    ``axis`` is the varying dimension; ``fixed_nm`` holds the other two
    coordinates in (x,y,z) axis order. A path may refer to a saved MPI cut.
    """
    paths = _as_files(paths)
    fig, ax = plt.subplots()
    try:
        for path in paths:
            field = load_potential(path, config_path=config_path, allow_full=allow_full)
            x, y = field.line(axis, fixed_nm)
            if limits_nm is not None:
                lo, hi = map(float, limits_nm)
                if hi <= lo:
                    raise ValueError('limits_nm must increase')
                select = (x >= lo) & (x <= hi)
                x, y = x[select], y[select]
                if len(x) == 0:
                    raise ValueError(f'{path.name}: requested range has no saved samples')
            label = f'{field.voltage:+g} V' if field.voltage is not None else path.stem
            # Disambiguate files with the same voltage but different movements.
            ax.plot(x, y, label=f'{path.stem} ({label})')
        ax.set(xlabel=f'{axis.lower()} (nm)', ylabel='Potential (V)',
               title=f'AFM potential along {axis.lower()}')
        ax.legend(fontsize='small')
        ax.grid(True, alpha=.25)
        fig.tight_layout()
        if show:
            plt.show()
        return fig
    except Exception:
        plt.close(fig)
        raise


def plot_potential_planes(paths: Sequence[str | Path], config_path: str | Path,
                          *, plane: str, at_nm: float,
                          allow_full: bool = False, show: bool = False):
    """Generate one 2-D plane per file (a series of independently sized figures)."""
    figures = []
    try:
        for path in _as_files(paths):
            field = load_potential(path, config_path=config_path, allow_full=allow_full)
            data, extent, labels = field.plane(plane, at_nm)
            fig, ax = plt.subplots()
            figures.append(fig)
            im = ax.imshow(data, extent=extent, origin='lower', aspect='equal',
                           interpolation='none', cmap='RdBu_r')
            ax.set(xlabel=f'{labels[0]} (nm)', ylabel=f'{labels[1]} (nm)',
                   title=f'{path.name}: {plane} at {float(at_nm):g} nm')
            fig.colorbar(im, ax=ax, label='Potential (V)')
            fig.tight_layout()
        if show:
            plt.show()
        return figures
    except Exception:
        for fig in figures:
            plt.close(fig)
        raise


def bulk_potential_plots():
    """Interactive local menu. Remains separate from batch/headless MPI runs."""
    folder = Path(input('Output directory [outputs]: ').strip() or 'outputs')
    files = discover_potentials(folder, prefer_cuts=True)
    if not files:
        print(f'No recognizable potential NPY files in {folder}')
        return []
    print('Available arrays (cuts preferred where present):')
    for i, p in enumerate(files):
        print(f'  {i}: {p.name}')
    indices = input('File indices, comma separated [0]: ').strip() or '0'
    selected = [files[int(v.strip())] for v in indices.split(',')]
    config = input('Effective AFM config JSON path: ').strip()
    if not config:
        raise ValueError('A configuration JSON is required for correct physical coordinates')
    kind = (input('Plot line or plane [line]: ').strip() or 'line').lower()
    if kind == 'line':
        axis = (input('Line axis [z]: ').strip() or 'z').lower()
        fixed_axes = [name for name in 'xyz' if name != axis]
        fixed = tuple(float(input(f'Fixed {name} coordinate (nm): ')) for name in fixed_axes)
        figs = [plot_potential_lines(selected, config, axis=axis, fixed_nm=fixed)]
    elif kind == 'plane':
        plane = (input('Plane (xy/xz/yz) [xy]: ').strip() or 'xy').lower()
        fixed = next(name for name in 'xyz' if name not in plane)
        at_nm = float(input(f'{fixed} plane coordinate (nm): '))
        figs = plot_potential_planes(selected, config, plane=plane, at_nm=at_nm)
    else:
        raise ValueError('Expected line or plane')
    if input('Save figures as PNG? [y/N] ').strip().lower() == 'y':
        saved = folder / 'saved_plots'
        saved.mkdir(parents=True, exist_ok=True)
        for i, fig in enumerate(figs):
            fig.savefig(saved / f'afm_{kind}_{i:03d}.png', dpi=160)
    plt.show()
    return figs
