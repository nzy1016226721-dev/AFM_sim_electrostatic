"""Single MPI-cut plane view for click-to-select potential line profiles."""
from __future__ import annotations

import matplotlib.pyplot as plt
from .potential_plot_data import load_potential


def single_plane_plotter(npy_path, config_path, *, plane='xy', at_nm=0.0,
                         allow_full=False, show=False):
    field = load_potential(npy_path, config_path=config_path, allow_full=allow_full)
    data, extent, axes = field.plane(plane, at_nm)
    fig, ax = plt.subplots()
    image = ax.imshow(data, origin='lower', extent=extent, interpolation='none',
                      aspect='equal', cmap='RdBu_r')
    ax.set(xlabel=f'{axes[0]} (nm)', ylabel=f'{axes[1]} (nm)',
           title=f'Potential in {plane} plane at {float(at_nm):g} nm')
    fig.colorbar(image, ax=ax, label='Potential (V)')
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax, field
