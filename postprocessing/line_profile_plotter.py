"""Interactive two-point line sampling of an MPI saved potential cut.

Adapted from the QoL Qt/IPython click plotter. Uses the current Matplotlib
backend; it does not force Qt in batch/Slurm sessions or change IPython state.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from .single_plane_plotter import single_plane_plotter


def sample_segment(plane_data: np.ndarray, bounds_nm, first_nm, last_nm, *, n=None):
    """Bilinearly sample the displayed 2-D plane along a physical line.

    ``plane_data`` is the transposed imshow matrix (rows correspond to y).
    Coordinates use the original solver sample nodes, not cut-bin boundaries.
    """
    if plane_data.ndim != 2:
        raise ValueError('Expected 2-D plane')
    left, right, bottom, top = map(float, bounds_nm)
    rows, cols = plane_data.shape
    if min(rows, cols) < 2 or right <= left or top <= bottom:
        raise ValueError('At least two points per plotted axis are required')
    x0, y0 = map(float, first_nm)
    x1, y1 = map(float, last_nm)
    # Clamp to actual centre coordinates. The edge half-voxel is not sampled.
    dx, dy = (right-left)/cols, (top-bottom)/rows
    xlo, xhi = left+dx/2, right-dx/2
    ylo, yhi = bottom+dy/2, top-dy/2
    for x,y in ((x0,y0),(x1,y1)):
        if x < xlo or x > xhi or y < ylo or y > yhi:
            raise ValueError('Line endpoints must be inside saved sample centres')
    length = float(np.hypot(x1-x0, y1-y0))
    if n is None:
        n = max(2, 1+int(np.ceil(length/min(dx, dy))))
    elif n < 2:
        raise ValueError('n must be >= 2')
    # Prevent an unexpectedly long drag from allocating unbounded samples.
    n = min(int(n), 1_000_000)
    x = np.linspace(x0, x1, n)
    y = np.linspace(y0, y1, n)
    ix = (x-xlo)/dx
    iy = (y-ylo)/dy
    values = map_coordinates(plane_data, (iy, ix), order=1, mode='nearest')
    return np.linspace(0, length, n), values


class InteractiveLineProfile:
    """Register click handling on a plane; first/second clicks define a line."""
    def __init__(self, fig, ax):
        if not ax.images:
            raise ValueError('The given plot has no potential image')
        self.fig, self.ax = fig, ax
        self.image = ax.images[0]
        self.points = []
        self.last_profile_figure = None
        self._cid = fig.canvas.mpl_connect('button_release_event', self.on_click)

    def on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        self.points.append((float(event.xdata), float(event.ydata)))
        if len(self.points) < 2:
            return
        first, last = self.points
        self.points.clear()
        distances, phi = sample_segment(np.asarray(self.image.get_array()),
                                        self.image.get_extent(), first, last)
        fig, ax = plt.subplots()
        ax.plot(distances, phi)
        ax.set(xlabel='Distance along selected line (nm)', ylabel='Potential (V)',
               title='Potential along selected line')
        ax.grid(True, alpha=.25)
        fig.tight_layout()
        self.last_profile_figure = fig
        fig.canvas.draw_idle()

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self._cid)


def line_profile_plotter(npy_path=None, config_path=None, *, plane='xy', at_nm=0.0,
                         show=True):
    """Open a saved cut for graphical line selection (requires GUI backend)."""
    if npy_path is None:
        npy_path = input('Potential NPY file: ').strip()
    if config_path is None:
        config_path = input('Effective AFM config JSON path: ').strip()
    fig, ax, _ = single_plane_plotter(npy_path, config_path,
                                     plane=plane, at_nm=at_nm, show=False)
    callback = InteractiveLineProfile(fig, ax)
    # Keep callback reachable until the figure is closed.
    fig._afm_line_profile_callback = callback
    if show:
        plt.show()
    return fig, callback
