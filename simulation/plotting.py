import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors


def plot_phi_plane(phi_matrix, boundary_mask=None, plane=(True, True, 0.5), cmap='RdBu_r',
                   tip_mask=None, apex=None):
    """Plot a 2D slice of the electrostatic potential.

    Parameters
    ----------
    phi_matrix : np.ndarray
        3D potential array.
    boundary_mask : np.ndarray (bool) or None, optional
        Boundary mask (default: None).
    plane : tuple, optional
        Slice spec: two True + one float (default: (True, True, 0.5)).
    cmap : str, optional
        Colormap name (default: 'RdBu_r').

    Returns
    -------
    matplotlib.figure.Figure
    """

    nx, ny, nz = phi_matrix.shape
    px, py, pz = plane

    if px is True and py is True and isinstance(pz, float):
        iz = int(pz * (nz - 1))
        data2d = phi_matrix[:, :, iz].T
        mask2d = boundary_mask[:, :, iz].T if boundary_mask is not None else None
        label = f"XY plane at z={pz:.2f}"
    elif px is True and isinstance(py, float) and pz is True:
        iy = int(py * (ny - 1))
        data2d = phi_matrix[:, iy, :].T
        mask2d = boundary_mask[:, iy, :].T if boundary_mask is not None else None
        label = f"XZ plane at y={py:.2f}"
    elif isinstance(px, float) and py is True and pz is True:
        ix = int(px * (nx - 1))
        data2d = phi_matrix[ix, :, :].T
        mask2d = boundary_mask[ix, :, :].T if boundary_mask is not None else None
        label = f"YZ plane at x={px:.2f}"
    else:
        raise ValueError("Plane must have two True axes and one float coordinate value.")

    if mask2d is not None:
        data2d = np.ma.masked_where(mask2d, data2d)

    cmap_mod = matplotlib.colormaps.get_cmap(cmap).copy()
    cmap_mod.set_bad(color='black')

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data2d, origin='lower', cmap=cmap_mod, aspect='equal')
    fig.colorbar(im, ax=ax, label='Potential phi (V)')
    _draw_tip_overlay(ax, tip_mask, apex, plane, (nx, ny, nz))
    ax.set_title(f"Potential Distribution ({label})")
    ax.set_xlabel("Grid index X or Y")
    ax.set_ylabel("Grid index Y or Z")
    fig.tight_layout(pad=0.5)
    return fig


def plot_residual_plane(residual_matrix, boundary_mask=None, plane=(True, True, 0.5),
                        vmin=None, vmax=None, tip_mask=None, apex=None):
    """Plot a log-scale 2D slice of the residual (with boundary masked black).

    Parameters
    ----------
    residual_matrix : np.ndarray
        3D residual array (NaN at boundaries).
    boundary_mask : np.ndarray (bool) or None, optional
        Boundary mask (default: None).
    plane : tuple, optional
        Slice spec (default: (True, True, 0.5)).
    vmin, vmax : float or None, optional
        Colour scale limits (default: None = auto).

    Returns
    -------
    matplotlib.figure.Figure
    """

    nx, ny, nz = residual_matrix.shape
    px, py, pz = plane

    if px is True and py is True and isinstance(pz, float):
        iz = int(pz * (nz - 1))
        data2d = residual_matrix[:, :, iz].T
        mask2d = boundary_mask[:, :, iz].T if boundary_mask is not None else None
        label = f"XY plane at z={pz:.2f}"
    elif px is True and isinstance(py, float) and pz is True:
        iy = int(py * (ny - 1))
        data2d = residual_matrix[:, iy, :].T
        mask2d = boundary_mask[:, iy, :].T if boundary_mask is not None else None
        label = f"XZ plane at y={py:.2f}"
    elif isinstance(px, float) and py is True and pz is True:
        ix = int(px * (nx - 1))
        data2d = residual_matrix[ix, :, :].T
        mask2d = boundary_mask[ix, :, :].T if boundary_mask is not None else None
        label = f"YZ plane at x={px:.2f}"
    else:
        raise ValueError("Plane must have two True axes and one float coordinate value.")

    if mask2d is not None:
        data2d = np.ma.masked_where(mask2d, data2d)

    if vmin is None or vmax is None:
        clean = data2d[~data2d.mask] if np.ma.is_masked(data2d) else data2d.ravel()
        clean = clean[~np.isnan(clean)]
        if len(clean) == 0:
            vmin_auto, vmax_auto = 1e-12, 1e-12
        else:
            vmax_auto = np.max(clean)
            vmin_auto = max(vmax_auto * 1e-3, np.min(clean[clean > 0]) if np.any(clean > 0) else 1e-12)
    else:
        vmin_auto, vmax_auto = vmin, vmax

    norm = mcolors.LogNorm(vmin=vmin_auto, vmax=vmax_auto)
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color='black')

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data2d, origin='lower', cmap=cmap, norm=norm, aspect='equal')
    fig.colorbar(im, ax=ax, label='Residual Magnitude (log scale)')
    _draw_tip_overlay(ax, tip_mask, apex, plane, (nx, ny, nz))
    ax.set_title(f"Residual Heatmap ({label})\n Local max residual: {vmax_auto:.3e}")
    ax.set_xlabel("Grid index X or Y")
    ax.set_ylabel("Grid index Y or Z")
    fig.tight_layout(pad=0.5)
    return fig


def plot_residual_line(residual_matrix, line):
    """Plot a 1D line-out of the residual magnitude.

    Parameters
    ----------
    residual_matrix : np.ndarray
        3D residual array.
    line : tuple
        (x_frac, y_frac, z_frac) with exactly one None for the slice axis.

    Returns
    -------
    matplotlib.figure.Figure
    """

    nx, ny, nz = residual_matrix.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)

    x_frac, y_frac, z_frac = line
    if sum(v is None for v in (x_frac, y_frac, z_frac)) != 1:
        raise ValueError("Exactly one coordinate must be None for line plot.")

    if z_frac is None:
        ix = int(x_frac * (nx - 1))
        iy = int(y_frac * (ny - 1))
        res_line = residual_matrix[ix, iy, :]
        coord = z
        xlabel = "Z (fraction)"
    elif y_frac is None:
        ix = int(x_frac * (nx - 1))
        iz = int(z_frac * (nz - 1))
        res_line = residual_matrix[ix, :, iz]
        coord = y
        xlabel = "Y (fraction)"
    elif x_frac is None:
        iy = int(y_frac * (ny - 1))
        iz = int(z_frac * (nz - 1))
        res_line = residual_matrix[:, iy, iz]
        coord = x
        xlabel = "X (fraction)"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(coord, res_line, 'b-', linewidth=2)
    ax.set_title("Residual Magnitude Along Line")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residual (log scale)")
    ax.grid(True, which='both', alpha=0.4)
    fig.tight_layout(pad=0.5)
    return fig


def visualize_afm_results(results, x_frac=0.5, y_frac=0.5, z_frac=0.3):
    """Comprehensive 6-panel visualisation of AFM simulation results.

    Shows potential, field magnitude, tip geometry, and line-outs.

    Parameters
    ----------
    results : dict
        Simulation results dict.
    x_frac, y_frac, z_frac : float, optional
        Slice coordinates for line-outs (default: 0.5, 0.5, 0.3).

    Returns
    -------
    matplotlib.figure.Figure
    """

    phi = results['phi']
    Ex, Ey, Ez = np.gradient(-phi)
    tip_mask = results['tip_mask']
    nx, ny, nz = results['parameters']['nx'], results['parameters']['ny'], results['parameters']['nz']
    tip_pos = results['parameters']['tip_pos']

    ix = int(x_frac * (nx - 1))
    iy = int(y_frac * (ny - 1))
    iz = int(z_frac * (nz - 1))

    x = np.linspace(-0.5, 0.5, nx)
    y = np.linspace(-0.5, 0.5, ny)
    z = np.linspace(0, 1, nz)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax1 = axes[0, 0]
    im1 = ax1.imshow(phi[:, :, iz].T, origin='lower', extent=[-0.5, 0.5, -0.5, 0.5],
                     cmap='RdBu', aspect='equal')
    ax1.scatter(x[ix], y[iy], color='yellow', marker='x', s=80, label='Sample position x')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title(f'Potential Distribution (z={z[iz]:.2f})')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y')
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    ax2 = axes[0, 1]
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    im2 = ax2.imshow(E_mag[:, :, iz].T, origin='lower', extent=[-0.5, 0.5, -0.5, 0.5],
                     cmap='hot', aspect='equal')
    ax2.scatter(x[ix], y[iy], color='cyan', marker='x', s=80)
    ax2.set_title(f'Electric Field Magnitude (z={z[iz]:.2f})')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    ax3 = axes[0, 2]
    tip_points = np.where(tip_mask)
    x_tip = tip_points[0] / (nx - 1) - 0.5
    y_tip = tip_points[1] / (ny - 1) - 0.5
    z_tip = tip_points[2] / (nz - 1)
    scatter = ax3.scatter(x_tip, y_tip, c=z_tip, cmap='plasma', s=40, alpha=0.8)
    ax3.set_title('AFM Tip Geometry (Top View)')
    ax3.set_xlabel('X'); ax3.set_ylabel('Y')
    ax3.set_aspect('equal')
    fig.colorbar(scatter, ax=ax3, shrink=0.8)

    ax4 = axes[1, 0]
    phi_line = phi[ix, iy, :]
    ax4.plot(z, phi_line, 'b-', lw=2)
    ax4.set_xlabel('Z'); ax4.set_ylabel('Potential (V)')
    ax4.set_title(f'Vertical Potential at (x={x_frac:.2f}, y={y_frac:.2f})')
    ax4.grid(True, alpha=0.3)

    ax5 = axes[1, 1]
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    E_line = E_mag[ix, iy, :]
    ax5.plot(z, E_line, 'r-', lw=2)
    ax5.set_xlabel('Z'); ax5.set_ylabel('|E| (V/unit)')
    ax5.set_title(f'Vertical |E| at (x={x_frac:.2f}, y={y_frac:.2f})')
    ax5.grid(True, alpha=0.3)

    ax6 = axes[1, 2]
    ax6.scatter(x_tip, z_tip, c=z_tip, cmap='plasma', s=30, alpha=0.8)
    ax6.axhline(y=0.0, color='brown', lw=3, alpha=0.7, label='Sample Surface')
    ax6.axhline(y=tip_pos, color='red', ls='--', alpha=0.8, label='Tip Position')
    ax6.set_xlabel('X'); ax6.set_ylabel('Z')
    ax6.set_title('Tip Side View (Downward Pointing)')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)

    fig.tight_layout(pad=0.5)
    return fig


def combine_phi_and_residual(phi, residual, boundary_mask, plane, title_tag=""):
    """Create a side-by-side figure of potential and residual slices.

    Parameters
    ----------
    phi : np.ndarray
        3D potential.
    residual : np.ndarray
        3D residual.
    boundary_mask : np.ndarray (bool)
        Boundary mask.
    plane : tuple
        Slice specification.
    title_tag : str, optional
        Additional title text (default: "").

    Returns
    -------
    None
    """

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    fig_phi = plt.figure()
    plot_phi_plane(phi, boundary_mask, plane=plane)
    fig_phi_axes = fig_phi.get_axes()[0]
    for im in fig_phi_axes.get_images():
        ax1.imshow(im.get_array(),
                   cmap=im.get_cmap(),
                   origin=im.origin,
                   extent=im.get_extent())
    ax1.set_title(f"Potential phi - {title_tag}")
    ax1.set_xlabel(fig_phi_axes.get_xlabel())
    ax1.set_ylabel(fig_phi_axes.get_ylabel())
    plt.close(fig_phi)

    fig_res = plt.figure()
    plot_residual_plane(residual, boundary_mask, plane=plane)
    fig_res_axes = fig_res.get_axes()[0]
    for im in fig_res_axes.get_images():
        ax2.imshow(im.get_array(),
                   cmap=im.get_cmap(),
                   origin=im.origin,
                   extent=im.get_extent())
    ax2.set_title(f"Residual - {title_tag}")
    ax2.set_xlabel(fig_res_axes.get_xlabel())
    ax2.set_ylabel(fig_res_axes.get_ylabel())
    plt.close(fig_res)

    fig.suptitle(f"Voltage + Residual - {title_tag}", fontsize=14)
    fig.tight_layout()
    plt.show()


def _draw_tip_overlay(ax, tip_mask, apex, plane, shape):
    """Overlay tip silhouette contour + apex marker on a 2D slice.

    Silhouette = projection of the 3D tip mask onto the slice plane
    (dark-goldenrod contours). Apex = black 'x' with dashed axis lines.
    """
    nx, ny, nz = shape
    px, py, pz = plane
    if tip_mask is not None:
        if px is True and py is True and isinstance(pz, float):
            sil = tip_mask.any(axis=2).T
        elif px is True and isinstance(py, float) and pz is True:
            sil = tip_mask.any(axis=1).T
        elif isinstance(px, float) and py is True and pz is True:
            sil = tip_mask.any(axis=0).T
        else:
            raise ValueError("Plane must have two True axes and one float.")
        ax.contour(sil.astype(float), levels=[0.5], colors='darkgoldenrod', linewidths=1.6)
    if apex is not None:
        if px is True and py is True and isinstance(pz, float):
            axx, ayy = apex[0] * (nx - 1), apex[1] * (ny - 1)
        elif px is True and isinstance(py, float) and pz is True:
            axx, ayy = apex[0] * (nx - 1), apex[2] * (nz - 1)
        else:
            axx, ayy = apex[1] * (ny - 1), apex[2] * (nz - 1)
        ax.plot(axx, ayy, 'x', color='black', ms=10, mew=1.5)
        ax.axhline(y=ayy, color='black', ls='--', lw=0.8)
        ax.axvline(x=axx, color='black', ls='--', lw=0.8)
