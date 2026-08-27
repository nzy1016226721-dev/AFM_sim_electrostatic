import numpy as np
import matplotlib.pyplot as plt


def plot_afm_from_npy(
    phi_file,
    ex_file=None, ey_file=None, ez_file=None,
    x_frac=0.5, y_frac=0.5, z_frac=0.5,
    axis='z',
    cmap_phi='RdBu_r',
    cmap_E='hot',
    cmap_comp='RdBu_r',
    show=True,
    save_prefix=None,
    show_component_slices=False,
    Lx_nm=512.0, Ly_nm=512.0, Lz_nm=512.0
):
    """Load and plot AFM potential (and optionally field) from .npy files.

    Displays a 2D slice of the potential and electric field magnitude,
    plus line-out plots (or component slices if requested).

    Parameters
    ----------
    phi_file : str
        Path to potential .npy file.
    ex_file, ey_file, ez_file : str or None, optional
        Paths to field component .npy files (default: None = compute from phi).
    x_frac, y_frac, z_frac : float, optional
        Slice coordinates (default: 0.5).
    axis : str, optional
        Slice axis 'x', 'y', or 'z' (default: 'z').
    cmap_phi : str, optional
        Colormap for potential (default: 'RdBu_r').
    cmap_E : str, optional
        Colormap for |E| (default: 'hot').
    cmap_comp : str, optional
        Colormap for E components (default: 'RdBu_r').
    show : bool, optional
        If True, display the figure (default: True).
    save_prefix : str or None, optional
        If set, save figure with this prefix (default: None).
    show_component_slices : bool, optional
        If True, show Ex, Ey, Ez slices instead of line-outs (default: False).
    Lx_nm, Ly_nm, Lz_nm : float, optional
        Box dimensions (nm) for gradient computation (default: 512.0).

    Returns
    -------
    matplotlib.figure.Figure or None
    """

    phi = np.load(phi_file)
    nx, ny, nz = phi.shape

    if ex_file and ey_file and ez_file:
        Ex = np.load(ex_file)
        Ey = np.load(ey_file)
        Ez = np.load(ez_file)
    else:
        dx_m = float(Lx_nm) * 1e-9 / (nx - 1)
        dy_m = float(Ly_nm) * 1e-9 / (ny - 1)
        dz_m = float(Lz_nm) * 1e-9 / (nz - 1)
        edge_order = 2 if min(phi.shape) > 2 else 1
        Ex, Ey, Ez = np.gradient(-phi, dx_m, dy_m, dz_m, edge_order=edge_order)

    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    x = np.linspace(-0.5, 0.5, nx)
    y = np.linspace(-0.5, 0.5, ny)
    z = np.linspace( 0.0, 1.0, nz)

    ix = int(round(x_frac * (nx-1)))
    iy = int(round(y_frac * (ny-1)))
    iz = int(round(z_frac * (nz-1)))

    axis = axis.lower()
    if axis == 'z':
        phi_slice = phi[:,:,iz]
        E_slice   = E_mag[:,:,iz]
        slice_extent = [-0.5,0.5, -0.5,0.5]
        slice_xlabel, slice_ylabel = 'x', 'y'
        slice_title_pos = f'(z = {z[iz]:.2f})'
        scatter_x, scatter_y = x[ix], y[iy]
        if show_component_slices:
            Ex_slice = Ex[:,:,iz]
            Ey_slice = Ey[:,:,iz]
            Ez_slice = Ez[:,:,iz]
    elif axis == 'y':
        phi_slice = phi[:,iy,:]
        E_slice   = E_mag[:,iy,:]
        slice_extent = [-0.5,0.5, 0.0,1.0]
        slice_xlabel, slice_ylabel = 'x', 'z'
        slice_title_pos = f'(y = {y[iy]:.2f})'
        scatter_x, scatter_y = x[ix], z[iz]
        if show_component_slices:
            Ex_slice = Ex[:,iy,:]
            Ey_slice = Ey[:,iy,:]
            Ez_slice = Ez[:,iy,:]
    elif axis == 'x':
        phi_slice = phi[ix,:,:]
        E_slice   = E_mag[ix,:,:]
        slice_extent = [-0.5,0.5, 0.0,1.0]
        slice_xlabel, slice_ylabel = 'y', 'z'
        slice_title_pos = f'(x = {x[ix]:.2f})'
        scatter_x, scatter_y = y[iy], z[iz]
        if show_component_slices:
            Ex_slice = Ex[ix,:,:]
            Ey_slice = Ey[ix,:,:]
            Ez_slice = Ez[ix,:,:]
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    if not show_component_slices:
        if axis == 'z':
            phi_line = phi[ix,iy,:]
            E_line   = E_mag[ix,iy,:]
            line_coord = z
            line_xlabel = 'z (fraction)'
            line_title_pos = f'at x = {x_frac:.2f}, y = {y_frac:.2f}'
        elif axis == 'y':
            phi_line = phi[ix,:,iz]
            E_line   = E_mag[ix,:,iz]
            line_coord = y
            line_xlabel = 'y (fraction)'
            line_title_pos = f'at x = {x_frac:.2f}, z = {z_frac:.2f}'
        elif axis == 'x':
            phi_line = phi[:,iy,iz]
            E_line   = E_mag[:,iy,iz]
            line_coord = x
            line_xlabel = 'x (fraction)'
            line_title_pos = f'at y = {y_frac:.2f}, z = {z_frac:.2f}'

    if show_component_slices:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        (ax_phi, ax_Emag, ax_dummy), (ax_Ex, ax_Ey, ax_Ez) = axes
        ax_dummy.set_visible(False)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax_phi, ax_Emag, ax_phi_line, ax_Emag_line = axes.ravel()

    im_phi = ax_phi.imshow(phi_slice.T, origin='lower', extent=slice_extent,
                           cmap=cmap_phi, aspect='auto')
    ax_phi.scatter(scatter_x, scatter_y, color='yellow', marker='x', s=80)
    ax_phi.set_title(f'Potential phi {slice_title_pos}')
    ax_phi.set_xlabel(slice_xlabel); ax_phi.set_ylabel(slice_ylabel)
    fig.colorbar(im_phi, ax=ax_phi, shrink=0.8)

    im_Emag = ax_Emag.imshow(E_slice.T, origin='lower', extent=slice_extent,
                             cmap=cmap_E, aspect='auto')
    ax_Emag.scatter(scatter_x, scatter_y, color='cyan', marker='x', s=80)
    ax_Emag.set_title(f'|E| field {slice_title_pos}')
    ax_Emag.set_xlabel(slice_xlabel); ax_Emag.set_ylabel(slice_ylabel)
    fig.colorbar(im_Emag, ax=ax_Emag, shrink=0.8)

    if show_component_slices:
        im_Ex = ax_Ex.imshow(Ex_slice.T, origin='lower', extent=slice_extent,
                             cmap=cmap_comp, aspect='auto')
        ax_Ex.scatter(scatter_x, scatter_y, color='black', marker='x', s=40)
        ax_Ex.set_title(f'Ex component {slice_title_pos}')
        ax_Ex.set_xlabel(slice_xlabel); ax_Ex.set_ylabel(slice_ylabel)
        fig.colorbar(im_Ex, ax=ax_Ex, shrink=0.8)

        im_Ey = ax_Ey.imshow(Ey_slice.T, origin='lower', extent=slice_extent,
                             cmap=cmap_comp, aspect='auto')
        ax_Ey.scatter(scatter_x, scatter_y, color='black', marker='x', s=40)
        ax_Ey.set_title(f'Ey component {slice_title_pos}')
        ax_Ey.set_xlabel(slice_xlabel); ax_Ey.set_ylabel(slice_ylabel)
        fig.colorbar(im_Ey, ax=ax_Ey, shrink=0.8)

        im_Ez = ax_Ez.imshow(Ez_slice.T, origin='lower', extent=slice_extent,
                             cmap=cmap_comp, aspect='auto')
        ax_Ez.scatter(scatter_x, scatter_y, color='black', marker='x', s=40)
        ax_Ez.set_title(f'Ez component {slice_title_pos}')
        ax_Ez.set_xlabel(slice_xlabel); ax_Ez.set_ylabel(slice_ylabel)
        fig.colorbar(im_Ez, ax=ax_Ez, shrink=0.8)
    else:
        ax_phi_line.plot(line_coord, phi_line, 'b-', lw=2)
        ax_phi_line.set_xlabel(line_xlabel)
        ax_phi_line.set_ylabel('Potential (V)')
        ax_phi_line.set_title(f'phi({axis}) {line_title_pos}')
        ax_phi_line.grid(True, alpha=0.3)

        ax_Emag_line.plot(line_coord, E_line, 'r-', lw=2)
        ax_Emag_line.set_xlabel(line_xlabel)
        ax_Emag_line.set_ylabel('|E| (V/m)')
        ax_Emag_line.set_title(f'|E|({axis}) {line_title_pos}')
        ax_Emag_line.grid(True, alpha=0.3)

    fig.tight_layout(pad=1.0)

    if save_prefix:
        fig.savefig(save_prefix + f"_{axis}_slice.png", dpi=150)

    if show:
        plt.show()
    else:
        return fig
