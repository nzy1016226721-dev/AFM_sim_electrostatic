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
    print(phi.shape)



    x = np.linspace(-0.5, 0.5, nx)
    y = np.linspace(-0.5, 0.5, ny)
    z = np.linspace( 0.0, 1.0, nz)

    ix = int(x_frac * (nx-1))
    iy = int(y_frac * (ny-1))
    iz = int(z_frac * (nz-1))
    print(iz)
    
    axis = axis.lower()
    
    if axis == 'z':
        phi_slice = phi[:,:,iz]

        slice_extent = [-0.5,0.5, -0.5,0.5]
        slice_xlabel, slice_ylabel = 'x', 'y'
        slice_title_pos = f'(z = {z[iz]:.2f})'
        scatter_x, scatter_y = x[ix], y[iy]

    elif axis == 'y':
        phi_slice = phi[:,iy,:]

        slice_extent = [-0.5,0.5, 0.0,1.0]
        slice_xlabel, slice_ylabel = 'x', 'z'
        slice_title_pos = f'(y = {y[iy]:.2f})'
        scatter_x, scatter_y = x[ix], z[iz]

    elif axis == 'x':
        phi_slice = phi[ix,:,:]

        slice_extent = [-0.5,0.5, 0.0,1.0]
        slice_xlabel, slice_ylabel = 'y', 'z'
        slice_title_pos = f'(x = {x[ix]:.2f})'
        scatter_x, scatter_y = y[iy], z[iz]

    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    if axis == 'z':
        phi_line = phi[ix,iy,:]

        line_coord = z
        line_xlabel = 'z (fraction)'
        line_title_pos = f'at x = {x_frac:.2f}, y = {y_frac:.2f}'
    elif axis == 'y':
        phi_line = phi[ix,:,iz]

        line_coord = y
        line_xlabel = 'y (fraction)'
        line_title_pos = f'at x = {x_frac:.2f}, z = {z_frac:.2f}'
    elif axis == 'x':
        phi_line = phi[:,iy,iz]

        line_coord = x
        line_xlabel = 'x (fraction)'
        line_title_pos = f'at y = {y_frac:.2f}, z = {z_frac:.2f}'

    fig, axes = plt.subplots(1, 2)
    ax_phi, ax_phi_line = axes.ravel()  

    im_phi = ax_phi.imshow(phi_slice.T, origin='lower', extent=slice_extent,
                           cmap=cmap_phi, aspect='auto')
    #
    np.savetxt("3_unrounded_output_phi_postprocessing_matrix.csv", phi_slice.T, delimiter=",")
    #
    ax_phi.scatter(scatter_x, scatter_y, color='yellow', marker='x', s=80)
    ax_phi.set_title(f'Potential phi {slice_title_pos}')
    ax_phi.set_xlabel(slice_xlabel); ax_phi.set_ylabel(slice_ylabel)
    fig.colorbar(im_phi, ax=ax_phi, shrink=0.8)

    ax_phi_line.plot(line_coord, phi_line, 'b-', lw=2)
    ax_phi_line.set_xlabel(line_xlabel)
    ax_phi_line.set_ylabel('Potential (V)')
    ax_phi_line.set_title(f'phi({axis}) {line_title_pos}')
    ax_phi_line.grid(True, alpha=0.3)

    fig.tight_layout(pad=1.0)

    if save_prefix:
        fig.savefig(save_prefix + f"_{axis}_slice.png", dpi=150)

    if show:
        plt.show()
    else:
        return fig
