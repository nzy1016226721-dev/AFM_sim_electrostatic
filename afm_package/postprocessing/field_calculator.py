import numpy as np


def compute_residual_Laplace(phi, eps_cell, boundary_mask=None):
    """Compute the Poisson equation residual: div(eps * grad(phi)).

    Parameters
    ----------
    phi : np.ndarray
        3D potential (Nx x Ny x Nz).
    eps_cell : np.ndarray
        Epsilon on cells ((Nx-1) x (Ny-1) x (Nz-1)).
    boundary_mask : np.ndarray (bool) or None, optional
        True at Dirichlet nodes (default: None).

    Returns
    -------
    L2 : float
        RMS residual over interior free nodes.
    max_res : float
        Maximum absolute residual.
    resid_full : np.ndarray
        3D residual array (NaN at boundaries).
    """
    Nx, Ny, Nz = phi.shape
    ec = eps_cell.astype(np.float64)
    axp = 0.25*(ec[1:,:-1,:-1]+ec[1:,1:,:-1]+ec[1:,:-1,1:]+ec[1:,1:,1:])
    axm = 0.25*(ec[:-1,:-1,:-1]+ec[:-1,1:,:-1]+ec[:-1,:-1,1:]+ec[:-1,1:,1:])
    ayp = 0.25*(ec[:-1,1:,:-1]+ec[1:,1:,:-1]+ec[:-1,1:,1:]+ec[1:,1:,1:])
    aym = 0.25*(ec[:-1,:-1,:-1]+ec[1:,:-1,:-1]+ec[:-1,:-1,1:]+ec[1:,:-1,1:])
    azp = 0.25*(ec[:-1,:-1,1:]+ec[1:,:-1,1:]+ec[:-1,1:,1:]+ec[1:,1:,1:])
    azm = 0.25*(ec[:-1,:-1,:-1]+ec[1:,:-1,:-1]+ec[:-1,1:,:-1]+ec[1:,1:,:-1])
    a0 = axp+axm+ayp+aym+azp+azm

    num = (axp*phi[2:,1:-1,1:-1]+axm*phi[:-2,1:-1,1:-1]+
           ayp*phi[1:-1,2:,1:-1]+aym*phi[1:-1,:-2,1:-1]+
           azp*phi[1:-1,1:-1,2:]+azm*phi[1:-1,1:-1,:-2])
    resid_interior = num - a0*phi[1:-1,1:-1,1:-1]

    resid_full = np.full_like(phi, np.nan)
    resid_full[1:-1,1:-1,1:-1] = resid_interior

    if boundary_mask is not None:
        free = ~boundary_mask[1:-1,1:-1,1:-1]
        resid_free = resid_interior[free]
        resid_full[1:-1,1:-1,1:-1][~free] = np.nan
    else:
        resid_free = resid_interior.ravel()

    L2 = np.sqrt(np.mean(resid_free**2)) if resid_free.size>0 else 0.0
    max_res = np.max(np.abs(resid_free)) if resid_free.size>0 else 0.0
    return L2, max_res, resid_full


def check_curl_E(phi, dx, dy, dz):
    """Check that curl of the electric field is approximately zero.

    Computes E = -grad(phi) and then curl(E) via central differences.

    Parameters
    ----------
    phi : np.ndarray
        3D potential.
    dx, dy, dz : float
        Grid spacings in m.

    Returns
    -------
    mx, my, mz : float
        Maximum absolute curl_x, curl_y, curl_z.
    mag : float
        Maximum curl magnitude.
    curl_full : np.ndarray
        3D curl magnitude (NaN at boundaries).
    """

    Ex, Ey, Ez = np.gradient(-phi, dx, dy, dz, edge_order=2)
    curl_x = (Ez[1:-1,2:,1:-1]-Ez[1:-1,:-2,1:-1])/(2*dy) - (Ey[1:-1,1:-1,2:]-Ey[1:-1,1:-1,:-2])/(2*dz)
    curl_y = (Ex[1:-1,1:-1,2:]-Ex[1:-1,1:-1,:-2])/(2*dz) - (Ez[2:,1:-1,1:-1]-Ez[:-2,1:-1,1:-1])/(2*dx)
    curl_z = (Ey[2:,1:-1,1:-1]-Ey[:-2,1:-1,1:-1])/(2*dx) - (Ex[1:-1,2:,1:-1]-Ex[1:-1,:-2,1:-1])/(2*dy)
    curl_mag_int = np.sqrt(curl_x**2+curl_y**2+curl_z**2)
    curl_full = np.full_like(phi, np.nan)
    curl_full[1:-1,1:-1,1:-1] = curl_mag_int
    return (np.max(np.abs(curl_x)), np.max(np.abs(curl_y)), np.max(np.abs(curl_z)),
            np.max(curl_mag_int), curl_full)


def check_dirichlet(phi, tip_voltage, backgate_voltage=0.0, tip_mask=None):
    """Check Dirichlet boundary condition fidelity.

    Parameters
    ----------
    phi : np.ndarray
        3D potential.
    tip_voltage : float
        Applied tip voltage (V).
    backgate_voltage : float, optional
        Back-gate voltage (default: 0.0).
    tip_mask : np.ndarray (bool) or None, optional
        Tip voxel mask (default: None).

    Returns
    -------
    err_bot : float
        Max deviation of phi[:,:,0] from backgate_voltage.
    tip_err : float
        Max deviation of phi[tip_mask] from tip_voltage.
    max_phi, min_phi : float
        Global extrema of phi on the tip (or full array).
    """

    err_bot = np.max(np.abs(phi[:,:,0]-backgate_voltage))
    if tip_mask is not None and np.any(tip_mask):
        tip_err = np.max(np.abs(phi[tip_mask]-tip_voltage))
        max_phi = np.max(phi[tip_mask])
        min_phi = np.min(phi[tip_mask])
    else:
        max_phi = np.max(phi)
        min_phi = np.min(phi)
        tip_err = abs(max_phi-tip_voltage) if tip_voltage>0 else abs(min_phi-tip_voltage)
    return err_bot, tip_err, max_phi, min_phi


def check_neumann(phi, tip_mask=None):
    """Check homogeneous Neumann BC fidelity on all six faces.

    Measures |phi_boundary - phi_interior| as a proxy for dphi/dn = 0.

    Parameters
    ----------
    phi : np.ndarray
        3D potential.
    tip_mask : np.ndarray (bool) or None, optional
        Tip mask (used to exclude tip from top-face check).

    Returns
    -------
    e0, e1, e2, e3, ez1 : float
        Max face differences for x=0, x=1, y=0, y=1, z=1 respectively.
    """

    e0=np.max(np.abs(phi[0]-phi[1])); e1=np.max(np.abs(phi[-1]-phi[-2]))
    e2=np.max(np.abs(phi[:,0]-phi[:,1])); e3=np.max(np.abs(phi[:,-1]-phi[:,-2]))
    top_diff = np.abs(phi[:,:,-1]-phi[:,:,-2])
    if tip_mask is not None and tip_mask.shape==phi.shape:
        top_diff = np.ma.masked_where(tip_mask[:,:,-1], top_diff)
        ez1 = np.max(top_diff) if not top_diff.mask.all() else 0.0
    else:
        ez1 = np.max(top_diff)
    return e0,e1,e2,e3,ez1


def cell_center_to_node(sigma_cell):
    """Interpolate cell-centred data to nodes via 8-point averaging.

    Parameters
    ----------
    sigma_cell : np.ndarray
        Cell-centred array (Nx x Ny x Nz).

    Returns
    -------
    np.ndarray
        Node-centred array (Nx+1 x Ny+1 x Nz+1) with zeros at boundaries.
    """

    Nx = sigma_cell.shape[0] + 1
    Ny = sigma_cell.shape[1] + 1
    Nz = sigma_cell.shape[2] + 1
    node = np.zeros((Nx, Ny, Nz), dtype=sigma_cell.dtype)
    node[1:-1, 1:-1, 1:-1] = (
        sigma_cell[:-1, :-1, :-1] + sigma_cell[1:, :-1, :-1] +
        sigma_cell[:-1, 1:, :-1] + sigma_cell[1:, 1:, :-1] +
        sigma_cell[:-1, :-1, 1:] + sigma_cell[1:, :-1, 1:] +
        sigma_cell[:-1, 1:, 1:] + sigma_cell[1:, 1:, 1:]) / 8.0
    return node


def compute_current_divergence(phi, sigma_cell, dx, dy, dz):
    """Compute div(J) = div(sigma * E) as a check of current conservation.

    Parameters
    ----------
    phi : np.ndarray
        3D potential.
    sigma_cell : np.ndarray
        Conductivity on cells.
    dx, dy, dz : float
        Grid spacings (m).

    Returns
    -------
    max_div : float
        Maximum |div J|.
    rms_div : float
        RMS div J.
    full : np.ndarray
        3D div J array (NaN at boundaries).
    """

    Ex, Ey, Ez = np.gradient(-phi, dx, dy, dz, edge_order=2)
    sn = cell_center_to_node(sigma_cell)
    Jx = sn*Ex; Jy = sn*Ey; Jz = sn*Ez
    divJ = (Jx[2:,1:-1,1:-1]-Jx[:-2,1:-1,1:-1])/(2*dx) + \
           (Jy[1:-1,2:,1:-1]-Jy[1:-1,:-2,1:-1])/(2*dy) + \
           (Jz[1:-1,1:-1,2:]-Jz[1:-1,1:-1,:-2])/(2*dz)
    full = np.full_like(phi, np.nan)
    full[1:-1,1:-1,1:-1] = divJ
    max_div = np.nanmax(np.abs(divJ))
    rms_div = np.sqrt(np.nanmean(divJ**2))
    return max_div, rms_div, full


def integrate_power(phi, sigma_cell, dx, dy, dz):
    """Compute total Joule power P = integral(sigma * |E|^2 dV).

    Parameters
    ----------
    phi : np.ndarray
        3D potential.
    sigma_cell : np.ndarray
        Conductivity on cells.
    dx, dy, dz : float
        Grid spacings (m).

    Returns
    -------
    float
        Total power in watts.
    """

    Ex, Ey, Ez = np.gradient(-phi, dx, dy, dz, edge_order=2)
    Ex_c = (Ex[:-1,:-1,:-1]+Ex[1:,:-1,:-1]+Ex[:-1,1:,:-1]+Ex[1:,1:,:-1]+
            Ex[:-1,:-1,1:]+Ex[1:,:-1,1:]+Ex[:-1,1:,1:]+Ex[1:,1:,1:])/8.0
    Ey_c = (Ey[:-1,:-1,:-1]+Ey[1:,:-1,:-1]+Ey[:-1,1:,:-1]+Ey[1:,1:,:-1]+
            Ey[:-1,:-1,1:]+Ey[1:,:-1,1:]+Ey[:-1,1:,1:]+Ey[1:,1:,1:])/8.0
    Ez_c = (Ez[:-1,:-1,:-1]+Ez[1:,:-1,:-1]+Ez[:-1,1:,:-1]+Ez[1:,1:,:-1]+
            Ez[:-1,:-1,1:]+Ez[1:,:-1,1:]+Ez[:-1,1:,1:]+Ez[1:,1:,1:])/8.0
    p_dens = sigma_cell*(Ex_c**2+Ey_c**2+Ez_c**2)
    return np.sum(p_dens)*(dx*dy*dz)
