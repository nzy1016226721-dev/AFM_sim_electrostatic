import numpy as np
import matplotlib.pyplot as plt
import csv
import os

E_CHARGE_C = 1.602176634e-19
AS_DOPED_SI_DEFAULT_N_CM3 = 7.0e20
AS_DOPED_SI_DEFAULT_MOBILITY_CM2_VS = 30.0

JOULE_SUMMARY_HEADER = [
    "config", "Vtip (V)", "P_instantaneous (W)", "P_average (W)",
    "E_diss_per_cycle (J)", "frequency (Hz)", "time_mode", "sigma_source",
]


def conductivity_from_carrier_density(n_cm3, mobility_cm2_v_s=AS_DOPED_SI_DEFAULT_MOBILITY_CM2_VS):
    """Compute electrical conductivity from carrier density and mobility.

    sigma = e * n * mu

    Parameters
    ----------
    n_cm3 : float
        Carrier density in cm^-3.
    mobility_cm2_v_s : float, optional
        Mobility in cm^2/V/s (default: 30.0).

    Returns
    -------
    float
        Conductivity in S/m.
    """
    n_m3 = float(n_cm3) * 1e6
    mobility_m2_v_s = float(mobility_cm2_v_s) * 1e-4
    return E_CHARGE_C * n_m3 * mobility_m2_v_s


def resolve_sigma_value(block, default_mobility_cm2_v_s=AS_DOPED_SI_DEFAULT_MOBILITY_CM2_VS):
    """Resolve conductivity from a block, computing from doping if needed.

    Parameters
    ----------
    block : dict
        Block dict with 'sigma_val' or carrier density keys.
    default_mobility_cm2_v_s : float, optional
        Default mobility (default: 30.0).

    Returns
    -------
    float
        Conductivity in S/m.
    """
    if "sigma_val" in block:
        return float(block["sigma_val"])

    material = str(block.get("material", "")).replace("_", "").replace("-", "").lower()
    n_cm3 = block.get("carrier_density_cm3", block.get("doping_cm3", block.get("n_cm3")))
    if n_cm3 is None and material in {"sias", "asdopedsi", "si:as"}:
        n_cm3 = AS_DOPED_SI_DEFAULT_N_CM3
    if n_cm3 is None:
        raise KeyError("Conductivity block needs sigma_val or carrier_density_cm3/doping_cm3/n_cm3")

    mobility = block.get("mobility_cm2_v_s",
                         block.get("mobility_cm2_V_s", default_mobility_cm2_v_s))
    return conductivity_from_carrier_density(n_cm3, mobility)


def joule_time_settings(cfg):
    """Extract Joule heating time-domain settings from config.

    Determines frequency, period, time mode (instantaneous / peak_phasor /
    rms_phasor), and the power averaging scale factor.

    Parameters
    ----------
    cfg : dict
        Configuration dict with optional 'joule_heating' sub-dict.

    Returns
    -------
    frequency_hz : float or None
        Oscillation frequency.
    period_s : float or None
        Oscillation period.
    time_mode : str
        'instantaneous', 'peak_phasor', or 'rms_phasor'.
    average_scale : float
        Scale factor for average power (0.5 for peak, 1.0 for RMS/instantaneous).
    """
    settings = cfg.get("joule_heating", {}) if isinstance(cfg, dict) else {}
    frequency_hz = settings.get(
        "frequency_hz",
        cfg.get("oscillation_frequency_hz", cfg.get("f_osc_hz")) if isinstance(cfg, dict) else None,
    )
    period_s = settings.get("period_s", None)
    if period_s is None and frequency_hz not in (None, 0):
        period_s = 1.0 / float(frequency_hz)

    time_mode = str(settings.get("time_mode", cfg.get("joule_time_mode", "instantaneous"))).lower()
    if time_mode in {"peak", "peak_phasor", "phasor_peak"}:
        average_scale = 0.5
        time_mode = "peak_phasor"
    elif time_mode in {"rms", "rms_phasor"}:
        average_scale = 1.0
        time_mode = "rms_phasor"
    else:
        average_scale = 1.0
        time_mode = "instantaneous"

    return frequency_hz, period_s, time_mode, average_scale


def joule_energy_summary(P_instantaneous, cfg):
    """Compute average power and energy-per-cycle from instantaneous power.

    Parameters
    ----------
    P_instantaneous : float
        Instantaneous Joule power (W).
    cfg : dict
        Config dict (used for time settings).

    Returns
    -------
    P_average : float
        Time-averaged power (W).
    E_cycle : float
        Energy dissipated per cycle (J).
    frequency_hz : float or None
        Frequency used.
    time_mode : str
        Time mode label.
    """
    frequency_hz, period_s, time_mode, average_scale = joule_time_settings(cfg)
    P_average = float(P_instantaneous) * average_scale
    E_cycle = P_average * period_s if period_s is not None else np.nan
    return P_average, E_cycle, frequency_hz, time_mode


def append_joule_summary(csv_path, row):
    """Append a row to the Joule heating summary CSV.

    Parameters
    ----------
    csv_path : str
        Path to CSV file.
    row : list
        Data row matching JOULE_SUMMARY_HEADER.

    Returns
    -------
    None
    """
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(JOULE_SUMMARY_HEADER)
        writer.writerow(row)


def compute_joule_heating(phi, sigma_cell, Lx_nm, Ly_nm, Lz_nm):
    """Compute Joule heating power density and total power.

    Calculates E = -grad(phi) at nodes, interpolates to cell centres,
    then computes p = sigma * |E|^2 and integrates over the cell volume.

    Parameters
    ----------
    phi : np.ndarray
        3D potential array (Nx x Ny x Nz).
    sigma_cell : np.ndarray
        Conductivity on cells ((Nx-1) x (Ny-1) x (Nz-1)).
    Lx_nm : float
        Box length in x (nm).
    Ly_nm : float
        Box length in y (nm).
    Lz_nm : float
        Box length in z (nm).

    Returns
    -------
    power_density : np.ndarray
        3D power density on cells (W/m^3).
    P_total : float
        Total integrated power (W).
    Jx, Jy, Jz : np.ndarray
        Current density components on cells (A/m^2).
    Ex_cell, Ey_cell, Ez_cell : np.ndarray
        Electric field components on cells (V/m).
    """
    phi = np.asarray(phi, dtype=np.float64)
    sigma_cell = np.asarray(sigma_cell, dtype=np.float64)
    Nx, Ny, Nz = phi.shape
    expected_shape = (Nx - 1, Ny - 1, Nz - 1)
    if sigma_cell.shape != expected_shape:
        raise ValueError(f"sigma_cell shape {sigma_cell.shape} does not match {expected_shape}")
    if min(Nx, Ny, Nz) < 2:
        raise ValueError("phi must have at least two nodes along every axis")

    dx_m = float(Lx_nm) * 1e-9 / (Nx - 1)
    dy_m = float(Ly_nm) * 1e-9 / (Ny - 1)
    dz_m = float(Lz_nm) * 1e-9 / (Nz - 1)
    edge_order = 2 if min(Nx, Ny, Nz) > 2 else 1

    Ex_phys, Ey_phys, Ez_phys = np.gradient(-phi, dx_m, dy_m, dz_m, edge_order=edge_order)

    Ex_cell = (Ex_phys[:-1, :-1, :-1] + Ex_phys[1:, :-1, :-1] +
               Ex_phys[:-1, 1:, :-1] + Ex_phys[1:, 1:, :-1] +
               Ex_phys[:-1, :-1, 1:] + Ex_phys[1:, :-1, 1:] +
               Ex_phys[:-1, 1:, 1:] + Ex_phys[1:, 1:, 1:]) / 8.0

    Ey_cell = (Ey_phys[:-1, :-1, :-1] + Ey_phys[1:, :-1, :-1] +
               Ey_phys[:-1, 1:, :-1] + Ey_phys[1:, 1:, :-1] +
               Ey_phys[:-1, :-1, 1:] + Ey_phys[1:, :-1, 1:] +
               Ey_phys[:-1, 1:, 1:] + Ey_phys[1:, 1:, 1:]) / 8.0

    Ez_cell = (Ez_phys[:-1, :-1, :-1] + Ez_phys[1:, :-1, :-1] +
               Ez_phys[:-1, 1:, :-1] + Ez_phys[1:, 1:, :-1] +
               Ez_phys[:-1, :-1, 1:] + Ez_phys[1:, :-1, 1:] +
               Ez_phys[:-1, 1:, 1:] + Ez_phys[1:, 1:, 1:]) / 8.0

    Jx = sigma_cell * Ex_cell
    Jy = sigma_cell * Ey_cell
    Jz = sigma_cell * Ez_cell

    E_sq = Ex_cell**2 + Ey_cell**2 + Ez_cell**2
    power_density = sigma_cell * E_sq

    cell_volume = dx_m * dy_m * dz_m
    P_total = float(np.sum(power_density, dtype=np.float64) * cell_volume)

    return power_density.astype(np.float32), P_total, Jx, Jy, Jz, Ex_cell, Ey_cell, Ez_cell


def plot_scalar_plane(data3d, boundary_mask=None, plane=(True, True, 0.5),
                      cmap='hot', label='', vmin=None, vmax=None):
    """Plot a 2D slice of a 3D scalar field with optional boundary masking.

    Parameters
    ----------
    data3d : np.ndarray
        3D scalar field.
    boundary_mask : np.ndarray (bool) or None, optional
        Boundary mask (default: None).
    plane : tuple, optional
        Slice specification: two True + one float (default: (True, True, 0.5)).
    cmap : str, optional
        Colormap name (default: 'hot').
    label : str, optional
        Colorbar label (default: '').
    vmin, vmax : float or None, optional
        Colour scale limits (default: None = auto).

    Returns
    -------
    matplotlib.figure.Figure
    """

    nx, ny, nz = data3d.shape
    px, py, pz = plane

    if px is True and py is True and isinstance(pz, float):
        iz = int(pz * (nz - 1))
        data2d = data3d[:, :, iz].T
        mask2d = boundary_mask[:, :, iz].T if boundary_mask is not None else None
        plane_label = f"XY plane at z={pz:.2f}"
    elif px is True and isinstance(py, float) and pz is True:
        iy = int(py * (ny - 1))
        data2d = data3d[:, iy, :].T
        mask2d = boundary_mask[:, iy, :].T if boundary_mask is not None else None
        plane_label = f"XZ plane at y={py:.2f}"
    elif isinstance(px, float) and py is True and pz is True:
        ix = int(px * (nx - 1))
        data2d = data3d[ix, :, :].T
        mask2d = boundary_mask[ix, :, :].T if boundary_mask is not None else None
        plane_label = f"YZ plane at x={px:.2f}"
    else:
        raise ValueError("Invalid plane specification")

    if mask2d is not None:
        data2d = np.ma.masked_where(mask2d, data2d)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data2d, origin='lower', cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title(f"{label} - {plane_label}")
    ax.set_xlabel("Grid index X or Y")
    ax.set_ylabel("Grid index Y or Z")
    fig.tight_layout(pad=0.5)
    return fig
