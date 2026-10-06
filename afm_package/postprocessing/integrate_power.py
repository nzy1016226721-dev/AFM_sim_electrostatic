import numpy as np
import matplotlib.pyplot as plt
import os


def energy_per_cycle(power_w, frequency_hz=None, period_s=None):
    """Compute energy per cycle from power and frequency/period.

    Parameters
    ----------
    power_w : float
        Power in watts.
    frequency_hz : float or None, optional
        Frequency in Hz (default: None).
    period_s : float or None, optional
        Period in seconds (default: None).

    Returns
    -------
    float
        Energy per cycle in joules, or NaN if neither frequency nor period given.
    """
    if period_s is None:
        if frequency_hz in (None, 0):
            return np.nan
        period_s = 1.0 / float(frequency_hz)
    return float(power_w) * float(period_s)


def load_array(path):
    """Load a .npy array and return it with its shape.

    Parameters
    ----------
    path : str
        Path to .npy file.

    Returns
    -------
    arr : np.ndarray
        Loaded array.
    shape : tuple
        Array shape.
    """
    arr = np.load(path)
    return arr, arr.shape


def compute_power(p_dens, Lx, Ly, Lz,
                  x_min=0.0, x_max=1.0,
                  y_min=0.0, y_max=1.0,
                  z_min=0.0, z_max=1.0):
    """Integrate power density over a sub-region of the box.

    Parameters
    ----------
    p_dens : np.ndarray
        3D power density array (W/m^3) on cell centres.
    Lx, Ly, Lz : float
        Box dimensions in nm.
    x_min, x_max, y_min, y_max, z_min, z_max : float, optional
        Integration region in fractional coordinates (default: full box [0,1]).

    Returns
    -------
    P : float
        Total integrated power (W).
    indices : tuple
        (i0, i1, j0, j1, k0, k1) array index bounds.
    """

    if any(v <= 0 for v in (Lx, Ly, Lz)):
        raise ValueError("Box dimensions must be positive")
    Nx, Ny, Nz = p_dens.shape

    dx_f = 1.0 / Nx
    dy_f = 1.0 / Ny
    dz_f = 1.0 / Nz

    i0 = int(np.ceil((x_min / dx_f) - 0.5))
    i1 = int(np.floor((x_max / dx_f) - 0.5)) + 1
    j0 = int(np.ceil((y_min / dy_f) - 0.5))
    j1 = int(np.floor((y_max / dy_f) - 0.5)) + 1
    k0 = int(np.ceil((z_min / dz_f) - 0.5))
    k1 = int(np.floor((z_max / dz_f) - 0.5)) + 1

    i0 = max(0, i0); i1 = min(Nx, i1)
    j0 = max(0, j0); j1 = min(Ny, j1)
    k0 = max(0, k0); k1 = min(Nz, k1)

    sub = p_dens[i0:i1, j0:j1, k0:k1]
    cell_vol = (Lx*1e-9 / Nx) * (Ly*1e-9 / Ny) * (Lz*1e-9 / Nz)
    P = np.sum(sub) * cell_vol
    return P, (i0, i1, j0, j1, k0, k1)


def slice_plot(p_dens, Lx, Ly, Lz,
               plane='xy', coord=0.5, zoom=1.0,
               region_bounds=None):
    """Plot a 2D slice of power density with optional zoom and region overlay.

    Parameters
    ----------
    p_dens : np.ndarray
        3D power density (W/m^3).
    Lx, Ly, Lz : float
        Box dimensions in nm.
    plane : str, optional
        'xy', 'xz', or 'yz' (default: 'xy').
    coord : float, optional
        Slice coordinate fraction (default: 0.5).
    zoom : float, optional
        Zoom factor around centre (default: 1.0).
    region_bounds : tuple or None, optional
        (xmin, xmax, ymin, ymax, zmin, zmax) for overlay lines.

    Returns
    -------
    matplotlib.figure.Figure
    """

    Nx, Ny, Nz = p_dens.shape
    if plane == 'xy':
        iz = int(coord * Nz)
        iz = max(0, min(Nz-1, iz))
        data = p_dens[:, :, iz].T
        extent = [0, Lx, 0, Ly]
        xlabel, ylabel = 'x (nm)', 'y (nm)'
        center_x = 0.5 * Lx
        center_y = 0.5 * Ly
    elif plane == 'xz':
        iy = int(coord * Ny)
        iy = max(0, min(Ny-1, iy))
        data = p_dens[:, iy, :].T
        extent = [0, Lx, 0, Lz]
        xlabel, ylabel = 'x (nm)', 'z (nm)'
        center_x = 0.5 * Lx
        center_y = 0.5 * Lz
    elif plane == 'yz':
        ix = int(coord * Nx)
        ix = max(0, min(Nx-1, ix))
        data = p_dens[ix, :, :].T
        extent = [0, Ly, 0, Lz]
        xlabel, ylabel = 'y (nm)', 'z (nm)'
        center_x = 0.5 * Ly
        center_y = 0.5 * Lz
    else:
        raise ValueError("Invalid plane")

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(data, origin='lower', cmap='hot', aspect='auto', extent=extent)

    if zoom > 1.0:
        half_w = (extent[1] - extent[0]) / (2.0 * zoom)
        half_h = (extent[3] - extent[2]) / (2.0 * zoom)
        ax.set_xlim(center_x - half_w, center_x + half_w)
        ax.set_ylim(center_y - half_h, center_y + half_h)

    if region_bounds is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = region_bounds
        if plane == 'xy':
            if xmin > 0 or xmax < 1:
                ax.axvline(xmin * Lx, color='cyan', ls='--')
                ax.axvline(xmax * Lx, color='cyan', ls='--')
            if ymin > 0 or ymax < 1:
                ax.axhline(ymin * Ly, color='cyan', ls='--')
                ax.axhline(ymax * Ly, color='cyan', ls='--')
        elif plane == 'xz':
            if xmin > 0 or xmax < 1:
                ax.axvline(xmin * Lx, color='cyan', ls='--')
                ax.axvline(xmax * Lx, color='cyan', ls='--')
            if zmin > 0 or zmax < 1:
                ax.axhline(zmin * Lz, color='cyan', ls='--')
                ax.axhline(zmax * Lz, color='cyan', ls='--')
        elif plane == 'yz':
            if ymin > 0 or ymax < 1:
                ax.axvline(ymin * Ly, color='cyan', ls='--')
                ax.axvline(ymax * Ly, color='cyan', ls='--')
            if zmin > 0 or zmax < 1:
                ax.axhline(zmin * Lz, color='cyan', ls='--')
                ax.axhline(zmax * Lz, color='cyan', ls='--')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Power density (W/m3) - {plane.upper()} slice at coord={coord:.3f}, zoom={zoom:.0f}x')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
    return fig


def line_plot(p_dens, Lx, Ly, Lz,
              axis='z', coord1=0.5, coord2=0.5, zoom=1.0):
    """Plot a 1D line-out of power density along a given axis.

    Parameters
    ----------
    p_dens : np.ndarray
        3D power density.
    Lx, Ly, Lz : float
        Box dimensions in nm.
    axis : str, optional
        Axis for line-out: 'x', 'y', or 'z' (default: 'z').
    coord1, coord2 : float, optional
        Other two coordinates (fractional, default: 0.5).
    zoom : float, optional
        Zoom factor (default: 1.0).

    Returns
    -------
    matplotlib.figure.Figure
    """

    Nx, Ny, Nz = p_dens.shape
    if axis == 'z':
        ix = int(coord1 * Nx)
        iy = int(coord2 * Ny)
        ix = max(0, min(Nx-1, ix))
        iy = max(0, min(Ny-1, iy))
        y_data = p_dens[ix, iy, :]
        x_vals = np.linspace(0, Lz, Nz)
        xlabel = 'z (nm)'
        center = 0.5 * Lz
    elif axis == 'x':
        iy = int(coord1 * Ny)
        iz = int(coord2 * Nz)
        iy = max(0, min(Ny-1, iy))
        iz = max(0, min(Nz-1, iz))
        y_data = p_dens[:, iy, iz]
        x_vals = np.linspace(0, Lx, Nx)
        xlabel = 'x (nm)'
        center = 0.5 * Lx
    elif axis == 'y':
        ix = int(coord1 * Nx)
        iz = int(coord2 * Nz)
        ix = max(0, min(Nx-1, ix))
        iz = max(0, min(Nz-1, iz))
        y_data = p_dens[ix, :, iz]
        x_vals = np.linspace(0, Ly, Ny)
        xlabel = 'y (nm)'
        center = 0.5 * Ly
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x_vals, y_data, 'b-')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Power density (W/m3)')
    ax.set_title(f'Line-out along {axis}')
    if zoom > 1.0:
        half = (x_vals[-1] - x_vals[0]) / (2.0 * zoom)
        ax.set_xlim(center - half, center + half)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def get_float(prompt, default=None):
    """Prompt for a float with an optional default.

    Parameters
    ----------
    prompt : str
        Input prompt.
    default : float or None, optional
        Default value if Enter is pressed.

    Returns
    -------
    float
    """
    s = input(prompt).strip()
    if s == "" and default is not None:
        return default
    return float(s)


def get_optional_float(prompt, default=None):
    """Prompt for a float; returns None if blank (no default fallback).

    Parameters
    ----------
    prompt : str
        Input prompt.
    default : float or None, optional
        Default value (unused if blank).

    Returns
    -------
    float or None
    """
    s = input(prompt).strip()
    if s == "":
        return default
    return float(s)


def interactive_main():
    """Interactive console tool for loading, viewing, and integrating power density.

    Walks through file loading, box dimension entry, frequency setting,
    region selection, slab integration, 2D slice plots, and 1D line-outs.

    Returns
    -------
    None
    """
    print("=== Interactive Power Density Integrator & Viewer ===\n")

    npy_path = None
    p_dens = None
    Lx = Ly = Lz = 512.0
    frequency_hz = None
    reg = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    while True:
        change_file = input("Change power density .npy file? (y/n, default n): ").strip().lower()
        if change_file == 'y':
            npy_path = input("  Enter file path: ").strip()
            if not os.path.isfile(npy_path):
                alt = os.path.join("outputs", os.path.basename(npy_path))
                if os.path.isfile(alt):
                    print(f"  Using {alt}")
                    npy_path = alt
                else:
                    print("  File not found, keeping previous.")
            else:
                p_dens, shape = load_array(npy_path)
                print(f"  Loaded array of shape {shape}")

        if p_dens is None:
            if npy_path is None:
                npy_path = input("Enter initial .npy file path: ").strip()
            if not os.path.isfile(npy_path):
                alt = os.path.join("outputs", os.path.basename(npy_path))
                if os.path.isfile(alt):
                    print(f"  Using {alt}")
                    npy_path = alt
                else:
                    print("File not found. Exiting.")
                    break
            p_dens, shape = load_array(npy_path)
            print(f"Loaded array of shape {shape}")

        change_box = input("Change box dimensions? (y/n, default n): ").strip().lower()
        if change_box == 'y':
            Lx = get_float("  Lx (nm): ", Lx)
            Ly = get_float("  Ly (nm): ", Ly)
            Lz = get_float("  Lz (nm): ", Lz)

        change_freq = input("Change AFM frequency for energy/cycle? (y/n, default n): ").strip().lower()
        if change_freq == 'y':
            frequency_hz = get_optional_float("  Frequency (Hz, blank for none): ", frequency_hz)

        change_reg = input("Change integration region? (y/n, default n): ").strip().lower()
        if change_reg == 'y':
            print("  Enter fractions (0-1), press Enter for default:")
            reg[0] = get_float("    x_min (default 0): ", 0.0)
            reg[1] = get_float("    x_max (default 1): ", 1.0)
            reg[2] = get_float("    y_min (default 0): ", 0.0)
            reg[3] = get_float("    y_max (default 1): ", 1.0)
            reg[4] = get_float("    z_min (default 0): ", 0.0)
            reg[5] = get_float("    z_max (default 1): ", 1.0)

        P, idx = compute_power(p_dens, Lx, Ly, Lz, *reg)
        E_cycle = energy_per_cycle(P, frequency_hz=frequency_hz)
        print(f"\n  Region: X [{reg[0]*Lx:.1f}, {reg[1]*Lx:.1f}] nm")
        print(f"          Y [{reg[2]*Ly:.1f}, {reg[3]*Ly:.1f}] nm")
        print(f"          Z [{reg[4]*Lz:.1f}, {reg[5]*Lz:.1f}] nm")
        print(f"  Integrated power = {P:.6e} W\n")
        if np.isfinite(E_cycle):
            print(f"  Dissipated energy/cycle = {E_cycle:.6e} J at {frequency_hz:.6e} Hz\n")

        do_slab = input("Slice integration (thin slab)? (y/n, default n): ").strip().lower()
        slab_reg = None
        if do_slab == 'y':
            plane = input("  Plane (xy, xz, yz) [xy]: ").strip().lower()
            if plane not in ('xy','xz','yz'): plane = 'xy'
            coord = get_float(f"  Coordinate fraction for {plane} plane (default 0.5): ", 0.5)
            thick_nm = get_float("  Thickness (nm, default 0.1): ", 0.1)
            if plane == 'xy':
                dz_f = thick_nm / Lz
                z_min = max(0.0, coord - dz_f/2)
                z_max = min(1.0, coord + dz_f/2)
                print("  Lateral bounds (fraction, press Enter for full):")
                x_min = get_float("    x_min (default 0): ", 0.0)
                x_max = get_float("    x_max (default 1): ", 1.0)
                y_min = get_float("    y_min (default 0): ", 0.0)
                y_max = get_float("    y_max (default 1): ", 1.0)
                slab_reg = [x_min, x_max, y_min, y_max, z_min, z_max]
            elif plane == 'xz':
                dy_f = thick_nm / Ly
                y_min = max(0.0, coord - dy_f/2)
                y_max = min(1.0, coord + dy_f/2)
                print("  Lateral bounds (x, z) - press Enter for full:")
                x_min = get_float("    x_min (default 0): ", 0.0)
                x_max = get_float("    x_max (default 1): ", 1.0)
                z_min = get_float("    z_min (default 0): ", 0.0)
                z_max = get_float("    z_max (default 1): ", 1.0)
                slab_reg = [x_min, x_max, y_min, y_max, z_min, z_max]
            else:
                dx_f = thick_nm / Lx
                x_min = max(0.0, coord - dx_f/2)
                x_max = min(1.0, coord + dx_f/2)
                print("  Lateral bounds (y, z) - press Enter for full:")
                y_min = get_float("    y_min (default 0): ", 0.0)
                y_max = get_float("    y_max (default 1): ", 1.0)
                z_min = get_float("    z_min (default 0): ", 0.0)
                z_max = get_float("    z_max (default 1): ", 1.0)
                slab_reg = [x_min, x_max, y_min, y_max, z_min, z_max]

            P_slab, _ = compute_power(p_dens, Lx, Ly, Lz, *slab_reg)
            E_slab = energy_per_cycle(P_slab, frequency_hz=frequency_hz)
            print(f"  Slice integrated power = {P_slab:.6e} W")
            if np.isfinite(E_slab):
                print(f"  Slice dissipated energy/cycle = {E_slab:.6e} J")

            reg_to_plot = slab_reg
        else:
            reg_to_plot = reg

        plane_plot = input("Slice plane for plot (xy, xz, yz) [default xy]: ").strip().lower()
        if plane_plot not in ('xy', 'xz', 'yz'):
            plane_plot = 'xy'
        coord_plot = get_float(f"Coordinate fraction for {plane_plot} slice (default 0.5): ", 0.5)
        zoom = get_float("Zoom factor (default 1 = full box): ", 1.0)

        region_bounds = None
        if not (reg_to_plot[0]==0 and reg_to_plot[1]==1 and reg_to_plot[2]==0 and reg_to_plot[3]==1 and reg_to_plot[4]==0 and reg_to_plot[5]==1):
            region_bounds = tuple(reg_to_plot)

        slice_plot(p_dens, Lx, Ly, Lz, plane_plot, coord_plot, zoom, region_bounds)

        do_line = input("Show a 1D line-out? (y/n, default n): ").strip().lower()
        if do_line == 'y':
            axis = input("  Axis (x, y, z) [default z]: ").strip().lower()
            if axis not in ('x', 'y', 'z'):
                axis = 'z'
            if axis == 'z':
                c1 = get_float("    x fraction (default 0.5): ", 0.5)
                c2 = get_float("    y fraction (default 0.5): ", 0.5)
            elif axis == 'x':
                c1 = get_float("    y fraction (default 0.5): ", 0.5)
                c2 = get_float("    z fraction (default 0.5): ", 0.5)
            else:
                c1 = get_float("    x fraction (default 0.5): ", 0.5)
                c2 = get_float("    z fraction (default 0.5): ", 0.5)
            zoom_l = get_float("  Zoom factor (default 1): ", 1.0)
            line_plot(p_dens, Lx, Ly, Lz, axis, c1, c2, zoom_l)

        again = input("\nContinue (y/n)? ").strip().lower()
        if again != 'y':
            print("Exiting.")
            break
