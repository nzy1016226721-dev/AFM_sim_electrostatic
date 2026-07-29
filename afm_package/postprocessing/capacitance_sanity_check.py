
#!/usr/bin/env python3
"""
Sanity check for AFM simulation: compare simulated lever arm with analytical
sphere‑plane model, accounting for a multi‑layer dielectric stack between the
back gate and the quantum dot.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob

# Physical constants
EPS0 = 8.8541878128e-12  # F/m


def sphere_plane_capacitance(Z_m, R_m, eps0=EPS0, tol=1e-15):
    """
    Exact capacitance of a sphere of radius R_m at height Z_m above
    an infinite grounded plane.
    Uses the infinite series:
        C = 4π ε0 R sinh(α) Σ_{n=1}∞ 1/sinh(nα),
        α = arcosh(1 + Z/R).
    """
    h = R_m + Z_m
    if h < R_m:
        raise ValueError("Sphere centre must be above the plane.")
    alpha = np.arccosh(h / R_m)
    sinh_alpha = np.sinh(alpha)

    series_sum = 0.0
    n = 1
    while True:
        term = 1.0 / np.sinh(n * alpha)
        series_sum += term
        if term < tol * series_sum:
            break
        n += 1
    return 4.0 * np.pi * eps0 * R_m * sinh_alpha * series_sum


def parse_phi_filename(fname):
    """Extract metadata from a phi .npy filename."""
    import re
    m = re.match(r'afm_phi_zoom_(\d+)x_(-?[\d.]+)V_(\d+)\.npy', fname)
    if m:
        return {'type': 'zoom', 'mag': int(m.group(1)),
                'Vtip': float(m.group(2)), 'config_idx': int(m.group(3))}
    m = re.match(r'afm_phi_(\d+)_(-?[\d.]+)V\.npy', fname)
    if m:
        return {'type': 'normal', 'config_idx': int(m.group(1)),
                'Vtip': float(m.group(2))}
    return None


def extract_phi_qd(phi, qd_xr, qd_yr, qd_zr, Lx, Ly, Lz, zoom_bounds=None):
    """Extract potential values inside a QD region."""
    nx, ny, nz = phi.shape
    if zoom_bounds is None:
        ix0 = max(0, int(np.ceil(qd_xr[0] / Lx * (nx - 1))))
        ix1 = min(nx - 1, int(np.floor(qd_xr[1] / Lx * (nx - 1))))
        iy0 = max(0, int(np.ceil(qd_yr[0] / Ly * (ny - 1))))
        iy1 = min(ny - 1, int(np.floor(qd_yr[1] / Ly * (ny - 1))))
        iz0 = max(0, int(np.ceil(qd_zr[0] / Lz * (nz - 1))))
        iz1 = min(nz - 1, int(np.floor(qd_zr[1] / Lz * (nz - 1))))
    else:
        x0f, x1f, y0f, y1f, z0f, z1f = zoom_bounds
        x_min = x0f * Lx
        x_max = x1f * Lx
        y_min = y0f * Ly
        y_max = y1f * Ly
        z_min = z0f * Lz
        z_max = z1f * Lz
        ox0 = max(qd_xr[0], x_min)
        ox1 = min(qd_xr[1], x_max)
        oy0 = max(qd_yr[0], y_min)
        oy1 = min(qd_yr[1], y_max)
        oz0 = max(qd_zr[0], z_min)
        oz1 = min(qd_zr[1], z_max)
        if ox1 <= ox0 or oy1 <= oy0 or oz1 <= oz0:
            return np.array([])
        ix0 = max(0, int(np.ceil((ox0 - x_min) / (x_max - x_min) * (nx - 1))))
        ix1 = min(nx - 1, int(np.floor((ox1 - x_min) / (x_max - x_min) * (nx - 1))))
        iy0 = max(0, int(np.ceil((oy0 - y_min) / (y_max - y_min) * (ny - 1))))
        iy1 = min(ny - 1, int(np.floor((oy1 - y_min) / (y_max - y_min) * (ny - 1))))
        iz0 = max(0, int(np.ceil((oz0 - z_min) / (z_max - z_min) * (nz - 1))))
        iz1 = min(nz - 1, int(np.floor((oz1 - z_min) / (z_max - z_min) * (nz - 1))))
    if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
        return np.array([])
    return phi[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1].ravel()


def compute_stats(phi_values):
    """Compute mean and max-absolute value."""
    if len(phi_values) == 0:
        return 0.0, 0.0
    avg = float(np.mean(phi_values))
    idx = int(np.argmax(np.abs(phi_values)))
    max_val = float(phi_values.flat[idx])
    return avg, max_val


def find_qd_block(nm_cfg, default_eps=12.0):
    """Find the quantum dot block in the nm config by eps_val match."""
    blocks = nm_cfg.get('blocks_nm', [])
    for i, b in enumerate(blocks):
        if b.get('eps_val') == default_eps and all(
                k in b for k in ('x_range_nm', 'y_range_nm', 'z_range_nm')):
            return i, b
    # Fallback: find the first block that has a z_range_nm and is not the full box
    for i, b in enumerate(blocks):
        if all(k in b for k in ('x_range_nm', 'y_range_nm', 'z_range_nm')):
            # If it has finite lateral extent (not full box), likely the QD
            if b['x_range_nm'] != [0, nm_cfg.get('Lx_nm', 100)] or b['y_range_nm'] != [0, nm_cfg.get('Ly_nm', 100)]:
                return i, b
    # If still none, return None
    return None, None


def create_default_qd_block(Lx, Ly, Lz, dot_diameter=25.0, dot_height=5.0, dot_bottom=46.0):
    """Create a default QD block for testing."""
    center = Lx / 2.0
    half = dot_diameter / 2.0
    return {
        'eps_val': 1.0,
        'x_range_nm': [center - half, center + half],
        'y_range_nm': [center - half, center + half],
        'z_range_nm': [dot_bottom, dot_bottom + dot_height]
    }


#!/usr/bin/env python3
"""
Sanity check for AFM simulation: compare simulated lever arm with analytical
sphere‑plane model, accounting for a multi‑layer dielectric stack between the
back gate and the quantum dot.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob

# Physical constants
EPS0 = 8.8541878128e-12  # F/m


def sphere_plane_capacitance(Z_m, R_m, eps0=EPS0, tol=1e-15):
    """
    Exact capacitance of a sphere of radius R_m at height Z_m above
    an infinite grounded plane.
    Uses the infinite series:
        C = 4π ε0 R sinh(α) Σ_{n=1}∞ 1/sinh(nα),
        α = arcosh(1 + Z/R).
    """
    h = R_m + Z_m
    if h < R_m:
        raise ValueError("Sphere centre must be above the plane.")
    alpha = np.arccosh(h / R_m)
    sinh_alpha = np.sinh(alpha)

    series_sum = 0.0
    n = 1
    while True:
        term = 1.0 / np.sinh(n * alpha)
        series_sum += term
        if term < tol * series_sum:
            break
        n += 1
    return 4.0 * np.pi * eps0 * R_m * sinh_alpha * series_sum


def parse_phi_filename(fname):
    """Extract metadata from a phi .npy filename."""
    import re
    m = re.match(r'afm_phi_zoom_(\d+)x_(-?[\d.]+)V_(\d+)\.npy', fname)
    if m:
        return {'type': 'zoom', 'mag': int(m.group(1)),
                'Vtip': float(m.group(2)), 'config_idx': int(m.group(3))}
    m = re.match(r'afm_phi_(\d+)_(-?[\d.]+)V\.npy', fname)
    if m:
        return {'type': 'normal', 'config_idx': int(m.group(1)),
                'Vtip': float(m.group(2))}
    return None


def extract_phi_qd(phi, qd_xr, qd_yr, qd_zr, Lx, Ly, Lz, zoom_bounds=None):
    """Extract potential values inside a QD region."""
    nx, ny, nz = phi.shape
    if zoom_bounds is None:
        ix0 = max(0, int(np.ceil(qd_xr[0] / Lx * (nx - 1))))
        ix1 = min(nx - 1, int(np.floor(qd_xr[1] / Lx * (nx - 1))))
        iy0 = max(0, int(np.ceil(qd_yr[0] / Ly * (ny - 1))))
        iy1 = min(ny - 1, int(np.floor(qd_yr[1] / Ly * (ny - 1))))
        iz0 = max(0, int(np.ceil(qd_zr[0] / Lz * (nz - 1))))
        iz1 = min(nz - 1, int(np.floor(qd_zr[1] / Lz * (nz - 1))))
    else:
        x0f, x1f, y0f, y1f, z0f, z1f = zoom_bounds
        x_min = x0f * Lx
        x_max = x1f * Lx
        y_min = y0f * Ly
        y_max = y1f * Ly
        z_min = z0f * Lz
        z_max = z1f * Lz
        ox0 = max(qd_xr[0], x_min)
        ox1 = min(qd_xr[1], x_max)
        oy0 = max(qd_yr[0], y_min)
        oy1 = min(qd_yr[1], y_max)
        oz0 = max(qd_zr[0], z_min)
        oz1 = min(qd_zr[1], z_max)
        if ox1 <= ox0 or oy1 <= oy0 or oz1 <= oz0:
            return np.array([])
        ix0 = max(0, int(np.ceil((ox0 - x_min) / (x_max - x_min) * (nx - 1))))
        ix1 = min(nx - 1, int(np.floor((ox1 - x_min) / (x_max - x_min) * (nx - 1))))
        iy0 = max(0, int(np.ceil((oy0 - y_min) / (y_max - y_min) * (ny - 1))))
        iy1 = min(ny - 1, int(np.floor((oy1 - y_min) / (y_max - y_min) * (ny - 1))))
        iz0 = max(0, int(np.ceil((oz0 - z_min) / (z_max - z_min) * (nz - 1))))
        iz1 = min(nz - 1, int(np.floor((oz1 - z_min) / (z_max - z_min) * (nz - 1))))
    if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
        return np.array([])
    return phi[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1].ravel()


def compute_stats(phi_values):
    """Compute mean and max-absolute value."""
    if len(phi_values) == 0:
        return 0.0, 0.0
    avg = float(np.mean(phi_values))
    idx = int(np.argmax(np.abs(phi_values)))
    max_val = float(phi_values.flat[idx])
    return avg, max_val


def find_qd_block(nm_cfg, default_eps=12.0):
    """Find the quantum dot block in the nm config by eps_val match."""
    blocks = nm_cfg.get('blocks_nm', [])
    for i, b in enumerate(blocks):
        if b.get('eps_val') == default_eps and all(
                k in b for k in ('x_range_nm', 'y_range_nm', 'z_range_nm')):
            return i, b
    # Fallback: find the first block that has a z_range_nm and is not the full box
    for i, b in enumerate(blocks):
        if all(k in b for k in ('x_range_nm', 'y_range_nm', 'z_range_nm')):
            if b['x_range_nm'] != [0, nm_cfg.get('Lx_nm', 100)] or b['y_range_nm'] != [0, nm_cfg.get('Ly_nm', 100)]:
                return i, b
    return None, None


def create_default_qd_block(Lx, Ly, Lz, dot_diameter=25.0, dot_height=5.0, dot_bottom=46.0):
    """Create a default QD block for testing."""
    center = Lx / 2.0
    half = dot_diameter / 2.0
    return {
        'eps_val': 1.0,
        'x_range_nm': [center - half, center + half],
        'y_range_nm': [center - half, center + half],
        'z_range_nm': [dot_bottom, dot_bottom + dot_height]
    }


def sanity_check_comparison_multi(phi_files, qd_block, qd_top_z_nm, Lx_nm, Ly_nm, Lz_nm,
                                  R_nm, config_dir, nm_cfg, use_max=False):
    """
    Compare simulated lever arm with analytical sphere‑plane model,
    using a multi‑layer dielectric stack for C_dot.
    Includes normalised comparison and agreement statistics.
    """
    # ---- Gather simulation data ----
    spacings = []
    alpha_sim = []
    for fpath in phi_files:
        fname = os.path.basename(fpath)
        info = parse_phi_filename(fname)
        if info is None:
            print(f"Skipping {fname}: could not parse filename.")
            continue
        Vtip = info['Vtip']
        config_idx = info['config_idx']
        is_zoom = info['type'] == 'zoom'
        config_path = os.path.join(config_dir, f'afm_config_{config_idx}.json')
        if not os.path.isfile(config_path):
            print(f"Config {config_path} not found, skipping {fname}.")
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        tip_z_frac = cfg.get('tip_z', 0.5)
        spacing_nm = (tip_z_frac - qd_top_z_nm / Lz_nm) * Lz_nm
        phi = np.load(fpath)
        nx, ny, nz = phi.shape
        qd_xr = qd_block['x_range_nm']
        qd_yr = qd_block['y_range_nm']
        qd_zr = qd_block['z_range_nm']
        zoom_bounds = None
        if is_zoom:
            cut = cfg.get('zoom_simulation', {}).get('cut', {})
            if cut and all(k in cut for k in ('x_range', 'y_range', 'z_range')):
                zoom_bounds = (cut['x_range'] + cut['y_range'] + cut['z_range'])
        phi_qd = extract_phi_qd(phi, qd_xr, qd_yr, qd_zr, Lx_nm, Ly_nm, Lz_nm, zoom_bounds)
        if len(phi_qd) == 0:
            print(f"    QD region outside grid – skipping {fname}")
            continue
        avg, mx = compute_stats(phi_qd)
        v_qd = avg if not use_max else mx
        alpha = v_qd / Vtip if Vtip != 0 else 0.0
        spacings.append(spacing_nm)
        alpha_sim.append(alpha)

    if not spacings:
        print("No valid data for sanity check.")
        return

    idx_sort = np.argsort(spacings)
    spacings_np = np.array(spacings)[idx_sort]
    alpha_sim_np = np.array(alpha_sim)[idx_sort]

    # ---- Analytical model with multi‑layer dielectric stack ----
    dot_bottom_nm = qd_block['z_range_nm'][0]

    blocks_nm = nm_cfg.get('blocks_nm', [])
    layers = []
    for blk in blocks_nm:
        xr = blk.get('x_range_nm', [0, Lx_nm])
        yr = blk.get('y_range_nm', [0, Ly_nm])
        if xr == [0, Lx_nm] and yr == [0, Ly_nm]:
            if 'z_range_nm' not in blk:
                continue
            z0, z1 = blk['z_range_nm']
            if z0 < dot_bottom_nm and z1 <= dot_bottom_nm:
                eps = blk.get('eps_val', 1.0)
                d_nm = z1 - z0
                if d_nm > 0:
                    layers.append((z0, z1, eps, d_nm))
    layers.sort(key=lambda x: x[0])

    sum_d_over_eps = 0.0
    print("\nDielectric layers below dot (from back gate to dot bottom):")
    for z0, z1, eps, d_nm in layers:
        d_m = d_nm * 1e-9
        sum_d_over_eps += d_m / eps
        print(f"  {z0:.1f}–{z1:.1f} nm, ε = {eps:.2f}, d = {d_nm:.1f} nm")

    if sum_d_over_eps == 0:
        d_m = dot_bottom_nm * 1e-9
        eps_r = 11.7
        sum_d_over_eps = d_m / eps_r
        print(f"  No full‑XY layers found below dot. Using uniform Si (ε={eps_r}) of thickness {dot_bottom_nm:.1f} nm.")

    dot_radius_nm = (qd_block['x_range_nm'][1] - qd_block['x_range_nm'][0]) / 2.0
    A_dot_m2 = np.pi * (dot_radius_nm * 1e-9)**2
    C_dot = EPS0 * A_dot_m2 / sum_d_over_eps

    print(f"\nAnalytical model parameters (multi‑layer):")
    print(f"  Dot diameter = {2*dot_radius_nm:.1f} nm")
    print(f"  Dot area = {A_dot_m2*1e18:.2f} nm²")
    print(f"  Equivalent dielectric thickness (Σ d/ε) = {sum_d_over_eps*1e9:.2f} nm")
    print(f"  C_dot = {C_dot:.3e} F\n")

    R_m = R_nm * 1e-9
    print("Spacing (nm)   C_tip (F)        alpha_theory")
    print("---------------------------------------------")
    alpha_theory = []
    for Z_nm in spacings_np:
        Z_m = Z_nm * 1e-9
        if Z_m < 1e-12:
            Z_m = 1e-12
        C_tip = sphere_plane_capacitance(Z_m, R_m, eps0=EPS0)
        alpha = C_tip / (C_tip + C_dot)
        alpha_theory.append(alpha)
        print(f"{Z_nm:12.1f}   {C_tip:.3e}   {alpha:.6f}")

    alpha_theory = np.array(alpha_theory)

    # ---- Normalise both for comparison ----
    alpha_sim_norm = (alpha_sim_np - alpha_sim_np.min()) / (alpha_sim_np.max() - alpha_sim_np.min() + 1e-12)
    alpha_theory_norm = (alpha_theory - alpha_theory.min()) / (alpha_theory.max() - alpha_theory.min() + 1e-12)

    # ---- Agreement statistics ----
    abs_diff = np.abs(alpha_sim_norm - alpha_theory_norm)
    mean_abs_diff = np.mean(abs_diff)
    max_abs_diff = np.max(abs_diff)
    rms_diff = np.sqrt(np.mean(abs_diff**2))
    agreement_percent = (1 - mean_abs_diff) * 100

    print("\n===== Agreement statistics =====")
    print(f"Mean absolute difference  : {mean_abs_diff:.6e}")
    print(f"Max absolute difference   : {max_abs_diff:.6e}")
    print(f"RMS difference            : {rms_diff:.6e}")
    print(f"Percent agreement         : {agreement_percent:.4f}%")

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: raw alpha vs spacing
    ax1.plot(spacings_np, alpha_sim_np, 'o-', color='blue', label='Simulation')
    ax1.plot(spacings_np, alpha_theory, 's--', color='red', label='Analytical model')
    ax1.set_xlabel('Tip‑dot spacing (nm)', fontsize=12)
    ax1.set_ylabel(r'Lever arm $\alpha$', fontsize=12)
    ax1.set_title('Raw lever arm', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Right: normalised comparison
    ax2.plot(spacings_np, alpha_sim_norm, 'o-', color='blue', label='Simulation (norm)')
    ax2.plot(spacings_np, alpha_theory_norm, 's--', color='red', label='Analytical (norm)')
    ax2.set_xlabel('Tip‑dot spacing (nm)', fontsize=12)
    ax2.set_ylabel('Normalised value', fontsize=12)
    ax2.set_title('Normalised comparison', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Agreement stats box
    stats_text = (f"Mean abs diff = {mean_abs_diff:.2e}\n"
                  f"Max abs diff   = {max_abs_diff:.2e}\n"
                  f"Agreement = {agreement_percent:.2f}%")
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Parameters annotation (shared)
    params_text = (f"R = {R_nm:.1f} nm\n"
                   f"Dot diameter = {2*dot_radius_nm:.1f} nm\n"
                   f"Equivalent d/ε = {sum_d_over_eps*1e9:.2f} nm\n"
                   f"C_dot = {C_dot:.3e} F")
    fig.suptitle(params_text, fontsize=10, y=1.02)

    plt.tight_layout()
    plt.show(block=True)

    # Error stats for raw values (optional)
    rel_err = np.abs(alpha_sim_np - alpha_theory) / (np.maximum(alpha_theory, 1e-12))
    rms_raw = np.sqrt(np.mean((alpha_sim_np - alpha_theory)**2))
    print(f"\nRaw comparison:")
    print(f"RMS error: {rms_raw:.4e}")
    print(f"Mean relative error: {np.mean(rel_err)*100:.2f}%")
    print(f"Max relative error: {np.max(rel_err)*100:.2f}%")


def main():
    """
    Interactive entry point for the multi‑layer sanity check.
    """
    print("\n=== Sanity Check (Multi‑layer Dielectric) ===\n")

    output_dir = input("Enter folder containing .npy files (default: outputs): ").strip()
    if not output_dir:
        output_dir = "outputs"
    if not os.path.isdir(output_dir):
        print(f"Folder '{output_dir}' not found.")
        return

    # Auto-detect phi files
    phi_files = sorted(glob.glob(os.path.join(output_dir, 'afm_phi_*.npy')))
    if not phi_files:
        print("No afm_phi_*.npy files found.")
        return

    print("\nAvailable phi files:")
    for i, f in enumerate(phi_files):
        print(f"  {i+1}. {os.path.basename(f)}")
    choice = input("Select file number (or press Enter for all): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(phi_files):
            phi_files = [phi_files[idx]]
        else:
            print("Invalid number. Using all.")
    # else all files

    # Load nm config
    nm_config_path = input("Path to nm config JSON (default: afm_config_nm.json): ").strip()
    if not nm_config_path:
        nm_config_path = "afm_config_nm.json"
    if not os.path.isfile(nm_config_path):
        print(f"nm config not found: {nm_config_path}")
        return

    with open(nm_config_path) as f:
        nm_cfg = json.load(f)

    Lx = nm_cfg.get('Lx_nm', 512)
    Ly = nm_cfg.get('Ly_nm', 512)
    Lz = nm_cfg.get('Lz_nm', 512)
    R_nm = nm_cfg.get('R_nm', 7.0)

    # Find QD block
    idx, qd_block = find_qd_block(nm_cfg)
    if qd_block is None:
        print("No QD block found. Creating a default QD block for testing.")
        qd_block = create_default_qd_block(Lx, Ly, Lz, dot_diameter=25.0, dot_height=5.0, dot_bottom=46.0)
        print(f"Default QD: x={qd_block['x_range_nm']}, y={qd_block['y_range_nm']}, z={qd_block['z_range_nm']}")

    qd_top_z_nm = float(qd_block['z_range_nm'][1])
    print(f"\nQD region: x={qd_block['x_range_nm']}, y={qd_block['y_range_nm']}, z={qd_block['z_range_nm']}")
    print(f"QD top at z = {qd_top_z_nm} nm\n")

    config_dir = input("Directory containing afm_config_*.json files (default: .): ").strip()
    if not config_dir:
        config_dir = "."

    use_max = input("Use max (M) or average (A) for QD potential? (M/A, default A): ").strip().upper()
    use_max = (use_max == 'M')

    # Run the sanity check
    sanity_check_comparison_multi(phi_files, qd_block, qd_top_z_nm, Lx, Ly, Lz,
                                  R_nm, config_dir, nm_cfg, use_max=use_max)


if __name__ == "__main__":
    main()
