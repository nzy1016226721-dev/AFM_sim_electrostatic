import numpy as np
import json
import os
import csv
import re
import glob
import matplotlib.pyplot as plt


def parse_phi_filename(fname):
    """Extract metadata from a standard AFM phi filename.

    Supports 'afm_phi_{idx}_{V}V.npy' and 'afm_phi_zoom_{mag}x_{V}V_{idx}.npy'.

    Parameters
    ----------
    fname : str
        Filename to parse.

    Returns
    -------
    dict or None
        Dict with keys 'type', 'config_idx', 'Vtip', and optionally 'mag'.
    """
    m = re.match(r'afm_phi_zoom_(\d+)x_(-?[\d.]+)V_(\d+)\.npy', fname)
    if m:
        return {'type': 'zoom', 'mag': int(m.group(1)),
                'Vtip': float(m.group(2)), 'config_idx': int(m.group(3))}
    m = re.match(r'afm_phi_(\d+)_(-?[\d.]+)V\.npy', fname)
    if m:
        return {'type': 'normal', 'config_idx': int(m.group(1)),
                'Vtip': float(m.group(2))}
    return None


def find_qd_block(nm_cfg, default_eps=12.0):
    """Find the quantum dot block in the nm config by eps_val match.

    Parameters
    ----------
    nm_cfg : dict
        nm-scale config with 'blocks_nm'.
    default_eps : float, optional
        Expected epsilon value for the QD block (default: 12.0).

    Returns
    -------
    index : int or None
        Block index.
    block : dict or None
        Block dict.
    """
    blocks = nm_cfg.get('blocks_nm', [])
    for i, b in enumerate(blocks):
        if b.get('eps_val') == default_eps and all(
                k in b for k in ('x_range_nm', 'y_range_nm', 'z_range_nm')):
            return i, b
    for i, b in enumerate(blocks):
        if all(k in b for k in ('x_range_nm', 'y_range_nm', 'z_range_nm')):
            if b.get('eps_val', 1) > 1:
                return i, b
    return None, None


def extract_phi_qd(phi, qd_xr, qd_yr, qd_zr, Lx, Ly, Lz, zoom_bounds=None):
    """Extract the potential values inside a QD region.

    Handles both full-grid and zoom-grid coordinates via zoom_bounds.

    Parameters
    ----------
    phi : np.ndarray
        3D potential array.
    qd_xr, qd_yr, qd_zr : list/tuple of float
        QD region bounds in nm.
    Lx, Ly, Lz : float
        Box dimensions in nm.
    zoom_bounds : tuple or None, optional
        (x0f, x1f, y0f, y1f, z0f, z1f) fractional zoom bounds.

    Returns
    -------
    np.ndarray
        1D array of potential values inside the QD (empty if no overlap).
    """

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
    """Compute mean and max-absolute value of a 1D array.

    Parameters
    ----------
    phi_values : np.ndarray
        1D array of potential values.

    Returns
    -------
    avg : float
        Mean value.
    max_val : float
        Value with maximum absolute deviation from zero (signed).
    """
    if len(phi_values) == 0:
        return 0.0, 0.0
    avg = float(np.mean(phi_values))
    idx = int(np.argmax(np.abs(phi_values)))
    max_val = float(phi_values.flat[idx])
    return avg, max_val


def main():
    """QD lever arm calculation entry point.

    Parses command-line arguments, loads the nm config to locate the QD,
    iterates over phi .npy files, computes lever arm (alpha = Vdot/Vtip),
    saves results to CSV, and generates plots.

    Returns
    -------
    None
    """
    import argparse
    parser = argparse.ArgumentParser(description='QD lever arm calculator')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--nm_config', default='afm_config_nm.json')
    parser.add_argument('--config_dir', default='.',
                        help='Directory containing afm_config_*.json files')
    parser.add_argument('--csv', default='qd_lever_arm.csv')
    args = parser.parse_args()

    output_dir = args.output_dir
    config_dir = args.config_dir
    nm_config_path = args.nm_config

    if not os.path.isfile(nm_config_path):
        print(f"nm config not found: {nm_config_path}")
        return

    with open(nm_config_path) as f:
        nm_cfg = json.load(f)

    Lx = nm_cfg.get('Lx_nm', 512)
    Ly = nm_cfg.get('Ly_nm', 512)
    Lz = nm_cfg.get('Lz_nm', 512)

    idx, qd_block = find_qd_block(nm_cfg)
    if qd_block is None:
        print("No suitable QD block found. Defaulting to hardcoded values.")
        qd_block = {
            'x_range_nm': [128, 384],
            'y_range_nm': [128, 384],
            'z_range_nm': [250, 256]
        }
    else:
        print(f"\nFound QD candidate block (index {idx}):")
        print(f"  eps_val     = {qd_block.get('eps_val')}")
        print(f"  x_range_nm  = {qd_block.get('x_range_nm')}")
        print(f"  y_range_nm  = {qd_block.get('y_range_nm')}")
        print(f"  z_range_nm  = {qd_block.get('z_range_nm')}")
        ans = input("Use this block? (Y/n): ").strip().lower()
        if ans == 'n':
            blocks = nm_cfg.get('blocks_nm', [])
            print("\nAll blocks in nm config:")
            for i, b in enumerate(blocks):
                xr = b.get('x_range_nm', 'any')
                yr = b.get('y_range_nm', 'any')
                zr = b.get('z_range_nm', 'any')
                print(f"  [{i:2d}] eps={str(b.get('eps_val', '?')):>8s}  "
                      f"x={xr}  y={yr}  z={zr}")
            choice = input("Select block index: ").strip()
            if choice.isdigit() and int(choice) < len(blocks):
                qd_block = blocks[int(choice)]

    qd_xr = qd_block['x_range_nm']
    qd_yr = qd_block['y_range_nm']
    qd_zr = qd_block['z_range_nm']
    qd_top_z_nm = float(qd_zr[1])

    print(f"\nQD region: x={qd_xr} nm, y={qd_yr} nm, z={qd_zr} nm")
    print(f"QD top at z = {qd_top_z_nm} nm\n")

    phi_files = sorted(glob.glob(os.path.join(output_dir, 'afm_phi_*.npy')))
    if not phi_files:
        print(f"No phi files found in {output_dir}")
        return

    results = []

    for fpath in phi_files:
        fname = os.path.basename(fpath)
        info = parse_phi_filename(fname)
        if info is None:
            continue

        config_idx = info['config_idx']
        Vtip = info['Vtip']
        is_zoom = info['type'] == 'zoom'

        config_path = os.path.join(config_dir, f'afm_config_{config_idx}.json')
        if not os.path.isfile(config_path):
            print(f"  Config not found: {config_path}, skipping {fname}")
            continue

        with open(config_path) as f:
            cfg = json.load(f)

        tip_z_frac = cfg.get('tip_z', 0.5)
        tip_spacing_nm = (tip_z_frac - qd_top_z_nm / Lz) * Lz
        if tip_spacing_nm < 0:
            print(f"    WARNING: tip below QD top (spacing={tip_spacing_nm:.2f}nm)")

        zoom_bounds = None
        if is_zoom:
            cut = cfg.get('zoom_simulation', {}).get('cut', {})
            if cut and all(k in cut for k in ('x_range', 'y_range', 'z_range')):
                zoom_bounds = (cut['x_range'] + cut['y_range'] + cut['z_range'])

        phi = np.load(fpath)
        print(f"  {fname}: shape {phi.shape}, Vtip={Vtip}V, spacing={tip_spacing_nm:.2f}nm")

        phi_qd = extract_phi_qd(phi, qd_xr, qd_yr, qd_zr, Lx, Ly, Lz, zoom_bounds)
        if len(phi_qd) == 0:
            print(f"    QD region outside grid -- skipping")
            continue

        v_avg, v_max = compute_stats(phi_qd)
        alpha_max = v_max / Vtip if Vtip != 0 else 0.0
        alpha_avg = v_avg / Vtip if Vtip != 0 else 0.0

        results.append({
            'config_idx': config_idx,
            'Vtip': Vtip,
            'tip_spacing_nm': tip_spacing_nm,
            'Vdot_max': v_max,
            'Vdot_avg': v_avg,
            'alpha_max': alpha_max,
            'alpha_avg': alpha_avg,
            'is_zoom': is_zoom
        })
        print(f"    Vdot_max={v_max:.4e} V, Vdot_avg={v_avg:.4e} V, "
              f"alpha_max={alpha_max:.4e}")

    if not results:
        print("No results collected.")
        return

    csv_path = os.path.join(output_dir, args.csv)
    fieldnames = ['config_idx', 'Vtip', 'tip_spacing_nm',
                  'Vdot_max', 'Vdot_avg', 'alpha_max', 'alpha_avg', 'is_zoom']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {len(results)} rows to {csv_path}")

    plot_results(results, output_dir)


def plot_results(results, output_dir):
    """Plot lever arm (alpha) vs tip spacing, colour-coded by Vtip.

    Separates full-grid and zoom results with distinct markers.

    Parameters
    ----------
    results : list of dict
        List of result dicts with 'tip_spacing_nm', 'alpha_max',
        'alpha_avg', 'Vtip', 'is_zoom'.
    output_dir : str
        Directory for saving plots.

    Returns
    -------
    None
    """
    non_zoom = [r for r in results if not r['is_zoom']]
    zoom = [r for r in results if r['is_zoom']]
    cbar_label = '$V_{tip}$ (V)'

    for plot_var, xlabel, fname_out in [
        ('tip_spacing_nm', 'Tip-sample spacing (nm)', 'lever_arm_vs_spacing.png'),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))

        all_vtip = []
        if non_zoom:
            all_vtip += [r['Vtip'] for r in non_zoom]
        if zoom:
            all_vtip += [r['Vtip'] for r in zoom]
        vmin, vmax = (min(all_vtip), max(all_vtip)) if all_vtip else (-2, 2)
        if vmax == vmin:
            vmax = vmin + 1e-30

        if non_zoom:
            xs_max = [r[plot_var] for r in non_zoom]
            ys_max = [r['alpha_max'] for r in non_zoom]
            cs_max = [r['Vtip'] for r in non_zoom]
            s_max = ax.scatter(xs_max, ys_max, c=cs_max, cmap='plasma',
                               vmin=vmin, vmax=vmax, s=70, marker='o',
                               label='Max (full-grid)')

            xs_avg = [r[plot_var] for r in non_zoom]
            ys_avg = [r['alpha_avg'] for r in non_zoom]
            cs_avg = [r['Vtip'] for r in non_zoom]
            norm_c = plt.Normalize(vmin, vmax)
            avg_edge = [plt.cm.plasma(norm_c(c)) for c in cs_avg]
            s_avg = ax.scatter(xs_avg, ys_avg, s=70, marker='o',
                               facecolors='none', edgecolors=avg_edge,
                               linewidths=1.5, label='Avg (full-grid)')

        if zoom:
            xs_z = [r[plot_var] for r in zoom]
            ys_z = [r['alpha_max'] for r in zoom]
            cs_z = [r['Vtip'] for r in zoom]
            s_z = ax.scatter(xs_z, ys_z, c=cs_z, cmap='plasma',
                             vmin=vmin, vmax=vmax, s=90, marker='^',
                             label='Max (zoom)')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(r'Lever arm $\alpha = V_{\mathrm{dot}} / V_{\mathrm{tip}}$',
                      fontsize=12)
        ax.set_title(f'QD Lever Arm vs {xlabel}', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')

        mappable = ax.collections[0] if ax.collections else None
        if mappable is not None:
            cbar = fig.colorbar(mappable, ax=ax)
            cbar.set_label(cbar_label, fontsize=11)

        plt.tight_layout()
        out_path = os.path.join(output_dir, fname_out)
        fig.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
        plt.show()


if __name__ == '__main__':
    main()
