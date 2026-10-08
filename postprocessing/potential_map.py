import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
try:
    from ..simulation.coordinates import physical_domain_nm
except ImportError:
    from simulation.coordinates import physical_domain_nm
import os
import glob

from .npy_utils import parse_phi_filename


def plot_potential_map(phi, qd_block, tip_center_frac, R_nm,
                       Lx_nm, Ly_nm, Lz_nm,
                       z_slice='top', save_prefix=None):
    """Plot a 3-D electrostatic potential map or selected slice."""
    nx, ny, nz = phi.shape

    if z_slice == 'top' or z_slice == '':
        z_frac = qd_block['z_range_nm'][1] / Lz_nm
    elif z_slice == 'middle':
        z_frac = (qd_block['z_range_nm'][0] + qd_block['z_range_nm'][1]) / (2.0 * Lz_nm)
    else:
        z_frac = float(z_slice) / Lz_nm
    iz = int(z_frac * (nz - 1))
    iz = max(0, min(nz-1, iz))

    phi_slice = phi[:, :, iz]

    tip_x_nm = tip_center_frac[0] * Lx_nm
    tip_y_nm = tip_center_frac[1] * Ly_nm

    half = 2.0 * R_nm
    x_min_nm = tip_x_nm - half
    x_max_nm = tip_x_nm + half
    y_min_nm = tip_y_nm - half
    y_max_nm = tip_y_nm + half

    x_edges = np.linspace(0, Lx_nm, nx)
    y_edges = np.linspace(0, Ly_nm, ny)
    ix0 = max(0, int(np.ceil((x_min_nm) / (Lx_nm/(nx-1)))))
    ix1 = min(nx-1, int(np.floor((x_max_nm) / (Lx_nm/(nx-1)))))
    iy0 = max(0, int(np.ceil((y_min_nm) / (Ly_nm/(ny-1)))))
    iy1 = min(ny-1, int(np.floor((y_max_nm) / (Ly_nm/(ny-1)))))

    phi_crop = phi_slice[ix0:ix1+1, iy0:iy1+1]
    if phi_crop.size == 0:
        print("No data in the cropped region. Check tip center and R_nm.")
        return

    x_crop = np.linspace(x_min_nm, x_max_nm, phi_crop.shape[0])
    y_crop = np.linspace(y_min_nm, y_max_nm, phi_crop.shape[1])
    X, Y = np.meshgrid(x_crop, y_crop, indexing='ij')

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, phi_crop, cmap='RdBu_r', edgecolor='none')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    ax1.set_zlabel(r'$\phi$ (V)')
    ax1.set_title('Potential map (z = %.1f nm)' % (z_frac * Lz_nm))
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label=r'$\phi$ (V)')

    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X, Y, phi_crop, levels=50, cmap='RdBu_r')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    ax2.set_title('Potential (contour)')
    fig.colorbar(contour, ax=ax2, label=r'$\phi$ (V)')
    plt.tight_layout()

    if save_prefix:
        out_name = "%s_potential_map.png" % save_prefix
        plt.savefig(out_name, dpi=150)
        print("Saved map to %s" % out_name)

    plt.show()
    plt.close(fig)

    return {'x_nm': X, 'y_nm': Y, 'phi': phi_crop}


def _auto_detect(output_dir="outputs"):
    phi_files = sorted(glob.glob(os.path.join(output_dir, 'afm_phi_*.npy')))
    if not phi_files:
        return []
    results = []
    for f in phi_files:
        fname = os.path.basename(f)
        info = parse_phi_filename(fname)
        if info is not None:
            results.append({'path': f, 'name': fname, 'info': info})
    return results


def interactive_main():
    """Run the interactive potential-map plotting interface."""
    print("\n=== Potential Map Generator ===\n")

    detected = _auto_detect("outputs")
    if not detected:
        output_dir = input("Enter folder containing .npy files (default: .): ").strip()
        if not output_dir:
            output_dir = "."
        detected = _auto_detect(output_dir)
        if not detected:
            print("No phi files found. Exiting.")
            return
    else:
        print("Detected %d phi files in 'outputs/':" % len(detected))
        for i, d in enumerate(detected):
            print("  %d. %s  (Vtip=%.2f, cfg=%d)" % (
                i+1, d['name'], d['info']['Vtip'], d['info']['config_idx']))
        ans = input("Use these files? (y/n, default y): ").strip().lower()
        if ans == 'n':
            output_dir = input("Enter folder containing .npy files: ").strip()
            if not output_dir:
                return
            detected = _auto_detect(output_dir)
            if not detected:
                print("No phi files found. Exiting.")
                return
        else:
            output_dir = "outputs"

    print("\nAvailable phi files:")
    for i, d in enumerate(detected):
        print("  %d. %s" % (i+1, d['name']))
    print("\nPress Enter to process ALL files, or enter a number to select one.")
    choice = input("Select: ").strip()
    if choice == "":
        files_to_process = detected
    elif choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(detected):
            files_to_process = [detected[idx]]
        else:
            print("Invalid number.")
            return
    else:
        print("Invalid input.")
        return

    nm_config = "afm_config_nm.json"
    if not os.path.isfile(nm_config):
        nm_config = input("Path to nm config JSON: ").strip()
        if not nm_config or not os.path.isfile(nm_config):
            print("Config not found. Exiting.")
            return
    else:
        print("Using nm config: %s" % nm_config)

    with open(nm_config) as f:
        nm_cfg = json.load(f)

    Lx, Ly, Lz = physical_domain_nm(nm_cfg)
    R_nm = nm_cfg.get('R_nm', 7.0)
    movement = nm_cfg.get('movement', {})
    if "start_nm" in movement:
        origin = nm_cfg.get("coordinate_system", {}).get("origin_fraction", [0.5, 0.5, 0.0])
        tip_center_frac = [
            origin[0] + float(movement["start_nm"][0]) / Lx,
            origin[1] + float(movement["start_nm"][1]) / Ly,
        ]
    else:
        tip_center_frac = movement.get('start', [0.5, 0.5, 0.5])[:2]

    blocks = nm_cfg.get('blocks_nm', [])
    if blocks:
        print("\nAvailable dielectric blocks (for z-slice):")
        for i, b in enumerate(blocks):
            zr = b.get('z_range_nm', 'unknown')
            print("  [%d] eps=%s, z=%s" % (i, b.get('eps_val', '?'), zr))
        block_idx_str = input("Select block index to define z-slice (default 0): ").strip()
        if block_idx_str.isdigit():
            block_idx = int(block_idx_str)
            if 0 <= block_idx < len(blocks):
                qd_block = blocks[block_idx]
            else:
                qd_block = blocks[0]
        else:
            qd_block = blocks[0]
    else:
        qd_block = {'z_range_nm': [0, Lz]}

    z_slice = input("z-slice (top/middle, default top): ").strip().lower()
    if z_slice not in ('top', 'middle'):
        z_slice = 'top'

    maps_dir = os.path.join(output_dir, 'potential_maps')
    os.makedirs(maps_dir, exist_ok=True)

    print("\nAbout to process %d file(s):" % len(files_to_process))
    print("  nm config: %s" % nm_config)
    print("  Lx=%.1f, Ly=%.1f, Lz=%.1f nm" % (Lx, Ly, Lz))
    print("  R_nm=%.2f, tip center=(%.3f, %.3f)" % (R_nm, tip_center_frac[0], tip_center_frac[1]))
    print("  QD block z-range: %s" % str(qd_block.get('z_range_nm', '?')))
    print("  z-slice: %s" % z_slice)
    print("  Output: %s/" % maps_dir)
    ans = input("\nProceed? (y/n, default y): ").strip().lower()
    if ans == 'n':
        print("Cancelled.")
        return

    for entry in files_to_process:
        fname = entry['name']
        info = entry['info']
        Vtip = info['Vtip']
        config_idx = info['config_idx']

        phi = np.load(entry['path'])
        print("\nProcessing %s (Vtip = %.2f V, config %d)" % (fname, Vtip, config_idx))

        save_prefix = os.path.join(maps_dir, "config%d_V%.2f" % (config_idx, Vtip))
        plot_potential_map(phi, qd_block, tip_center_frac, R_nm, Lx, Ly, Lz,
                           z_slice=z_slice, save_prefix=save_prefix)

    print("\nAll maps saved to %s" % maps_dir)


if __name__ == "__main__":
    interactive_main()
