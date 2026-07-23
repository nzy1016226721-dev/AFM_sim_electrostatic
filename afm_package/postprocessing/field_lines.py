import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob

from simulation.solver import build_downward_pointing_tip
from .npy_utils import parse_phi_filename


def _find_interface_z_fracs(blocks):
    if not blocks:
        return []
    layers = [(b["z_range"][0], b["z_range"][1], b["eps_val"]) for b in blocks
              if list(b.get("x_range", [0, 1])) == [0, 1]
              and list(b.get("y_range", [0, 1])) == [0, 1]]
    if not layers:
        return []
    layers.sort(key=lambda t: t[0])
    interfaces = set()
    prev_eps = None
    for z0, z1, eps in layers:
        if prev_eps is not None and abs(eps - prev_eps) > 1e-6:
            interfaces.add(round(z0, 10))
        prev_eps = eps
    return sorted(interfaces)


def plot_field_lines(phi, plane='xz', coord=0.5,
                     Lx_nm=100, Ly_nm=100, Lz_nm=100,
                     crop_radius_nm=None, title="", save_path=None,
                     tip_params=None, field_sign=1.0,
                     tip_buffer_cells=2, blocks=None):
    from scipy.ndimage import binary_dilation

    nx, ny, nz = phi.shape

    if tip_params is None:
        tip_params = {'tip_z': 0.55, 'R': 0.07, 'r_tip': 0.3, 'aspect_ratio': 4}
    tip_mask, _, _ = build_downward_pointing_tip(
        nx, ny, nz,
        tip_z=tip_params['tip_z'],
        R=tip_params['R'],
        r_tip=tip_params['r_tip'],
        aspect_ratio=tip_params['aspect_ratio'],
        verbose=False
    )
    print(f"Tip mask voxels: {np.sum(tip_mask)}")

    tip_mask_buffered = binary_dilation(tip_mask, iterations=tip_buffer_cells)

    dx = Lx_nm * 1e-9 / (nx - 1)
    dy = Ly_nm * 1e-9 / (ny - 1)
    dz = Lz_nm * 1e-9 / (nz - 1)
    Ex, Ey, Ez = np.gradient(-phi * field_sign, dx, dy, dz)

    Ex[tip_mask] = 0.0
    Ey[tip_mask] = 0.0
    Ez[tip_mask] = 0.0

    if plane != 'xz':
        print("Warning: only 'xz' plane is fully supported. Forcing 'xz'.")
        plane = 'xz'
    iy = int(coord * (ny - 1))
    iy = max(0, min(ny - 1, iy))
    phi_slice = phi[:, iy, :]
    tip_slice = tip_mask[:, iy, :]
    tip_slice_buffered = tip_mask_buffered[:, iy, :]

    x_edges = np.linspace(0, Lx_nm, nx)
    y_edges = np.linspace(0, Lz_nm, nz)
    U = Ex[:, iy, :]
    V = Ez[:, iy, :]

    interface_z_fracs = _find_interface_z_fracs(blocks)
    interface_z_nm = [f * Lz_nm for f in interface_z_fracs]
    interface_buffer_nm = 2.5 * (Lz_nm / (nz - 1))
    if interface_z_nm:
        print(f"  Dielectric interfaces at z = "
              f"{[round(v,2) for v in interface_z_nm]} nm "
              f"(excluding +/-{interface_buffer_nm:.2f} nm band)")
    interface_row_mask_1d = np.zeros(nz, dtype=bool)
    for iz_nm in interface_z_nm:
        interface_row_mask_1d |= (np.abs(y_edges - iz_nm) < interface_buffer_nm)
    interface_row_mask = np.tile(interface_row_mask_1d[np.newaxis, :], (nx, 1))
    xlabel, ylabel = 'x (nm)', 'z (nm)'
    title_prefix = f'XZ plane at y={coord*Ly_nm:.1f} nm'
    if crop_radius_nm is not None:
        cx, cz = Lx_nm/2, Lz_nm/2
        xlim = (cx - crop_radius_nm, cx + crop_radius_nm)
        ylim = (cz - crop_radius_nm, cz + crop_radius_nm)
    else:
        xlim = (0, Lx_nm)
        ylim = (0, Lz_nm)

    phi_masked = np.ma.masked_where(tip_slice, phi_slice)

    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color='black')
    im = ax.imshow(phi_masked.T, origin='lower', cmap=cmap,
                   extent=extent, aspect='auto')
    fig.colorbar(im, ax=ax, label='Potential (V)')

    phi_contour = phi_slice.copy()
    phi_contour[tip_slice] = np.nan
    levels = np.linspace(phi_slice.min(), phi_slice.max(), 15)
    ax.contour(x_edges, y_edges, phi_contour.T, levels=levels,
               colors='k', linewidths=0.8, alpha=0.5)

    if np.sum(tip_slice) > 0:
        ax.contour(x_edges, y_edges, tip_slice.T, levels=[0.5],
                   colors='red', linewidths=2, linestyles='dashed', alpha=1.0)

    plot_exclude_mask = tip_slice_buffered | interface_row_mask
    U_masked = np.ma.masked_array(U, mask=plot_exclude_mask)
    V_masked = np.ma.masked_array(V, mask=plot_exclude_mask)

    E_mag = np.sqrt(U**2 + V**2)
    valid = ~plot_exclude_mask
    if np.any(valid):
        e_ref = np.percentile(E_mag[valid], 97)
    else:
        e_ref = 1.0
    e_ref = max(e_ref, 1e-30)
    lw = 0.4 + 3.0 * np.clip(E_mag / e_ref, 0.0, 1.0)

    ax.streamplot(x_edges, y_edges, U_masked.T, V_masked.T,
                  color='blue', linewidth=lw.T, density=2.2,
                  arrowstyle='->', arrowsize=1.2,
                  integration_direction='both')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title_prefix}\n{title}' if title else title_prefix)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show(block=True)
    plt.close(fig)


def _auto_detect(output_dir="outputs"):
    phi_files = sorted(glob.glob(os.path.join(output_dir, 'afm_phi_*.npy')))
    if not phi_files:
        print("  No afm_phi_*.npy files found in '%s'." % output_dir)
        return []

    cwd = os.getcwd()
    results = []
    for f in phi_files:
        fname = os.path.basename(f)
        info = parse_phi_filename(fname)
        if info is None:
            continue
        cfg_name = 'afm_config_%d.json' % info['config_idx']
        cfg_path = os.path.join(os.path.dirname(f) or '.', cfg_name)
        if not os.path.isfile(cfg_path):
            cfg_path = os.path.join(cwd, cfg_name)
        results.append({
            'path': f,
            'name': fname,
            'info': info,
            'config_path': cfg_path if os.path.isfile(cfg_path) else None
        })
    return results


def interactive_main():
    print("\n=== Electric Field Lines Plotter ===\n")

    detected = _auto_detect("outputs")
    if not detected:
        folder = input("Enter folder containing .npy files (default: .): ").strip()
        if not folder:
            folder = "."
        detected = _auto_detect(folder)
        if not detected:
            print("No parseable phi files found. Exiting.")
            return
    else:
        print("Detected %d phi files in 'outputs/':" % len(detected))
        for i, d in enumerate(detected):
            cfg_mark = "  [config OK]" if d['config_path'] else "  [NO CONFIG]"
            print("  %d. %s%s" % (i+1, d['name'], cfg_mark))
        ans = input("Use these files? (y/n, default y): ").strip().lower()
        if ans == 'n':
            folder = input("Enter folder containing .npy files: ").strip()
            if not folder:
                return
            detected = _auto_detect(folder)
            if not detected:
                print("No parseable phi files found. Exiting.")
                return

    print("\nAvailable files:")
    for i, d in enumerate(detected):
        print("  %d. %s" % (i+1, d['name']))
    choice = input("Select file number (or press Enter for first): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(detected):
            entry = detected[idx]
        else:
            entry = detected[0]
    else:
        entry = detected[0]

    fname = entry['name']
    info = entry['info']
    print("Config index: %d, Vtip = %.2f V" % (info['config_idx'], info['Vtip']))

    blocks = None
    if entry['config_path']:
        with open(entry['config_path']) as f:
            cfg = json.load(f)
        tip_params = {
            'tip_z': cfg.get('tip_z', 0.55),
            'R': cfg.get('R', 0.07),
            'r_tip': cfg.get('r_tip', 0.3),
            'aspect_ratio': cfg.get('aspect_ratio', 4)
        }
        Lx_nm = cfg.get('Lx_nm', 100)
        Ly_nm = cfg.get('Ly_nm', 100)
        Lz_nm = cfg.get('Lz_nm', 100)
        blocks = cfg.get('blocks', None)
        print("Using config: tip_z=%.6f, R=%.8f, r_tip=%.8f" % (
            tip_params['tip_z'], tip_params['R'], tip_params['r_tip']))
    else:
        print("WARNING: no matching config found. Using generic defaults "
              "(Lx=Ly=Lz=100nm, tip_z=0.55, R=0.07, r_tip=0.3).")
        tip_params = None
        Lx_nm = Ly_nm = Lz_nm = 100

    phi = np.load(entry['path'])
    print("Phi shape: %s, range [%.3f, %.3f]" % (
        str(phi.shape), phi.min(), phi.max()))

    plane = input("Plane (xy, xz, yz, default xz): ").strip().lower()
    if plane not in ('xy', 'xz', 'yz'):
        plane = 'xz'
    coord_str = input("Coordinate fraction for fixed axis (0-1, default 0.5): ").strip()
    coord = float(coord_str) if coord_str else 0.5

    crop = input("Crop radius (nm) around centre (press Enter for full view): ").strip()
    crop_radius_nm = float(crop) if crop else None

    flip = input("Flip field direction? (y/n, default n): ").strip().lower()
    field_sign = -1.0 if flip == 'y' else 1.0

    save = input("Save plot to file? (press Enter to skip): ").strip()
    save_path = save if save else None

    print("\nSettings summary:")
    print("  File: %s" % fname)
    print("  Plane: %s at coord %.3f" % (plane, coord))
    print("  Crop: %s" % ("full view" if crop_radius_nm is None else "%.1f nm" % crop_radius_nm))
    print("  Field sign: %s" % ("normal" if field_sign == 1.0 else "flipped"))
    print("  Config: %s" % ("found" if entry['config_path'] else "NONE (defaults)"))
    ans = input("\nProceed with plotting? (y/n, default y): ").strip().lower()
    if ans == 'n':
        print("Cancelled.")
        return

    plot_field_lines(phi, plane=plane, coord=coord,
                     Lx_nm=Lx_nm, Ly_nm=Ly_nm, Lz_nm=Lz_nm,
                     crop_radius_nm=crop_radius_nm,
                     title=fname,
                     save_path=save_path,
                     tip_params=tip_params,
                     field_sign=field_sign,
                     blocks=blocks)


if __name__ == "__main__":
    interactive_main()
