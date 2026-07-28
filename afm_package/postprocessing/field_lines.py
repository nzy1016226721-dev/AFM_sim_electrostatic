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
                      tip_buffer_cells=2, blocks=None,
                      streamplot_density=1.6,
                      arrow_spacing_nm=None,
                      max_arrow_len_nm=6.0,
                      min_arrow_len_nm=0,
                      mag_percentile=97):
    from scipy.ndimage import binary_dilation
    from scipy.interpolate import RegularGridInterpolator
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as pe

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
    E_mag_full = np.sqrt(Ex**2 + Ey**2 + Ez**2)

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
    E_mag_slice = E_mag_full[:, iy, :]

    # --- Exclusion mask ---
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

    exclude_mask = tip_slice_buffered | interface_row_mask

    U_masked = np.ma.masked_array(U, mask=exclude_mask)
    V_masked = np.ma.masked_array(V, mask=exclude_mask)

    # ---- Generate streamlines (throwaway) ----
    fig_tmp, ax_tmp = plt.subplots()
    res = ax_tmp.streamplot(x_edges, y_edges, U_masked.T, V_masked.T,
                             density=streamplot_density,
                             integration_direction='both')
    segments = res.lines.get_segments()
    plt.close(fig_tmp)

    # ---- Interpolator for |E| ----
    mag_interp = RegularGridInterpolator((x_edges, y_edges), E_mag_slice,
                                          bounds_error=False, fill_value=0.0)

    # ---- Place quiver anchors along streamlines ----
    domain_diag = np.hypot(Lx_nm, Lz_nm)
    target_spacing_nm = arrow_spacing_nm if arrow_spacing_nm is not None else domain_diag / 40

    anchors_x, anchors_y, dirs_u, dirs_v, mags = [], [], [], [], []

    for seg in segments:
        seg = np.asarray(seg)
        if seg.ndim != 2 or seg.shape[1] != 2 or seg.shape[0] < 2:
            continue

        diffs = np.diff(seg, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = cumlen[-1]
        if total_len < 1e-9:
            continue

        n_samples = max(1, int(total_len // target_spacing_nm))
        sample_positions = np.linspace(0, total_len, n_samples, endpoint=False) + target_spacing_nm / 2

        for s in sample_positions:
            idx = np.clip(np.searchsorted(cumlen, s) - 1, 0, len(seg) - 2)
            p0, p1 = seg[idx], seg[idx + 1]
            dxs, dys = (p1[0] - p0[0]), (p1[1] - p0[1])
            local_len = np.hypot(dxs, dys)
            if local_len < 1e-12:
                continue
            mx, my = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
            anchors_x.append(mx); anchors_y.append(my)
            dirs_u.append(dxs / local_len); dirs_v.append(dys / local_len)
            mags.append(mag_interp((mx, my)))

    anchors_x = np.array(anchors_x); anchors_y = np.array(anchors_y)
    dirs_u = np.array(dirs_u); dirs_v = np.array(dirs_v)
    mags = np.array(mags)

    print(f"  Streamlines: {len(segments)}, quiver anchors placed: {len(anchors_x)}")

    # ---- Scale arrow length ----
    mag_ref = 0.0
    if len(mags) > 0 and mags.max() > 0:
        mag_ref = np.percentile(mags[mags > 0], mag_percentile) if np.any(mags > 0) else mags.max()
        mag_ref = max(mag_ref, 1e-12)
        scaled_len = np.clip(mags / mag_ref, 0, 1) * max_arrow_len_nm
        scaled_len = np.maximum(scaled_len, min_arrow_len_nm)
        U_arrows = dirs_u * scaled_len
        V_arrows = dirs_v * scaled_len
    else:
        U_arrows = dirs_u
        V_arrows = dirs_v

    # ============ Main figure ============
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

    # ---- Equipotential contours ----
    phi_contour = phi_slice.copy()
    phi_contour[tip_slice] = np.nan
    levels = np.linspace(phi_slice.min(), phi_slice.max(), 15)
    ax.contour(x_edges, y_edges, phi_contour.T, levels=levels,
               colors='k', linewidths=0.8, alpha=0.5)

    # ---- Tip outline ----
    if np.sum(tip_slice) > 0:
        ax.contour(x_edges, y_edges, tip_slice.T, levels=[0.5],
                   colors='red', linewidths=2, linestyles='dashed', alpha=1.0)

    # ---- Quiver arrows ----
    if len(anchors_x) > 0:
        q = ax.quiver(anchors_x, anchors_y, U_arrows, V_arrows,
                      angles='xy', scale_units='xy', scale=1.0,
                      width=0.003, color='black', alpha=0.9, pivot='middle',
                      zorder=4)
        q.set_path_effects([pe.Stroke(linewidth=1.5, foreground='white'), pe.Normal()])

    # ---- Legend: properly sized box ----
    if len(mags) > 0 and mags.max() > 0:
        # Axes coordinates for bottom‑left
        ax_x = 0.02
        ax_y = 0.02

        # Format field magnitude
        if mag_ref >= 1e6:
            mag_str = f"{mag_ref/1e6:.2f}×10⁶"
        elif mag_ref >= 1e3:
            mag_str = f"{mag_ref/1e3:.2f}×10³"
        else:
            mag_str = f"{mag_ref:.2f}"

        # Box with enough width to contain full text
        box_width = 0.45
        box_height = 0.05
        box = FancyBboxPatch(
            (ax_x, ax_y),
            box_width, box_height,
            boxstyle="round,pad=0.01",
            linewidth=0.5,
            edgecolor='gray',
            facecolor='white',
            alpha=0.9,
            zorder=5,
            transform=ax.transAxes
        )
        ax.add_patch(box)

        # Draw arrow
        arrow_start_x = ax_x + 0.02
        arrow_start_y = ax_y + box_height/2
        arrow_len_ax = 0.045
        ax.arrow(arrow_start_x, arrow_start_y, arrow_len_ax, 0,
                 head_width=0.008, head_length=0.008,
                 fc='black', ec='black', linewidth=0.8,
                 zorder=6, transform=ax.transAxes)

        # Text label: full information
        label_text = f"|E| = {mag_str} V/m"
        ax.text(arrow_start_x + arrow_len_ax + 0.01, arrow_start_y,
                label_text,
                ha='left', va='center', fontsize=10, color='black',
                transform=ax.transAxes, zorder=6)

    # ---- Crop and labels ----
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

    # Auto-detect files in 'outputs'
    detected = _auto_detect("outputs")
    if detected:
        folder = "outputs"
    else:
        folder = input("Enter folder containing .npy files (default: .): ").strip()
        if not folder:
            folder = "."
        detected = _auto_detect(folder)
        if not detected:
            print("No parseable phi files found. Exiting.")
            return

    print(f"Detected {len(detected)} phi files in '{folder}':")
    for i, d in enumerate(detected):
        cfg_mark = "  [config OK]" if d['config_path'] else "  [NO CONFIG]"
        print(f"  {i+1}. {d['name']}{cfg_mark}")

    # ---- Ask for plotting settings once ----
    plane = input("Plane (xy, xz, yz, default xz): ").strip().lower()
    if plane not in ('xy', 'xz', 'yz'):
        plane = 'xz'
    coord_str = input("Coordinate fraction for fixed axis (0-1, default 0.5): ").strip()
    coord = float(coord_str) if coord_str else 0.5
    crop = input("Crop radius (nm) around centre (press Enter for full view): ").strip()
    crop_radius_nm = float(crop) if crop else None

    # ---- Process all files ----
    print(f"\nProcessing {len(detected)} files...")
    for entry in detected:
        fname = entry['name']
        info = entry['info']
        print(f"  {fname} (config {info['config_idx']}, Vtip = {info['Vtip']:.2f} V)")

        # Load config if available
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
            print(f"    tip_z={tip_params['tip_z']:.6f}, R={tip_params['R']:.8f}, r_tip={tip_params['r_tip']:.8f}")
        else:
            print("    WARNING: no matching config found. Using defaults.")
            tip_params = None
            Lx_nm = Ly_nm = Lz_nm = 100

        phi = np.load(entry['path'])
        print(f"    Phi shape: {phi.shape}, range [{phi.min():.3f}, {phi.max():.3f}]")

        # Generate plot (no extra prompts)
        plot_field_lines(phi, plane=plane, coord=coord,
                         Lx_nm=Lx_nm, Ly_nm=Ly_nm, Lz_nm=Lz_nm,
                         crop_radius_nm=crop_radius_nm,
                         title=fname,
                         save_path=None,
                         tip_params=tip_params,
                         field_sign=1.0,
                         blocks=blocks)

    print("\nAll files processed.")


if __name__ == "__main__":
    interactive_main()