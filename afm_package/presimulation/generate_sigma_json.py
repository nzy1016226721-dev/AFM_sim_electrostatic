import json
import os
import numpy as np
import matplotlib.pyplot as plt


def confirm_path(description, current_path, must_exist=True):
    """Prompt user to confirm or correct a file path.

    Parameters
    ----------
    description : str
        Label describing the file.
    current_path : str
        Proposed file path.
    must_exist : bool, optional
        If True, only accept existing paths (default: True).

    Returns
    -------
    str or None
        Confirmed path, or None if skipped.
    """
    if not current_path:
        print(f"{description}: (none)")
        return None
    while True:
        print(f"{description}: {current_path}")
        ans = input("Is this correct? (y/n): ").strip().lower()
        if ans == 'y':
            if must_exist and not os.path.isfile(current_path):
                print(f"  File not found: {current_path}")
                continue
            return current_path
        elif ans == 'n':
            new_path = input("  Enter correct path (or press Enter to skip): ").strip()
            if new_path == "":
                return None
            current_path = new_path
        else:
            print("  Please answer 'y' or 'n'.")


def get_float(prompt, positive=True):
    """Prompt user to enter a floating-point number.

    Parameters
    ----------
    prompt : str
        Input prompt text.
    positive : bool, optional
        If True, only accept values > 0 (default: True).

    Returns
    -------
    float
        The validated number.
    """
    while True:
        try:
            val = float(input(prompt).strip())
            if positive and val <= 0:
                print("  Value must be > 0.")
                continue
            return val
        except ValueError:
            print("  Please enter a valid number.")


def get_float_pair(prompt, low_lim=0.0, high_lim=None):
    """Prompt user to enter two numbers (a range pair).

    Parameters
    ----------
    prompt : str
        Input prompt text.
    low_lim : float, optional
        Minimum allowed value (default: 0.0).
    high_lim : float or None, optional
        Maximum allowed value (default: None).

    Returns
    -------
    list of float
        Two-element list [low, high], sorted.
    """
    while True:
        s = input(prompt).strip()
        try:
            parts = s.replace(",", " ").split()
            if len(parts) != 2:
                raise ValueError
            x0, x1 = float(parts[0]), float(parts[1])
            if x0 > x1:
                x0, x1 = x1, x0
            if x0 < low_lim:
                print(f"  Lower bound must be >= {low_lim}. Clipping to {low_lim}.")
                x0 = low_lim
            if high_lim is not None and x1 > high_lim:
                print(f"  Upper bound exceeds box size ({high_lim}). Clipping to {high_lim}.")
                x1 = high_lim
            return [x0, x1]
        except ValueError:
            print("  Enter two numbers separated by space or comma, e.g. '0 100'.")


def normalize_blocks(blocks, Lx, Ly, Lz):
    """Convert block coordinates to fractional, handling mixed nm/frac inputs.

    Parameters
    ----------
    blocks : list of dict
        Blocks with 'sigma_val' and optional '_nm' or fractional ranges.
    Lx : float
        Box length in x (nm).
    Ly : float
        Box length in y (nm).
    Lz : float
        Box length in z (nm).

    Returns
    -------
    list of dict
        Normalized blocks with fractional 'x_range', 'y_range', 'z_range'.
    """
    normalized = []
    for blk in blocks:
        sigma = blk["sigma_val"]
        nb = {"sigma_val": sigma}
        if "x_range_nm" in blk:
            nb["x_range"] = [blk["x_range_nm"][0]/Lx, blk["x_range_nm"][1]/Lx]
        elif "x_range" in blk:
            nb["x_range"] = blk["x_range"]
        else:
            nb["x_range"] = [0.0, 1.0]
        if "y_range_nm" in blk:
            nb["y_range"] = [blk["y_range_nm"][0]/Ly, blk["y_range_nm"][1]/Ly]
        elif "y_range" in blk:
            nb["y_range"] = blk["y_range"]
        else:
            nb["y_range"] = [0.0, 1.0]
        if "z_range_nm" in blk:
            nb["z_range"] = [blk["z_range_nm"][0]/Lz, blk["z_range_nm"][1]/Lz]
        elif "z_range" in blk:
            nb["z_range"] = blk["z_range"]
        else:
            nb["z_range"] = [0.0, 1.0]
        normalized.append(nb)
    return normalized


def preview_conductivity(blocks_frac, Lx_nm, Ly_nm, Lz_nm):
    """Interactive 2D/1D preview of conductivity block distribution.

    Parameters
    ----------
    blocks_frac : list of dict
        Fractional-coordinate conductivity blocks.
    Lx_nm : float
        Box length in x (nm).
    Ly_nm : float
        Box length in y (nm).
    Lz_nm : float
        Box length in z (nm).

    Returns
    -------
    None
    """
    if not blocks_frac:
        print("No blocks to preview.")
        return

    full_blocks = [b for b in blocks_frac
                   if np.allclose(b["x_range"], [0,1], atol=1e-9)
                   and np.allclose(b["y_range"], [0,1], atol=1e-9)]
    pillar_blocks = [b for b in blocks_frac
                     if not (np.allclose(b["x_range"], [0,1], atol=1e-9)
                             and np.allclose(b["y_range"], [0,1], atol=1e-9))]

    nz_samples = 20000
    z_frac = np.linspace(0.0, 1.0, nz_samples)
    z_nm = z_frac * Lz_nm

    sigma_bg = np.ones(nz_samples, dtype=np.float64) * 1e-12
    for blk in full_blocks:
        z0, z1 = blk["z_range"]
        mask = (z_frac >= z0) & (z_frac <= z1)
        sigma_bg[mask] = blk["sigma_val"]

    sigma_pillar = np.copy(sigma_bg) if pillar_blocks else None
    if sigma_pillar is not None:
        for blk in blocks_frac:
            z0, z1 = blk["z_range"]
            mask = (z_frac >= z0) & (z_frac <= z1)
            sigma_pillar[mask] = blk["sigma_val"]

    pillar_rect = None
    for blk in pillar_blocks:
        xr = blk["x_range"]
        yr = blk["y_range"]
        if pillar_rect is None:
            pillar_rect = (xr[0], xr[1], yr[0], yr[1])

    def inside_pillar(x, y):
        if pillar_rect is None:
            return 0.0
        x0, x1, y0, y1 = pillar_rect
        return 1.0 if (x0 <= x <= x1 and y0 <= y <= y1) else 0.0

    x0, y0, z0 = 0.5, 0.5, 0.5
    zoom = 10

    while True:
        half_x_nm = Lx_nm / (2.0 * zoom)
        half_y_nm = Ly_nm / (2.0 * zoom)
        half_z_nm = Lz_nm / (2.0 * zoom)

        xlim = (x0 * Lx_nm - half_x_nm, x0 * Lx_nm + half_x_nm)
        ylim_xy = (y0 * Ly_nm - half_y_nm, y0 * Ly_nm + half_y_nm)
        zlim_xz = (z0 * Lz_nm - half_z_nm, z0 * Lz_nm + half_z_nm)

        grid_size = 200
        xs_2d = np.linspace(0.0, 1.0, grid_size)
        ys_2d = np.linspace(0.0, 1.0, grid_size)
        zs_2d = np.linspace(0.0, 1.0, grid_size)

        W_xy = np.array([[inside_pillar(x, y) for y in ys_2d] for x in xs_2d])
        iz = np.searchsorted(z_frac, z0); iz = min(iz, nz_samples-1)
        bg_val = sigma_bg[iz]
        pil_val = sigma_pillar[iz] if sigma_pillar is not None else bg_val
        sigma_xy = (1 - W_xy) * bg_val + W_xy * pil_val

        W_xz = np.array([inside_pillar(x, y0) for x in xs_2d])[:, np.newaxis]
        bg_1d = sigma_bg[np.searchsorted(z_frac, zs_2d).clip(0, nz_samples-1)]
        pil_1d = sigma_pillar[np.searchsorted(z_frac, zs_2d).clip(0, nz_samples-1)] if sigma_pillar is not None else bg_1d
        sigma_xz = (1 - W_xz) * bg_1d[np.newaxis, :] + W_xz * pil_1d[np.newaxis, :]

        W_yz = np.array([inside_pillar(x0, y) for y in ys_2d])[:, np.newaxis]
        sigma_yz = (1 - W_yz) * bg_1d[np.newaxis, :] + W_yz * pil_1d[np.newaxis, :]

        w = inside_pillar(x0, y0)
        if sigma_pillar is not None:
            sigma_z_line = (1 - w) * sigma_bg + w * sigma_pillar
        else:
            sigma_z_line = sigma_bg

        x_line = np.linspace(0.0, 1.0, 500)
        w_x_line = np.array([inside_pillar(x, y0) for x in x_line])
        sigma_x_line = (1 - w_x_line) * bg_val + w_x_line * pil_val

        y_line = np.linspace(0.0, 1.0, 500)
        w_y_line = np.array([inside_pillar(x0, y) for y in y_line])
        sigma_y_line = (1 - w_y_line) * bg_val + w_y_line * pil_val

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        ax_xy = axes[0, 0]
        im = ax_xy.imshow(sigma_xy.T, origin='lower', extent=[0, Lx_nm, 0, Ly_nm],
                          cmap='viridis', aspect='auto')
        ax_xy.scatter(x0*Lx_nm, y0*Ly_nm, color='red', marker='x', s=80)
        ax_xy.set_xlim(*xlim); ax_xy.set_ylim(*ylim_xy)
        ax_xy.set_title(f'\u03c3(x,y) at z={z0:.4f}  (zoom {zoom}\u00d7)')
        ax_xy.set_xlabel('x (nm)'); ax_xy.set_ylabel('y (nm)')
        fig.colorbar(im, ax=ax_xy, label='\u03c3 (S/m)')

        ax_xz = axes[0, 1]
        im = ax_xz.imshow(sigma_xz.T, origin='lower', extent=[0, Lx_nm, 0, Lz_nm],
                          cmap='viridis', aspect='auto')
        ax_xz.scatter(x0*Lx_nm, z0*Lz_nm, color='red', marker='x', s=80)
        ax_xz.set_xlim(*xlim); ax_xz.set_ylim(*zlim_xz)
        ax_xz.set_title(f'\u03c3(x,z) at y={y0:.4f}  (zoom {zoom}\u00d7)')
        ax_xz.set_xlabel('x (nm)'); ax_xz.set_ylabel('z (nm)')
        fig.colorbar(im, ax=ax_xz, label='\u03c3 (S/m)')

        ax_yz = axes[0, 2]
        im = ax_yz.imshow(sigma_yz.T, origin='lower', extent=[0, Ly_nm, 0, Lz_nm],
                          cmap='viridis', aspect='auto')
        ax_yz.scatter(y0*Ly_nm, z0*Lz_nm, color='red', marker='x', s=80)
        ax_yz.set_xlim(ylim_xy); ax_yz.set_ylim(*zlim_xz)
        ax_yz.set_title(f'\u03c3(y,z) at x={x0:.4f}  (zoom {zoom}\u00d7)')
        ax_yz.set_xlabel('y (nm)'); ax_yz.set_ylabel('z (nm)')
        fig.colorbar(im, ax=ax_yz, label='\u03c3 (S/m)')

        ax_z = axes[1, 0]
        ax_z.plot(z_nm, sigma_z_line, 'b-', lw=1.5)
        ax_z.set_xlim(z0*Lz_nm - half_z_nm, z0*Lz_nm + half_z_nm)
        ax_z.set_xlabel('z (nm)'); ax_z.set_ylabel('\u03c3 (S/m)')
        ax_z.set_title(f'\u03c3(z) at (x={x0:.3f}, y={y0:.3f})')
        ax_z.grid(True, alpha=0.3)

        ax_x = axes[1, 1]
        ax_x.plot(x_line * Lx_nm, sigma_x_line, 'r-', lw=1.5)
        ax_x.set_xlim(x0*Lx_nm - half_x_nm, x0*Lx_nm + half_x_nm)
        ax_x.set_xlabel('x (nm)'); ax_x.set_ylabel('\u03c3 (S/m)')
        ax_x.set_title(f'\u03c3(x) at (y={y0:.3f}, z={z0:.3f})')
        ax_x.grid(True, alpha=0.3)

        ax_y = axes[1, 2]
        ax_y.plot(y_line * Ly_nm, sigma_y_line, 'g-', lw=1.5)
        ax_y.set_xlim(y0*Ly_nm - half_y_nm, y0*Ly_nm + half_y_nm)
        ax_y.set_xlabel('y (nm)'); ax_y.set_ylabel('\u03c3 (S/m)')
        ax_y.set_title(f'\u03c3(y) at (x={x0:.3f}, z={z0:.3f})')
        ax_y.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        ans = input(
            "\nEnter new centre (x y z fractions) and optional zoom factor, e.g. '0.5 0.5 0.49 20', or 'n' to finish: "
        ).strip().lower()
        if ans == 'n':
            break
        parts = ans.replace(',', ' ').split()
        if len(parts) >= 3:
            try:
                x0 = max(0.0, min(1.0, float(parts[0])))
                y0 = max(0.0, min(1.0, float(parts[1])))
                z0 = max(0.0, min(1.0, float(parts[2])))
                if len(parts) == 4:
                    zoom = max(1, int(float(parts[3])))
                elif len(parts) > 4:
                    print("Too many values -- using first three for centre, ignoring extra.")
                print(f"New centre: x={x0:.4f}, y={y0:.4f}, z={z0:.4f}  |  Zoom: {zoom}\u00d7")
            except ValueError:
                print("Invalid numbers -- keeping previous centre and zoom.")
        else:
            print("Please provide at least three numbers.")


def generate_sigma(source_json="afm_config_nm.json", sigma_csv="conductivity_profile.csv",
                   output_json="sigma_blocks.json", interactive=True):
    """Generate conductivity (sigma) block JSON from CSV profile.

    Reads a conductivity-vs-depth CSV, lets the user select or define a
    spatial region, subdivides it into merged layers, and writes the
    result as a sigma_blocks JSON.

    Parameters
    ----------
    source_json : str, optional
        Source nm config with box dimensions (default: afm_config_nm.json).
    sigma_csv : str, optional
        Conductivity profile CSV (default: conductivity_profile.csv).
    output_json : str, optional
        Output sigma blocks JSON path (default: sigma_blocks.json).
    interactive : bool, optional
        If True, prompt for user input (default: True).

    Returns
    -------
    None
    """
    if interactive:
        print("=== Conductivity (sigma) Block Generator ===\n")
        src = confirm_path("Source JSON (nm)", source_json, must_exist=True)
        if src is None:
            print("Source JSON required. Exiting.")
            return
        csv_path = confirm_path("Conductivity CSV", sigma_csv, must_exist=True)
        if csv_path is None:
            print("CSV required. Exiting.")
            return
        out = confirm_path("Output JSON", output_json, must_exist=False)
        if out is None:
            out = input("Enter output file name: ").strip()
            if not out:
                print("No output file specified. Exiting.")
                return
    else:
        src = source_json
        csv_path = sigma_csv
        out = output_json

    with open(src, 'r') as f:
        src_data = json.load(f)

    required = {"Lx_nm", "Ly_nm", "Lz_nm"}
    missing = required - set(src_data.keys())
    if missing:
        print(f"Error: Source JSON missing keys: {missing}")
        return

    Lx, Ly, Lz = src_data["Lx_nm"], src_data["Ly_nm"], src_data["Lz_nm"]

    try:
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        if data.ndim == 1:
            z_m = np.array([data[0]])
            sigma_csv_arr = np.array([data[1]])
        else:
            z_m = data[:, 0]
            sigma_csv_arr = data[:, 1]
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    sigma_Sm = sigma_csv_arr * 100.0
    z_offset_nm = z_m * 1e9

    existing_blocks_raw = src_data.get("sigma_blocks", [])
    if existing_blocks_raw:
        print(f"\nFound {len(existing_blocks_raw)} existing sigma_blocks in source JSON.")
        for i, blk in enumerate(existing_blocks_raw):
            sigma = blk.get("sigma_val", 0.0)
            if "x_range_nm" in blk:
                xr = blk["x_range_nm"]
            elif "x_range" in blk:
                xr = [blk["x_range"][0]*Lx, blk["x_range"][1]*Lx]
            else:
                xr = [0, Lx]
            if "y_range_nm" in blk:
                yr = blk["y_range_nm"]
            elif "y_range" in blk:
                yr = [blk["y_range"][0]*Ly, blk["y_range"][1]*Ly]
            else:
                yr = [0, Ly]
            if "z_range_nm" in blk:
                zr = blk["z_range_nm"]
            elif "z_range" in blk:
                zr = [blk["z_range"][0]*Lz, blk["z_range"][1]*Lz]
            else:
                zr = [0, Lz]
            print(f"  [{i}] sigma={sigma:.2f} S/m, X={xr}, Y={yr}, Z={zr}")

        while True:
            ans = input("\nDo you want to replace one of these blocks with the CSV profile? (y/n): ").strip().lower()
            if ans == 'y':
                idx = int(input("Index of block to replace: "))
                if 0 <= idx < len(existing_blocks_raw):
                    block_to_replace = existing_blocks_raw[idx]
                    if "x_range_nm" in block_to_replace:
                        x_range_nm = block_to_replace["x_range_nm"]
                    elif "x_range" in block_to_replace:
                        x_range_nm = [block_to_replace["x_range"][0]*Lx, block_to_replace["x_range"][1]*Lx]
                    else:
                        x_range_nm = [0, Lx]
                    if "y_range_nm" in block_to_replace:
                        y_range_nm = block_to_replace["y_range_nm"]
                    elif "y_range" in block_to_replace:
                        y_range_nm = [block_to_replace["y_range"][0]*Ly, block_to_replace["y_range"][1]*Ly]
                    else:
                        y_range_nm = [0, Ly]
                    if "z_range_nm" in block_to_replace:
                        z_range_nm = block_to_replace["z_range_nm"]
                    elif "z_range" in block_to_replace:
                        z_range_nm = [block_to_replace["z_range"][0]*Lz, block_to_replace["z_range"][1]*Lz]
                    else:
                        z_range_nm = [0, Lz]
                    break
                else:
                    print("Invalid index.")
            elif ans == 'n':
                block_to_replace = None
                break
            else:
                print("Please answer 'y' or 'n'.")
    else:
        block_to_replace = None

    if block_to_replace is None:
        print("\nDefine a new region (in nm) to fill with the conductivity profile.")
        print(f"Box dimensions: X = [0, {Lx:.1f}], Y = [0, {Ly:.1f}], Z = [0, {Lz:.1f}]")
        x_range_nm = get_float_pair("  X range (min max) nm: ", 0, Lx)
        y_range_nm = get_float_pair("  Y range (min max) nm: ", 0, Ly)
        z_range_nm = get_float_pair("  Z range (min max) nm: ", 0, Lz)
    else:
        print("\nUsing the selected block's region for the CSV profile.")

    x_frac = [x_range_nm[0] / Lx, x_range_nm[1] / Lx]
    y_frac = [y_range_nm[0] / Ly, y_range_nm[1] / Ly]
    z_frac = [z_range_nm[0] / Lz, z_range_nm[1] / Lz]

    z_top_nm = z_range_nm[1]
    z_abs_nm = z_top_nm - z_offset_nm
    z_abs_frac = z_abs_nm / Lz

    inside = (z_abs_frac >= z_frac[0] - 1e-12) & (z_abs_frac <= z_frac[1] + 1e-12)
    if not inside.any():
        print("  No CSV points fall inside the specified region. Aborting.")
        return

    z_abs_frac = z_abs_frac[inside]
    sigma_vals = sigma_Sm[inside]

    order = np.argsort(z_abs_frac)
    z_abs_frac = z_abs_frac[order]
    sigma_vals = sigma_vals[order]

    nan_mask = np.isnan(sigma_vals)
    if nan_mask.any():
        print("  Warning: NaN values found. Filling with previous valid value.")
        valid = ~nan_mask
        if valid.any():
            sigma_vals[nan_mask] = np.interp(
                z_abs_frac[nan_mask],
                z_abs_frac[valid],
                sigma_vals[valid]
            )
        else:
            sigma_vals[:] = 1.0

    boundaries = np.concatenate(([z_frac[0]],
                                 (z_abs_frac[:-1] + z_abs_frac[1:]) / 2,
                                 [z_frac[1]]))

    merged = []
    for i in range(len(sigma_vals)):
        sig = float(sigma_vals[i])
        zl, zh = boundaries[i], boundaries[i+1]
        if not merged:
            merged.append([sig, zl, zh])
        else:
            last = merged[-1]
            if abs(sig - last[0]) < 1e-9:
                last[2] = zh
            else:
                merged.append([sig, zl, zh])

    new_blocks = []
    for sig, zl, zh in merged:
        new_blocks.append({
            "sigma_val": sig,
            "x_range": x_frac.copy(),
            "y_range": y_frac.copy(),
            "z_range": [zl, zh]
        })

    if existing_blocks_raw and block_to_replace is not None:
        existing_norm = normalize_blocks(existing_blocks_raw, Lx, Ly, Lz)
        existing_norm.pop(idx)
        for j, blk in enumerate(new_blocks):
            existing_norm.insert(idx + j, blk)
        final_blocks = existing_norm
    else:
        existing_norm = normalize_blocks(existing_blocks_raw, Lx, Ly, Lz) if existing_blocks_raw else []
        final_blocks = existing_norm + new_blocks

    if final_blocks and interactive:
        print("\nPreviewing conductivity distribution...")
        preview_conductivity(final_blocks, Lx, Ly, Lz)

    output_data = {
        "Lx_nm": Lx,
        "Ly_nm": Ly,
        "Lz_nm": Lz,
        "sigma_blocks": final_blocks
    }

    with open(out, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"\nConductivity blocks saved to '{out}' ({len(final_blocks)} layers).")


if __name__ == "__main__":
    generate_sigma()
