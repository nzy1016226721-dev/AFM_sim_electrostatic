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
        If True, only accept paths that already exist (default: True).

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


def convert_blocks_nm_to_frac(blocks_nm, Lx, Ly, Lz):
    """Convert nanometre-scale block coordinates to fractional coordinates.

    Parameters
    ----------
    blocks_nm : list of dict
        Blocks with 'x_range_nm', 'y_range_nm', 'z_range_nm' and 'eps_val'.
    Lx : float
        Full box length in x (nm).
    Ly : float
        Full box length in y (nm).
    Lz : float
        Full box length in z (nm).

    Returns
    -------
    list of dict
        Blocks with fractional 'x_range', 'y_range', 'z_range' and 'eps_val'.
    """
    frac_blocks = []
    for blk in blocks_nm:
        if "eps_val" not in blk:
            print("Warning: skipping block missing 'eps_val'")
            continue
        frac_blk = {"eps_val": blk["eps_val"]}
        for axis, length in zip(['x', 'y', 'z'], [Lx, Ly, Lz]):
            key_nm = f"{axis}_range_nm"
            key_frac = f"{axis}_range"
            if key_nm in blk:
                lo, hi = blk[key_nm]
                frac_blk[key_frac] = [lo/length, hi/length]
            else:
                frac_blk[key_frac] = [0.0, 1.0]
        frac_blocks.append(frac_blk)
    return frac_blocks


def modify_block_from_csv(blocks_frac, Lz, csv_path):
    """Replace a dielectric block's epsilon profile with CSV data.

    Reads epsilon vs z from a CSV, identifies which block covers that depth,
    and subdivides it into merged layers with the CSV values.

    Parameters
    ----------
    blocks_frac : list of dict
        Fractional-coordinate blocks (will be modified).
    Lz : float
        Full box length in z (nm).
    csv_path : str
        Path to CSV with columns (z_m, epsilon).

    Returns
    -------
    list of dict
        Updated block list with the selected block subdivided.
    """
    try:
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        if data.ndim == 1:
            z_m = np.array([data[0]])
            eps_vals = np.array([data[1]])
        else:
            z_m = data[:, 0]
            eps_vals = data[:, 1]
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return blocks_frac

    z_offset_nm = z_m * 1e9

    print("\nExisting dielectric blocks:")
    for i, blk in enumerate(blocks_frac):
        print(f"  [{i}] eps={blk['eps_val']:.3f}  "
              f"x=[{blk['x_range'][0]:.4f}, {blk['x_range'][1]:.4f}]  "
              f"y=[{blk['y_range'][0]:.4f}, {blk['y_range'][1]:.4f}]  "
              f"z=[{blk['z_range'][0]:.4f}, {blk['z_range'][1]:.4f}]")

    while True:
        try:
            idx = int(input("Index of block to replace (or -1 to cancel): "))
            if idx == -1:
                return blocks_frac
            if 0 <= idx < len(blocks_frac):
                break
            print("  Index out of range.")
        except ValueError:
            print("  Enter a valid integer.")

    selected = blocks_frac[idx]
    original_eps = selected["eps_val"]

    eps_vals = np.where(np.isnan(eps_vals), original_eps, eps_vals)

    z0_frac = selected["z_range"][0]
    z1_frac = selected["z_range"][1]
    z_top_nm = z1_frac * Lz

    z_abs_nm = z_top_nm - z_offset_nm
    z_abs_frac = z_abs_nm / Lz

    inside = (z_abs_frac >= z0_frac - 1e-12) & (z_abs_frac <= z1_frac + 1e-12)
    if not inside.any():
        print("  No CSV points fall inside the block's z-range. Block unchanged.")
        return blocks_frac

    z_abs_frac = z_abs_frac[inside]
    eps_vals = eps_vals[inside]

    order = np.argsort(z_abs_frac)
    z_abs_frac = z_abs_frac[order]
    eps_vals = eps_vals[order]

    boundaries = np.concatenate(([z0_frac],
                                 (z_abs_frac[:-1] + z_abs_frac[1:]) / 2,
                                 [z1_frac]))

    merged = []
    for i in range(len(eps_vals)):
        eps_i = float(eps_vals[i])
        zl, zh = boundaries[i], boundaries[i+1]
        if not merged:
            merged.append([eps_i, zl, zh])
        else:
            last = merged[-1]
            if abs(eps_i - last[0]) < 1e-9:
                last[2] = zh
            else:
                merged.append([eps_i, zl, zh])

    new_blocks = []
    for eps_i, zl, zh in merged:
        new_blocks.append({
            "eps_val": eps_i,
            "x_range": selected["x_range"].copy(),
            "y_range": selected["y_range"].copy(),
            "z_range": [zl, zh]
        })

    blocks_frac.pop(idx)
    for j, blk in enumerate(new_blocks):
        blocks_frac.insert(idx + j, blk)

    print(f"  Replaced block [{idx}] with {len(new_blocks)} merged layer(s).")
    return blocks_frac


def preview_blocks(blocks, Lx_nm, Ly_nm, Lz_nm):
    """Interactive 2D/1D preview of dielectric block distribution.

    Displays XY, XZ, YZ slice plots plus line-outs, allowing the user to
    change the centre point and zoom level interactively.

    Parameters
    ----------
    blocks : list of dict
        Fractional-coordinate dielectric blocks.
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
    if not blocks:
        print("No blocks to preview.")
        return

    nz_samples = 20000
    z_frac = np.linspace(0.0, 1.0, nz_samples)
    z_nm = z_frac * Lz_nm

    full_blocks = [b for b in blocks if b["x_range"] == [0.0, 1.0] and b["y_range"] == [0.0, 1.0]]
    pillar_blocks = [b for b in blocks if b["x_range"] != [0.0, 1.0] or b["y_range"] != [0.0, 1.0]]

    eps_bg = np.ones(nz_samples, dtype=np.float32)
    for blk in full_blocks:
        z0b, z1b = blk["z_range"]
        eps_bg[(z_frac >= z0b) & (z_frac <= z1b)] = blk["eps_val"]

    eps_pillar = np.ones(nz_samples, dtype=np.float32) if pillar_blocks else None
    if eps_pillar is not None:
        for blk in blocks:
            z0b, z1b = blk["z_range"]
            eps_pillar[(z_frac >= z0b) & (z_frac <= z1b)] = blk["eps_val"]

    pillar_rect = None
    for blk in pillar_blocks:
        xr = blk["x_range"]
        yr = blk["y_range"]
        if pillar_rect is None:
            pillar_rect = (xr[0], xr[1], yr[0], yr[1])
        else:
            if (xr[0], xr[1], yr[0], yr[1]) != pillar_rect:
                print("Warning: multiple pillar shapes -- using the first one.")

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
        bg_val = eps_bg[iz]
        p_val = eps_pillar[iz] if eps_pillar is not None else bg_val
        eps_xy = (1 - W_xy) * bg_val + W_xy * p_val

        W_xz = np.array([inside_pillar(x, y0) for x in xs_2d])[:, np.newaxis]
        bg_1d = eps_bg[np.searchsorted(z_frac, zs_2d).clip(0, nz_samples-1)]
        p_1d = eps_pillar[np.searchsorted(z_frac, zs_2d).clip(0, nz_samples-1)] if eps_pillar is not None else bg_1d
        eps_xz = (1 - W_xz) * bg_1d[np.newaxis, :] + W_xz * p_1d[np.newaxis, :]

        W_yz = np.array([inside_pillar(x0, y) for y in ys_2d])[:, np.newaxis]
        eps_yz = (1 - W_yz) * bg_1d[np.newaxis, :] + W_yz * p_1d[np.newaxis, :]

        w = inside_pillar(x0, y0)
        if eps_pillar is not None:
            eps_z_line = (1 - w) * eps_bg + w * eps_pillar
        else:
            eps_z_line = eps_bg

        x_line = np.linspace(0.0, 1.0, 500)
        w_x_line = np.array([inside_pillar(x, y0) for x in x_line])
        eps_x_line = (1 - w_x_line) * bg_val + w_x_line * p_val

        y_line = np.linspace(0.0, 1.0, 500)
        w_y_line = np.array([inside_pillar(x0, y) for y in y_line])
        eps_y_line = (1 - w_y_line) * bg_val + w_y_line * p_val

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        ax_xy = axes[0, 0]
        im = ax_xy.imshow(eps_xy.T, origin='lower', extent=[0, Lx_nm, 0, Ly_nm],
                          cmap='viridis', aspect='auto')
        ax_xy.scatter(x0*Lx_nm, y0*Ly_nm, color='red', marker='x', s=80)
        ax_xy.set_xlim(*xlim)
        ax_xy.set_ylim(*ylim_xy)
        ax_xy.set_title(f'\u03b5(x,y) at z={z0:.4f}  (zoom {zoom}\u00d7)')
        ax_xy.set_xlabel('x (nm)'); ax_xy.set_ylabel('y (nm)')
        fig.colorbar(im, ax=ax_xy)

        ax_xz = axes[0, 1]
        im = ax_xz.imshow(eps_xz.T, origin='lower', extent=[0, Lx_nm, 0, Lz_nm],
                          cmap='viridis', aspect='auto')
        ax_xz.scatter(x0*Lx_nm, z0*Lz_nm, color='red', marker='x', s=80)
        ax_xz.set_xlim(*xlim)
        ax_xz.set_ylim(*zlim_xz)
        ax_xz.set_title(f'\u03b5(x,z) at y={y0:.4f}  (zoom {zoom}\u00d7)')
        ax_xz.set_xlabel('x (nm)'); ax_xz.set_ylabel('z (nm)')
        fig.colorbar(im, ax=ax_xz)

        ax_yz = axes[0, 2]
        im = ax_yz.imshow(eps_yz.T, origin='lower', extent=[0, Ly_nm, 0, Lz_nm],
                          cmap='viridis', aspect='auto')
        ax_yz.scatter(y0*Ly_nm, z0*Lz_nm, color='red', marker='x', s=80)
        ax_yz.set_xlim(ylim_xy)
        ax_yz.set_ylim(*zlim_xz)
        ax_yz.set_title(f'\u03b5(y,z) at x={x0:.4f}  (zoom {zoom}\u00d7)')
        ax_yz.set_xlabel('y (nm)'); ax_yz.set_ylabel('z (nm)')
        fig.colorbar(im, ax=ax_yz)

        ax_z = axes[1, 0]
        ax_z.plot(z_nm, eps_z_line, 'b-', lw=1.5)
        ax_z.set_xlim(z0*Lz_nm - half_z_nm, z0*Lz_nm + half_z_nm)
        ax_z.set_xlabel('z (nm)'); ax_z.set_ylabel('\u03b5')
        ax_z.set_title(f'\u03b5(z) at (x={x0:.3f}, y={y0:.3f})')
        ax_z.grid(True, alpha=0.3)

        ax_x = axes[1, 1]
        ax_x.plot(x_line * Lx_nm, eps_x_line, 'r-', lw=1.5)
        ax_x.set_xlim(x0*Lx_nm - half_x_nm, x0*Lx_nm + half_x_nm)
        ax_x.set_xlabel('x (nm)'); ax_x.set_ylabel('\u03b5')
        ax_x.set_title(f'\u03b5(x) at (y={y0:.3f}, z={z0:.3f})')
        ax_x.grid(True, alpha=0.3)

        ax_y = axes[1, 2]
        ax_y.plot(y_line * Ly_nm, eps_y_line, 'g-', lw=1.5)
        ax_y.set_xlim(y0*Ly_nm - half_y_nm, y0*Ly_nm + half_y_nm)
        ax_y.set_xlabel('y (nm)'); ax_y.set_ylabel('\u03b5')
        ax_y.set_title(f'\u03b5(y) at (x={x0:.3f}, z={z0:.3f})')
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


def generate_config(source_json="afm_config_nm.json", dest_json="afm_config_nm_frac.json",
                    eps_csv=None, interactive=True):
    """Generate a fractional-coordinate AFM configuration JSON from an nm-scale source.

    Converts block coordinates from nm to fractional, optionally applies an
    epsilon depth profile from CSV, compares with any existing destination,
    and writes the result.

    Parameters
    ----------
    source_json : str, optional
        Path to source JSON with nm coordinates (default: afm_config_nm.json).
    dest_json : str, optional
        Path to output fractional JSON (default: afm_config_nm_frac.json).
    eps_csv : str or None, optional
        Epsilon depth profile CSV to apply (default: None).
    interactive : bool, optional
        If True, prompts for user confirmation at each step (default: True).

    Returns
    -------
    None
    """
    if interactive:
        print("=== AFM Configuration Generator ===\n")
        src = confirm_path("Source JSON (nm)", source_json, must_exist=True)
        if src is None:
            print("Source JSON is required. Exiting.")
            return
        dst = confirm_path("Destination JSON", dest_json, must_exist=False)
        if dst is None:
            dst = input("Enter destination file name: ").strip()
            if not dst:
                print("No destination file specified. Exiting.")
                return
        csv_path = None
        if eps_csv:
            csv_path = confirm_path("Epsilon profile CSV", eps_csv, must_exist=True)
            if csv_path is None:
                print("CSV not provided -- skipping block modification.")
    else:
        src = source_json
        dst = dest_json
        csv_path = eps_csv

    with open(src, 'r') as f:
        src_data = json.load(f)

    required = {"Lx_nm", "Ly_nm", "Lz_nm", "tip_z_nm", "aspect_ratio", "R_nm", "r_tip_nm"}
    missing = required - set(src_data.keys())
    if missing:
        print(f"Error: Source JSON missing keys: {missing}")
        return

    Lx, Ly, Lz = src_data["Lx_nm"], src_data["Ly_nm"], src_data["Lz_nm"]

    tip_params = {
        "tip_z": src_data["tip_z_nm"] / Lz,
        "aspect_ratio": src_data["aspect_ratio"],
        "R": src_data["R_nm"] / Lx,
        "r_tip": src_data["r_tip_nm"] / Lx
    }
    if "res_tol_main" in src_data:
        tip_params["res_tol_main"] = src_data["res_tol_main"]
    if "res_tol_zoom" in src_data:
        tip_params["res_tol_zoom"] = src_data["res_tol_zoom"]

    if "blocks_nm" in src_data:
        blocks_frac = convert_blocks_nm_to_frac(src_data["blocks_nm"], Lx, Ly, Lz)
    else:
        blocks_frac = []
        print("No blocks_nm found in source JSON. Starting with empty block list.")

    if dst and os.path.isfile(dst):
        with open(dst, 'r') as f:
            dest_data = json.load(f)
        print(f"\nExisting destination JSON loaded ({len(dest_data)} keys).")
    else:
        dest_data = {}
        print("\nNo existing destination file -- a new one will be created.")

    if dest_data:
        print("\nCurrent destination tip parameters:")
        for k in ["tip_z", "aspect_ratio", "R", "r_tip"]:
            print(f"  {k}: {dest_data.get(k, '<not present>')}")
        dest_blocks = dest_data.get("blocks", [])
        print(f"Current destination block count: {len(dest_blocks)}")
        if dest_blocks:
            print("First few blocks:")
            for i, blk in enumerate(dest_blocks[:3]):
                print(f"  [{i}] eps={blk['eps_val']:.3f}  z=[{blk['z_range'][0]:.6f}, {blk['z_range'][1]:.6f}]")
    else:
        dest_blocks = []

    if csv_path and blocks_frac:
        print("\n--- Epsilon profile from CSV ---")
        blocks_frac = modify_block_from_csv(blocks_frac, Lz, csv_path)

    if blocks_frac and interactive:
        ans = input("\nPreview current dielectric blocks? (y/n): ").strip().lower()
        if ans == 'y':
            preview_blocks(blocks_frac, Lx, Ly, Lz)

    new_blocks = blocks_frac
    existing_blocks = dest_data.get("blocks", []) if dest_data else []

    if existing_blocks:
        print(f"\nComparing {len(new_blocks)} new block(s) with {len(existing_blocks)} existing block(s)...")

        def block_key(blk):
            xr = tuple(round(v, 9) for v in blk["x_range"])
            yr = tuple(round(v, 9) for v in blk["y_range"])
            zr = tuple(round(v, 9) for v in blk["z_range"])
            return (xr, yr, zr)

        existing_lookup = {}
        for i, blk in enumerate(existing_blocks):
            key = block_key(blk)
            existing_lookup[key] = (i, blk["eps_val"])

        changed = []
        unchanged = []

        for j, blk in enumerate(new_blocks):
            key = block_key(blk)
            if key in existing_lookup:
                i_ex, eps_ex = existing_lookup[key]
                if abs(blk["eps_val"] - eps_ex) > 1e-9:
                    changed.append((j, i_ex, blk["eps_val"], eps_ex))
                else:
                    unchanged.append((j, i_ex, blk["eps_val"]))

        if changed:
            print(f"\nBlocks with IDENTICAL GEOMETRY but DIFFERENT eps (will be overwritten):")
            print(f"   Total: {len(changed)}")
        else:
            print("\nNo blocks with identical geometry and differing eps.")

        if unchanged:
            print(f"\nBlocks with IDENTICAL GEOMETRY and SAME eps (already present, no change):")
            print(f"   Total: {len(unchanged)}")

        if changed and interactive:
            ans = input("\nOverwrite the changed blocks (different eps) with the new eps values? (y/n): ").strip().lower()
            if ans != 'y':
                print("Keeping the original eps values for these blocks.")
                for j, i_ex, new_eps, old_eps in changed:
                    new_blocks[j]["eps_val"] = old_eps
            else:
                print("Keeping the new eps values for these blocks.")

    print("\n--- Summary of changes ---")
    print("Tip parameters to write:")
    for k, v in tip_params.items():
        print(f"  {k}: {v}")
    print(f"Number of dielectric blocks: {len(new_blocks)}")

    if interactive:
        confirm = input("\nWrite these changes to the destination JSON? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Aborted.")
            return
    else:
        for k, v in tip_params.items():
            dest_data[k] = v
        if "output_dir" in src_data:
            dest_data["output_dir"] = src_data["output_dir"]
        dest_data["blocks"] = new_blocks
        with open(dst, 'w') as f:
            json.dump(dest_data, f, indent=4)
        print(f"Configuration written to {dst}")
        return

    dest_data.update(tip_params)
    if "output_dir" in src_data:
        dest_data["output_dir"] = src_data["output_dir"]
    dest_data["blocks"] = new_blocks

    with open(dst, 'w') as f:
        json.dump(dest_data, f, indent=4)

    print(f"\nConfiguration written to {dst}")


if __name__ == "__main__":
    generate_config()
