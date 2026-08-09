import json
import os
import glob
import re
import subprocess
import sys


def confirm(message):
    """Prompt user for a yes/no confirmation.

    Parameters
    ----------
    message : str
        Question to display.

    Returns
    -------
    bool
        True if user enters 'y', False otherwise.
    """
    ans = input(f"{message} (y/n): ").strip().lower()
    return ans == 'y'


def confirm_directory(description, dir_path):
    """Prompt user to confirm or correct a directory path.

    Loops until the user confirms the path; creates the directory if needed.

    Parameters
    ----------
    description : str
        Label describing what this directory is for.
    dir_path : str
        Proposed directory path.

    Returns
    -------
    str
        Confirmed directory path, or "" if skipped.
    """
    if not dir_path:
        return ""
    while True:
        print(f"{description}: {dir_path}")
        ans = input("Is this correct? (y/n): ").strip().lower()
        if ans == 'y':
            os.makedirs(dir_path, exist_ok=True)
            return dir_path
        elif ans == 'n':
            new_path = input("  Enter correct path (or press Enter to skip): ").strip()
            if new_path == "":
                return ""
            dir_path = new_path
        else:
            print("  Please answer 'y' or 'n'.")


def _is_newer(source, target):
    """Check if source was modified more recently than target.

    Parameters
    ----------
    source : str
        Path to the source file.
    target : str
        Path to the target file.

    Returns
    -------
    bool
        True if source doesn't exist (can't check), or if target
        doesn't exist, or if source is newer than target.
    """
    if not os.path.isfile(source):
        return True
    if not os.path.isfile(target):
        return True
    return os.path.getmtime(source) > os.path.getmtime(target)


def delete_old_numbered_configs(base_name="afm_config"):
    """
    Delete all numbered config files matching the pattern: afm_config_<number>.json.
    This will NOT delete afm_config_nm.json or afm_config_nm_frac.json.

    Parameters
    ----------
    base_name : str, optional
        Base name of the config files (default: "afm_config").
    """
    # Pattern: base_name + underscore + one or more digits + .json
    pattern = r"^" + re.escape(base_name) + r"_(\d+)\.json$"
    regex = re.compile(pattern)
    deleted = []
    for fname in os.listdir('.'):
        # Only delete if it matches the numbered pattern and doesn't contain "nm"
        if regex.match(fname) and "nm" not in fname:
            os.remove(fname)
            deleted.append(fname)
    if deleted:
        print(f"  Deleted {len(deleted)} old config file(s):")
        for f in deleted:
            print(f"    {f}")
    else:
        print("  No old numbered config files found to delete.")


def interactive_edit_json(source_json):
    """
    Interactively view and edit key parameters in the source JSON.

    Parameters
    ----------
    source_json : str
        Path to the source nm config JSON.

    Returns
    -------
    bool
        True if changes were made, False otherwise.
    """
    with open(source_json, 'r') as f:
        data = json.load(f)

    # Define the parameters we want to be able to edit
    param_list = [
        ('Lx_nm', 'Box length X (nm)'),
        ('Ly_nm', 'Box length Y (nm)'),
        ('Lz_nm', 'Box length Z (nm)'),
        ('tip_z_nm', 'Tip apex height (nm)'),
        ('R_nm', 'Tip radius (nm)'),
        ('r_tip_nm', 'Tip truncation radius (nm)'),
        ('aspect_ratio', 'Tip aspect ratio'),
        ('v_start', 'Start voltage (V)'),
        ('v_stop', 'Stop voltage (V)'),
        ('v_step', 'Voltage step (V)'),
        ('mg_max_runtime', 'Max runtime (s)'),
        ('grid_resolution.nx', 'Grid resolution Nx'),
        ('grid_resolution.ny', 'Grid resolution Ny'),
        ('grid_resolution.nz', 'Grid resolution Nz'),
    ]

    print("\n=== Editing Source JSON ===")
    print(f"File: {source_json}\n")

    # Show current values
    print("Current key parameters:")
    for key, desc in param_list:
        if '.' in key:
            parts = key.split('.')
            val = data
            for p in parts:
                val = val.get(p, 'N/A')
        else:
            val = data.get(key, 'N/A')
        print(f"  {desc}: {val}")

    ans = input("\nDo you want to edit any of these parameters? (y/n): ").strip().lower()
    if ans != 'y':
        print("  No changes made.")
        return False

    changed = False
    for key, desc in param_list:
        current_val = None
        if '.' in key:
            parts = key.split('.')
            val = data
            for p in parts:
                val = val.get(p, 'N/A')
            current_val = val
        else:
            current_val = data.get(key, 'N/A')

        new_val = input(f"  {desc} (current: {current_val}, press Enter to keep): ").strip()
        if new_val == "":
            continue

        # Try to convert to int or float
        try:
            if '.' in new_val:
                new_val_num = float(new_val)
            else:
                new_val_num = int(new_val)
        except ValueError:
            print(f"    Invalid number, keeping current value.")
            continue

        # Update the nested dict
        if '.' in key:
            parts = key.split('.')
            target = data
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = new_val_num
        else:
            data[key] = new_val_num
        changed = True
        print(f"    Updated {desc} to {new_val_num}")

    if changed:
        # Also allow editing offsets
        print("\nCurrent offsets_nm:", data.get('offsets_nm', []))
        ans2 = input("Do you want to edit the offset list? (y/n): ").strip().lower()
        if ans2 == 'y':
            new_offsets = input("Enter new comma-separated offsets in nm (e.g. -5,-4,-3,-2,-1,0): ").strip()
            try:
                offsets = [float(x.strip()) for x in new_offsets.split(",") if x.strip() != ""]
                if offsets:
                    data['offsets_nm'] = offsets
                    changed = True
                    print(f"  Updated offsets to {offsets}")
                else:
                    print("  No valid offsets entered, keeping current.")
            except ValueError:
                print("  Invalid input, keeping current offsets.")

        # Write back the updated JSON
        with open(source_json, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\n  Changes saved to {source_json}")
    else:
        print("  No changes made.")

    return changed


def interactive_set_output_dir(source_json):
    """
    Interactively view and set the output directory.

    Parameters
    ----------
    source_json : str
        Path to the source nm config JSON.

    Returns
    -------
    str
        The confirmed output directory path.
    """
    with open(source_json, 'r') as f:
        data = json.load(f)

    current_dir = data.get('output_dir', '.')
    print(f"\nCurrent output directory: {current_dir}")
    ans = input("Do you want to change the output directory? (y/n): ").strip().lower()
    if ans != 'y':
        print("  Keeping current output directory.")
        return current_dir

    new_dir = input("Enter new output directory path (press Enter to keep current): ").strip()
    if not new_dir:
        print("  Keeping current output directory.")
        return current_dir

    data['output_dir'] = new_dir
    with open(source_json, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"  Updated output directory to {new_dir}")
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def main():
    """Run the full presimulation pipeline interactively.

    Guides the user through each presimulation step: loading a source JSON,
    generating epsilon depth profile, fractional config, sigma blocks, tip
    sweep configs, high-res NPZ files, and grid arrays.

    Steps 3 (fractional config) and 6 (tip sweep) regenerate automatically
    when the source nm JSON is newer than the generated files.

    Returns
    -------
    None
    """
    print("=== Presimulation Master ===\n")

    # ---- Step 0: Delete old numbered configs ----
    print("[0] Clean up old numbered config files")
    if confirm("Delete old afm_config_*.json files (numbered configs only)?"):
        delete_old_numbered_configs()
    else:
        print("  Skipping deletion.")

    # ---- Step 1: Locate source JSON ----
    source_json = "afm_config_nm.json"
    if os.path.isfile(source_json):
        print(f"[1] Source JSON found: {source_json}")
    else:
        print(f"[1] Source JSON not found: {source_json}")
        source_json = input("Enter path to source JSON (nm config): ").strip()
        if not source_json:
            print("Source JSON required. Exiting.")
            return

    # ---- Step 2: Edit source JSON parameters ----
    print("\n[2] Edit source JSON parameters")
    interactive_edit_json(source_json)

    # ---- Step 3: Set output directory ----
    print("\n[3] Output directory")
    output_dir = interactive_set_output_dir(source_json)

    # ---- Step 4: Tip offsets ----
    print("\n[4] Tip offsets")
    with open(source_json, 'r') as f:
        data = json.load(f)
    current_offsets = data.get('offsets_nm', None)
    if current_offsets is not None:
        print(f"Current tip offsets (nm): {current_offsets}")
        ans = input("Do you want to change these offsets? (y/n): ").strip().lower()
        if ans == 'y':
            user_input = input("Enter new comma-separated offsets in nm (e.g. -5,-4,-3,-2,-1,0): ").strip()
            try:
                new_offsets = [float(x.strip()) for x in user_input.split(",") if x.strip() != ""]
                if new_offsets:
                    data['offsets_nm'] = new_offsets
                    with open(source_json, 'w') as f:
                        json.dump(data, f, indent=4)
                    print(f"  Updated offsets to {new_offsets}")
                else:
                    print("  No valid offsets entered, keeping current.")
            except ValueError:
                print("  Invalid input, keeping current offsets.")
        else:
            print("  Keeping current offsets.")
    else:
        print("  No 'offsets_nm' found in the source JSON. Skipping.")

    # ---- Ask if user wants material profiles (epsilon depth, sigma) ----
    print("\n[5] Material profiles (epsilon depth, conductivity)")
    include_material_profiles = confirm(
        "Do you want to include material profiles beyond the basic dielectric blocks?\n"
        "   (If you only need a simple electrostatic model without depth-dependent epsilon or conductivity, answer 'n')"
    )

    frac_config = "afm_config_nm_frac.json"

    # ---- Step 6: Epsilon depth profile (skipped if not wanted) ----
    eps_csv = "eps_z.csv"
    if include_material_profiles:
        if not os.path.isfile(eps_csv):
            print(f"\n[6] {eps_csv} not found.")
            if confirm("Generate epsilon depth profile?"):
                from .eps_z_gen import generate_eps_profile
                generate_eps_profile()
        else:
            print(f"\n[6] Epsilon profile found: {eps_csv}")
            if confirm("Regenerate?"):
                from .eps_z_gen import generate_eps_profile
                generate_eps_profile()
    else:
        print(f"\n[6] Skipping epsilon depth profile (material profiles disabled).")

    # ---- Step 7: Fractional config (always done, regardless of material profiles) ----
    if _is_newer(source_json, frac_config):
        print(f"\n[7] Source changed or {frac_config} missing — regenerating...")
        from .generate_json import generate_config
        generate_config(source_json=source_json, dest_json=frac_config,
                        eps_csv=eps_csv if os.path.isfile(eps_csv) and include_material_profiles else None,
                        interactive=False)
    else:
        print(f"\n[7] {frac_config} is up to date.")

    # ---- Step 8: Sigma blocks (skipped if not wanted) ----
    sigma_json = "sigma_blocks.json"
    sigma_csv = "conductivity_profile.csv"
    if include_material_profiles:
        if not os.path.isfile(sigma_json):
            print(f"\n[8] {sigma_json} not found.")
            if confirm("Generate sigma blocks?"):
                if not os.path.isfile(sigma_csv):
                    print(f"  {sigma_csv} not found. Skipping sigma generation.")
                else:
                    from .generate_sigma_json import generate_sigma
                    generate_sigma(source_json=source_json, sigma_csv=sigma_csv,
                                   output_json=sigma_json)
        else:
            print(f"\n[8] Sigma blocks found: {sigma_json}")
            if confirm("Regenerate?"):
                from .generate_sigma_json import generate_sigma
                generate_sigma(source_json=source_json, sigma_csv=sigma_csv,
                               output_json=sigma_json)
    else:
        print(f"\n[8] Skipping sigma blocks (material profiles disabled).")

    # ---- Step 9: Tip sweep configs ----
    print(f"\n[9] Tip sweep configs")
    tip_first = f"afm_config_1.json"
    if _is_newer(source_json, tip_first):
        print(f"  Source changed or {tip_first} missing — regenerating...")
        from .json_tips_gen import generate_tip_sweep
        generate_tip_sweep(source_json=source_json, interactive=False)
    else:
        print(f"  Tip sweep configs are up to date.")

    # ---- Step 10: High-res epsilon profiles (skipped if not wanted) ----
    eps_npz = "eps_highres.npz"
    if include_material_profiles:
        if not os.path.isfile(eps_npz):
            print(f"\n[10] {eps_npz} not found.")
            if confirm("Build high-res epsilon profiles?"):
                from .precompute_materials import build_eps_highres
                build_eps_highres(frac_config, eps_npz)
        else:
            print(f"\n[10] High-res epsilon profiles found: {eps_npz}")
            if confirm("Rebuild?"):
                from .precompute_materials import build_eps_highres
                build_eps_highres(frac_config, eps_npz)
    else:
        print(f"\n[10] Skipping high-res epsilon profiles (material profiles disabled).")

    # ---- Step 11: High-res sigma profiles (skipped if not wanted) ----
    sigma_npz = "sigma_highres.npz"
    if include_material_profiles:
        if not os.path.isfile(sigma_npz):
            print(f"\n[11] {sigma_npz} not found.")
            if confirm("Build high-res sigma profiles?"):
                if os.path.isfile(sigma_json):
                    from .precompute_materials import build_sigma_highres
                    build_sigma_highres(sigma_json, sigma_npz)
                else:
                    print("  sigma_blocks.json not found. Skipping.")
        else:
            print(f"\n[11] High-res sigma profiles found: {sigma_npz}")
            if confirm("Rebuild?"):
                if os.path.isfile(sigma_json):
                    from .precompute_materials import build_sigma_highres
                    build_sigma_highres(sigma_json, sigma_npz)
                else:
                    print("  sigma_blocks.json not found. Skipping.")
    else:
        print(f"\n[11] Skipping high-res sigma profiles (material profiles disabled).")

    # ---- Step 12: Grid arrays ----
    print(f"\n[12] Grid arrays")
    if confirm("Precompute grid arrays for epsilon?"):
        from .precompute_materials import precompute_grid_arrays
        precompute_grid_arrays(frac_config, output_dir=".", kind="eps")

    if os.path.isfile(sigma_json) and include_material_profiles and confirm("Precompute grid arrays for sigma?"):
        from .precompute_materials import precompute_grid_arrays
        precompute_grid_arrays(sigma_json, output_dir=".", kind="sigma")

    print("\nPresimulation complete.")


if __name__ == "__main__":
    main()