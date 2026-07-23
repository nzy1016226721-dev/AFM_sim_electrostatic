import json
import os


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


def generate_tip_sweep(template_json="afm_config_nm_frac.json",
                       offsets_nm=None, output_base="afm_config",
                       interactive=True):
    """Generate multiple config files with varying tip z-offsets.

    Takes a template JSON and creates numbered copies with shifted
    tip_z values based on the provided offset list.

    Parameters
    ----------
    template_json : str, optional
        Template JSON with fractional coordinates (default: afm_config_nm_frac.json).
    offsets_nm : list of float, optional
        Z-offsets in nanometres (default: sweep from 0 to +/-6 nm).
    output_base : str, optional
        Base filename for output configs (default: "afm_config").
    interactive : bool, optional
        If True, prompt for confirmation (default: True).

    Returns
    -------
    None
    """
    if offsets_nm is None:
        offsets_nm = [0.0000, 1.5529, -1.5529, 3.0000, -3.0000, 4.2426, -4.2426,
                      5.1962, -5.1962, 5.7956, -5.7956, 6.0000, -6.0000]

    if interactive:
        print("=== Tip Z-offset batch generator ===\n")
        template = confirm_path("Template JSON (fractional coords)", template_json)
        if template is None:
            print("Template JSON required. Exiting.")
            return
    else:
        template = template_json
        if not os.path.isfile(template):
            print(f"Template JSON not found: {template}")
            return

    with open(template, 'r') as f:
        cfg = json.load(f)

    required = {"Lx_nm", "Ly_nm", "Lz_nm", "tip_z"}
    missing = required - set(cfg.keys())
    if missing:
        print(f"Error: Template JSON missing keys: {missing}")
        return

    Lz = cfg["Lz_nm"]
    base_tip_z = cfg["tip_z"]

    if interactive:
        print(f"\nCurrent offset list (nm): {offsets_nm}")
        ans = input("Use this list? (y/n): ").strip().lower()
        if ans != 'y':
            user_input = input("Enter comma-separated offsets in nm (e.g. 0,0.5,1): ").strip()
            try:
                offsets = [float(x.strip()) for x in user_input.split(",") if x.strip() != ""]
            except ValueError:
                print("Invalid numbers. Exiting.")
                return
        else:
            offsets = offsets_nm
    else:
        offsets = offsets_nm

    for idx, offset_nm in enumerate(offsets, start=1):
        offset_frac = offset_nm / Lz
        new_tip_z = base_tip_z + offset_frac

        if not (0.0 <= new_tip_z <= 1.0):
            print(f"  Offset {offset_nm:.2f} nm -> tip_z = {new_tip_z:.6f} "
                  f"(out of [0,1]) -- skipping.")
            continue

        new_cfg = cfg.copy()
        new_cfg["tip_z"] = new_tip_z

        out_name = f"{output_base}_{idx}.json"
        with open(out_name, 'w') as f:
            json.dump(new_cfg, f, indent=4)

        print(f"  Saved {out_name}  (tip_z = {base_tip_z:.6f} + {offset_frac:.6f} = {new_tip_z:.6f})")

    print("\nDone.")


if __name__ == "__main__":
    generate_tip_sweep()
