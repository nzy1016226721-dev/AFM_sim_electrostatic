import json
import os

from .generate_json import convert_blocks_nm_to_frac


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


def generate_tip_sweep(source_json="afm_config_nm.json",
                       output_base="afm_config",
                       interactive=True):
    """Generate multiple config files with varying tip z-offsets.

    Reads the nm-scale source JSON (including offsets_nm, tip_z_nm, Lz_nm,
    blocks_nm, etc.), converts to fractional coordinates, and creates
    numbered copies each with a shifted tip_z value.

    Parameters
    ----------
    source_json : str, optional
        Source JSON with nm-scale coordinates (default: afm_config_nm.json).
    output_base : str, optional
        Base filename for output configs (default: "afm_config").
    interactive : bool, optional
        If True, prompt for confirmation (default: True).

    Returns
    -------
    None
    """
    if interactive:
        print("=== Tip Z-offset batch generator ===\n")
        src = confirm_path("Source JSON (nm)", source_json)
        if src is None:
            print("Source JSON required. Exiting.")
            return
    else:
        src = source_json
        if not os.path.isfile(src):
            print(f"Source JSON not found: {src}")
            return

    with open(src, 'r') as f:
        cfg = json.load(f)

    required = {"Lx_nm", "Ly_nm", "Lz_nm", "tip_z_nm", "R_nm", "r_tip_nm", "offsets_nm"}
    missing = required - set(cfg.keys())
    if missing:
        print(f"Error: Source JSON missing keys: {missing}")
        return

    Lx = cfg["Lx_nm"]
    Ly = cfg["Ly_nm"]
    Lz = cfg["Lz_nm"]
    base_tip_z_nm = cfg["tip_z_nm"]
    offsets_nm = cfg["offsets_nm"]

    if interactive:
        print(f"\nOffset list from config (nm): {offsets_nm}")
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
        tip_z = (base_tip_z_nm + offset_nm) / Lz
        if not (0.0 <= tip_z <= 1.0):
            print(f"  Offset {offset_nm:.2f} nm -> tip_z = {tip_z:.6f} "
                  f"(out of [0,1]) -- skipping.")
            continue

        out_cfg = {}
        for k, v in cfg.items():
            if k in ("tip_z_nm", "R_nm", "r_tip_nm", "blocks_nm"):
                continue
            out_cfg[k] = v

        out_cfg["tip_z"] = tip_z
        out_cfg["R"] = cfg["R_nm"] / Lx
        out_cfg["r_tip"] = cfg["r_tip_nm"] / Lx

        if "blocks_nm" in cfg:
            out_cfg["blocks"] = convert_blocks_nm_to_frac(
                cfg["blocks_nm"], Lx, Ly, Lz)

        out_name = f"{output_base}_{idx}.json"
        with open(out_name, 'w') as f:
            json.dump(out_cfg, f, indent=4)

        base_nm = base_tip_z_nm
        print(f"  Saved {out_name}  "
              f"(tip_z = {base_nm:.2f}nm + {offset_nm:.2f}nm -> {tip_z:.6f} frac)")

    print("\nDone.")


if __name__ == "__main__":
    generate_tip_sweep()
