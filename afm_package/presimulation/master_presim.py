import json
import os


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


def main():
    """Run the full presimulation pipeline interactively.

    Guides the user through each presimulation step: loading a source JSON,
    generating epsilon depth profile, fractional config, sigma blocks, tip
    sweep configs, high-res NPZ files, and grid arrays.

    Steps 3 (fractional config) and 5 (tip sweep) regenerate automatically
    when the source nm JSON is newer than the generated files.

    Returns
    -------
    None
    """
    print("=== Presimulation Master ===\n")

    source_json = "afm_config_nm.json"
    if os.path.isfile(source_json):
        print(f"[1] Source JSON found: {source_json}")
    else:
        print(f"[1] Source JSON not found: {source_json}")
        source_json = input("Enter path to source JSON (nm config): ").strip()
        if not source_json:
            print("Source JSON required. Exiting.")
            return

    with open(source_json, 'r') as f:
        src_data = json.load(f)
    output_dir = src_data.get("output_dir", "")
    if output_dir:
        output_dir = confirm_directory("Output directory", output_dir)

    eps_csv = "eps_z.csv"
    if not os.path.isfile(eps_csv):
        print(f"\n[2] {eps_csv} not found.")
        if confirm("Generate epsilon depth profile?"):
            from .eps_z_gen import generate_eps_profile
            generate_eps_profile()
    else:
        print(f"\n[2] Epsilon profile found: {eps_csv}")
        if confirm("Regenerate?"):
            from .eps_z_gen import generate_eps_profile
            generate_eps_profile()

    frac_config = "afm_config_nm_frac.json"
    if _is_newer(source_json, frac_config):
        print(f"\n[3] Source changed or {frac_config} missing — regenerating...")
        from .generate_json import generate_config
        generate_config(source_json=source_json, dest_json=frac_config,
                        eps_csv=eps_csv if os.path.isfile(eps_csv) else None,
                        interactive=False)
    else:
        print(f"\n[3] {frac_config} is up to date.")

    sigma_json = "sigma_blocks.json"
    sigma_csv = "conductivity_profile.csv"
    if not os.path.isfile(sigma_json):
        print(f"\n[4] {sigma_json} not found.")
        if confirm("Generate sigma blocks?"):
            if not os.path.isfile(sigma_csv):
                print(f"  {sigma_csv} not found. Skipping sigma generation.")
            else:
                from .generate_sigma_json import generate_sigma
                generate_sigma(source_json=source_json, sigma_csv=sigma_csv,
                               output_json=sigma_json)
    else:
        print(f"\n[4] Sigma blocks found: {sigma_json}")
        if confirm("Regenerate?"):
            from .generate_sigma_json import generate_sigma
            generate_sigma(source_json=source_json, sigma_csv=sigma_csv,
                           output_json=sigma_json)

    print(f"\n[5] Tip sweep configs")
    tip_first = f"afm_config_1.json"
    if _is_newer(source_json, tip_first):
        print(f"  Source changed or {tip_first} missing — regenerating...")
        from .json_tips_gen import generate_tip_sweep
        generate_tip_sweep(source_json=source_json, interactive=False)
    else:
        print(f"  Tip sweep configs are up to date.")

    eps_npz = "eps_highres.npz"
    if not os.path.isfile(eps_npz):
        print(f"\n[6] {eps_npz} not found.")
        if confirm("Build high-res epsilon profiles?"):
            from .precompute_materials import build_eps_highres
            build_eps_highres(frac_config, eps_npz)
    else:
        print(f"\n[6] High-res epsilon profiles found: {eps_npz}")
        if confirm("Rebuild?"):
            from .precompute_materials import build_eps_highres
            build_eps_highres(frac_config, eps_npz)

    sigma_npz = "sigma_highres.npz"
    if not os.path.isfile(sigma_npz):
        print(f"\n[7] {sigma_npz} not found.")
        if confirm("Build high-res sigma profiles?"):
            if os.path.isfile(sigma_json):
                from .precompute_materials import build_sigma_highres
                build_sigma_highres(sigma_json, sigma_npz)
            else:
                print("  sigma_blocks.json not found. Skipping.")
    else:
        print(f"\n[7] High-res sigma profiles found: {sigma_npz}")
        if confirm("Rebuild?"):
            if os.path.isfile(sigma_json):
                from .precompute_materials import build_sigma_highres
                build_sigma_highres(sigma_json, sigma_npz)
            else:
                print("  sigma_blocks.json not found. Skipping.")

    print(f"\n[8] Grid arrays")
    if confirm("Precompute grid arrays for epsilon?"):
        from .precompute_materials import precompute_grid_arrays
        precompute_grid_arrays(frac_config, output_dir=".", kind="eps")

    if os.path.isfile(sigma_json) and confirm("Precompute grid arrays for sigma?"):
        from .precompute_materials import precompute_grid_arrays
        precompute_grid_arrays(sigma_json, output_dir=".", kind="sigma")

    print("\nPresimulation complete.")


if __name__ == "__main__":
    main()
