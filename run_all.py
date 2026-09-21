import argparse
import os
import sys
from datetime import datetime

from simulation.runtime import (
    discover_afm_configs,
    is_spyder_like_ide,
    resolve_config_path,
)


def _build_parser():
    """Build the AFM launcher CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "AFM simulation launcher. Pass an explicit JSON to run it; with no "
            "JSON, non-interactive mode runs the newest AFM JSON."
        )
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=None,
        help="Explicit AFM JSON filename/path. No base-name lookup is performed.",
    )
    parser.add_argument(
        "--config-dir",
        default=".",
        help="Directory containing AFM JSON configs (default: current working directory).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output root. The selected config gets its own child directory. "
            "Overrides the JSON output directory."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="List all AFM JSON configs in the working directory and prompt for selection.",
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot", dest="plotting_override", action="store_true",
        help="Force interactive plots on.",
    )
    plot_group.add_argument(
        "--no-plot", dest="plotting_override", action="store_false",
        help="Force interactive plots off.",
    )
    parser.set_defaults(plotting_override=None)
    return parser


def select_config_interactively(config_dir="."):
    """List every AFM JSON in config_dir and return the user's selection."""
    configs = discover_afm_configs(config_dir)
    if not configs:
        raise FileNotFoundError(
            f"No AFM JSON configurations found in {os.path.abspath(config_dir)}"
        )

    print("\n" + "=" * 70)
    print("  AFM configuration selection")
    print("=" * 70)
    for idx, path in enumerate(configs, start=1):
        mtime = os.path.getmtime(path)
        print(f"  {idx:>2}. {os.path.basename(path)}  (modified {datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M:%S})")
    print("=" * 70)

    while True:
        raw = input(f"Select configuration [1-{len(configs)}]: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Please enter the number of a listed configuration.")
            continue
        if 1 <= choice <= len(configs):
            return configs[choice - 1]
        print("Selection out of range.")


def select_output_interactively(config_path, config_dir=".", default_output=None):
    """Prompt for an output root, preserving a sensible default."""
    if default_output is None:
        default_output = os.environ.get("AFM_OUTPUT_ROOT", "outputs")
    prompt = (
        f"Output root for {os.path.basename(config_path)} "
        f"(default: {default_output}): "
    )
    return input(prompt).strip() or default_output


def show_menu(config_dir="."):
    """Display the compact interactive launcher menu."""
    print("\n" + "=" * 50)
    print("  AFM Simulation Package - Launcher")
    print("=" * 50)
    print("  1. Presimulation")
    print("  2. Simulation")
    print("  3. Postprocessing")
    print("  q. Quit")
    print("=" * 50)


def run_presimulation():
    """Run presimulation for one explicit JSON path."""
    config_path = input("Base config JSON path: ").strip()
    if not config_path:
        print("No config selected.")
        return
    from simulation.presimulation import run_presimulation as _run_presim
    _run_presim(config_path)


def run_simulation(config_dir="."):
    """Select one AFM config and one output root, then run it."""
    config_path = select_config_interactively(config_dir)
    output_root = select_output_interactively(config_path, config_dir)
    _run_single_config(
        config_path,
        config_dir=config_dir,
        plotting_override=None,
        interactive=True,
        output_dir=output_root,
    )


def run_postprocessing():
    """Launch the available postprocessing and validation menu."""
    print("\n--- Postprocessing ---")
    print("1. Plot NPY potential file")
    print("2. Run sanity check")
    print("3. QD lever arm calculator")
    print("4. Plot electric field lines")
    print("5. 3D potential map")
    print("6. Capacitance sanity check")
    choice = input("Select option (1-6, or Enter to return): ").strip()

    if choice == "1":
        from postprocessing.plot_npy import plot_afm_from_npy
        path = input("Path to .npy potential file: ").strip()
        if not os.path.isfile(path):
            candidates = [os.path.join("outputs", os.path.basename(path))]
            if path != os.path.basename(path):
                candidates.insert(0, os.path.basename(path))
            for c in candidates:
                if os.path.isfile(c):
                    print(f"  Using {c}")
                    path = c
                    break
        axis = input("Slice axis (x/y/z, default z): ").strip().lower() or "z"
        fig = plot_afm_from_npy(path, axis=axis)
        import matplotlib.pyplot as plt
        plt.show()
    elif choice == "2":
        from postprocessing.sanity_check import run_sanity_check
        phi_file = input("Path to potential .npy file: ").strip()
        config_file = input("Path to config JSON: ").strip()
        run_sanity_check(phi_file=phi_file, config_file=config_file, check_type="e")
    elif choice == "3":
        from postprocessing.lever_arm_calc import main as lever_arm_main
        lever_arm_main()
    elif choice == "4":
        from postprocessing.field_lines import interactive_main as field_lines_main
        field_lines_main()
    elif choice == "5":
        from postprocessing.potential_map import interactive_main as potential_map_main
        potential_map_main()
    elif choice == "6":
        from postprocessing.capacitance_sanity_check import main as sanity_multi_main
        sanity_multi_main()


def _run_single_config(config_path, *, config_dir=".", plotting_override=None,
                       interactive=False, output_dir=None):
    """Run exactly one explicit AFM JSON configuration."""
    from simulation.main_loop import batch_main
    batch_main(
        config_path,
        config_dir=config_dir,
        plotting_override=plotting_override,
        interactive=interactive,
        output_dir_override=output_dir,
    )


def _run_cli(argv):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.interactive:
        config_path = select_config_interactively(args.config_dir)
        output_root = args.output_dir or select_output_interactively(config_path, args.config_dir)
    else:
        config_path = resolve_config_path(args.config_path, directory=args.config_dir)
        output_root = args.output_dir
        print(f"Selected AFM configuration: {config_path}")
        if output_root:
            print(f"Selected output root: {os.path.abspath(os.path.expanduser(output_root))}")

    _run_single_config(
        config_path,
        config_dir=args.config_dir,
        plotting_override=args.plotting_override,
        interactive=args.interactive,
        output_dir=output_root,
    )


def main(argv=None):
    """Main entry point.

    Non-interactive execution accepts one explicit JSON filename/path. With no
    filename, the newest AFM JSON in the working directory is selected.
    Interactive execution enumerates all AFM JSONs and prompts for one, then
    prompts for the output root.
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0].lower() in ("presim", "presimulation"):
        run_presimulation()
        return
    if argv and argv[0].lower() in ("post", "postprocessing"):
        run_postprocessing()
        return
    if argv and argv[0].lower() in ("sim", "simulation"):
        _run_cli(argv[1:])
        return

    if not argv and is_spyder_like_ide():
        show_menu()
        choice = input("Select option (1-3, or q to quit): ").strip().lower()
        if choice in ("1", "presim", "presimulation"):
            run_presimulation()
        elif choice in ("2", "sim", "simulation"):
            run_simulation(".")
        elif choice in ("3", "post", "postprocessing"):
            run_postprocessing()
        else:
            print("Exiting.")
        return

    _run_cli(argv)


if __name__ == "__main__":
    main()
