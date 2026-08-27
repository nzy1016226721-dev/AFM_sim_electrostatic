import os
import sys
import argparse


def _build_parser():
    """Build the direct simulation command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "AFM simulation launcher. Physical geometry is specified in nm relative "
            "to the configured origin; presimulation generates tip-offset JSONs."
        )
    )
    parser.add_argument(
        "config_source",
        nargs="?",
        default="afm_config_nm",
        help=(
            "JSON file path, config directory, or legacy base name. "
            "Direct JSON paths run only that configuration."
        ),
    )
    parser.add_argument(
        "--config-dir",
        default=".",
        help="Directory used for legacy base-name lookup.",
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot",
        dest="plotting_override",
        action="store_true",
        help="Force interactive plots on.",
    )
    plot_group.add_argument(
        "--no-plot",
        dest="plotting_override",
        action="store_false",
        help="Force interactive plots off.",
    )
    parser.set_defaults(plotting_override=None)
    return parser


def show_menu():
    """Display the main AFM simulation package launcher menu."""
    print("\n" + "=" * 50)
    print("  AFM Simulation Package - Launcher")
    print("=" * 50)
    print("  1. Presimulation [generate tip-z offset JSONs]")
    print("  2. Simulation")
    print("     Run AFM simulation with movement + voltage sweep")
    print("  3. Postprocessing")
    print("     Plot NPY files and run validation tools")
    print("  q. Quit")
    print("=" * 50)


def run_presimulation():
    """Generate per-offset JSON files from a base physical-coordinate config."""
    config_path = input(
        "Base config path (default: afm_config_nm.json): "
    ).strip() or "afm_config_nm.json"
    from simulation.presimulation import run_presimulation as _run_presim
    _run_presim(config_path)


def run_simulation():
    """Launch the simulation pipeline: batch main loop with movement + voltage sweep."""
    config_base = input("Config base name (default: afm_config_nm): ").strip()
    if not config_base:
        config_base = "afm_config_nm"
    from simulation.main_loop import batch_main
    batch_main(config_base)


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
        phi_file = input("Path to potential .npy file (default: afm_phi_1_-2.00V.npy): ").strip()
        if not phi_file:
            phi_file = "afm_phi_1_-2.00V.npy"
        config_file = input("Path to config JSON (default: afm_config_nm.json): ").strip()
        if not config_file:
            config_file = "afm_config_nm.json"
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


    else:
        return


def main(argv=None):
    """Main entry point.

    Legacy behavior:
      python run_all.py
      -> interactive menu
      python run_all.py sim afm_config_nm
      -> legacy simulation mode

    New direct-JSON behavior:
      python run_all.py afm_config_nm.json
      python run_all.py /path/to/config.json
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    # Keep the legacy explicit mode syntax first.
    if argv and argv[0].lower() in ("presim", "presimulation", "1"):
        from simulation.presimulation import run_presimulation as _run_presim
        config_path = argv[1] if len(argv) > 1 else "afm_config_nm.json"
        _run_presim(config_path)
        return
    if argv and argv[0].lower() in ("post", "postprocessing", "3"):
        run_postprocessing()
        return
    if argv and argv[0].lower() in ("sim", "simulation", "2"):
        config_source = argv[1] if len(argv) > 1 else "afm_config"
        extra = argv[2:]
        parser = _build_parser()
        args = parser.parse_args([config_source] + extra)
        from simulation.main_loop import batch_main
        batch_main(
            args.config_source,
            config_dir=args.config_dir,
            plotting_override=args.plotting_override,
        )
        return

    # A direct JSON/config source is now also accepted.  No argument still
    # preserves the original interactive menu.
    if argv:
        parser = _build_parser()
        args = parser.parse_args(argv)
        from simulation.main_loop import batch_main
        batch_main(
            args.config_source,
            config_dir=args.config_dir,
            plotting_override=args.plotting_override,
        )
        return

    while True:
        show_menu()
        choice = input("Select option: ").strip()
        if choice == "1":
            run_presimulation()
        elif choice == "2":
            run_simulation()
        elif choice == "3":
            run_postprocessing()
        elif choice.lower() in ("q", "quit", "exit"):
            print("Exiting.")
            break


if __name__ == "__main__":
    main()
