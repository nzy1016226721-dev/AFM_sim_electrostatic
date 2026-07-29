import os
import sys


def show_menu():
    """Display the main AFM simulation package launcher menu."""
    print("\n" + "=" * 50)
    print("  AFM Simulation Package - Launcher")
    print("=" * 50)
    print("  1. Presimulation")
    print("     Generate configs, epsilon/sigma profiles, high-res NPZ files")
    print("  2. Simulation")
    print("     Run AFM simulation with movement + voltage sweep")
    print("  3. Postprocessing")
    print("     Plot NPY files, integrate power, sanity checks")
    print("  q. Quit")
    print("=" * 50)


def run_presimulation():
    """Launch the presimulation pipeline (config generation, material profiles)."""
    from presimulation.master_presim import main as presim_main
    presim_main()


def run_simulation():
    """Launch the simulation pipeline: batch main loop with movement + voltage sweep."""
    config_base = input("Config base name (default: afm_config): ").strip()
    if not config_base:
        config_base = "afm_config"
    from simulation.main_loop import batch_main
    batch_main(config_base)


def run_postprocessing():
    """Launch postprocessing menu: plotting, power integration, sanity checks, lever arm."""
    print("\n--- Postprocessing ---")
    print("1. Plot NPY potential file")
    print("2. Integrate power density")
    print("3. Run sanity check")
    print("4. QD lever arm calculator")
    print("5. Plot electric field lines")
    print("6. 3D potential map")
    print("7. Capacitance sanity check")
    choice = input("Select option (1-7, or Enter to return): ").strip()

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
        from postprocessing.integrate_power import interactive_main
        interactive_main()

    elif choice == "3":
        from postprocessing.sanity_check import run_sanity_check
        phi_file = input("Path to potential .npy file (default: afm_phi_1_-2.00V.npy): ").strip()
        if not phi_file:
            phi_file = "afm_phi_1_-2.00V.npy"
        config_file = input("Path to config JSON (default: afm_config_nm_frac.json): ").strip()
        if not config_file:
            config_file = "afm_config_nm_frac.json"
        check_type = input("Check epsilon (e) or conductivity (c)? (default e): ").strip().lower() or "e"
        run_sanity_check(phi_file=phi_file, config_file=config_file, check_type=check_type)

    elif choice == "4":
        from postprocessing.lever_arm_calc import main as lever_arm_main
        lever_arm_main()

    elif choice == "5":
        from postprocessing.field_lines import interactive_main as field_lines_main
        field_lines_main()

    elif choice == "6":
        from postprocessing.potential_map import interactive_main as potential_map_main
        potential_map_main()
        
    elif choice == "7":
        from postprocessing.capacitance_sanity_check import main as sanity_multi_main
        sanity_multi_main()


    else:
        return


def main():
    """Main entry point for the AFM Simulation Package.

    Supports command-line mode (presim/simulation/postprocessing) and
    interactive menu mode.

    Returns
    -------
    None
    """
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode in ("presim", "presimulation", "1"):
            run_presimulation()
        elif mode in ("sim", "simulation", "2"):
            config_base = sys.argv[2] if len(sys.argv) > 2 else "afm_config"
            from simulation.main_loop import batch_main
            batch_main(config_base)
        elif mode in ("post", "postprocessing", "3"):
            run_postprocessing()
        else:
            print(f"Unknown mode: {mode}")
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
