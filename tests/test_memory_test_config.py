import json
from copy import deepcopy
from pathlib import Path

from simulation.mpi_config import load_afm_config


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_all_package_jsons_use_qd_centered_100nm_cut():
    paths = sorted(PACKAGE_ROOT.glob("*.json"))
    assert paths
    for path in paths:
        # Compact run variants may inherit the package-wide cut policy. Test
        # the effective configuration rather than only literal JSON keys.
        _, config = load_afm_config(path)
        assert config["save_cut"] is True, path.name
        assert config["save_cut_reference"] == "movement_center", path.name
        assert config["save_cut_box_nm"] == [
            -50.0, 50.0, -50.0, 50.0, -20.0, 80.0
        ], path.name


def test_memory_test_config_is_bounded_and_resource_aligned():
    config = json.loads((PACKAGE_ROOT / "afm_memory_test.json").read_text(encoding="utf-8"))
    policy = config["memory_test"]

    assert policy["enabled"] is True
    assert policy["levels"] == [8192]
    assert max(policy["levels"]) == policy["max_level"] == 8192
    assert config["res_tol_main"] == 1e-4
    assert config["res_tol_zoom"] == 1e-4
    assert config["mg_max_runtime"] == 1200.0
    assert config["cpu_threads"] == 192
    assert config["initial_grid_level"] == 4096
    assert config["memory_live_interval_seconds"] == 1.0
    assert policy["expected_tolerance"] == 1e-4
    assert policy["expected_runtime_limit_seconds"] == 1200.0
    assert policy["initial_grid_level"] == 4096
    assert policy["live_rss_interval_seconds"] == 1.0
    assert config["save_full"] is False
    assert config["save_cut"] is True
    assert config["save_cut_reference"] == "movement_center"
    assert config["save_cut_box_nm"] == [-50.0, 50.0, -50.0, 50.0, -20.0, 80.0]
    assert config["zoom_simulation"]["enabled"] is False
    assert config["plotting"]["enabled"] is False
    assert config["memory_tracking"] is True


def test_memory_test_job_uses_fir_high_memory_node_and_level_diagnostics():
    script = (PACKAGE_ROOT / "jobs" / "run_memory_test.sh").read_text(encoding="utf-8")

    assert "#SBATCH --partition=cpularge_bynode_b1" in script
    assert "#SBATCH --cpus-per-task=192" in script
    assert "#SBATCH --mem=4096G" in script
    assert "#SBATCH --time=01:00:00" in script
    assert "#SBATCH --account=rrg-hongguo-ad" in script
    assert "target_level=${level}" in script
    assert "[MEMORY-TEST][START]" in script
    assert "[MEMORY-TEST][FAIL]" in script
    assert "current_level.txt" in script
    assert "had_failure=0" in script
    assert "AFM_MEMORY_CONTINUE_AFTER_FAILURE" in script
    assert "AFM_MEMORY_CHILD_DEADLINE_SECONDS" in script
    assert "[MEMORY-TEST][TIME-BUDGET]" in script
    assert "timeout --signal=TERM" in script
    assert "AFM_JOB_OUTPUT_ROOT=\"${output_base}/job_${SLURM_JOB_ID}\"" in script


def test_current_2048_serial_mpi_gate_has_matched_cut_only_physics():
    serial_path = PACKAGE_ROOT / "afm_config_nm_serial_mpi_comparison_serial_2048.json"
    mpi_path = PACKAGE_ROOT / "afm_config_nm_serial_mpi_comparison_mpi_2048.json"
    _, serial = load_afm_config(serial_path)
    _, mpi = load_afm_config(mpi_path)

    assert serial["grid_resolution"] == {"nx": 2048, "ny": 2048, "nz": 2048}
    assert serial["cpu_threads"] == 96
    assert mpi["cpu_threads"] == 2
    assert serial["res_tol_main"] == mpi["res_tol_main"] == 1e-5
    assert serial["save_full"] is mpi["save_full"] is False
    assert serial["save_cut"] is mpi["save_cut"] is True
    assert serial["movement"]["start"] == serial["movement"]["end"]
    assert mpi["movement"]["start"] == mpi["movement"]["end"]
    assert mpi["mpi"]["process_grid"] == [4, 4, 4]
    assert mpi["mpi"]["residual_check_interval"] == 10

    serial_science = deepcopy(serial)
    mpi_science = deepcopy(mpi)
    serial_science.pop("cpu_threads")
    mpi_science.pop("cpu_threads")
    mpi_science.pop("mpi")
    assert serial_science == mpi_science


def test_current_2048_gate_resources_preserve_four_gib_per_cpu():
    serial_script = (
        PACKAGE_ROOT / "jobs" / "run_afm_serial_mpi_reference_2048.sh"
    ).read_text(encoding="utf-8")
    mpi_script = (
        PACKAGE_ROOT / "jobs" / "run_afm_serial_mpi_candidate_2048.sh"
    ).read_text(encoding="utf-8")
    compare_script = (
        PACKAGE_ROOT / "jobs" / "run_compare_serial_mpi_current_2048.sh"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --nodes=1" in serial_script
    assert "#SBATCH --cpus-per-task=96" in serial_script
    assert "#SBATCH --mem-per-cpu=4G" in serial_script
    assert "#SBATCH --time=01:00:00" in serial_script
    assert "#SBATCH --nodes=2" in mpi_script
    assert "#SBATCH --ntasks=64" in mpi_script
    assert "#SBATCH --ntasks-per-node=32" in mpi_script
    assert "#SBATCH --cpus-per-task=2" in mpi_script
    assert "#SBATCH --mem-per-cpu=4G" in mpi_script
    assert "#SBATCH --time=01:00:00" in mpi_script
    assert "--atol 0 --rtol 0" in compare_script
    assert "--atol 1e-6 --rtol 0" in compare_script
    assert "cut_from_grid2048x2048x2048.npy" in compare_script
