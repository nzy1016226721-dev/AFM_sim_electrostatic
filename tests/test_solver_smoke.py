import csv

import numpy as np

from simulation.main_loop import run_afm_simulation


def test_small_serial_and_parallel_runs_are_equivalent(tmp_path):
    common = {
        "Vtip": -1.0,
        "nx": 16,
        "ny": 16,
        "nz": 16,
        "tip_z": 0.2,
        "R": 0.08,
        "r_tip": 0.25,
        "damping": 0.8,
        "max_iter": None,
        "tol": 1e-5,
        "mg_max_runtime": 10.0,
        "verbose": False,
        "eps": True,
        "blocks": [],
        "Vgate": [{
            "x_range": [0, 1],
            "y_range": [0, 1],
            "z_range": [0, 0.01],
            "Vgate_val": 0.0,
        }],
        "plotting_enabled": False,
        "memory_tracking": False,
        "eps_reference_resolution": 16,
    }
    serial = run_afm_simulation(
        output_dir=tmp_path / "serial", cpu_threads=1, **common
    )
    parallel = run_afm_simulation(
        output_dir=tmp_path / "parallel", cpu_threads=2, **common
    )

    np.testing.assert_allclose(serial["phi"], parallel["phi"], atol=2e-6)
    np.testing.assert_array_equal(serial["tip_mask"], parallel["tip_mask"])
    np.testing.assert_array_equal(
        serial["boundary_mask"], parallel["boundary_mask"]
    )


def test_legacy_max_iter_does_not_stop_a_time_limited_solver(tmp_path):
    """A legacy iteration setting must not truncate the current solver."""
    output_dir = tmp_path / "time_only"
    run_afm_simulation(
        Vtip=-1.0,
        nx=8,
        ny=8,
        nz=8,
        tip_z=0.2,
        R=0.08,
        r_tip=0.25,
        damping=0.8,
        max_iter=1,
        tol=1e-12,
        mg_max_runtime=1.0,
        verbose=False,
        eps=True,
        blocks=[],
        Vgate=[{
            "x_range": [0, 1],
            "y_range": [0, 1],
            "z_range": [0, 0.01],
            "Vgate_val": 0.0,
        }],
        output_dir=output_dir,
        plotting_enabled=False,
        memory_tracking=False,
        eps_reference_resolution=8,
        initial_grid_level=8,
    )
    with (output_dir / "residual_history.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert int(rows[-1]["iteration"]) > 1
