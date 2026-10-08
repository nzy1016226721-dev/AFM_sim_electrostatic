import csv

import numpy as np
import pytest

from simulation.main_loop import run_afm_simulation
from simulation.numerics import (
    diagnostic_iteration_limit,
    prepare_drive_voltages,
    resolve_solver_dtype,
)


def _small_run(tmp_path, name, voltage, **overrides):
    options = {
        "Vtip": voltage,
        "nx": 8,
        "ny": 8,
        "nz": 8,
        "tip_z": 0.2,
        "R": 0.08,
        "r_tip": 0.25,
        "damping": 1.0,
        "tol": 1e-4,
        "verbose": False,
        "eps": True,
        "blocks": [],
        "Vgate": [{
            "x_range": [0, 1],
            "y_range": [0, 1],
            "z_range": [0, 0.01],
            "Vgate_val": 0.0,
        }],
        "output_dir": tmp_path / name,
        "plotting_enabled": False,
        "memory_tracking": False,
        "eps_reference_resolution": 8,
        "cpu_threads": 2,
        "return_residual": False,
        "initial_grid_level": 8,
    }
    options.update(overrides)
    return run_afm_simulation(**options)


def test_relative_drive_solves_unit_field_and_restores_physical_voltage(tmp_path):
    unit = _small_run(tmp_path, "unit", -1.0)
    scaled = _small_run(
        tmp_path,
        "scaled",
        -9.0,
        residual_tolerance_mode="relative_drive",
    )

    expected = np.multiply(unit["phi"], np.float32(9.0))
    np.testing.assert_array_equal(scaled["phi"], expected)
    assert scaled["parameters"]["drive_scale_volts"] == 9.0
    assert scaled["parameters"]["residual_tolerance_mode"] == "relative_drive"


def test_float64_diagnostic_runs_only_requested_iterations(tmp_path):
    output_dir = tmp_path / "float64_smoke"
    result = _small_run(
        tmp_path,
        "float64_smoke",
        -1.0,
        nx=16,
        ny=16,
        nz=16,
        initial_grid_level=8,
        solver_dtype="float64",
        tol=1e-30,
        diagnostic={
            "mode": "float64_memory_smoke",
            "iterations_by_level": {"8": 2, "16": 3},
        },
    )
    assert result["phi"].dtype == np.float64
    with (output_dir / "float64_memory_smoke_levels.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["requested_iterations"]) for row in rows] == [2, 3]
    assert [int(row["completed_iterations"]) for row in rows] == [2, 3]
    assert all(row["result"] == "diagnostic_iteration_limit" for row in rows)


def test_fixed_iterations_are_rejected_outside_float64_smoke():
    with pytest.raises(ValueError, match="requires solver_dtype"):
        diagnostic_iteration_limit(
            {
                "mode": "float64_memory_smoke",
                "iterations_by_level": {"8": 2},
            },
            (8, 8, 8),
            solver_dtype=resolve_solver_dtype("float32"),
        )


def test_relative_drive_scales_tip_and_all_gate_voltages():
    tip, gates, scale = prepare_drive_voltages(
        -9.0,
        [{"Vgate_val": 3.0}, {"Vgate_val": -4.5}],
        "relative_drive",
    )
    assert scale == 9.0
    assert tip == -1.0
    assert [gate["Vgate_val"] for gate in gates] == pytest.approx([1 / 3, -0.5])

