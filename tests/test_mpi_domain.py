import io
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import zoom

from simulation.materials import _fine_eps, generate_eps_level
from simulation.main_loop import default_initial_grid_level
from simulation.mpi_config import load_mpi_config
from simulation.mpi_domain import (
    _prolongation_axis,
    _jacobi_owned,
    _epsilon_halo_from_global,
    build_distributed_epsilon_halo,
    build_local_face_fields,
    build_local_tip_mask,
    estimate_rank_peak_bytes,
    partition_axis,
    rank_layout_table,
    rasterize_epsilon_halo,
    suggest_process_grid,
    validate_process_grid,
)
from simulation.mpi_io import numpy_header_bytes, physical_cut_slices
from simulation.solver import _build_dielectric_face_fields, build_downward_pointing_tip


def test_axis_partitions_cover_without_overlap():
    pieces = [partition_axis(17, 4, coordinate) for coordinate in range(4)]
    assert pieces == [(0, 5), (5, 4), (9, 4), (13, 4)]
    covered = [index for start, count in pieces for index in range(start, start + count)]
    assert covered == list(range(17))


def test_default_initial_grid_level_keeps_large_hierarchies_between_six_and_eight_levels():
    def levels(target, initial):
        current = initial
        count = 1
        while current < target:
            current = min(current * 2, target)
            count += 1
        return count

    assert default_initial_grid_level(512, 512, 512) == 8
    assert default_initial_grid_level(2048, 1024, 2048) == 64
    assert default_initial_grid_level((2049, 2048, 1024)) == 64
    assert default_initial_grid_level(4096, 4096, 4096) == 64
    assert default_initial_grid_level(8192, 8192, 8192) == 64
    assert default_initial_grid_level(16384, 16384, 16384) == 128
    assert default_initial_grid_level(32768, 32768, 32768) == 256
    for target in (2048, 2049, 4096, 8192, 16384, 32768):
        assert 6 <= levels(target, default_initial_grid_level(target)) <= 8


def test_process_grid_suggestion_is_balanced_and_complete():
    dims = suggest_process_grid(64, (16384, 16384, 16384))
    assert np.prod(dims) == 64
    validate_process_grid(dims, 64, (16384, 16384, 16384))
    assert max(dims) / min(dims) <= 2


def test_rank_layout_reports_every_rank_and_positive_memory():
    rows = rank_layout_table((65, 66, 67), (2, 2, 2))
    assert len(rows) == 8
    assert {row["rank"] for row in rows} == set(range(8))
    assert all(row["estimated_peak_bytes"] > 0 for row in rows)
    assert max(row["estimated_peak_bytes"] for row in rows) == estimate_rank_peak_bytes((33, 33, 34))


def test_local_epsilon_halo_matches_global_direct_rasterization():
    global_shape = (9, 8, 7)
    blocks = [
        {"eps_val": 2.0, "x_range": [0.0, 1.0], "y_range": [0.0, 0.5], "z_range": [0.0, 1.0]},
        {"eps_val": 7.0, "x_range": [0.25, 0.75], "y_range": [0.25, 1.0], "z_range": [0.2, 0.8]},
    ]
    full = _fine_eps((8, 7, 6), blocks)
    starts = (2, 1, 2)
    counts = (4, 4, 3)
    local = rasterize_epsilon_halo(global_shape, starts, counts, blocks)
    np.testing.assert_array_equal(local, full[1:6, 0:5, 1:5])


def test_volume_averaged_epsilon_halo_matches_serial_production_level():
    global_shape = (9, 8, 7)
    blocks = [
        {"eps_val": 2.5, "x_range": [0.0, 1.0], "y_range": [0.0, 0.47], "z_range": [0.0, 1.0]},
        {"eps_val": 11.7, "x_range": [0.23, 0.81], "y_range": [0.31, 1.0], "z_range": [0.17, 0.76]},
    ]
    full = generate_eps_level(global_shape, blocks, reference_shape=(32, 32, 32))
    starts = (2, 1, 2)
    counts = (4, 4, 3)
    local = _epsilon_halo_from_global(full, starts, counts)
    np.testing.assert_array_equal(local, full[1:6, 0:5, 1:5])

    # The regression is meaningful only if this geometry exposes the old MPI
    # behaviour: direct coarse rasterization must differ at material interfaces.
    direct = rasterize_epsilon_halo(global_shape, starts, counts, blocks)
    assert not np.array_equal(local, direct)


def test_single_rank_distributed_epsilon_and_faces_match_serial_production():
    class SingleRankComm:
        def bcast(self, value, root=0):
            return value

        def gather(self, value, root=0):
            return [value]

    global_shape = (9, 8, 7)
    blocks = [
        {"eps_val": 3.2, "x_range": [0.0, 1.0], "y_range": [0.0, 0.53], "z_range": [0.0, 1.0]},
        {"eps_val": 13.1, "x_range": [0.19, 0.77], "y_range": [0.28, 1.0], "z_range": [0.13, 0.74]},
    ]
    decomposition = SimpleNamespace(
        cart=SingleRankComm(),
        rank=0,
        global_shape=global_shape,
        starts=(0, 0, 0),
        counts=global_shape,
    )
    halo = build_distributed_epsilon_halo(
        decomposition, blocks, reference_shape=(32, 32, 32)
    )
    full_eps = generate_eps_level(
        global_shape, blocks, reference_shape=(32, 32, 32)
    )
    serial_x, serial_y, serial_z = _build_dielectric_face_fields(full_eps)
    local_x, local_y, local_z = build_local_face_fields(halo)
    nx, ny, nz = global_shape
    np.testing.assert_array_equal(local_x[2:nx, 1:ny-1, 1:nz-1], serial_x[1:, :, :])
    np.testing.assert_array_equal(local_x[1:nx-1, 1:ny-1, 1:nz-1], serial_x[:-1, :, :])
    np.testing.assert_array_equal(local_y[1:nx-1, 2:ny, 1:nz-1], serial_y[:, 1:, :])
    np.testing.assert_array_equal(local_y[1:nx-1, 1:ny-1, 1:nz-1], serial_y[:, :-1, :])
    np.testing.assert_array_equal(local_z[1:nx-1, 1:ny-1, 2:nz], serial_z[:, :, 1:])
    np.testing.assert_array_equal(local_z[1:nx-1, 1:ny-1, 1:nz-1], serial_z[:, :, :-1])


def test_local_face_coefficients_match_serial_interior_coefficients():
    global_shape = (9, 8, 7)
    blocks = [
        {"eps_val": 12.5, "x_range": [0.0, 1.0], "y_range": [0.0, 1.0], "z_range": [0.0, 0.4]},
        {"eps_val": 15.0, "x_range": [0.3, 0.7], "y_range": [0.2, 0.8], "z_range": [0.4, 0.7]},
    ]
    full_eps = _fine_eps(tuple(v - 1 for v in global_shape), blocks)
    serial_x, serial_y, serial_z = _build_dielectric_face_fields(full_eps)
    halo = rasterize_epsilon_halo(global_shape, (0, 0, 0), global_shape, blocks)
    local_x, local_y, local_z = build_local_face_fields(halo)

    nx, ny, nz = global_shape
    np.testing.assert_array_equal(local_x[2:nx, 1:ny-1, 1:nz-1], serial_x[1:, :, :])
    np.testing.assert_array_equal(local_x[1:nx-1, 1:ny-1, 1:nz-1], serial_x[:-1, :, :])
    np.testing.assert_array_equal(local_y[1:nx-1, 2:ny, 1:nz-1], serial_y[:, 1:, :])
    np.testing.assert_array_equal(local_y[1:nx-1, 1:ny-1, 1:nz-1], serial_y[:, :-1, :])
    np.testing.assert_array_equal(local_z[1:nx-1, 1:ny-1, 2:nz], serial_z[:, :, 1:])
    np.testing.assert_array_equal(local_z[1:nx-1, 1:ny-1, 1:nz-1], serial_z[:, :, :-1])


def test_distributed_jacobi_kernel_matches_numpy_reference_for_owned_interior():
    rng = np.random.default_rng(42)
    counts = (7, 6, 5)
    current = rng.normal(size=tuple(v + 2 for v in counts)).astype(np.float32)
    target = np.empty_like(current)
    mask = np.zeros(counts, dtype=bool)
    mask[3, 2, 2] = True
    eps = rng.uniform(1.0, 15.0, size=tuple(v + 1 for v in counts)).astype(np.float32)
    x_faces, y_faces, z_faces = build_local_face_fields(eps)
    omega = np.float32(0.8)

    _jacobi_owned(
        current,
        target,
        mask,
        x_faces,
        y_faces,
        z_faces,
        0,
        0,
        0,
        *counts,
        omega,
    )
    expected = current.copy()
    for i in range(1, counts[0] - 1):
        for j in range(1, counts[1] - 1):
            for k in range(1, counts[2] - 1):
                if mask[i, j, k]:
                    continue
                axm, axp = x_faces[i, j, k], x_faces[i + 1, j, k]
                aym, ayp = y_faces[i, j, k], y_faces[i, j + 1, k]
                azm, azp = z_faces[i, j, k], z_faces[i, j, k + 1]
                denom = axp
                denom = denom + axm
                denom = denom + ayp
                denom = denom + aym
                denom = denom + azp
                denom = denom + azm
                numerator = axp * current[i + 2, j + 1, k + 1]
                numerator = numerator + axm * current[i, j + 1, k + 1]
                numerator = numerator + ayp * current[i + 1, j + 2, k + 1]
                numerator = numerator + aym * current[i + 1, j, k + 1]
                numerator = numerator + azp * current[i + 1, j + 1, k + 2]
                numerator = numerator + azm * current[i + 1, j + 1, k]
                value = np.float32(numerator / denom)
                residual = np.float32(value - current[i + 1, j + 1, k + 1])
                update = np.float32(omega * residual)
                expected[i + 1, j + 1, k + 1] = np.float32(
                    current[i + 1, j + 1, k + 1] + update
                )
    np.testing.assert_array_equal(
        target[2:-2, 2:-2, 2:-2], expected[2:-2, 2:-2, 2:-2]
    )


def test_distributed_tip_mask_matches_serial_mask_in_every_partition():
    shape = (65, 63, 61)
    physical = {
        "domain_nm": (200.0, 180.0, 100.0),
        "origin_fraction": (0.5, 0.5, 0.0),
        "tip_z_nm": 45.0,
        "R_nm": 5.0,
        "r_tip_nm": 10000.0,
    }
    serial, _, _ = build_downward_pointing_tip(
        *shape,
        tip_z=0.2,
        R=0.05,
        r_tip=0.15,
        aspect_ratio=4.0,
        verbose=False,
        tip_z_nm=physical["tip_z_nm"],
        R_nm=physical["R_nm"],
        r_tip_nm=physical["r_tip_nm"],
        domain_nm=physical["domain_nm"],
        center_fraction=physical["origin_fraction"],
    )
    for cx in range(2):
        for cy in range(2):
            for cz in range(2):
                pieces = [
                    partition_axis(n, 2, coordinate)
                    for n, coordinate in zip(shape, (cx, cy, cz))
                ]
                starts = tuple(item[0] for item in pieces)
                counts = tuple(item[1] for item in pieces)
                decomposition = SimpleNamespace(
                    global_shape=shape,
                    starts=starts,
                    counts=counts,
                )
                local, _, _ = build_local_tip_mask(
                    decomposition,
                    tip_z=0.2,
                    R=0.05,
                    r_tip=0.15,
                    aspect_ratio=4.0,
                    physical_params=physical,
                )
                expected = serial[tuple(slice(s, s + c) for s, c in zip(starts, counts))]
                np.testing.assert_array_equal(local, expected)


def test_prolongation_matches_scipy_and_never_reads_upper_physical_ghost():
    rng = np.random.default_rng(73)
    for old_n, new_n in ((8, 16), (9, 17), (16, 25)):
        coarse_owned = rng.normal(size=(old_n, old_n, old_n)).astype(np.float32)
        coarse_halo = np.full(
            (old_n + 2, old_n + 2, old_n + 2), np.nan, dtype=np.float32
        )
        coarse_halo[1:-1, 1:-1, 1:-1] = coarse_owned
        low, weights = _prolongation_axis(old_n, new_n, 0, new_n, 0, old_n)
        fine = np.empty((new_n, new_n, new_n), dtype=np.float32)
        from simulation.mpi_domain import _trilinear_prolongate

        _trilinear_prolongate(
            coarse_halo,
            fine,
            low,
            weights,
            low,
            weights,
            low,
            weights,
        )
        reference = zoom(
            coarse_owned, (new_n / old_n,) * 3, order=1
        )
        assert np.isfinite(fine).all()
        np.testing.assert_array_equal(fine[-1, -1, -1], coarse_owned[-1, -1, -1])
        np.testing.assert_array_equal(fine, reference)


def test_numpy_header_can_be_loaded_without_allocating_global_shape(tmp_path):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    path = tmp_path / "distributed.npy"
    path.write_bytes(numpy_header_bytes(data.shape) + data.tobytes(order="C"))
    np.testing.assert_array_equal(np.load(path), data)


def test_physical_cut_slices_match_expected_voxel_box():
    slices, bounds = physical_cut_slices(
        (100, 100, 100),
        (0.0, 0.0, 20.0),
        (-10.0, 10.0, -20.0, 20.0, -20.0, 30.0),
        (-50.0, 50.0, -50.0, 50.0, 0.0, 100.0),
    )
    assert tuple(s.stop - s.start for s in slices) == (20, 40, 50)
    assert bounds == (-10.0, 10.0, -20.0, 20.0, 0.0, 50.0)


def test_mpi_template_extends_production_geometry():
    _, cfg = load_mpi_config("afm_config_nm_mpi_16384_template.json")
    assert cfg["grid_resolution"] == {"nx": 16384, "ny": 16384, "nz": 16384}
    assert cfg["mpi"]["process_grid"] == [16, 16, 16]
    assert "initial_grid_level" not in cfg["mpi"]
    assert default_initial_grid_level(tuple(cfg["grid_resolution"].values())) == 128
    assert cfg["mpi"]["require_convergence"] is True
    assert cfg["cpu_threads"] == 12
    assert cfg["zoom_simulation"]["enabled"] is False
    assert len(cfg["blocks"]) > 10


def test_mpi_2048_trial_restricts_only_the_first_production_case():
    _, cfg = load_mpi_config("afm_config_nm_mpi_trial_2048.json")
    assert cfg["grid_resolution"] == {"nx": 2048, "ny": 2048, "nz": 2048}
    assert (cfg["v_start"], cfg["v_stop"], cfg["v_step"]) == (-1.0, -1.0, 0.5)
    assert cfg["movement"]["start"] == cfg["movement"]["end"]
    assert cfg["cpu_threads"] == 2
    assert cfg["mpi"]["process_grid"] == [4, 4, 4]
    assert "initial_grid_level" not in cfg["mpi"]
    assert cfg["save_full"] is False
    assert cfg["save_cut"] is True
    assert len(cfg["blocks"]) > 10


def test_16384_template_refinement_needs_only_one_coarse_halo():
    dims = (16, 16, 16)
    levels = (512, 1024, 2048, 4096, 8192, 16384)
    for old_n, new_n in zip(levels, levels[1:]):
        for parts in dims:
            for coordinate in range(parts):
                coarse_start, coarse_count = partition_axis(old_n, parts, coordinate)
                fine_start, fine_count = partition_axis(new_n, parts, coordinate)
                _prolongation_axis(
                    old_n,
                    new_n,
                    fine_start,
                    fine_count,
                    coarse_start,
                    coarse_count,
                )


def test_4096_comparison_layouts_resolve_to_the_same_scientific_case():
    paths = (
        "afm_config_nm_mpi_comparison_4096_layout_8x8x8.json",
        "afm_config_nm_mpi_comparison_4096_layout_16x8x4.json",
    )
    configs = [load_mpi_config(path)[1] for path in paths]
    for cfg in configs:
        assert cfg["grid_resolution"] == {"nx": 4096, "ny": 4096, "nz": 4096}
        assert cfg["cpu_threads"] == 1
        assert cfg["res_tol_main"] == 1e-6
        assert cfg["v_start"] == cfg["v_stop"] == -1.0
        assert cfg["movement"]["start"] == cfg["movement"]["end"]
        assert cfg["zoom_simulation"]["enabled"] is False
        assert cfg["save_full"] is False
        assert cfg["save_cut"] is True
        assert "initial_grid_level" not in cfg["mpi"]
        assert default_initial_grid_level(tuple(cfg["grid_resolution"].values())) == 64
        assert cfg["mpi"]["residual_check_interval"] == 10
        assert np.prod(cfg["mpi"]["process_grid"]) == 512

    left = dict(configs[0])
    right = dict(configs[1])
    left_mpi = dict(left.pop("mpi"))
    right_mpi = dict(right.pop("mpi"))
    left_mpi.pop("process_grid")
    right_mpi.pop("process_grid")
    assert left == right
    assert left_mpi == right_mpi
