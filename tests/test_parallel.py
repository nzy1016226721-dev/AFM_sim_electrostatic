import numpy as np

from simulation.parallel import (
    configure_cpu_threads,
    residual_scalars_faces_parallel,
    residual_scalars_parallel,
    weighted_jacobi_residual_parallel,
    weighted_jacobi_step_parallel,
    weighted_jacobi_update_faces_parallel,
)
from simulation.solver import _build_dielectric_face_fields


def _case(shape=(8, 7, 6), seed=20260829):
    rng = np.random.default_rng(seed)
    field = rng.normal(size=shape).astype(np.float32)
    mask = rng.random(shape) < 0.12
    core = tuple(size - 2 for size in shape)
    coeff = [rng.uniform(0.5, 4.0, size=core).astype(np.float32)
             for _ in range(6)]
    return field, mask, coeff


def test_parallel_jacobi_matches_numpy_reference():
    field, _, coeff = _case()
    axp, axm, ayp, aym, azp, azm = coeff
    a0 = sum(coeff)
    numerator = (
        axp * field[2:, 1:-1, 1:-1]
        + axm * field[:-2, 1:-1, 1:-1]
        + ayp * field[1:-1, 2:, 1:-1]
        + aym * field[1:-1, :-2, 1:-1]
        + azp * field[1:-1, 1:-1, 2:]
        + azm * field[1:-1, 1:-1, :-2]
    )
    expected_new = numerator / a0
    expected_residual = expected_new - field[1:-1, 1:-1, 1:-1]
    got_new = np.empty_like(expected_new)
    got_residual = np.empty_like(expected_residual)

    configure_cpu_threads(2, slurm_cap=False)
    weighted_jacobi_step_parallel(
        field, axp, axm, ayp, aym, azp, azm, a0,
        np.float32(1.0), got_new, got_residual,
    )

    np.testing.assert_allclose(got_new, expected_new, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        got_residual, expected_residual, rtol=2e-6, atol=2e-6
    )


def test_parallel_residual_scalars_match_numpy_reference():
    field, mask, coeff = _case(seed=19)
    axp, axm, ayp, aym, azp, azm = coeff
    a0 = sum(coeff)
    numerator = (
        axp * field[2:, 1:-1, 1:-1]
        + axm * field[:-2, 1:-1, 1:-1]
        + ayp * field[1:-1, 2:, 1:-1]
        + aym * field[1:-1, :-2, 1:-1]
        + azp * field[1:-1, 1:-1, 2:]
        + azm * field[1:-1, 1:-1, :-2]
    )
    residual = numerator - a0 * field[1:-1, 1:-1, 1:-1]
    free = ~mask[1:-1, 1:-1, 1:-1]
    expected_l2 = np.sqrt(np.mean(residual[free] ** 2))
    expected_max = np.max(np.abs(residual[free]))

    got_l2, got_max = residual_scalars_parallel(
        field, mask, axp, axm, ayp, aym, azp, azm, a0
    )

    np.testing.assert_allclose(got_l2, expected_l2, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(got_max, expected_max, rtol=2e-6, atol=2e-6)


def test_parallel_residual_only_kernel_matches_reference():
    field, _, coeff = _case(seed=23)
    axp, axm, ayp, aym, azp, azm = coeff
    a0 = sum(coeff)
    numerator = (
        axp * field[2:, 1:-1, 1:-1]
        + axm * field[:-2, 1:-1, 1:-1]
        + ayp * field[1:-1, 2:, 1:-1]
        + aym * field[1:-1, :-2, 1:-1]
        + azp * field[1:-1, 1:-1, 2:]
        + azm * field[1:-1, 1:-1, :-2]
    )
    expected = numerator / a0 - field[1:-1, 1:-1, 1:-1]
    got = np.empty_like(expected)

    configure_cpu_threads(2, slurm_cap=False)
    weighted_jacobi_residual_parallel(
        field, axp, axm, ayp, aym, azp, azm, a0, got
    )
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=2e-6)


def test_low_memory_parallel_update_matches_explicit_float32_order():
    field, _, coeff = _case(seed=41)
    axp, axm, ayp, aym, azp, azm = coeff
    omega = np.float32(0.8)
    expected = np.full_like(field, np.nan)
    for i in range(field.shape[0] - 2):
        for j in range(field.shape[1] - 2):
            for k in range(field.shape[2] - 2):
                cs = [value[i, j, k] for value in coeff]
                denom = np.float32(cs[0] + cs[1])
                for value in cs[2:]:
                    denom = np.float32(denom + value)
                neighbors = (
                    field[i + 2, j + 1, k + 1],
                    field[i, j + 1, k + 1],
                    field[i + 1, j + 2, k + 1],
                    field[i + 1, j, k + 1],
                    field[i + 1, j + 1, k + 2],
                    field[i + 1, j + 1, k],
                )
                numerator = np.float32(cs[0] * neighbors[0])
                for coefficient, neighbor in zip(cs[1:], neighbors[1:]):
                    numerator = np.float32(
                        numerator + np.float32(coefficient * neighbor)
                    )
                old = field[i + 1, j + 1, k + 1]
                candidate = np.float32(numerator / denom)
                residual = np.float32(candidate - old)
                update = np.float32(omega * residual)
                expected[i + 1, j + 1, k + 1] = np.float32(old + update)

    got = np.full_like(field, np.nan)
    configure_cpu_threads(2, slurm_cap=False)
    weighted_jacobi_update_faces_parallel(
        field, got, axp, axm, ayp, aym, azp, azm, omega
    )
    np.testing.assert_array_equal(got[1:-1, 1:-1, 1:-1], expected[1:-1, 1:-1, 1:-1])


def test_low_memory_residual_scalars_match_persistent_denominator_path():
    field, mask, coeff = _case(seed=43)
    axp, axm, ayp, aym, azp, azm = coeff
    a0 = np.empty_like(axp)
    np.add(axp, axm, out=a0)
    for value in (ayp, aym, azp, azm):
        np.add(a0, value, out=a0)

    expected = residual_scalars_parallel(
        field, mask, axp, axm, ayp, aym, azp, azm, a0
    )
    got = residual_scalars_faces_parallel(
        field, mask, axp, axm, ayp, aym, azp, azm
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_oriented_face_fields_match_legacy_coefficients_bit_for_bit():
    rng = np.random.default_rng(29)
    ec = rng.uniform(1.0, 15.0, size=(9, 8, 7)).astype(np.float32)
    x_faces, y_faces, z_faces = _build_dielectric_face_fields(ec)

    expected_x = 0.25 * (
        ec[:, :-1, :-1] + ec[:, 1:, :-1]
        + ec[:, :-1, 1:] + ec[:, 1:, 1:]
    )
    expected_y = 0.25 * (
        ec[:-1, :, :-1] + ec[1:, :, :-1]
        + ec[:-1, :, 1:] + ec[1:, :, 1:]
    )
    expected_z = 0.25 * (
        ec[:-1, :-1, :] + ec[1:, :-1, :]
        + ec[:-1, 1:, :] + ec[1:, 1:, :]
    )

    np.testing.assert_array_equal(x_faces, expected_x)
    np.testing.assert_array_equal(y_faces, expected_y)
    np.testing.assert_array_equal(z_faces, expected_z)
