"""Pyramid tip cross-section tests.

The default ``"pyramid"`` tip shape replaces the legacy circular cone disk
with a square whose sides are tangent to the cone, keeping the apex curvature
radius ``R`` for the vertical-edge fillets.  ``"cone"`` preserves the legacy
disk exactly.
"""

from types import SimpleNamespace

import numpy as np

from simulation.coordinates import normalize_config
from simulation.mpi_domain import build_local_tip_mask, partition_axis
from simulation.numerics import (
    pyramid_tip_corner_radius,
    pyramid_tip_half_side,
    pyramid_tip_inside,
    resolve_tip_shape,
)
from simulation.solver import build_downward_pointing_tip


def _pyramid_mask(n=65, R=0.04, r_tip=0.20, aspect_ratio=2.0, tip_z=0.45):
    mask, z_tip, z_base = build_downward_pointing_tip(
        n, n, n,
        tip_z=tip_z, R=R, r_tip=r_tip, aspect_ratio=aspect_ratio,
        verbose=False, tip_shape="pyramid",
    )
    return mask, z_tip, z_base


def _cone_mask(n=65, R=0.04, r_tip=0.20, aspect_ratio=2.0, tip_z=0.45):
    mask, z_tip, z_base = build_downward_pointing_tip(
        n, n, n,
        tip_z=tip_z, R=R, r_tip=r_tip, aspect_ratio=aspect_ratio,
        verbose=False, tip_shape="cone",
    )
    return mask, z_tip, z_base


def test_resolve_tip_shape_defaults_and_rejects():
    assert resolve_tip_shape(None) == "pyramid"
    assert resolve_tip_shape("") == "pyramid"
    assert resolve_tip_shape("pyramid") == "pyramid"
    assert resolve_tip_shape("Cone") == "cone"
    for bad in ("square", "0", "cylinder"):
        try:
            resolve_tip_shape(bad)
        except ValueError:
            continue
        raise AssertionError(f"resolve_tip_shape accepted {bad!r}")


def test_normalize_config_defaults_tip_shape_to_pyramid():
    cfg = normalize_config({"grid_resolution": {"nx": 32, "ny": 32, "nz": 32},
                            "voxel_nm3": 0.5})
    assert cfg["tip_shape"] == "pyramid"


def test_pyramid_base_slice_is_square_with_rounded_corners():
    n, R, r_tip = 65, 0.04, 0.20
    mask, _, _ = _pyramid_mask(n=n, R=R, r_tip=r_tip)
    filled = [k for k in range(n) if mask[:, :, k].any()]
    assert filled
    sl = mask[:, :, filled[-1]]
    xs = np.where(sl.any(axis=1))[0]
    ys = np.where(sl.any(axis=0))[0]
    # Square: equal x/y extents, half-side == r_tip within one voxel.
    assert np.ptp(xs) == np.ptp(ys)
    half = np.ptp(xs) / 2.0 / (n - 1)
    assert abs(half - r_tip) <= 1.0 / (n - 1) + 1e-9
    cx = cy = (n - 1) // 2
    ex = xs.max() - cx
    # Edge midpoint is inside, the sharp square corner is cut by the fillet.
    assert sl[cx + ex, cy]
    assert not sl[cx + ex, cy + ex]


def test_pyramid_near_apex_matches_cone_disk():
    n, R, r_tip = 65, 0.04, 0.20
    pyr, _, _ = _pyramid_mask(n=n, R=R, r_tip=r_tip)
    cone, _, _ = _cone_mask(n=n, R=R, r_tip=r_tip)
    # First filled slice above the apex: half-side s < R so the fillet
    # consumes the flats and the section must equal the cone disk.
    filled = [k for k in range(n) if pyr[:, :, k].any()]
    assert filled
    np.testing.assert_array_equal(pyr[:, :, filled[0]], cone[:, :, filled[0]])


def test_pyramid_z_base_matches_cone_formula():
    n, R, r_tip, aspect = 65, 0.04, 0.20, 2.0
    _, _, z_base = _pyramid_mask(n=n, R=R, r_tip=r_tip, aspect_ratio=aspect)
    theta = np.arctan(aspect)
    a = R * np.tan(theta)
    b = R * np.tan(theta) ** 2
    tip_idx = int(np.clip(0.45, 0, 1) * (n - 1))
    z_tip = tip_idx / (n - 1)
    expected = (z_tip - b) + np.sqrt(b ** 2 * (1 + (r_tip ** 2 / a ** 2)))
    assert abs(z_base - expected) < 1e-9


def test_pyramid_side_center_slope_matches_cone():
    # At side-plane centres the pyramid meets the cone, so ds/dz == drho/dz.
    R, aspect = 0.04, 2.0
    theta = np.arctan(aspect)
    a = R * np.tan(theta)
    b = R * np.tan(theta) ** 2
    for dz in (0.25, 0.50, 0.80):
        rho = a * np.sqrt((dz ** 2 / b ** 2) - 1.0)
        slope = a * dz / (b ** 2 * np.sqrt((dz ** 2 / b ** 2) - 1.0))
        s = pyramid_tip_half_side(rho, 10.0)
        assert s == rho
        h = 1e-6
        numeric = (pyramid_tip_half_side(a * np.sqrt(((dz + h) ** 2 / b ** 2) - 1.0), 10.0) - s) / h
        assert abs(numeric - slope) < 1e-3


def test_pyramid_differs_from_cone_only_at_corners():
    n, R, r_tip = 65, 0.04, 0.20
    pyr, _, _ = _pyramid_mask(n=n, R=R, r_tip=r_tip)
    cone, _, _ = _cone_mask(n=n, R=R, r_tip=r_tip)
    assert not np.array_equal(pyr, cone)
    # The square circumscribes the disk: every cone voxel is inside the
    # pyramid, and the flats add voxels the disk lacks.
    assert np.all(~cone | pyr)
    assert np.any(pyr & ~cone)


def test_pyramid_serial_matches_distributed_with_domain_sized_truncation():
    shape = (33, 31, 29)
    physical = {
        "domain_nm": (64.0, 64.0, 64.0),
        "origin_fraction": (0.5, 0.5, 0.0),
        "tip_z_nm": 20.0,
        "R_nm": 2.0,
        "r_tip_nm": 8.0,
    }
    serial, _, _ = build_downward_pointing_tip(
        *shape,
        tip_z=0.2, R=0.05, r_tip=0.15, aspect_ratio=2.0, verbose=False,
        tip_z_nm=physical["tip_z_nm"],
        R_nm=physical["R_nm"],
        r_tip_nm=physical["r_tip_nm"],
        domain_nm=physical["domain_nm"],
        center_fraction=physical["origin_fraction"],
        tip_shape="pyramid",
    )
    assert serial.any()
    # The square base must be wider than the cone disk somewhere.
    cone, _, _ = build_downward_pointing_tip(
        *shape,
        tip_z=0.2, R=0.05, r_tip=0.15, aspect_ratio=2.0, verbose=False,
        tip_z_nm=physical["tip_z_nm"],
        R_nm=physical["R_nm"],
        r_tip_nm=physical["r_tip_nm"],
        domain_nm=physical["domain_nm"],
        center_fraction=physical["origin_fraction"],
        tip_shape="cone",
    )
    assert np.any(serial & ~cone)
    for cz in range(2):
        for cy in range(2):
            for cx in range(2):
                pieces = [partition_axis(n, 2, c) for n, c in zip(shape, (cx, cy, cz))]
                starts = tuple(p[0] for p in pieces)
                counts = tuple(p[1] for p in pieces)
                decomp = SimpleNamespace(global_shape=shape, starts=starts, counts=counts)
                local, _, _ = build_local_tip_mask(
                    decomp, tip_z=0.2, R=0.05, r_tip=0.15, aspect_ratio=2.0,
                    physical_params=physical, tip_shape="pyramid",
                )
                np.testing.assert_array_equal(
                    local, serial[tuple(slice(s, s + c) for s, c in zip(starts, counts))]
                )


def test_pyramid_helper_corner_radius_clamps_to_half_side():
    assert pyramid_tip_corner_radius(0.01, 0.05) == 0.01
    assert pyramid_tip_corner_radius(0.10, 0.05) == 0.05
    assert pyramid_tip_half_side(0.03, 0.20) == 0.03
    assert pyramid_tip_half_side(0.30, 0.20) == 0.20
