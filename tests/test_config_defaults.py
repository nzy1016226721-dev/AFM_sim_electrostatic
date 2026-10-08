import json

import numpy as np

from simulation.coordinates import normalize_config, ordered_config_for_json
from simulation.io_utils import (
    gate_slices,
    make_gate_mask,
    movement_suffix_nm,
    resolution_tagged_cut_filename,
    save_potential_physical_cut,
)
from simulation.materials import generate_eps_level
from simulation.presimulation import generate_tip_offset_configs
from postprocessing.npy_utils import parse_phi_filename


def _physical_config(**overrides):
    cfg = {
        "voxel_nm3": 1.0,
        "grid_resolution": {"nx": 16, "ny": 20, "nz": 24},
        "v_start": -1.0,
        "v_stop": -1.0,
        "v_step": 1.0,
        "blocks_nm": [],
    }
    cfg.update(overrides)
    return cfg


def test_partial_physical_objects_default_missing_axes_to_full_domain():
    cfg = normalize_config(_physical_config(
        blocks_nm=[{"eps_val": 12.5, "z_range_nm": [0.0, 2.0]}],
        Vgate_nm=[{"z_range_nm": [0.0, 0.0]}],
    ))

    assert cfg["blocks"][0]["x_range"] == [0.0, 1.0]
    assert cfg["blocks"][0]["y_range"] == [0.0, 1.0]
    assert cfg["Vgate"][0]["x_range"] == [0.0, 1.0]
    assert cfg["Vgate"][0]["y_range"] == [0.0, 1.0]
    assert cfg["Vgate"][0]["z_range"] == [0.0, 0.0]
    assert cfg["Vgate"][0]["Vgate_val"] == 0.0


def test_direct_gate_helpers_default_missing_axes_to_full_domain():
    gate = {"z_range": [0.0, 0.0]}
    slices = gate_slices(16, 20, 24, gate)
    mask = make_gate_mask(16, 20, 24, gate)

    assert slices == (slice(0, 16), slice(0, 20), slice(0, 1))
    assert int(mask.sum()) == 16 * 20
    assert mask[:, :, 0].all()
    assert not mask[:, :, 1].any()


def test_epsilon_block_with_only_z_range_covers_full_x_and_y():
    eps = generate_eps_level(
        (8, 9, 10),
        blocks=[{"eps_val": 12.5, "z_range": [0.0, 1.0]}],
        reference_shape=(8, 8, 8),
    )

    assert eps.dtype == np.float32
    assert eps.shape == (7, 8, 9)
    assert np.all(eps == np.float32(12.5))


def test_zoom_cut_missing_axes_default_to_full_domain():
    cfg = normalize_config(_physical_config(
        zoom_simulation={
            "enabled": True,
            "zoom_factor": 2,
            "zoom_limit": 2,
            "cut": {"z_range": [0.0, 0.5]},
        }
    ))
    cut = cfg["zoom_simulation"]["cut"]
    assert cut["x_range"] == [0.0, 1.0]
    assert cut["y_range"] == [0.0, 1.0]
    assert cut["z_range"] == [0.0, 0.5]


def test_empty_saved_cut_selects_the_full_field(tmp_path):
    phi = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    path, bounds = save_potential_physical_cut(
        phi,
        center_nm=(0.0, 0.0, 0.0),
        box_offsets_nm=[],
        field_bounds_nm=(-2.0, 2.0, -3.0, 3.0, 0.0, 2.0),
        filename="full_cut.npy",
        output_dir=tmp_path,
    )

    np.testing.assert_array_equal(np.load(path), phi)
    assert bounds == (-2.0, 2.0, -3.0, 3.0, 0.0, 2.0)


def test_movement_centered_100nm_cut_offsets_z_to_zero_through_100(tmp_path):
    movement_center = (40.0, -30.0, 20.0)
    phi = np.arange(20 * 20 * 20, dtype=np.float32).reshape(20, 20, 20)
    filename = resolution_tagged_cut_filename(
        "afm_phi_1_0nm_-9.00V.npy", phi.shape
    )
    path, bounds = save_potential_physical_cut(
        phi,
        center_nm=movement_center,
        box_offsets_nm=[-50.0, 50.0, -50.0, 50.0, -20.0, 80.0],
        field_bounds_nm=(-100.0, 100.0, -100.0, 100.0, 0.0, 200.0),
        filename=filename,
        output_dir=tmp_path,
    )

    assert str(path).endswith(
        "afm_phi_1_0nm_-9.00V_"
        "cut_from_grid20x20x20.npy"
    )
    assert np.load(path).shape == (10, 10, 10)
    assert bounds == (-10.0, 90.0, -80.0, 20.0, 0.0, 100.0)


def test_movement_suffix_uses_unrounded_signed_physical_nm_offsets():
    domain_nm = (100.0, 100.0, 100.0)
    center = (0.5, 0.5, 0.2)

    assert movement_suffix_nm(center, center, domain_nm) == "_0nm"
    assert movement_suffix_nm((0.50001, 0.5, 0.2), center, domain_nm) == "_0.001nm"
    assert movement_suffix_nm((0.49999, 0.5, 0.2), center, domain_nm) == "-0.001nm"
    assert movement_suffix_nm((0.50002, 0.5, 0.2), center, domain_nm) == "_0.002nm"


def test_current_nm_movement_filename_metadata_is_parseable():
    name = resolution_tagged_cut_filename(
        "afm_phi_3_0.001nm_-9.00V.npy",
        (512, 256, 128),
    )
    assert parse_phi_filename(name) == {
        "type": "normal",
        "config_idx": 3,
        "Vtip": -9.0,
        "movement_offset_nm": 0.001,
        "is_cut": True,
        "source_grid": (512, 256, 128),
    }

    zoom_name = resolution_tagged_cut_filename(
        "afm_phi_zoom_4x_-1.0V_3-0.001nm.npy",
        (512, 512, 512),
    )
    assert parse_phi_filename(zoom_name) == {
        "type": "zoom",
        "config_idx": 3,
        "Vtip": -1.0,
        "mag": 4,
        "movement_offset_nm": -0.001,
        "is_cut": True,
        "source_grid": (512, 512, 512),
    }


def test_resolution_tagged_cut_filename_retains_parseable_metadata():
    name = resolution_tagged_cut_filename(
        "afm_phi_3_cx0.70_cy0.80_cz0.08_-9.00V.npy",
        (512, 256, 128),
    )
    assert parse_phi_filename(name) == {
        "type": "normal",
        "config_idx": 3,
        "Vtip": -9.0,
        "position": (0.7, 0.8, 0.08),
        "is_cut": True,
        "source_grid": (512, 256, 128),
    }

    zoom_name = resolution_tagged_cut_filename(
        "afm_phi_zoom_4x_-1.0V_3_cx0.70_cy0.80_cz0.08.npy",
        (512, 512, 512),
    )
    assert parse_phi_filename(zoom_name) == {
        "type": "zoom",
        "config_idx": 3,
        "Vtip": -1.0,
        "mag": 4,
        "position": (0.7, 0.8, 0.08),
        "is_cut": True,
        "source_grid": (512, 512, 512),
    }


def test_generated_json_is_cpu_first_and_blocks_last(tmp_path):
    source = tmp_path / "base.json"
    source.write_text(json.dumps({
        "tip_z_nm": 10.0,
        "blocks_nm": [{"eps_val": 1.0}],
        "output_dir": "outputs",
        "presimulation": {"tip_z_offsets_nm": [0.0]},
    }), encoding="utf-8")

    generated = generate_tip_offset_configs(source)
    written = json.loads((tmp_path / "base_0nm.json").read_text(encoding="utf-8"))
    assert generated == [str(tmp_path / "base_0nm.json")]
    assert list(written)[0] == "cpu_threads"
    assert list(written)[-1] == "blocks_nm"
    assert written["cpu_threads"] == 1


def test_ordering_helper_does_not_mutate_input():
    source = {
        "blocks_nm": [],
        "value": {"nested": True},
        "cpu_threads": 4,
    }
    ordered = ordered_config_for_json(source)
    assert list(ordered) == ["cpu_threads", "value", "blocks_nm"]
    assert source == {
        "blocks_nm": [],
        "value": {"nested": True},
        "cpu_threads": 4,
    }
