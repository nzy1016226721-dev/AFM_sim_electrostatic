"""The maintained JSON entry points default to lossless RAM storage."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from simulation.config_defaults import apply_solver_defaults
from simulation.coordinates import normalize_config
from simulation.mpi_config import load_afm_config
from simulation.presimulation import generate_tip_offset_configs
from simulation.ram_first import resolve_storage_options
from simulation.runtime import resolve_plotting_enabled


ROOT = Path(__file__).resolve().parents[1]


def test_missing_keys_choose_compact_without_mutating_input():
    config = {"plotting": {"disable_in_non_ide": False}, "blocks_nm": []}
    before = deepcopy(config)
    effective = apply_solver_defaults(config)
    assert config == before
    assert effective["memory_mode"] == "ram_compact"
    assert effective["phi_update_mode"] == "in_place"
    assert effective["residual_accumulation"] == "scalar"
    assert effective["plotting"]["enabled"] is False
    assert effective["plotting"]["disable_in_non_ide"] is False
    assert apply_solver_defaults(effective) == effective


def test_physical_dict_normalization_has_same_defaults():
    cfg = normalize_config({"grid_resolution": {"nx": 8, "ny": 8, "nz": 8},
                            "voxel_nm3": 1.0, "blocks_nm": []})
    assert cfg["memory_mode"] == "ram_compact"
    assert resolve_storage_options(cfg["memory_mode"], cfg["phi_update_mode"],
                                   cfg["residual_accumulation"]) == ("in_place", "scalar")


@pytest.mark.parametrize("mode,expected", [("standard", None), ("ram_first", "row_array"),
                                         ("ram_compact", "scalar")])
def test_explicit_modes_remain_available(mode, expected):
    effective = apply_solver_defaults({"memory_mode": mode})
    assert effective["memory_mode"] == mode
    assert effective.get("residual_accumulation") == expected


def test_inheritance_resolves_before_policy_defaults(tmp_path):
    base = tmp_path / "base.json"
    child = tmp_path / "child.json"
    base.write_text(json.dumps({"blocks_nm": [], "memory_mode": "ram_compact",
                               "phi_update_mode": "in_place", "residual_accumulation": "scalar"}), encoding="utf-8")
    child.write_text(json.dumps({"extends": "base.json", "memory_mode": "standard",
                                "phi_update_mode": None, "residual_accumulation": None,
                                "solver_dtype": "float64"}), encoding="utf-8")
    _, effective = load_afm_config(child, normalize=False)
    assert effective["memory_mode"] == "standard"
    assert effective["solver_dtype"] == "float64"
    assert effective["phi_update_mode"] is None
    resolve_storage_options(effective["memory_mode"], effective["phi_update_mode"],
                            effective["residual_accumulation"])


def test_raw_loader_applies_defaults_to_an_old_config(tmp_path):
    path = tmp_path / "old.json"
    before = {"blocks_nm": [], "v_start": -1, "v_stop": -1, "v_step": 1}
    path.write_text(json.dumps(before), encoding="utf-8")
    _, cfg = load_afm_config(path, normalize=False)
    assert cfg["memory_mode"] == "ram_compact"
    assert json.loads(path.read_text(encoding="utf-8")) == before


def test_explicit_higher_ram_selectors_are_respected():
    cfg = apply_solver_defaults({"phi_update_mode": "buffered", "residual_accumulation": "array"})
    assert resolve_storage_options(cfg["memory_mode"], cfg["phi_update_mode"],
                                   cfg["residual_accumulation"]) == ("buffered", "array")


def test_generated_config_records_ram_policy_and_key_order(tmp_path):
    source = tmp_path / "base.json"
    source.write_text(json.dumps({"tip_z_nm": 10, "cpu_threads": 8, "blocks_nm": []}), encoding="utf-8")
    generated = generate_tip_offset_configs(source, [0.0])
    cfg = json.loads(Path(generated[0]).read_text(encoding="utf-8"))
    assert cfg["memory_mode"] == "ram_compact"
    assert cfg["phi_update_mode"] == "in_place"
    assert cfg["residual_accumulation"] == "scalar"
    assert cfg["plotting"]["enabled"] is False
    assert list(cfg)[0] == "cpu_threads" and list(cfg)[-1] == "blocks_nm"


def test_ram_default_stays_headless_even_in_an_ide(monkeypatch):
    monkeypatch.setenv("SPYDER_KERNEL_ID", "test-only")
    assert resolve_plotting_enabled({}) is False
    assert resolve_plotting_enabled({"memory_mode": "standard"}) is True
    with pytest.raises(ValueError, match="local_post.py"):
        resolve_plotting_enabled({"plotting": {"enabled": True}})
    assert resolve_plotting_enabled({"plotting": {"enabled": True}}, cli_override=False) is False


def test_all_shipped_configs_record_ram_policy_or_named_float64_exception():
    paths = sorted(ROOT.glob("*.json")) + sorted((ROOT / "tests/data").glob("*.json"))
    paths += sorted((ROOT / "configs").glob("*.json"))
    assert len(paths) >= 33
    exceptions = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        _, cfg = load_afm_config(path)
        assert list(raw)[0] == "cpu_threads", path
        for key in ("blocks_nm", "blocks"):
            if key in raw:
                assert list(raw)[-1] == key, path
        assert cfg["plotting"]["enabled"] is False, path
        if "float64_memory_smoke" in path.name:
            exceptions.append(path.name)
            assert cfg["memory_mode"] == "standard"
            assert cfg["solver_dtype"] == "float64"
            assert cfg["phi_update_mode"] is cfg["residual_accumulation"] is None
        else:
            assert cfg["memory_mode"] == "ram_compact", path
            assert cfg["phi_update_mode"] == "in_place", path
            assert cfg["residual_accumulation"] == "scalar", path
            assert cfg.get("solver_dtype", "float32") == "float32", path
    assert len(exceptions) == 2
