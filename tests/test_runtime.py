import json

import pytest

from simulation.runtime import (
    discover_afm_configs,
    resolve_config_path,
    resolve_output_dir,
)


def _config():
    return {
        "blocks_nm": [],
        "v_start": -1,
        "v_stop": -1,
        "v_step": 1,
    }


def test_explicit_config_resolution_and_discovery(tmp_path):
    config = tmp_path / "afm_config_nm.json"
    config.write_text(json.dumps(_config()), encoding="utf-8")
    assert resolve_config_path(config) == str(config.resolve())
    assert discover_afm_configs(tmp_path) == [str(config.resolve())]


def test_extensionless_config_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        resolve_config_path("afm_config_nm", directory=tmp_path)


def test_slurm_output_override_isolated_by_config(tmp_path, monkeypatch):
    root = tmp_path / "job_42"
    monkeypatch.setenv("AFM_JOB_OUTPUT_ROOT", str(root))
    resolved = resolve_output_dir("outputs", "afm_config_nm.json", {})
    assert resolved == str(root / "afm_config_nm")


def test_local_output_keeps_json_default_without_batch_override(monkeypatch):
    monkeypatch.delenv("AFM_JOB_OUTPUT_ROOT", raising=False)
    assert resolve_output_dir("outputs", "afm_config_nm.json", {}) == "outputs"
