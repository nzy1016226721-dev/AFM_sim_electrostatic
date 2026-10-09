"""Public-interface checks for the adapted, local-only plotting merge."""
import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest

from local_visualization.data import PotentialData as LocalData
from postprocessing.potential_plot_data import load_potential
from postprocessing.bulk_potential_plotter import bulk_potential_plots, plot_potential_lines
from postprocessing.single_plane_plotter import single_plane_plotter
from postprocessing.line_profile_plotter import InteractiveLineProfile, sample_segment
from simulation.output_coordinates import write_coordinate_receipt


@pytest.fixture
def saved_cut(tmp_path):
    config = tmp_path / "exact.json"
    record = dict(grid_resolution=dict(nx=16, ny=12, nz=8), voxel_nm3=2.,
                  coordinate_system=dict(origin_fraction=[.5, .5, 0.]),
                  movement=dict(start_nm=[0., 0., 4.], end_nm=[0., 0., 4.], spacing_nm=1.),
                  save_cut=True, save_full=False, save_cut_box_nm=[-4., 4., -6., 6., -2., 6.],
                  blocks_nm=[], v_start=-1., v_stop=-1., v_step=1.)
    config.write_text(json.dumps(record), encoding="utf-8")
    axes = (-16 + np.arange(6, 10)*32/15, -12 + np.arange(3, 9)*24/11, np.arange(1, 5)*16/7)
    values = (.002*axes[0][:, None, None] + .003*axes[1][None, :, None]
              + .004*axes[2][None, None, :] - 1).astype(np.float32)
    path = tmp_path / "afm_phi_1_0nm_-1.00V_cut_from_grid16x12x8.npy"
    np.save(path, values)
    write_coordinate_receipt(path, (16, 12, 8), (-16, 16, -12, 12, 0, 16),
                             (slice(6, 10), slice(3, 9), slice(1, 5)))
    return config, path


def test_receipt_only_reader_and_detached_selected_data(saved_cut):
    _, path = saved_cut
    with load_potential(path) as volume:
        mapped = volume.array
        x, line = volume.line("x", (-3, 5))
        plane, extent, labels = volume.plane("xy", 5)
        assert not np.shares_memory(line, mapped)
        assert not np.shares_memory(plane, mapped)
        assert volume.node_bounds_nm != volume.bounds_nm
        assert labels == ("x", "y") and extent[0] < x[0]
    assert mapped._mmap.closed
    assert np.isfinite(line).all() and np.isfinite(plane).all()
    volume.close()  # Idempotent, including after context exit.


def test_same_shape_wrong_json_rejected_without_overriding_receipt(saved_cut):
    config, path = saved_cut
    original = hashlib.sha256(path_receipt(path).read_bytes()).hexdigest()
    wrong = json.loads(config.read_text())
    wrong["coordinate_system"]["origin_fraction"][0] = .515625
    wrong["save_cut_box_nm"][:2] = [-4.5, 3.5]
    config.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_potential(path, config_path=config)
    with load_potential(path) as volume:
        assert volume.centres[0][0] == pytest.approx(-3.2)
    assert hashlib.sha256(path_receipt(path).read_bytes()).hexdigest() == original


def path_receipt(path):
    return path.with_name(path.name + ".coords.json")


def test_missing_receipt_requires_explicit_node_bounds_even_with_json(saved_cut):
    config, path = saved_cut
    bounds = json.loads(path_receipt(path).read_text())["node_bounds_nm"]
    path_receipt(path).unlink()
    with pytest.raises(ValueError, match="No coordinate receipt"):
        load_potential(path, config_path=config)
    with load_potential(path, config_path=config, bounds_nm=bounds) as volume:
        assert volume.node_bounds_nm == pytest.approx(bounds)


@pytest.mark.parametrize("entry,value", [
    ("shape", [2, 2, 2]), ("potential_units", "mV"),
    ("node_bounds_nm", [0, 1, 0, 1, 0, 1]), ("source_index_slices", [[6, 11], [3, 9], [1, 5]]),
    ("shape", None), ("source_shape", None), ("source_index_slices", [None, None, None]),
])
@pytest.mark.parametrize("reader", [LocalData, load_potential])
def test_malformed_receipt_rejected_by_both_interfaces(saved_cut, entry, value, reader):
    _, path = saved_cut
    record = json.loads(path_receipt(path).read_text())
    record[entry] = value
    path_receipt(path).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        reader(path)


def test_explicit_bounds_cannot_silently_override_existing_receipt(saved_cut):
    _, path = saved_cut
    for reader in (LocalData, load_potential):
        with pytest.raises(ValueError, match="does not match"):
            reader(path, bounds_nm=[0, 1, 0, 1, 0, 1])


@pytest.mark.parametrize("variable", ["SLURM_JOB_ID", "SLURM_JOBID", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MPI_LOCALNRANKS"])
def test_all_new_plotting_entry_points_reject_mpi_and_slurm(saved_cut, monkeypatch, variable):
    config, path = saved_cut
    monkeypatch.setenv(variable, "1")
    for action in (lambda: load_potential(path), lambda: plot_potential_lines([path], config, axis="z", fixed_nm=(0, 0)),
                   lambda: single_plane_plotter(path, config, plane="xy", at_nm=5), bulk_potential_plots):
        with pytest.raises(RuntimeError):
            action()


def test_invalid_dtype_and_failed_config_validation_close_mapping(saved_cut, monkeypatch):
    config, path = saved_cut
    mappings = []
    real_load = np.load
    def observed_load(*args, **kwargs):
        result = real_load(*args, **kwargs)
        mappings.append(result)
        return result
    monkeypatch.setattr(np, "load", observed_load)
    record = json.loads(config.read_text())
    record["voxel_nm3"] = 3
    config.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        load_potential(path, config_path=config)
    np.save(path, np.zeros((4, 6, 4), np.int32))
    with pytest.raises(ValueError, match="floating-point"):
        load_potential(path)
    assert mappings and all(array._mmap.closed for array in mappings)


def test_menu_saves_without_overwriting_existing_png(saved_cut, monkeypatch):
    _, path = saved_cut
    destination = path.parent / "saved_plots"
    destination.mkdir()
    old = destination / "afm_line_000.png"
    old.write_bytes(b"existing-figure")
    answers = iter([str(path.parent), "0", "", "line", "z", "0", "0", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(plt, "show", lambda: None)
    figures = bulk_potential_plots()
    try:
        assert old.read_bytes() == b"existing-figure"
        assert (destination / "afm_line_000_1.png").stat().st_size > 1000
    finally:
        for figure in figures:
            plt.close(figure)


@pytest.mark.parametrize("plane,position", [("xy", 5), ("xz", -3), ("yz", 0)])
def test_single_plane_is_detached_metadata_only_and_read_only(saved_cut, plane, position):
    config, path = saved_cut
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    fig, ax, metadata = single_plane_plotter(path, config, plane=plane, at_nm=position)
    try:
        assert isinstance(metadata, dict) and "actual_nm" in metadata
        assert np.isfinite(ax.images[0].get_array()).all()
    finally:
        plt.close(fig)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original


def test_profile_sampler_is_bounded_finite_and_float64():
    plane = np.arange(9, dtype=np.float32).reshape(3, 3)
    distance, values = sample_segment(plane, (0, 6, 0, 6), (1, 1), (5, 5), n=1_000_001)
    assert len(values) == 100000 and values.dtype == np.float64
    assert distance[-1] == pytest.approx(np.sqrt(32))
    with pytest.raises(ValueError):
        sample_segment(plane, (0, 6, 0, 6), (np.nan, 1), (5, 5))


def test_close_disconnects_callback_and_outside_click_is_handled(saved_cut):
    config, path = saved_cut
    fig, ax, _ = single_plane_plotter(path, config, plane="xy", at_nm=5)
    handler = InteractiveLineProfile(fig, ax)
    first = SimpleNamespace(inaxes=ax, xdata=-4.2, ydata=-6.4)
    last = SimpleNamespace(inaxes=ax, xdata=3., ydata=5.)
    handler.on_click(first)
    handler.on_click(last)
    assert handler.last_profile_figure is None and handler.points == []
    fig.canvas.callbacks.process("close_event", SimpleNamespace(canvas=fig.canvas))
    assert handler._cid is None
    handler.disconnect()
    plt.close(fig)


def test_closed_reader_methods_raise_instead_of_accessing_unmapped_memory(saved_cut):
    _, path = saved_cut
    for reader in (LocalData, load_potential):
        volume = reader(path)
        volume.close()
        with pytest.raises(ValueError, match="closed"):
            volume.plane("xy", 5)


@pytest.mark.parametrize("offset", [-2, 0, 2])
def test_signed_movement_with_inherited_config_and_non_cubic_receipt(saved_cut, offset):
    config, path = saved_cut
    raw = json.loads(config.read_text())
    raw["movement"]["end_nm"] = [offset, 0, 4]
    config.write_text(json.dumps(raw), encoding="utf-8")
    child = path.parent / "child.json"
    child.write_text(json.dumps({"extends":config.name}), encoding="utf-8")
    suffix = f"-{abs(offset)}nm" if offset < 0 else f"_{offset}nm"
    moved = path.parent / f"afm_phi_9{suffix}_-1.00V_cut_from_grid16x12x8.npy"
    np.save(moved, np.zeros((4, 6, 4), np.float32))
    first = 6 + offset//2
    write_coordinate_receipt(moved, (16, 12, 8), (-16, 16, -12, 12, 0, 16),
                             (slice(first,first+4),slice(3,9),slice(1,5)))
    with load_potential(moved, config_path=child) as volume:
        assert volume.centres[0][0] == pytest.approx(-16+first*32/15)


def test_partial_multifile_failure_closes_figures_and_every_mapping(saved_cut, monkeypatch):
    config, path = saved_cut
    mappings = []
    original_load = np.load
    def tracked(*args, **kwargs):
        mapped = original_load(*args, **kwargs)
        mappings.append(mapped)
        return mapped
    monkeypatch.setattr(np, "load", tracked)
    from postprocessing.bulk_potential_plotter import plot_potential_planes
    existing = set(plt.get_fignums())
    with pytest.raises(FileNotFoundError):
        plot_potential_planes([path,path.parent/"missing.npy"],config,plane="xy",at_nm=5)
    assert set(plt.get_fignums())==existing
    assert mappings and all(mapped._mmap.closed for mapped in mappings)


@pytest.mark.parametrize("module", ["potential_plot_data", "bulk_potential_plotter", "single_plane_plotter", "line_profile_plotter"])
def test_fresh_plotting_import_rejects_mpi_environment(module):
    env = os.environ.copy()
    env["PMI_SIZE"] = "2"
    result = subprocess.run([sys.executable,"-B","-c",f"import postprocessing.{module}"],
                            env=env,capture_output=True,text=True)
    assert result.returncode!=0 and "MPI launcher" in result.stderr


def test_post_menu_rejects_mpi_before_any_input(monkeypatch):
    import run_all
    monkeypatch.setenv("PMI_SIZE","2")
    monkeypatch.setattr("builtins.input",lambda *args: pytest.fail("Cluster visualization must not prompt"))
    with pytest.raises(RuntimeError,match="MPI launcher"):
        run_all.main(["post"])


def test_simulation_launcher_help_stays_headless_under_mpi():
    env = os.environ.copy()
    env["PMI_SIZE"] = "2"
    result = subprocess.run([sys.executable,"-B","run_all.py","--help"],env=env,capture_output=True,text=True)
    assert result.returncode==0 and "AFM simulation launcher" in result.stdout


@pytest.mark.parametrize("kind",["line","planes","single","profile"])
def test_display_failure_does_not_leave_partial_figures(saved_cut,monkeypatch,kind):
    from postprocessing.bulk_potential_plotter import plot_potential_lines,plot_potential_planes
    from postprocessing.line_profile_plotter import line_profile_plotter
    config,path = saved_cut
    existing = set(plt.get_fignums())
    def failed_display():
        raise RuntimeError("test display failure")
    monkeypatch.setattr(plt,"show",failed_display)
    with pytest.raises(RuntimeError,match="test display failure"):
        if kind=="line":
            plot_potential_lines([path],config,axis="z",fixed_nm=(0,0),show=True)
        elif kind=="planes":
            plot_potential_planes([path],config,plane="xy",at_nm=5,show=True)
        elif kind=="single":
            single_plane_plotter(path,config,plane="xy",at_nm=5,show=True)
        else:
            line_profile_plotter(path,config,plane="xy",at_nm=5,show=True)
    assert set(plt.get_fignums())==existing
