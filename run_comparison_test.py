"""Run and analyse the legacy-vs-parallel AFM alignment comparison.

This is deliberately a sequential runner.  A 512^3 solve can consume most of
the available memory on a laptop, so running the legacy and parallel legs one
at a time gives a useful comparison without creating two simultaneous solver
peaks.  The runner writes temporary configs and all reports below one run
directory; it does not modify either package's source configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates


LEVELS = (128, 256, 512)
VOLTAGES = (-1.0, -9.0)


def _json_dump(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def _set_common(cfg: dict[str, Any], output_dir: Path, threads: int,
                single_position: bool, zoom_enabled: bool,
                clamp: bool | None = None) -> dict[str, Any]:
    """Apply only run controls; retain the source geometry and material data."""
    cfg = json.loads(json.dumps(cfg))
    cfg["output_dir"] = str(output_dir.resolve())
    cfg["save_all_levels"] = True
    cfg["save_full"] = True
    cfg["cpu_threads"] = int(threads)
    cfg["res_tol_main"] = 5e-6
    cfg["res_tol_zoom"] = 5e-6
    cfg["v_start"] = -1.0
    cfg["v_stop"] = -9.0
    cfg["v_step"] = 8.0
    cfg["plotting"] = {"enabled": False}
    cfg["plotting_enabled"] = False
    cfg["plot_zoom_residuals"] = False
    cfg["memory_tracking"] = False
    zoom = cfg.setdefault("zoom_simulation", {})
    zoom["enabled"] = bool(zoom_enabled)
    zoom["plotting_enabled"] = False
    zoom["plot_residuals"] = False
    zoom["res_tol"] = 5e-6
    if clamp is not None:
        zoom["clamp"] = bool(clamp)
    if single_position:
        movement = cfg.setdefault("movement", {})
        if "start_nm" in movement:
            movement["end_nm"] = list(movement["start_nm"])
        else:
            movement["end"] = list(movement.get("start", [0.5, 0.5, 0.5]))
    return cfg


def _make_old_config(source: Path, dest: Path, output_dir: Path,
                     threads: int, single_position: bool) -> None:
    cfg = json.loads(source.read_text(encoding="utf-8"))
    cfg = _set_common(cfg, output_dir, threads, single_position, False)
    # The old package consumes fractional geometry and scans afm_config_N.json.
    # Keep the old file shape and key ordering, while making its single
    # alignment position identical to the first position of the physical file.
    _json_dump(dest / "afm_config_1.json", cfg)


def _make_new_config(source: Path, dest: Path, output_dir: Path,
                     threads: int, single_position: bool,
                     zoom_enabled: bool, clamp: bool | None = None) -> Path:
    cfg = json.loads(source.read_text(encoding="utf-8"))
    cfg = _set_common(cfg, output_dir, threads, single_position,
                      zoom_enabled, clamp)
    path = dest / ("config_zoom.json" if zoom_enabled else "config_main.json")
    _json_dump(path, cfg)
    return path


# This code is executed inside each package's own Python process.  In
# addition to disabling GUI work, it wraps mg_3d_masked so every multigrid
# call gets a separate residual CSV.  The normal solver still writes its
# usual aggregate residual_history.csv as well.
CHILD_COMMON = r'''
import csv, os, sys
from pathlib import Path

PACK = Path(sys.argv[1]).resolve()
STAGE = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(PACK))
os.chdir(STAGE)
os.environ.setdefault("MPLBACKEND", "Agg")

def _rows(path):
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if rows and rows[0] and rows[0][0].lower() in {"iteration", "iter"}:
        return rows[:1], rows[1:]
    return [], rows

def _wrap(kind, real, counter):
    def wrapped(*args, **kwargs):
        out = kwargs.get("output_dir") or os.environ.get("AFM_CAPTURE_OUTPUT", ".")
        out = Path(out)
        residual = out / "residual_history.csv"
        before_header, before = _rows(residual)
        result = real(*args, **kwargs)
        after_header, after = _rows(residual)
        # The solver appends rows.  If a future implementation rotates the
        # file, retaining the complete post-call rows is still more useful
        # than silently losing the convergence trace.
        delta = after[len(before):] if len(after) >= len(before) else after
        if delta:
            shape = "unknown"
            for obj in args:
                shp = getattr(obj, "shape", None)
                if shp is not None and len(shp) == 3:
                    shape = "x".join(str(int(v)) for v in shp)
                    break
            target = out / (f"residual_{kind}_call{counter[0]:03d}_{shape}.csv")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(after_header[0] if after_header else
                                 ["iteration", "avg_residual", "max_residual"])
                writer.writerows(delta)
            counter[0] += 1
        return result
    return wrapped

def _disable_plots(module):
    class _Fig:
        axes = []
    module.preview_tip_only = lambda *a, **k: None
    module.preview_before_run = lambda *a, **k: True
    module.plot_phi_plane = lambda *a, **k: _Fig()
    module.plot_residual_plane = lambda *a, **k: _Fig()

try:
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None
    plt.close = lambda *a, **k: None
except Exception:
    pass
'''


def _old_code() -> str:
    return CHILD_COMMON + r'''
import simulation.main_loop as main_loop
import simulation.solver as solver
_disable_plots(main_loop)
solver.plot_convergence = lambda *a, **k: None
counter_main = [0]
main_loop.mg_3d_masked = _wrap("main", main_loop.mg_3d_masked, counter_main)
try:
    import simulation.zoom as zoom
    if hasattr(zoom, "mg_3d_masked"):
        zoom.mg_3d_masked = _wrap("zoom", zoom.mg_3d_masked, [0])
except Exception:
    pass

import builtins
answers = iter(["y", ""])
builtins.input = lambda *a, **k: next(answers, "")
main_loop.batch_main("afm_config")
'''


def _new_code(config_path: Path) -> str:
    return CHILD_COMMON + f'''
import simulation.main_loop as main_loop
counter_main = [0]
main_loop.mg_3d_masked = _wrap("main", main_loop.mg_3d_masked, counter_main)
try:
    import simulation.zoom as zoom
    if hasattr(zoom, "mg_3d_masked"):
        zoom.mg_3d_masked = _wrap("zoom", zoom.mg_3d_masked, [0])
except Exception:
    pass
_disable_plots(main_loop)
main_loop.batch_main({str(config_path.resolve())!r}, config_dir={str(config_path.parent.resolve())!r},
                     plotting_override=False, interactive=False)
'''


def _zoom_code(config_path: Path, main_output: Path) -> str:
    """Run layered zooms only, reusing the completed main-grid NPYs."""
    return CHILD_COMMON + f'''
import json
import numpy as np
from simulation.coordinates import normalize_config
import simulation.zoom as zoom

_disable_plots(zoom)
counter_zoom = [0]
zoom.mg_3d_masked = _wrap("zoom", zoom.mg_3d_masked, counter_zoom)
with open({str(config_path.resolve())!r}, "r", encoding="utf-8") as f:
    cfg = normalize_config(json.load(f))
center = tuple(cfg.get("movement", {{}}).get("start", [0.5, 0.5, 0.5]))
main_output = Path({str(main_output.resolve())!r})
zoom_output = Path(os.environ.get("AFM_CAPTURE_OUTPUT", ".")).resolve()
zoom_output.mkdir(parents=True, exist_ok=True)
for V in (-1.0, -9.0):
    candidates = sorted(main_output.glob(f"afm_phi_1_{{V:.2f}}V.npy"))
    if not candidates:
        raise FileNotFoundError(f"main field missing for {{V}} V in {{main_output}}")
    phi = np.load(candidates[0], mmap_mode="r")
    zoom.run_zoom_simulation(
        cfg, {{"phi": phi}}, V, 1, time_log=[],
        output_dir=str(zoom_output),
        movement_active=False, center=center, center0=center,
        plotting_enabled=False,
    )
    del phi
'''


def _stream_process(label: str, cmd: list[str], cwd: Path, env: dict[str, str],
                    log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print(f"\n[{label}] starting: {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            stripped = line.rstrip()
            # Keep the live parent output useful without duplicating every
            # inner SOR iteration; the complete transcript remains in log.
            keys = ("Found ", "Processing configuration", "Starting 3D MG",
                    "converged", "Converged", "NOT converged", "Saved",
                    "zoom", "Zoom", "ERROR", "Traceback", "Finished",
                    "elapsed", "residual")
            if any(k in stripped for k in keys):
                print(f"[{label}] {stripped}", flush=True)
        rc = proc.wait()
    elapsed = time.time() - start
    result = {"label": label, "returncode": rc, "elapsed_s": elapsed,
              "log": str(log_path.resolve())}
    print(f"[{label}] finished returncode={rc} elapsed={elapsed:.1f}s", flush=True)
    return result


def _env(pack: Path, output: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pack.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    env["AFM_CAPTURE_OUTPUT"] = str(output.resolve())
    return env


def _run_old(pack: Path, stage: Path, output: Path, log: Path) -> dict[str, Any]:
    return _stream_process("old", [sys.executable, "-c", _old_code(),
                                    str(pack), str(stage)], stage,
                           _env(pack, output), log)


def _run_new(pack: Path, stage: Path, config: Path, output: Path,
             log: Path, label: str) -> dict[str, Any]:
    return _stream_process(label, [sys.executable, "-c", _new_code(config),
                                   str(pack), str(stage)],
                           stage, _env(pack, output), log)


def _run_zoom_only(pack: Path, stage: Path, config: Path, main_output: Path,
                   output: Path, log: Path, label: str) -> dict[str, Any]:
    return _stream_process(label, [sys.executable, "-c",
                                   _zoom_code(config, main_output),
                                   str(pack), str(stage)],
                           stage, _env(pack, output), log)


def _voltage_from_name(name: str) -> float | None:
    m = re.search(r"_(-?\d+(?:\.\d+)?)V(?:_|\.)", name)
    return float(m.group(1)) if m else None


def _level_from_name(name: str) -> int | None:
    m = re.search(r"_level(\d+)x\1x\1\.npy$", name)
    return int(m.group(1)) if m else None


def _array_index(output: Path) -> dict[tuple[float, int], Path]:
    found: dict[tuple[float, int], Path] = {}
    for path in output.glob("*.npy"):
        voltage = _voltage_from_name(path.name)
        if voltage is None:
            continue
        level = _level_from_name(path.name)
        if level is None:
            try:
                arr_shape = np.load(path, mmap_mode="r").shape
            except Exception:
                continue
            if len(arr_shape) == 3 and arr_shape[0] == arr_shape[1] == arr_shape[2]:
                level = int(arr_shape[0])
        if level in LEVELS:
            found[(voltage, level)] = path
    return found


def _array_diff(a_path: Path, b_path: Path) -> dict[str, Any]:
    a = np.load(a_path, mmap_mode="r")
    b = np.load(b_path, mmap_mode="r")
    result: dict[str, Any] = {
        "old_file": str(a_path), "new_file": str(b_path),
        "old_shape": list(a.shape), "new_shape": list(b.shape),
        "old_dtype": str(a.dtype), "new_dtype": str(b.dtype),
    }
    if a.shape != b.shape:
        result["shape_match"] = False
        return result
    result["shape_match"] = True
    # Avoid a float64-sized 512^3 temporary.  The potentials are float32 by
    # design; a float32 difference is sufficient for the requested comparison
    # tolerance and keeps peak analysis memory modest.
    work = np.empty(a.shape, dtype=np.float32)
    np.subtract(a, b, out=work, dtype=np.float32)
    np.abs(work, out=work)
    result["max_abs"] = float(np.max(work))
    result["mean_abs"] = float(np.mean(work, dtype=np.float64))
    np.square(work, out=work)
    result["rms_abs"] = float(np.sqrt(np.mean(work, dtype=np.float64)))
    result["within_5e-6"] = bool(result["max_abs"] <= 5e-6)
    return result


def _compare_arrays(old_output: Path, new_output: Path,
                    report_dir: Path) -> list[dict[str, Any]]:
    old_idx, new_idx = _array_index(old_output), _array_index(new_output)
    rows: list[dict[str, Any]] = []
    for voltage in VOLTAGES:
        for level in LEVELS:
            row = {"voltage_V": voltage, "level": level}
            a, b = old_idx.get((voltage, level)), new_idx.get((voltage, level))
            if a is None or b is None:
                row.update({"status": "missing", "old_file": str(a) if a else "",
                            "new_file": str(b) if b else ""})
            else:
                row.update({"status": "compared", **_array_diff(a, b)})
            rows.append(row)
    report_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with (report_dir / "main_npy_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _residual_files(output: Path, kind: str) -> list[Path]:
    return sorted(output.glob(f"residual_{kind}_call*.csv"))


def _read_residual(path: Path) -> np.ndarray:
    vals: list[tuple[float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        for row in rows:
            try:
                avg = float(row.get("avg_residual",
                                    row.get("average_residual",
                                            row.get("residual_avg", "nan"))))
                mx = float(row.get("max_residual",
                                   row.get("residual_max", "nan")))
            except (TypeError, ValueError):
                continue
            vals.append((avg, mx))
    return np.asarray(vals, dtype=np.float64)


def _last_residual_iteration(path: Path) -> int | None:
    last: int | None = None
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                last = int(float(row.get("iteration", "")))
            except (TypeError, ValueError):
                continue
    return last


def _residual_summary(old_output: Path, new_output: Path,
                      report_dir: Path) -> list[dict[str, Any]]:
    old_files, new_files = _residual_files(old_output, "main"), _residual_files(new_output, "main")
    rows: list[dict[str, Any]] = []
    for call in range(max(len(old_files), len(new_files))):
        # With two voltages and save_all_levels, calls are ordered by voltage
        # then grid level.  This maps calls to the requested 128/256/512 rows.
        voltage = VOLTAGES[(call // 7) % len(VOLTAGES)] if call // 7 < len(VOLTAGES) else None
        level_sequence = (8, 16, 32, 64, 128, 256, 512)
        level = level_sequence[call % 7] if call % 7 < len(level_sequence) else None
        if level not in LEVELS:
            continue
        row: dict[str, Any] = {"call": call, "voltage_V": voltage, "level": level}
        for side, files in (("old", old_files), ("new", new_files)):
            if call >= len(files):
                row[f"{side}_status"] = "missing"
                continue
            arr = _read_residual(files[call])
            row[f"{side}_status"] = "ok"
            row[f"{side}_file"] = str(files[call])
            row[f"{side}_iterations"] = _last_residual_iteration(files[call])
            row[f"{side}_residual_rows"] = int(arr.shape[0])
            if arr.size:
                row[f"{side}_last_avg"] = float(arr[-1, 0])
                row[f"{side}_last_max"] = float(arr[-1, 1])
                row[f"{side}_min_avg"] = float(np.nanmin(arr[:, 0]))
                row[f"{side}_min_max"] = float(np.nanmin(arr[:, 1]))
        if call < len(old_files) and call < len(new_files):
            oa, na = _read_residual(old_files[call]), _read_residual(new_files[call])
            n = min(len(oa), len(na))
            if n:
                d = np.abs(oa[:n] - na[:n])
                row["residual_max_abs_diff"] = float(np.nanmax(d))
                row["residual_last_avg_abs_diff"] = float(abs(oa[n-1, 0] - na[n-1, 0]))
                row["residual_last_max_abs_diff"] = float(abs(oa[n-1, 1] - na[n-1, 1]))
        rows.append(row)
    report_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with (report_dir / "main_residual_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _physical_qd_ranges(source_cfg: Path) -> list[tuple[float, float, float, float, float, float]]:
    """Return eps=15 ranges in nm relative to the simulation origin."""
    cfg = json.loads(source_cfg.read_text(encoding="utf-8"))
    origin = cfg.get("coordinate_system", {}).get("origin_fraction", [0.5, 0.5, 0.0])
    grid = cfg.get("grid_resolution", {})
    if isinstance(grid, dict):
        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    else:
        nx, ny, nz = (int(v) for v in grid)
    # The current alignment config is physical; fall back to the old unit
    # cube convention if a caller supplies a fractional source.
    domain = cfg.get("domain_nm", {})
    if not domain:
        voxel = cfg.get("voxel_nm3", 1.0)
        if isinstance(voxel, (list, tuple)):
            vx, vy, vz = (float(v) for v in voxel)
        else:
            vx = vy = vz = float(voxel)
        domain = {"Lx_nm": float(nx) * vx, "Ly_nm": float(ny) * vy,
                  "Lz_nm": float(nz) * vz}
    Lx, Ly, Lz = float(domain.get("Lx_nm", 256.0)), float(domain.get("Ly_nm", 256.0)), float(domain.get("Lz_nm", 256.0))
    out = []
    blocks = cfg.get("blocks_nm", cfg.get("blocks", []))
    for block in blocks:
        if float(block.get("eps_val", block.get("eps", 0.0))) != 15.0:
            continue
        def _axis(name: str, L: float, o: float) -> tuple[float, float]:
            values = block.get(f"{name}_range_nm")
            if values is None:
                values = block.get(f"{name}_range", [0.0, 1.0])
                return ((float(values[0]) - o) * L, (float(values[1]) - o) * L)
            # blocks_nm in the parallel package are origin-relative nm.
            return float(values[0]), float(values[1])
        x0, x1 = _axis("x", Lx, float(origin[0]))
        y0, y1 = _axis("y", Ly, float(origin[1]))
        z0, z1 = _axis("z", Lz, float(origin[2]))
        out.append((min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1), min(z0, z1), max(z0, z1)))
    if not out:
        raise ValueError("No eps=15 blocks found in the source configuration")
    return out


def _field_bounds(kind: str) -> tuple[float, float, float, float, float, float]:
    if kind == "main":
        return (-128.0, 128.0, -128.0, 128.0, 0.0, 256.0)
    if kind == "zoom2":
        return (-64.0, 64.0, -64.0, 64.0, 0.0, 128.0)
    if kind == "zoom4":
        return (-32.0, 32.0, -32.0, 32.0, 0.0, 64.0)
    raise ValueError(kind)


def _qd_coordinates(shape: tuple[int, int, int], bounds: tuple[float, ...],
                    qd_ranges: list[tuple[float, ...]]) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(bounds[0], bounds[1], shape[0], dtype=np.float64)
    ys = np.linspace(bounds[2], bounds[3], shape[1], dtype=np.float64)
    zs = np.linspace(bounds[4], bounds[5], shape[2], dtype=np.float64)
    # Build only the mask; meshgrid arrays would triple peak memory for the
    # 512^3 main field.  The selected QD coordinates are normally <1M points.
    mx = np.zeros(shape, dtype=bool)
    for x0, x1, y0, y1, z0, z1 in qd_ranges:
        ix = (xs >= x0) & (xs <= x1)
        iy = (ys >= y0) & (ys <= y1)
        iz = (zs >= z0) & (zs <= z1)
        if ix.any() and iy.any() and iz.any():
            mx |= ix[:, None, None] & iy[None, :, None] & iz[None, None, :]
    coords = np.nonzero(mx)
    return mx, np.vstack(coords).astype(np.float64, copy=False)


def _load_final(output: Path, prefix: str, voltage: float) -> Path:
    candidates = []
    level_candidates = []
    for p in output.glob("*.npy"):
        if prefix not in p.name:
            continue
        if prefix == "afm_phi_" and "afm_phi_zoom_" in p.name:
            continue
        v = _voltage_from_name(p.name)
        if v is None or abs(v - voltage) > 1e-8:
            continue
        # Zoom 2x is an intermediate level when the configured zoom limit is
        # 4x, so retain it as a fallback.  For the main field and final 4x
        # field the no-_level file is preferred.
        if "_level" in p.name:
            level_candidates.append(p)
        else:
            candidates.append(p)
    if not candidates:
        candidates = level_candidates
    if not candidates:
        raise FileNotFoundError(f"No final {prefix} array for {voltage} V in {output}")
    return sorted(candidates)[0]


def _zoom_drift(source_cfg: Path, main_output: Path,
                zoom_outputs: dict[str, Path], report_dir: Path) -> list[dict[str, Any]]:
    qd_ranges = _physical_qd_ranges(source_cfg)
    rows: list[dict[str, Any]] = []
    for voltage in VOLTAGES:
        main_path = _load_final(main_output, "afm_phi_", voltage)
        main = np.load(main_path, mmap_mode="r")
        main_bounds = _field_bounds("main")
        main_mask, main_idx = _qd_coordinates(main.shape, main_bounds, qd_ranges)
        main_values = np.asarray(main[tuple(main_idx.astype(np.int64))], dtype=np.float64)
        base = {"voltage_V": voltage, "qd_nodes_main": int(main_values.size),
                "main_mean": float(np.mean(main_values)),
                "main_std": float(np.std(main_values)),
                "main_min": float(np.min(main_values)),
                "main_max": float(np.max(main_values))}
        del main_mask, main_idx, main_values
        for clamp_label, output in zoom_outputs.items():
            for zoom_kind, prefix in (("zoom2", "afm_phi_zoom_2x_"),
                                      ("zoom4", "afm_phi_zoom_4x_")):
                zoom_path = _load_final(output, prefix, voltage)
                zoom = np.load(zoom_path, mmap_mode="r")
                bounds = _field_bounds(zoom_kind)
                mask, idx = _qd_coordinates(zoom.shape, bounds, qd_ranges)
                idx_int = idx.astype(np.int64, copy=False)
                values = np.asarray(zoom[tuple(idx_int)], dtype=np.float64)
                # Map zoom QD nodes into physical coordinates and interpolate
                # the main solution at those points.  This is a like-for-like
                # comparison despite the different zoom domains/resolutions.
                coords = np.vstack((
                    (np.linspace(bounds[0], bounds[1], zoom.shape[0])[idx_int[0]] - main_bounds[0]) /
                    (main_bounds[1] - main_bounds[0]) * (main.shape[0] - 1),
                    (np.linspace(bounds[2], bounds[3], zoom.shape[1])[idx_int[1]] - main_bounds[2]) /
                    (main_bounds[3] - main_bounds[2]) * (main.shape[1] - 1),
                    (np.linspace(bounds[4], bounds[5], zoom.shape[2])[idx_int[2]] - main_bounds[4]) /
                    (main_bounds[5] - main_bounds[4]) * (main.shape[2] - 1),
                ))
                main_sample = map_coordinates(main, coords, order=1,
                                              mode="nearest", prefilter=False)
                diff = values - main_sample
                row = dict(base)
                row.update({"clamp": clamp_label, "grid": zoom_kind,
                            "file": str(zoom_path), "qd_nodes": int(values.size),
                            "zoom_mean": float(np.mean(values)),
                            "zoom_std": float(np.std(values)),
                            "zoom_min": float(np.min(values)),
                            "zoom_max": float(np.max(values)),
                            "mean_drift": float(np.mean(diff)),
                            "mean_abs_drift": float(np.mean(np.abs(diff))),
                            "rms_drift": float(np.sqrt(np.mean(diff * diff))),
                            "max_abs_drift": float(np.max(np.abs(diff))),
                            "relative_to_main_range_pct": float(
                                100.0 * np.max(np.abs(diff)) /
                                max(np.ptp(main_sample), 1e-30))})
                rows.append(row)
                del zoom, mask, idx, idx_int, values, coords, main_sample, diff
        del main_path, main
    report_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with (report_dir / "zoom_qd_drift.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-pack", type=Path, default=Path(r"D:\afm_pack_v1\afm_package"))
    parser.add_argument("--new-pack", type=Path, default=Path(r"D:\afm_pack_v1\afm_parallel"))
    parser.add_argument("--new-source", type=Path,
                        default=Path(r"D:\afm_pack_v1\afm_parallel\afm_config_nm_512_alignment.json"))
    parser.add_argument("--old-source", type=Path,
                        default=Path(r"D:\afm_pack_v1\afm_package\afm_config_1.json"))
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--two-positions", action="store_true",
                        help="retain the source movement endpoint instead of using one alignment position")
    parser.add_argument("--zoom-only", action="store_true",
                        help="reuse an existing run root's new_output and run only clamp off/on zooms")
    parser.add_argument("--zoom-runtime", type=float, default=600.0,
                        help="per-zoom-level wall-time guard in seconds (default: 600)")
    args = parser.parse_args()
    single = not args.two_positions
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = (args.run_root or (Path(r"D:\afm_pack_v1") / f"afm_comparison_{timestamp}")).resolve()
    if args.zoom_only:
        if not run_root.is_dir() or not (run_root / "new_output").is_dir():
            raise FileNotFoundError(f"--zoom-only requires an existing run root with new_output: {run_root}")
        results_path = run_root / "run_summary.json"
        results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {"runs": []}
        new_output = run_root / "new_output"
        zoom_outputs: dict[str, Path] = {}
        for clamp in (False, True):
            label = "parallel-zoom-clamp-" + ("on" if clamp else "off")
            output = run_root / ("zoom_clamp_on" if clamp else "zoom_clamp_off")
            stage = run_root / ("stage_zoom_clamp_on" if clamp else "stage_zoom_clamp_off")
            stage.mkdir(parents=True, exist_ok=True)
            config = _make_new_config(args.new_source, stage, output, args.threads,
                                      single, True, clamp)
            zoom_cfg = json.loads(config.read_text(encoding="utf-8"))
            zoom_cfg["mg_max_runtime"] = float(args.zoom_runtime)
            _json_dump(config, zoom_cfg)
            run = _run_zoom_only(args.new_pack, stage, config, new_output, output,
                                 run_root / (label + ".log"), label)
            results.setdefault("runs", []).append(run)
            if run["returncode"] != 0:
                _write_json(results_path, results)
                return run["returncode"]
            zoom_outputs["clamp_on" if clamp else "clamp_off"] = output
        results["zoom_qd_drift"] = _zoom_drift(args.new_source, new_output,
                                               zoom_outputs, run_root / "zoom_report")
        _write_json(results_path, results)
        print(f"\nCompleted zoom-only comparison. Reports are under {run_root}", flush=True)
        return 0
    if run_root.exists():
        raise FileExistsError(f"Refusing to reuse existing run directory: {run_root}")
    run_root.mkdir(parents=True)
    stages = run_root / "stages"
    old_stage, new_stage = stages / "old", stages / "new"
    old_output = run_root / "old_output"
    new_output = run_root / "new_output"
    old_stage.mkdir(parents=True)
    new_stage.mkdir(parents=True)

    _make_old_config(args.old_source, old_stage, old_output, args.threads, single)
    main_config = _make_new_config(args.new_source, new_stage, new_output,
                                   args.threads, single, False)
    manifest = {"run_root": str(run_root), "old_pack": str(args.old_pack),
                "new_pack": str(args.new_pack), "old_source": str(args.old_source),
                "new_source": str(args.new_source), "threads": args.threads,
                "single_position": single, "voltages": VOLTAGES,
                "levels_compared": LEVELS, "res_tol": 5e-6}
    _write_json(run_root / "manifest.json", manifest)

    results = {"manifest": manifest, "runs": []}
    old_run = _run_old(args.old_pack, old_stage, old_output, run_root / "old.log")
    results["runs"].append(old_run)
    if old_run["returncode"] != 0:
        _write_json(run_root / "run_summary.json", results)
        return old_run["returncode"]
    new_run = _run_new(args.new_pack, new_stage, main_config, new_output,
                       run_root / "new.log", "parallel-main")
    results["runs"].append(new_run)
    if new_run["returncode"] != 0:
        _write_json(run_root / "run_summary.json", results)
        return new_run["returncode"]

    main_rows = _compare_arrays(old_output, new_output, run_root / "main_report")
    residual_rows = _residual_summary(old_output, new_output, run_root / "main_report")
    results["main_npy"] = main_rows
    results["main_residuals"] = residual_rows

    zoom_outputs: dict[str, Path] = {}
    for clamp in (False, True):
        label = "parallel-zoom-clamp-" + ("on" if clamp else "off")
        output = run_root / ("zoom_clamp_on" if clamp else "zoom_clamp_off")
        stage = run_root / ("stage_zoom_clamp_on" if clamp else "stage_zoom_clamp_off")
        stage.mkdir(parents=True)
        config = _make_new_config(args.new_source, stage, output, args.threads,
                                  single, True, clamp)
        zoom_cfg = json.loads(config.read_text(encoding="utf-8"))
        zoom_cfg["mg_max_runtime"] = float(args.zoom_runtime)
        _json_dump(config, zoom_cfg)
        run = _run_zoom_only(args.new_pack, stage, config, new_output, output,
                             run_root / (label + ".log"), label)
        results["runs"].append(run)
        if run["returncode"] != 0:
            _write_json(run_root / "run_summary.json", results)
            return run["returncode"]
        zoom_outputs["clamp_on" if clamp else "clamp_off"] = output

    results["zoom_qd_drift"] = _zoom_drift(args.new_source, new_output,
                                           zoom_outputs, run_root / "zoom_report")
    _write_json(run_root / "run_summary.json", results)
    print(f"\nCompleted comparison. Reports are under {run_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
