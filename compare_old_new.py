#!/usr/bin/env python3
"""Bit-for-bit AFM compatibility checker for the legacy and parallel packages.

This utility runs the *same source JSON* through:
  1. the old/reference AFM package,
  2. the new package in cpu_threads=1 reference mode, and
  3. the new package in multi-threaded CPU-parallel mode.

Each run gets an isolated temporary output directory. The source JSON itself is
never modified. All generated .npy files are compared both as arrays and as
raw file bytes; numerical differences are quantified when exact equality fails.

Example
-------
python compare_old_new.py \
    --old-pack /path/to/old/afm_package \
    --new-pack /path/to/new/afm_package \
    --config /path/to/afm_config_nm.json \
    --parallel-threads 4 \
    --max-iter 50 \
    --no-plot
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from simulation.coordinates import ordered_config_for_json


def _make_config(source: Path, destination: Path, output_dir: Path, cpu_threads: int | None, disable_plot: bool) -> None:
    cfg = json.loads(source.read_text())
    cfg["output_dir"] = str(output_dir)
    if cpu_threads is not None:
        cfg["cpu_threads"] = int(cpu_threads)
    if disable_plot:
        plotting = cfg.setdefault("plotting", {})
        if isinstance(plotting, dict):
            plotting["enabled"] = False
            plotting["disable_in_non_ide"] = True
        cfg["plot_zoom_residuals"] = False
    destination.write_text(
        json.dumps(
            ordered_config_for_json(
                cfg, ensure_cpu_threads=cpu_threads is not None
            ),
            indent=4,
        )
        + "\n"
    )


def _run(pack: Path, config: Path, output_dir: Path, label: str,
         max_iter: int | None = None, single_position: bool = False) -> float:
    env = os.environ.copy()
    env["AFM_NONINTERACTIVE"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"\n[{label}] running {pack / 'run_all.py'}")
    start = time.perf_counter()
    if max_iter is None:
        command = [sys.executable, "run_all.py", str(config)]
    else:
        # Keep the benchmark bounded without modifying either package's
        # production launcher. batch_main resolves run_afm_simulation through
        # its module global, so this wrapper can impose the same iteration cap
        # on old, new-serial, and new-parallel runs.
        limited_runner = (
            "import sys\n"
            "from simulation import main_loop\n"
            "_real = main_loop.run_afm_simulation\n"
            "def _limited(*args, **kwargs):\n"
            "    kwargs['max_iter'] = int(sys.argv[2])\n"
            "    return _real(*args, **kwargs)\n"
            "main_loop.run_afm_simulation = _limited\n"
        )
        if single_position:
            limited_runner += (
                "main_loop.compute_block_positions = "
                "lambda start, end, spacing, domain_nm=None: [tuple(start)]\n"
            )
        limited_runner += (
            "main_loop.batch_main(sys.argv[1], plotting_override=False, interactive=False)\n"
        )
        command = [sys.executable, "-c", limited_runner, str(config), str(max_iter)]

    proc = subprocess.run(
        command,
        cwd=pack,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wall_time = time.perf_counter() - start
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")
    if not output_dir.exists():
        raise RuntimeError(f"{label} did not create output directory: {output_dir}")
    return wall_time


def _timing_summary(output_dir: Path) -> dict[str, object]:
    """Read the per-level solver log and combine it with harness wall time."""
    timing_file = output_dir / "mg_timing_log.csv"
    if not timing_file.is_file():
        return {"solver_time_sec": float("nan"), "levels": 0}

    total = 0.0
    levels = 0
    with timing_file.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                total += float(row["time_sec"])
                levels += 1
            except (KeyError, TypeError, ValueError):
                continue
    return {"solver_time_sec": total, "levels": levels}


def _write_timing_report(temp: Path, rows: list[dict[str, object]]) -> None:
    """Persist timing data so kept comparison workspaces are auditable."""
    report = temp / "comparison_timing.csv"
    with report.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "wall_time_sec", "solver_time_sec", "mg_levels"])
        for row in rows:
            writer.writerow([row["mode"], f"{row['wall_time_sec']:.6f}",
                             f"{row['solver_time_sec']:.6f}", row["mg_levels"]])
    print(f"Timing report: {report}")


def _npy_map(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)): p for p in sorted(root.rglob("*.npy"))}


def _compare_pair(a: Path, b: Path) -> dict[str, object]:
    raw_a = a.read_bytes()
    raw_b = b.read_bytes()
    raw_equal = raw_a == raw_b

    arr_a = np.load(a, allow_pickle=False)
    arr_b = np.load(b, allow_pickle=False)
    shape_equal = arr_a.shape == arr_b.shape
    dtype_equal = arr_a.dtype == arr_b.dtype
    array_equal = shape_equal and dtype_equal and np.array_equal(arr_a, arr_b, equal_nan=True)

    if shape_equal and arr_a.size:
        a64 = arr_a.astype(np.float64, copy=False)
        b64 = arr_b.astype(np.float64, copy=False)
        diff = np.abs(a64 - b64)
        max_abs = float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0
        same_values = int(np.count_nonzero(arr_a == arr_b))
        total = int(arr_a.size)
    else:
        max_abs = float("nan")
        same_values = 0
        total = int(arr_a.size) if hasattr(arr_a, "size") else 0

    return {
        "raw_file_equal": raw_equal,
        "array_equal": array_equal,
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "max_abs_diff": max_abs,
        "same_values": same_values,
        "total_values": total,
        "shape_a": arr_a.shape,
        "shape_b": arr_b.shape,
        "dtype_a": str(arr_a.dtype),
        "dtype_b": str(arr_b.dtype),
    }


def _report(title: str, root_a: Path, root_b: Path) -> bool:
    files_a = _npy_map(root_a)
    files_b = _npy_map(root_b)
    all_names = sorted(set(files_a) | set(files_b))

    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"A: {root_a}")
    print(f"B: {root_b}")
    print(f"NPY files: A={len(files_a)}, B={len(files_b)}")

    ok = files_a.keys() == files_b.keys()
    if files_a.keys() != files_b.keys():
        print("FILE SET MISMATCH")
        print("Only in A:", sorted(set(files_a) - set(files_b)))
        print("Only in B:", sorted(set(files_b) - set(files_a)))
        ok = False

    for name in all_names:
        if name not in files_a or name not in files_b:
            continue
        result = _compare_pair(files_a[name], files_b[name])
        exact = result["raw_file_equal"] and result["array_equal"]
        print(
            f"{'PASS' if exact else 'DIFF'}  {name}: "
            f"raw_bytes={'yes' if result['raw_file_equal'] else 'no'}, "
            f"array={'yes' if result['array_equal'] else 'no'}, "
            f"shape={result['shape_a']}, dtype={result['dtype_a']}, "
            f"max_abs_diff={result['max_abs_diff']:.6e}"
        )
        if not exact:
            ok = False

    return ok


def _read_residual_log(root: Path) -> list[tuple[int, float, float]] | None:
    path = root / "residual_history.csv"
    if not path.is_file():
        return None
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append((int(row["iteration"]), float(row["residual_avg"]),
                             float(row["residual_max"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _report_residuals(title: str, root_a: Path, root_b: Path,
                      atol: float = 1e-7) -> bool:
    """Compare residual histories, allowing tiny parallel sum-order drift."""
    rows_a = _read_residual_log(root_a)
    rows_b = _read_residual_log(root_b)
    print(f"\n{title} residual history")
    if rows_a is None or rows_b is None:
        print("DIFF  residual_history.csv missing")
        return False

    iterations_equal = [row[0] for row in rows_a] == [row[0] for row in rows_b]
    count = min(len(rows_a), len(rows_b))
    if count:
        avg_diff = max(abs(rows_a[i][1] - rows_b[i][1]) for i in range(count))
        max_diff = max(abs(rows_a[i][2] - rows_b[i][2]) for i in range(count))
        last_a = rows_a[-1]
        last_b = rows_b[-1]
    else:
        avg_diff = max_diff = float("nan")
        last_a = last_b = None
    ok = (len(rows_a) == len(rows_b) and iterations_equal and
          avg_diff <= atol and max_diff <= atol)
    print(
        f"{'PASS' if ok else 'DIFF'}  rows={len(rows_a)}/{len(rows_b)}, "
        f"iterations={'same' if iterations_equal else 'different'}, "
        f"max_avg_abs_diff={avg_diff:.6e}, max_residual_abs_diff={max_diff:.6e}, "
        f"last_avg={last_a[1] if last_a else float('nan'):.6e}/"
        f"{last_b[1] if last_b else float('nan'):.6e}, "
        f"last_max={last_a[2] if last_a else float('nan'):.6e}/"
        f"{last_b[2] if last_b else float('nan'):.6e}"
    )
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-pack", required=True, type=Path, help="Root directory of the old AFM package.")
    ap.add_argument("--new-pack", required=True, type=Path, help="Root directory of the new AFM package.")
    ap.add_argument("--config", required=True, type=Path, help="One AFM JSON used for all runs.")
    ap.add_argument("--parallel-threads", type=int, default=4, help="CPU threads for the new parallel run (default: 4).")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="Optional identical per-level iteration cap for a bounded benchmark.")
    ap.add_argument("--single-position", action="store_true",
                    help="Run only the first movement position in a bounded benchmark.")
    ap.add_argument("--keep-work", action="store_true", help="Keep temporary run directories for inspection.")
    ap.add_argument("--no-plot", action="store_true", help="Disable plotting in temporary configs.")
    args = ap.parse_args()

    for label, p in (("old", args.old_pack), ("new", args.new_pack), ("config", args.config)):
        if not p.exists():
            ap.error(f"{label} path does not exist: {p}")

    if args.parallel_threads < 2:
        ap.error("--parallel-threads must be >= 2")
    if args.max_iter is not None and args.max_iter < 1:
        ap.error("--max-iter must be >= 1")

    temp = Path(tempfile.mkdtemp(prefix="afm_compat_"))
    if args.keep_work:
        print(f"Keeping compatibility test workspace: {temp}")
    try:
        old_out = temp / "old_out"
        new_serial_out = temp / "new_serial_out"
        new_parallel_out = temp / "new_parallel_out"
        old_cfg = temp / "old_config.json"
        serial_cfg = temp / "new_serial_config.json"
        parallel_cfg = temp / "new_parallel_config.json"

        # These configs originate from the exact same source JSON. Only output_dir,
        # and the new CPU setting where applicable, are changed in the copies.
        _make_config(args.config, old_cfg, old_out, None, args.no_plot)
        _make_config(args.config, serial_cfg, new_serial_out, 1, args.no_plot)
        _make_config(args.config, parallel_cfg, new_parallel_out, args.parallel_threads, args.no_plot)

        old_wall = _run(args.old_pack.resolve(), old_cfg.resolve(), old_out, "OLD PACKAGE",
                        args.max_iter, args.single_position)
        serial_wall = _run(args.new_pack.resolve(), serial_cfg.resolve(), new_serial_out,
                           "NEW PACKAGE / 1 CPU", args.max_iter, args.single_position)
        parallel_wall = _run(
            args.new_pack.resolve(), parallel_cfg.resolve(), new_parallel_out,
            f"NEW PACKAGE / {args.parallel_threads} CPUs",
            args.max_iter,
            args.single_position,
        )

        if args.max_iter is not None:
            print(f"Bounded benchmark: max_iter={args.max_iter} at every grid level; "
                  "1e-6 remains the convergence tolerance.")
        if args.single_position:
            print("Movement benchmark reduced to the first position.")

        timing_rows = []
        for mode, wall, output in (
            ("old", old_wall, old_out),
            ("new_serial", serial_wall, new_serial_out),
            (f"new_parallel_{args.parallel_threads}", parallel_wall, new_parallel_out),
        ):
            summary = _timing_summary(output)
            timing_rows.append({
                "mode": mode,
                "wall_time_sec": wall,
                "solver_time_sec": summary["solver_time_sec"],
                "mg_levels": summary["levels"],
            })
        _write_timing_report(temp, timing_rows)

        print("\nTIMING SUMMARY")
        print(f"{'mode':<22} {'wall s':>12} {'solver s':>12} {'MG levels':>10}")
        for row in timing_rows:
            print(f"{row['mode']:<22} {row['wall_time_sec']:12.3f} "
                  f"{row['solver_time_sec']:12.3f} {row['mg_levels']:10d}")
        serial_solver = float(timing_rows[1]["solver_time_sec"])
        parallel_solver = float(timing_rows[2]["solver_time_sec"])
        if serial_solver > 0 and parallel_solver > 0:
            print(f"Solver-time speedup (serial / parallel): {serial_solver / parallel_solver:.3f}x")
        if serial_wall > 0 and parallel_wall > 0:
            print(f"Wall-time speedup (serial / parallel): {serial_wall / parallel_wall:.3f}x")

        old_vs_serial = _report("OLD PACKAGE  <->  NEW PACKAGE (cpu_threads=1)", old_out, new_serial_out)
        serial_vs_parallel = _report(
            f"NEW PACKAGE (cpu_threads=1)  <->  NEW PACKAGE (cpu_threads={args.parallel_threads})",
            new_serial_out,
            new_parallel_out,
        )
        old_vs_serial_residuals = _report_residuals(
            "OLD PACKAGE <-> NEW PACKAGE (cpu_threads=1)", old_out, new_serial_out
        )
        serial_vs_parallel_residuals = _report_residuals(
            f"NEW PACKAGE (cpu_threads=1) <-> NEW PACKAGE (cpu_threads={args.parallel_threads})",
            new_serial_out, new_parallel_out,
        )

        print("\n" + "=" * 78)
        print("COMPATIBILITY SUMMARY")
        print("=" * 78)
        print(f"Old vs new serial bit-for-bit compatible: {'YES' if old_vs_serial else 'NO'}")
        print(f"New serial vs parallel bit-for-bit compatible: {'YES' if serial_vs_parallel else 'NO'}")
        print(f"Old vs new serial residuals aligned: {'YES' if old_vs_serial_residuals else 'NO'}")
        print(f"New serial vs parallel residuals aligned: {'YES' if serial_vs_parallel_residuals else 'NO'}")
        print("Note: raw .npy equality is stricter than numerical array equality.")

        return 0 if (old_vs_serial and serial_vs_parallel and
                     old_vs_serial_residuals and serial_vs_parallel_residuals) else 2
    finally:
        if not args.keep_work:
            shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
