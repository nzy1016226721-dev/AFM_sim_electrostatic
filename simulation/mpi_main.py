"""Non-interactive MPI driver for one distributed AFM configuration."""

from __future__ import annotations

import csv
import gc
import os
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from .io_utils import movement_suffix_nm, resolution_tagged_cut_filename
from .main_loop import (
    apply_relative_blocks,
    apply_relative_vgates,
    compute_block_positions,
    default_initial_grid_level,
    precompute_relative_offsets_for_blocks,
    precompute_relative_offsets_for_vgates,
)
from .mpi_domain import (
    DistributedField,
    DomainDecomposition,
    apply_fixed_geometry,
    build_distributed_epsilon_halo,
    build_local_face_fields,
    human_bytes,
    prolongate_distributed,
    solve_distributed_level,
    verify_node_memory,
)
from .mpi_io import (
    distributed_probe,
    save_distributed_npy,
    save_distributed_physical_cut,
)
from .mpi_config import load_mpi_config
from .numerics import (
    diagnostic_iteration_limit,
    prepare_drive_voltages,
    resolve_residual_tolerance_mode,
    resolve_solver_dtype,
)
from .runtime import resolve_output_dir


def _require_mpi():
    try:
        from mpi4py import MPI
    except ImportError as exc:  # pragma: no cover - exercised on cluster
        raise RuntimeError(
            "mpi4py is required. Run jobs/setup_afm_mpi_env.sh before submitting "
            "jobs/run_afm_mpi.sh."
        ) from exc
    return MPI


def _parse_process_grid(value, world_size: int):
    env_value = os.environ.get("AFM_MPI_PROCESS_GRID", "").strip()
    if env_value:
        value = [int(item.strip()) for item in env_value.split(",")]
    if value is None:
        return None
    dims = tuple(int(v) for v in value)
    if len(dims) != 3:
        raise ValueError("MPI process grid must contain x,y,z rank counts")
    if all(v == 0 for v in dims):
        return None
    if any(v < 1 for v in dims):
        raise ValueError("MPI process-grid entries must all be positive, or all zero for auto")
    if int(np.prod(dims)) != int(world_size):
        raise ValueError(
            f"MPI process grid {dims} has {int(np.prod(dims))} ranks but the job launched "
            f"{world_size}"
        )
    return dims


def _append_csv(path: str, header: Sequence[str], row: Sequence[object]) -> None:
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def _log_memory_snapshot(comm, output_dir: str, case_tag: str, level_shape, stage: str) -> None:
    try:
        import psutil

        rss = int(psutil.Process().memory_info().rss)
    except Exception:
        rss = 0
    hostname = _require_mpi().Get_processor_name()
    records = comm.gather((hostname, comm.Get_rank(), rss), root=0)
    if comm.Get_rank() != 0:
        return
    per_node = {}
    for host, rank, value in records:
        per_node[host] = per_node.get(host, 0) + int(value)
    _append_csv(
        os.path.join(output_dir, "memory_usage_mpi.csv"),
        (
            "case",
            "level",
            "stage",
            "rank_count",
            "max_rank_rss_gib",
            "max_node_rss_gib",
            "sum_rss_gib",
        ),
        (
            case_tag,
            "x".join(str(v) for v in level_shape),
            stage,
            len(records),
            f"{max(v for _, _, v in records) / 1024**3:.6f}",
            f"{max(per_node.values()) / 1024**3:.6f}",
            f"{sum(v for _, _, v in records) / 1024**3:.6f}",
        ),
    )


def _next_shape(current: Sequence[int], target: Sequence[int]) -> tuple[int, int, int]:
    return tuple(min(int(old) * 2, int(final)) for old, final in zip(current, target))


def _initial_shape(target: Sequence[int], initial_level: int) -> tuple[int, int, int]:
    return tuple(min(int(initial_level), int(v)) for v in target)


def run_distributed_hierarchy(
    cfg: dict,
    *,
    Vtip: float,
    output_dir: str,
    case_tag: str,
    comm,
) -> tuple[DistributedField, dict]:
    """Run all refinement levels for one voltage/position on every MPI rank."""
    mpi_cfg = cfg.get("mpi", {})
    field_dtype = resolve_solver_dtype(cfg.get("solver_dtype", "float32"))
    tolerance_mode = resolve_residual_tolerance_mode(
        cfg.get("residual_tolerance_mode", "absolute")
    )
    solver_vtip, solver_vgate, drive_scale = prepare_drive_voltages(
        Vtip, cfg.get("Vgate", []), tolerance_mode
    )
    target = tuple(int(cfg["grid_resolution"][key]) for key in ("nx", "ny", "nz"))
    process_grid = _parse_process_grid(mpi_cfg.get("process_grid"), comm.Get_size())
    configured_initial = mpi_cfg.get("initial_grid_level", cfg.get("initial_grid_level"))
    initial_level = (
        default_initial_grid_level(target)
        if configured_initial is None
        else int(configured_initial)
    )
    if initial_level < 4:
        raise ValueError("mpi.initial_grid_level must be at least 4")
    current_shape = _initial_shape(target, initial_level)
    decomposition = DomainDecomposition.create(comm, current_shape, process_grid)
    final_decomposition = decomposition.for_shape(target)

    memory_fraction = float(mpi_cfg.get("memory_fraction", 0.80))
    node_plan = verify_node_memory(
        final_decomposition, memory_fraction, dtype=field_dtype
    )
    if decomposition.rank == 0:
        worst = max(int(item["estimated_bytes"]) for item in node_plan.values())
        print(
            f"MPI memory preflight: {len(node_plan)} nodes, "
            f"worst estimated node peak={human_bytes(worst)}, "
            f"safety fraction={memory_fraction:.0%}"
        )
        print(f"MPI solver dtype: {field_dtype.name}")
        if tolerance_mode == "relative_drive":
            print(
                "Voltage-normalized residual mode: "
                f"drive scale={drive_scale:.12g} V; solver tip={solver_vtip:.12g}; "
                "saved potential is rescaled to physical volts."
            )

    field = DistributedField(
        decomposition,
        decomposition.allocate(0.001, dtype=field_dtype),
    )
    reference_value = cfg.get("epsilon_material", {}).get(
        "reference_resolution", 512
    )
    reference_shape = (
        (int(reference_value),) * 3
        if not isinstance(reference_value, (list, tuple))
        else tuple(int(v) for v in reference_value)
    )
    level = 1
    level_reports = []
    while True:
        decomp = field.decomposition
        if decomp.rank == 0:
            print(
                f"\n[MPI level {level}] global={decomp.global_shape}, "
                f"rank-0 local={decomp.counts}, process_grid={decomp.dims}"
            )
        _log_memory_snapshot(comm, output_dir, case_tag, decomp.global_shape, "before_geometry")

        mask, fixed_values, tip_pos, base_pos = apply_fixed_geometry(
            field,
            Vtip=solver_vtip,
            Vgate=solver_vgate,
            tip_z=cfg["tip_z"],
            R=cfg["R"],
            r_tip=cfg["r_tip"],
            aspect_ratio=cfg["aspect_ratio"],
            physical_params=cfg.get("_physical"),
            tip_shape=cfg.get("tip_shape", "pyramid"),
        )
        eps_halo = build_distributed_epsilon_halo(
            decomp,
            cfg.get("blocks", []),
            reference_shape=reference_shape,
            dtype=field_dtype,
        )
        faces = build_local_face_fields(eps_halo)
        _log_memory_snapshot(
            comm,
            output_dir,
            case_tag,
            decomp.global_shape,
            "faces_ready_with_epsilon",
        )
        del eps_halo
        gc.collect()
        _log_memory_snapshot(comm, output_dir, case_tag, decomp.global_shape, "solver_ready")

        diagnostic_iterations = diagnostic_iteration_limit(
            cfg.get("diagnostic"),
            decomp.global_shape,
            solver_dtype=field_dtype,
        )
        field, report = solve_distributed_level(
            field,
            mask,
            fixed_values,
            faces,
            omega=float(mpi_cfg.get("damping", 1.0)),
            tol=float(cfg.get("res_tol_main", 5e-5)),
            # Fixed iteration caps are deliberately ignored by the MPI solver;
            # preserve the legacy value only for a clear compatibility notice.
            max_iter=mpi_cfg.get("max_iter"),
            max_runtime=cfg.get("mg_max_runtime"),
            cpu_threads=int(cfg.get("cpu_threads", 1)),
            output_dir=output_dir,
            residual_check_interval=int(mpi_cfg.get("residual_check_interval", 10)),
            diagnostic_iterations=diagnostic_iterations,
        )
        report["shape"] = decomp.global_shape
        report["solver_dtype"] = field_dtype.name
        report["residual_tolerance_mode"] = tolerance_mode
        report["drive_scale_volts"] = drive_scale
        report["physical_residual_volts"] = float(report["residual"]) * drive_scale
        level_reports.append(report)
        _log_memory_snapshot(comm, output_dir, case_tag, decomp.global_shape, "level_complete")
        if decomp.rank == 0:
            _append_csv(
                os.path.join(output_dir, "mg_timing_mpi.csv"),
                (
                    "case", "level", "Nx", "Ny", "Nz", "time_sec",
                    "iterations", "residual", "result", "solver_dtype",
                    "residual_tolerance_mode", "drive_scale_V",
                    "physical_residual_V",
                ),
                (
                    case_tag,
                    level,
                    *decomp.global_shape,
                    f"{report['elapsed']:.6f}",
                    report["iterations"],
                    f"{report['residual']:.12e}",
                    report["reason"],
                    field_dtype.name,
                    tolerance_mode,
                    f"{drive_scale:.12e}",
                    f"{report['physical_residual_volts']:.12e}",
                ),
            )

        diagnostic_ok = (
            diagnostic_iterations is not None
            and report["reason"] == "diagnostic_iteration_limit"
        )
        if (
            report["reason"] != "converged"
            and not diagnostic_ok
            and bool(mpi_cfg.get("require_convergence", True))
        ):
            raise RuntimeError(
                f"MPI level {decomp.global_shape} ended with {report['reason']} after "
                f"{report['iterations']} iterations; residual={report['residual']:.6e}. "
                "No finer level or scientific output was produced. Increase the "
                "per-level mg_max_runtime or the Slurm wall time, adjust the solver "
                "only after validation, or explicitly set mpi.require_convergence=false "
                "for diagnostic runs."
            )

        if decomp.global_shape == target:
            if drive_scale != 1.0:
                np.multiply(
                    field.data,
                    field_dtype.type(drive_scale),
                    out=field.data,
                )
            return field, {
                "levels": level,
                "tip_pos": tip_pos,
                "base_pos": base_pos,
                "level_reports": level_reports,
                "solver_dtype": field_dtype.name,
                "residual_tolerance_mode": tolerance_mode,
                "drive_scale_volts": drive_scale,
            }

        new_shape = _next_shape(decomp.global_shape, target)
        new_decomposition = decomp.for_shape(new_shape)
        del mask, fixed_values, faces
        gc.collect()
        if decomp.rank == 0:
            print(f"[MPI level {level}] prolongating to {new_shape}")
        field = prolongate_distributed(field, new_decomposition)
        gc.collect()
        level += 1


def batch_main_mpi(
    config_path: str,
    *,
    output_dir_override: str | None = None,
    comm=None,
) -> None:
    """Run one explicit AFM JSON with every rank cooperating on every case."""
    MPI = _require_mpi()
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())

    if rank == 0:
        resolved, cfg = load_mpi_config(config_path)
        if not cfg.get("mpi", {}).get("enabled", True):
            raise ValueError("configuration has mpi.enabled=false; use run_all.py instead")
        if cfg.get("zoom_simulation", {}).get("enabled", False):
            raise ValueError(
                "distributed MPI mode currently supports the main grid only; "
                "set zoom_simulation.enabled=false"
            )
        if output_dir_override:
            output_dir = os.path.join(
                os.path.abspath(os.path.expanduser(output_dir_override)), Path(resolved).stem
            )
        else:
            output_dir = resolve_output_dir(cfg.get("output_dir", "outputs"), resolved, cfg)
        payload = (resolved, cfg, os.path.abspath(output_dir))
    else:
        payload = None
    config_path, cfg, output_dir = comm.bcast(payload, root=0)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"MPI configuration: {config_path}")
        print(f"MPI output directory: {output_dir}")
    comm.Barrier()

    mov = cfg.get("movement", {})
    start_center = tuple(mov.get("start", (0.5, 0.5, 0.5)))
    end_center = tuple(mov.get("end", start_center))
    physical = cfg.get("_physical", {})
    physical_mov = physical.get("movement", {})
    if "spacing_nm" in physical_mov:
        centers = compute_block_positions(
            start_center,
            end_center,
            physical_mov["spacing_nm"],
            domain_nm=physical.get("domain_nm"),
        )
    else:
        centers = compute_block_positions(start_center, end_center, mov.get("spacing", 0.1))

    v_start = float(cfg["v_start"])
    v_stop = float(cfg["v_stop"])
    v_step = abs(float(cfg["v_step"]))
    if v_start <= v_stop:
        voltages = np.arange(v_start, v_stop + 1e-12, v_step)
    else:
        voltages = np.arange(v_start, v_stop - 1e-12, -v_step)
    if not len(voltages):
        raise ValueError("configuration voltage sweep is empty")

    stationary_upto = cfg.get("fixed_blocks", [2])[0]
    mobile_block_indices = list(range(stationary_upto, len(cfg["blocks"])))
    mobile_vgate_indices = list(range(len(cfg.get("Vgate", []))))
    center0 = centers[0]
    movement_active = len(centers) > 1
    movement_named = isinstance(cfg.get("movement"), dict)
    physical_cfg = cfg.get("_physical", {})
    domain_nm = tuple(float(v) for v in physical_cfg["domain_nm"])
    origin = tuple(float(v) for v in physical_cfg.get("origin_fraction", (0.5, 0.5, 0.0)))
    relative_blocks = precompute_relative_offsets_for_blocks(
        cfg["blocks"], mobile_block_indices, center0
    )
    relative_vgates = precompute_relative_offsets_for_vgates(
        cfg.get("Vgate", []), mobile_vgate_indices, center0
    )
    max_cut_bytes = int(float(cfg.get("mpi", {}).get("max_cut_gather_gib", 8.0)) * 1024**3)

    for center in centers:
        cfg["blocks"] = apply_relative_blocks(cfg["blocks"], relative_blocks, center)
        if cfg.get("Vgate") is not None:
            cfg["Vgate"] = apply_relative_vgates(cfg["Vgate"], relative_vgates, center)
        cx, cy, cz = (float(v) for v in center)
        movement_suffix = movement_suffix_nm(center, center0, domain_nm)

        for voltage in voltages:
            voltage = float(voltage)
            case_tag = f"move{movement_suffix}_V{voltage:.6f}"
            case_dir = os.path.join(output_dir, "mpi_logs", case_tag)
            if rank == 0:
                os.makedirs(case_dir, exist_ok=True)
                print(f"\n=== MPI AFM case {case_tag} ===")
            comm.Barrier()
            started = time.monotonic()
            field, metadata = run_distributed_hierarchy(
                cfg,
                Vtip=voltage,
                output_dir=case_dir,
                case_tag=case_tag,
                comm=comm,
            )
            shape = field.decomposition.global_shape
            probe_index = tuple(
                min(n - 1, max(0, int(frac * (n - 1))))
                for frac, n in zip(center, shape)
            )
            phi_value = distributed_probe(field, probe_index)
            npy_name = f"afm_phi_1_{voltage:.2f}V.npy"
            if movement_named:
                npy_name = f"afm_phi_1{movement_suffix}_{voltage:.2f}V.npy"

            center_nm = tuple((center[i] - origin[i]) * domain_nm[i] for i in range(3))
            field_bounds_nm = tuple(
                value
                for axis in range(3)
                for value in (-origin[axis] * domain_nm[axis], (1.0 - origin[axis]) * domain_nm[axis])
            )
            saved = []
            if bool(cfg.get("save_full", False)):
                saved.append(save_distributed_npy(field, npy_name, output_dir=output_dir))
            if bool(cfg.get("save_cut", False)):
                cut_name = resolution_tagged_cut_filename(npy_name, shape)
                cut_path, _ = save_distributed_physical_cut(
                    field,
                    center_nm,
                    cfg.get("save_cut_box_nm"),
                    field_bounds_nm,
                    filename=cut_name,
                    output_dir=output_dir,
                    max_root_bytes=max_cut_bytes,
                )
                if cut_path:
                    saved.append(cut_path)
            elapsed = time.monotonic() - started
            if rank == 0:
                print(
                    f"MPI case complete: phi({cx:.4f},{cy:.4f},{cz:.4f})={phi_value:.8e} V, "
                    f"elapsed={elapsed:.2f} s, outputs={saved or 'none'}"
                )
                _append_csv(
                    os.path.join(output_dir, cfg.get("csv_filename", "phi_vs_Vtip.csv")),
                    (
                        "cx_fraction",
                        "cy_fraction",
                        "cz_fraction",
                        "Vtip_V",
                        "phi_V",
                        "elapsed_s",
                        "levels",
                        "final_residual",
                        "final_result",
                    ),
                    (
                        cx,
                        cy,
                        cz,
                        voltage,
                        f"{phi_value:.12e}",
                        f"{elapsed:.6f}",
                        metadata["levels"],
                        f"{metadata['level_reports'][-1]['residual']:.12e}",
                        metadata["level_reports"][-1]["reason"],
                    ),
                )
            del field
            gc.collect()
            comm.Barrier()

    if rank == 0:
        print("MPI configuration processed successfully.")


__all__ = ["batch_main_mpi", "run_distributed_hierarchy"]
