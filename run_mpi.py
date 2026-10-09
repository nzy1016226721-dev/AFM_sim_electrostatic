#!/usr/bin/env python3
"""Distributed-memory MPI entry point and offline resource planner."""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback

# Keep per-rank Numba caches on node-local storage. This must be set before
# importing the cached kernels from simulation.mpi_domain.
if not os.environ.get("NUMBA_CACHE_DIR") and os.environ.get("SLURM_TMPDIR"):
    process_id = os.environ.get("SLURM_PROCID", "batch")
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(
        os.environ["SLURM_TMPDIR"], f"afm_numba_cache_{process_id}"
    )

from simulation.mpi_config import load_mpi_config
from simulation.mpi_domain import (
    human_bytes,
    rank_layout_table,
    suggest_process_grid,
    validate_process_grid,
)
from simulation.numerics import resolve_solver_dtype


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one AFM grid with 3-D MPI domain decomposition, or calculate "
            "a safe rank/node layout without allocating the grid."
        )
    )
    parser.add_argument("config_path", help="Explicit AFM JSON configuration")
    parser.add_argument("--output-dir", help="Output root override")
    parser.add_argument("--no-plot", action="store_true", help="Accepted for launcher compatibility")
    parser.add_argument("--plan", action="store_true", help="Print a no-allocation memory/resource plan")
    parser.add_argument("--ranks", type=int, help="Total MPI ranks for --plan")
    parser.add_argument(
        "--process-grid",
        help="Cartesian rank grid PX,PY,PZ; runtime also accepts AFM_MPI_PROCESS_GRID",
    )
    parser.add_argument("--ranks-per-node", type=int, help="Rank packing for --plan")
    parser.add_argument("--node-memory-gib", type=float, help="Physical/requested GiB per node for --plan")
    parser.add_argument(
        "--memory-fraction",
        type=float,
        help="Usable fraction of node memory; defaults to JSON mpi.memory_fraction",
    )
    return parser


def _parse_grid(text: str | None):
    if not text:
        return None
    dims = tuple(int(item.strip()) for item in text.split(","))
    if len(dims) != 3:
        raise ValueError("--process-grid must be PX,PY,PZ")
    return dims


def _read_config(path: str) -> tuple[str, dict]:
    return load_mpi_config(path)


def _plan(args) -> int:
    resolved, cfg = _read_config(args.config_path)
    target = tuple(int(cfg["grid_resolution"][key]) for key in ("nx", "ny", "nz"))
    field_dtype = resolve_solver_dtype(cfg.get("solver_dtype", "float32"))
    mpi_cfg = cfg.get("mpi", {})
    ranks = int(args.ranks or mpi_cfg.get("planner_ranks", 0))
    if ranks < 1:
        raise ValueError("--plan requires --ranks (or mpi.planner_ranks in JSON)")
    dims = _parse_grid(args.process_grid)
    if dims is None:
        json_dims = mpi_cfg.get("process_grid")
        if json_dims and any(int(v) for v in json_dims):
            dims = tuple(int(v) for v in json_dims)
        else:
            dims = suggest_process_grid(ranks, target)
    validate_process_grid(dims, ranks, target)
    reference_value = cfg.get("epsilon_material", {}).get("reference_resolution", 512)
    reference_shape = ((int(reference_value),)*3 if not isinstance(reference_value,(list,tuple))
                       else tuple(int(v) for v in reference_value))
    rows = rank_layout_table(
        target, dims, dtype=field_dtype,
        memory_mode=cfg.get("memory_mode", "standard"),
        cpu_threads=int(cfg.get("cpu_threads", 1)),
        reference_shape=reference_shape,
        phi_update_mode=cfg.get("phi_update_mode"),
        residual_accumulation=cfg.get("residual_accumulation"),
    )

    ranks_per_node = int(args.ranks_per_node or mpi_cfg.get("planner_ranks_per_node", 1))
    if ranks_per_node < 1:
        raise ValueError("--ranks-per-node must be positive")
    node_memory_gib = float(args.node_memory_gib or mpi_cfg.get("planner_node_memory_gib", 0.0))
    fraction = float(
        args.memory_fraction
        if args.memory_fraction is not None
        else mpi_cfg.get("memory_fraction", 0.80)
    )
    if not 0.1 <= fraction <= 0.95:
        raise ValueError("memory fraction must be between 0.1 and 0.95")

    per_node = []
    for node in range(math.ceil(ranks / ranks_per_node)):
        group = rows[node * ranks_per_node : (node + 1) * ranks_per_node]
        per_node.append(sum(int(row["estimated_peak_bytes"]) for row in group))
    worst_rank = max(int(row["estimated_peak_bytes"]) for row in rows)
    worst_node = max(per_node)
    total_estimated = sum(int(row["estimated_peak_bytes"]) for row in rows)
    field_bytes = math.prod(target) * field_dtype.itemsize
    safe = None
    if node_memory_gib > 0:
        safe = worst_node <= node_memory_gib * 1024**3 * fraction

    print("AFM MPI no-allocation resource plan")
    print(f"Config: {resolved}")
    print(f"Global grid: {target[0]}x{target[1]}x{target[2]}")
    print(f"One {field_dtype.name} global field: {human_bytes(field_bytes)}")
    print(f"MPI ranks: {ranks}; process grid: {dims}")
    print(f"Ranks per node: {ranks_per_node}; nodes: {len(per_node)}")
    print(f"Worst local block: {max(rows, key=lambda row: int(row['estimated_peak_bytes']))['counts']}")
    print(f"Worst estimated rank peak: {human_bytes(worst_rank)}")
    print(f"Worst estimated node peak: {human_bytes(worst_node)}")
    print(f"Sum of rank peak estimates: {human_bytes(total_estimated)}")
    if node_memory_gib > 0:
        allowed = node_memory_gib * 1024**3 * fraction
        print(
            f"Node memory check: {human_bytes(node_memory_gib * 1024**3)} physical/requested, "
            f"{human_bytes(allowed)} allowed at {fraction:.0%}: {'PASS' if safe else 'FAIL'}"
        )
    else:
        print("Node memory check: not evaluated; pass --node-memory-gib")
    return 0 if safe is not False else 2


def _run(args) -> int:
    if args.ranks or args.ranks_per_node or args.node_memory_gib:
        raise ValueError("planning resource arguments require --plan")
    import mpi4py

    mpi4py.rc.thread_level = "funneled"
    from mpi4py import MPI

    from simulation.mpi_main import batch_main_mpi

    comm = MPI.COMM_WORLD
    try:
        if _parse_grid(args.process_grid) is not None:
            os.environ["AFM_MPI_PROCESS_GRID"] = args.process_grid
        batch_main_mpi(args.config_path, output_dir_override=args.output_dir, comm=comm)
        return 0
    except BaseException:
        print(f"[MPI rank {comm.Get_rank()}] fatal AFM error:", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        if comm.Get_size() > 1:
            comm.Abort(1)
        raise


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _plan(args) if args.plan else _run(args)
    except (ValueError, RuntimeError, MemoryError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
