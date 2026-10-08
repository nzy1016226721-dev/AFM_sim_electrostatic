#!/usr/bin/env python3
"""Memory-bounded numerical comparison for serial and MPI AFM ``.npy`` fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def compare_npy_fields(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    atol: float = 2e-5,
    rtol: float = 2e-5,
    chunk_planes: int = 1,
) -> dict[str, Any]:
    """Compare two numeric NPY arrays without loading an entire field into RAM.

    Arrays are memory-mapped and compared one first-axis slab at a time. This
    permits the same command to compare small physical cuts or a full 2048^3
    output without creating another full in-memory array.
    """
    reference_path = Path(reference_path).expanduser().resolve()
    candidate_path = Path(candidate_path).expanduser().resolve()
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")
    if int(chunk_planes) < 1:
        raise ValueError("chunk_planes must be at least one")

    reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    summary: dict[str, Any] = {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "atol": float(atol),
        "rtol": float(rtol),
        "chunk_planes": int(chunk_planes),
    }
    if reference.shape != candidate.shape:
        summary.update(
            {
                "compared_values": 0,
                "finite_values": 0,
                "nonfinite_values": 0,
                "max_abs_diff": None,
                "rms_abs_diff": None,
                "max_relative_diff": None,
                "passed": False,
                "reason": "shape_mismatch",
            }
        )
        return summary

    if reference.dtype.kind not in "fiu" or candidate.dtype.kind not in "fiu":
        raise TypeError("both NPY arrays must have integer or floating dtypes")

    if reference.ndim == 0:
        chunks = [(Ellipsis,)]
    else:
        chunks = [
            (slice(start, min(start + int(chunk_planes), reference.shape[0])),)
            + (slice(None),) * (reference.ndim - 1)
            for start in range(0, reference.shape[0], int(chunk_planes))
        ]

    compared = 0
    finite_values = 0
    nonfinite_values = 0
    max_abs = 0.0
    max_relative = 0.0
    sum_sq = 0.0
    value_passed = True
    for index in chunks:
        reference_chunk = np.asarray(reference[index], dtype=np.float64)
        candidate_chunk = np.asarray(candidate[index], dtype=np.float64)
        finite = np.isfinite(reference_chunk) & np.isfinite(candidate_chunk)
        compared += int(reference_chunk.size)
        finite_values += int(np.count_nonzero(finite))
        nonfinite_values += int(reference_chunk.size - np.count_nonzero(finite))
        if not np.any(finite):
            value_passed = False
            continue
        difference = np.abs(candidate_chunk[finite] - reference_chunk[finite])
        allowed = float(atol) + float(rtol) * np.abs(reference_chunk[finite])
        if np.any(difference > allowed):
            value_passed = False
        max_abs = max(max_abs, float(np.max(difference)))
        scale = np.maximum(np.abs(reference_chunk[finite]), np.finfo(np.float64).tiny)
        max_relative = max(max_relative, float(np.max(difference / scale)))
        sum_sq += float(np.dot(difference, difference))

    if nonfinite_values:
        value_passed = False
    dtype_match = reference.dtype == candidate.dtype
    summary.update(
        {
            "compared_values": compared,
            "finite_values": finite_values,
            "nonfinite_values": nonfinite_values,
            "max_abs_diff": max_abs if finite_values else None,
            "rms_abs_diff": (sum_sq / finite_values) ** 0.5 if finite_values else None,
            "max_relative_diff": max_relative if finite_values else None,
            "dtype_match": bool(dtype_match),
            "value_within_tolerance": bool(value_passed),
            "passed": bool(dtype_match and value_passed),
            "reason": "ok" if dtype_match and value_passed else "numeric_or_dtype_mismatch",
        }
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare serial and MPI AFM NPY fields with memory-mapped chunks."
    )
    parser.add_argument("reference", help="Trusted serial/reference NPY file")
    parser.add_argument("candidate", help="MPI candidate NPY file")
    parser.add_argument("--atol", type=float, default=2e-5, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=2e-5, help="Relative tolerance")
    parser.add_argument(
        "--chunk-planes",
        type=int,
        default=1,
        help="First-axis planes held in memory at once; default is safest for full fields",
    )
    parser.add_argument("--report", help="Optional JSON report path")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = compare_npy_fields(
            args.reference,
            args.candidate,
            atol=args.atol,
            rtol=args.rtol,
            chunk_planes=args.chunk_planes,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.report:
        report = Path(args.report).expanduser().resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(text + "\n", encoding="utf-8")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
