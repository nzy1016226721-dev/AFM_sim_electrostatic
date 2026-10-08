import numpy as np

from postprocessing.compare_mpi_npy import compare_npy_fields


def test_streaming_npy_comparison_accepts_equal_arrays(tmp_path):
    reference = np.arange(120, dtype=np.float32).reshape(5, 4, 6)
    candidate = reference.copy()
    reference_path = tmp_path / "reference.npy"
    candidate_path = tmp_path / "candidate.npy"
    np.save(reference_path, reference)
    np.save(candidate_path, candidate)

    result = compare_npy_fields(reference_path, candidate_path, chunk_planes=1)

    assert result["passed"] is True
    assert result["compared_values"] == reference.size
    assert result["max_abs_diff"] == 0.0


def test_streaming_npy_comparison_reports_numeric_mismatch(tmp_path):
    reference = np.zeros((3, 4, 5), dtype=np.float32)
    candidate = reference.copy()
    candidate[2, 3, 4] = 1e-2
    reference_path = tmp_path / "reference.npy"
    candidate_path = tmp_path / "candidate.npy"
    np.save(reference_path, reference)
    np.save(candidate_path, candidate)

    result = compare_npy_fields(reference_path, candidate_path, atol=1e-5, rtol=1e-5)

    assert result["passed"] is False
    assert np.isclose(result["max_abs_diff"], 1e-2)
    assert result["reason"] == "numeric_or_dtype_mismatch"


def test_streaming_npy_comparison_rejects_shape_mismatch(tmp_path):
    reference_path = tmp_path / "reference.npy"
    candidate_path = tmp_path / "candidate.npy"
    np.save(reference_path, np.zeros((3, 4, 5), dtype=np.float32))
    np.save(candidate_path, np.zeros((3, 4, 6), dtype=np.float32))

    result = compare_npy_fields(reference_path, candidate_path)

    assert result["passed"] is False
    assert result["reason"] == "shape_mismatch"
