import numpy as np
import pytest

from simulation.parallel import configure_cpu_threads, weighted_jacobi_update_faces_parallel, residual_scalars_faces_parallel
from simulation.solver import _build_dielectric_face_fields
from simulation.ram_first import allocate_stream_scratch, jacobi_cells_in_place, residual_cells_shared
from simulation.materials import average_reference_to_cells
from simulation.materials_bounded import average_reference_bounded


def neumann(a):
    a[0,:,:] = a[1,:,:]; a[-1,:,:] = a[-2,:,:]
    a[:,0,:] = a[:,1,:]; a[:,-1,:] = a[:,-2,:]
    a[:,:,0] = a[:,:,1]; a[:,:,-1] = a[:,:,-2]


@pytest.mark.parametrize("shape", [(5,7,9), (17,14,11), (32,32,32)])
@pytest.mark.parametrize("threads", [1,2,8])
def test_snapshot_field_and_residual_are_bit_exact(shape, threads):
    configure_cpu_threads(threads)
    rng = np.random.default_rng(719)
    original = rng.uniform(-1, 0, size=shape).astype(np.float32)
    epsilon = rng.uniform(1, 15, size=tuple(v-1 for v in shape)).astype(np.float32)
    mask = rng.random(shape) < .05
    mask[:,:,0] = True
    fixed = original[mask]
    x, y, z = _build_dielectric_face_fields(epsilon)
    faces = (x[1:],x[:-1],y[:,1:],y[:,:-1],z[:,:,1:],z[:,:,:-1])
    reference, streamed = original.copy(), original.copy()
    target = reference.copy()
    scratch = allocate_stream_scratch(streamed, mask[1:-1,1:-1,1:-1], threads)
    for _ in range(11):
        weighted_jacobi_update_faces_parallel(reference, target, *faces, np.float32(.8))
        neumann(target); target[mask] = fixed
        reference, target = target, reference
        jacobi_cells_in_place(streamed, epsilon, mask[1:-1,1:-1,1:-1],
                             (1,1,1), shape, np.float32(.8), scratch, True)
        neumann(streamed); streamed[mask] = fixed
        np.testing.assert_array_equal(streamed, reference)
        assert residual_cells_shared(streamed, epsilon, mask) == residual_scalars_faces_parallel(reference, mask, *faces)


@pytest.mark.parametrize("target", [(7,5,9),(11,9,13),(13,5,13),(13,11,13)])
def test_bounded_rebin_is_bit_exact_and_cleans_files(tmp_path, target):
    reference = np.random.default_rng(181).uniform(1,15,size=(13,11,13)).astype(np.float32)
    path = tmp_path / "reference.npy"
    np.save(path, reference)
    mmap = np.load(path, mmap_mode="r")
    expected = average_reference_to_cells(reference, target)
    actual = average_reference_bounded(mmap, target, directory=tmp_path, scratch_budget_bytes=4000)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    mmap._mmap.close()
    assert list(tmp_path.glob("afm_rebin_*.npy")) == []


def test_rejects_precision_downgrade():
    with pytest.raises(ValueError, match="float32"):
        allocate_stream_scratch(np.zeros((5,5,5), dtype=np.float64), np.zeros((3,3,3),bool), 1)
