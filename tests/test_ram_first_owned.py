from types import SimpleNamespace
import numpy as np
import pytest
from simulation.parallel import configure_cpu_threads
from simulation.mpi_domain import build_local_face_fields, _jacobi_owned, _residual_rows, _apply_neumann_boundaries
from simulation.ram_first import allocate_stream_scratch, jacobi_cells_in_place, residual_cells_rows


@pytest.mark.parametrize("starts", [(0,0,0),(8,7,6),(16,15,14)])
@pytest.mark.parametrize("threads", [1,2,8])
def test_owned_halos_fixed_masks_boundaries_and_residuals_exact(starts,threads):
    configure_cpu_threads(threads)
    counts = (7,6,5)
    shape = (23,21,19)
    rng = np.random.default_rng(55)
    reference = rng.uniform(-1,0,size=tuple(n+2 for n in counts)).astype(np.float32)
    streamed = reference.copy()
    target = reference.copy()
    epsilon = rng.uniform(1,15,size=tuple(n+1 for n in counts)).astype(np.float32)
    mask = rng.random(counts) < .1
    fixed = reference[1:-1,1:-1,1:-1][mask]
    faces = build_local_face_fields(epsilon)
    decomp = SimpleNamespace(starts=starts,stops=tuple(s+n for s,n in zip(starts,counts)),global_shape=shape)
    scratch = allocate_stream_scratch(streamed,mask,threads)
    for _ in range(9):
        _jacobi_owned(reference,target,mask,*faces,*starts,*shape,np.float32(.8))
        _apply_neumann_boundaries(target,decomp)
        target[1:-1,1:-1,1:-1][mask] = fixed
        reference,target = target,reference
        jacobi_cells_in_place(streamed,epsilon,mask,starts,shape,np.float32(.8),scratch)
        _apply_neumann_boundaries(streamed,decomp)
        streamed[1:-1,1:-1,1:-1][mask] = fixed
        np.testing.assert_array_equal(streamed[1:-1,1:-1,1:-1],reference[1:-1,1:-1,1:-1])
        # Supply exactly identical current-epoch remote halos for the next pass.
        reference[0] = streamed[0]; reference[-1] = streamed[-1]
        reference[:,0] = streamed[:,0]; reference[:,-1] = streamed[:,-1]
        reference[:,:,0] = streamed[:,:,0]; reference[:,:,-1] = streamed[:,:,-1]
        sums = np.zeros(counts[0],np.float64)
        maxima = sums.copy()
        numbers = np.zeros(counts[0],np.int64)
        _residual_rows(reference,mask,*faces,*starts,*shape,sums,maxima,numbers)
        rows = residual_cells_rows(streamed,epsilon,mask,starts,shape)
        for actual,expected in zip(rows,(sums,maxima,numbers)):
            np.testing.assert_array_equal(actual,expected)


def test_large_rank_estimate_saves_ram_but_counts_workspaces():
    from simulation.mpi_domain import estimate_rank_peak_bytes
    counts = (512,512,512)
    standard = estimate_rank_peak_bytes(counts)
    streamed = estimate_rank_peak_bytes(counts,memory_mode="ram_first",cpu_threads=8)
    assert streamed < standard
    assert streamed > 4*(2*514**3+513**3+512**3)
