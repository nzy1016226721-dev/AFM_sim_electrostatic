"""Storage options must change allocations, never the physical problem/norm."""
import numpy as np
import pytest
from simulation import ram_first as ram
from simulation.solver import mg_3d_masked, MG_TIME


@pytest.mark.parametrize("nx", [1, 7, 8, 15, 16, 127, 128, 129, 255, 510])
@pytest.mark.parametrize("starts", [(1,1,1), (0,0,0)])
def test_scalar_and_volume_match_both_existing_reductions(nx, starts):
    rng = np.random.default_rng(20261008+nx)
    shape = (nx+2, 7, 8)
    field = rng.normal(size=shape).astype(np.float32)
    epsilon = rng.uniform(1,15,size=tuple(n-1 for n in shape)).astype(np.float32)
    mask = rng.random((nx,5,6)) < .17
    global_shape = (nx+3,9,10)
    workspace = np.empty(mask.shape, np.float32)
    for pairwise in (False, True):
        reference = ram.residual_cells_stats(field,epsilon,mask,starts,global_shape,pairwise=pairwise)
        scalar = ram.residual_cells_stats(field,epsilon,mask,starts,global_shape,"scalar",pairwise=pairwise)
        array = ram.residual_cells_stats(field,epsilon,mask,starts,global_shape,"array",workspace,pairwise=pairwise)
        assert scalar == reference
        assert array == reference
    mask[:] = True
    assert ram.residual_cells_stats(field,epsilon,mask,starts,global_shape,"scalar") == (0.,0.,0)


@pytest.mark.parametrize("threads", [1,2,8])
@pytest.mark.parametrize("damping", [.8,1.])
def test_all_storage_combinations_are_bit_exact(tmp_path, threads, damping):
    rng = np.random.default_rng(45)
    phi = rng.normal(size=(19,15,13)).astype(np.float32)
    mask = rng.random(phi.shape) < .12
    mask[:,:,0] = True
    epsilon = rng.uniform(1,15,size=(18,14,12)).astype(np.float32)
    expected = None
    expected_csv = None
    for update in ("in_place", "buffered"):
        for accumulation in ("row_array", "scalar", "array"):
            current = phi.copy()
            destination = tmp_path / f"{update}_{accumulation}"
            destination.mkdir()
            result, residual = mg_3d_masked(-1,current,mask,eps_r=epsilon,damping=damping,
                cpu_threads=threads,return_residual=False,plotting_enabled=False,
                memory_mode="ram_first",phi_update_mode=update,residual_accumulation=accumulation,
                diagnostic_iterations=11,output_dir=str(destination))
            assert residual is None
            assert result.dtype == np.float32
            if update == "in_place":
                assert result is current
                assert MG_TIME["phi_buffer_bytes"] == 0
            else:
                assert MG_TIME["phi_buffer_bytes"] == current.nbytes
            assert MG_TIME["residual_workspace_bytes"] == (17*13*11*4 if accumulation == "array" else 0)
            history = (destination / "residual_history.csv").read_bytes()
            if expected is None:
                expected, expected_csv = result.copy(), history
            np.testing.assert_array_equal(result,expected)
            assert history == expected_csv


def test_scalar_never_calls_row_or_volume_kernels(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("scalar mode must not allocate residual/row arrays")
    monkeypatch.setattr(ram,"residual_cells_rows",forbidden)
    monkeypatch.setattr(ram,"fill_residual_volume",forbidden)
    field = np.ones((5,5,5), np.float32)
    assert ram.residual_cells_shared(field,np.ones((4,4,4),np.float32),np.zeros(field.shape,bool),"scalar") == (0.,0.)


def test_option_defaults_and_validation():
    assert ram.resolve_storage_options("ram_first") == ("in_place", "row_array")
    assert ram.resolve_storage_options("ram_first","in_place","scalar") == ("in_place","scalar")
    with pytest.raises(ValueError,match="require memory_mode"):
        ram.resolve_storage_options("standard",residual_accumulation="scalar")
    with pytest.raises(ValueError,match="phi_update_mode"):
        ram.resolve_storage_options("ram_first","gauss_seidel")
    with pytest.raises(ValueError,match="residual_accumulation"):
        ram.resolve_storage_options("ram_first",residual_accumulation="signed_mean")


def test_array_mode_needs_correct_workspace():
    field = np.ones((5,5,5),np.float32)
    with pytest.raises(ValueError,match="workspace"):
        ram.residual_cells_stats(field,np.ones((4,4,4),np.float32),np.zeros((3,3,3),bool),
                                 (1,1,1),field.shape,"array")
