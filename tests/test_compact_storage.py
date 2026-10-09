"""Representation changes must replay the dense cells/masks/stencil bit-for-bit."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from simulation.compact_storage import (PlaneEpsilon,PackedMask,TipPredicate,
    rasterize_planes,generate_eps_compact,build_reference_planes,restore_fixed)
from simulation.materials import _fine_eps,generate_eps_level,release_eps_reference,average_reference_to_cells
from simulation.materials_bounded import average_reference_bounded
from simulation.solver import build_downward_pointing_tip,mg_3d_masked,MG_TIME
from simulation.mpi_domain import build_local_tip_mask,rasterize_epsilon_halo
from simulation import ram_first as ram


def bits_equal(a,b):
    np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))


def test_bank_deduplicates_exact_bits_and_cleans(tmp_path):
    a = np.ones((6,5,7),np.float32)
    a[1,2,3] = np.float32(1.0000001)
    a[2,2,3] = np.float32(-0.)
    a[3,2,3] = np.float32(0.)
    a[4] = a[1]
    bank = PlaneEpsilon.from_dense(a,directory=tmp_path)
    try:
        assert bank.bank.shape[0] == 4
        assert bank.ids[1] == bank.ids[4]
        assert bank.ids[2] != bank.ids[3]
        bits_equal(bank.to_dense(),a)
        assert bank.nbytes == 4*5*7*4+6*4
    finally:
        bank.close()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("shape",[(8,9,11),(13,7,5)])
def test_exact_raster_and_rank_padding(shape,tmp_path):
    blocks=[{"eps_val":12.5,"z_range":[.4,0.]},
            {"eps_val":15.,"x_range":[.2,.9],"y_range":[.25,.65],"z_range":[.11,.42]},
            {"eps_val":3.0000002,"x_range":[.7,.3],"y_range":None},
            {"eps_val":9,"x_range":[.5,.5]}, {"eps_val":7,"z_range":[2,3]}]
    bank = rasterize_planes(shape,blocks,directory=tmp_path)
    try:
        bits_equal(bank.to_dense(),_fine_eps(shape,blocks))
    finally:
        bank.close()
    global_shape = tuple(v+1 for v in shape)
    for starts,counts in [((0,0,0),(3,4,2)),((shape[0]-2,shape[1]-2,shape[2]-2),(3,3,3))]:
        bank = rasterize_planes(shape,blocks,starts=tuple(v-1 for v in starts),
                                counts=tuple(v+1 for v in counts),directory=tmp_path)
        try:
            bits_equal(bank.to_dense(),rasterize_epsilon_halo(global_shape,starts,counts,blocks))
        finally:
            bank.close()


@pytest.mark.parametrize("target",[(7,5,9),(11,9,13),(13,11,13),(15,9,13)])
def test_reference_rebin_preserves_original_bits(target,tmp_path):
    blocks=[{"eps_val":12.5,"z_range":[0,.123]},
            {"eps_val":15,"x_range":[.24,.73],"y_range":[.18,.81],"z_range":[.11,.44]},
            {"eps_val":5.0625,"x_range":[.63,.77]}]
    reference_shape=(13,11,13)
    path, reference = build_reference_planes(reference_shape,blocks,tmp_path)
    try:
        bits_equal(reference,_fine_eps(reference_shape,blocks))
        bank = generate_eps_compact(tuple(v+1 for v in target),blocks,reference_shape,reference,directory=tmp_path)
        try:
            bits_equal(bank.to_dense(),generate_eps_level(tuple(v+1 for v in target),blocks,reference_shape,reference))
        finally:
            bank.close()
    finally:
        release_eps_reference(path,reference)
    assert list(tmp_path.iterdir()) == []


def test_file_backed_final_matches_dense_and_cleanup(tmp_path):
    ref=np.random.default_rng(34).uniform(1,15,(13,11,13)).astype(np.float32)
    target=(7,5,9)
    path=tmp_path/"final.npy"
    final=average_reference_bounded(ref,target,directory=tmp_path,final_path=path,scratch_budget_bytes=4000)
    try:
        assert isinstance(final,np.memmap)
        bits_equal(final,average_reference_to_cells(ref,target))
    finally:
        final._mmap.close()
        path.unlink()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("shape",[(5,7,9),(3,2,1),(7,11,17)])
def test_mask_packing_fixed_order_and_box(shape,tmp_path):
    rng=np.random.default_rng(54)
    mask=rng.random(shape)<.4
    packed=PackedMask.from_dense(mask)
    np.testing.assert_array_equal(packed.to_dense(),mask)
    assert packed.nbytes == shape[0]*((shape[1]*shape[2]+7)//8)
    field=rng.normal(size=tuple(n+2 for n in shape)).astype(np.float32)
    expected=field.copy()
    values,prefixes=packed.fixed_values(field,(1,1,1))
    bits_equal(values,field[1:-1,1:-1,1:-1][mask])
    field[1:-1,1:-1,1:-1][mask]=987
    restore_fixed(field,packed.bits,shape,values,prefixes,(1,1,1))
    bits_equal(field,expected)
    box=(slice(0,shape[0]),slice(0,1),slice(0,shape[2]))
    packed.mark_box(box); mask[box]=True
    np.testing.assert_array_equal(packed.to_dense(),mask)


@pytest.mark.parametrize("shape",[(13,19,17),(33,27,21)])
@pytest.mark.parametrize("tip_shape",["cone","pyramid"])
@pytest.mark.parametrize("physical",[False,True])
def test_tip_predicate_matches_both_dense_builders(shape,tip_shape,physical):
    params=dict(domain_nm=(133.,97.,80.),origin_fraction=(.31,.63,.14),
                tip_z_nm=10.,R_nm=2.,r_tip_nm=45.) if physical else None
    values=dict(tip_z=.233,R=.07,r_tip=.33,aspect_ratio=2.,tip_shape=tip_shape)
    compact=TipPredicate(shape,physical_params=params,**values)
    extra=dict(tip_z_nm=params["tip_z_nm"],R_nm=params["R_nm"],r_tip_nm=params["r_tip_nm"],
               domain_nm=params["domain_nm"],center_fraction=params["origin_fraction"]) if physical else {}
    dense,tip,base=build_downward_pointing_tip(*shape,verbose=False,**values,**extra)
    np.testing.assert_array_equal(compact.to_dense(),dense)
    assert compact.tip_pos==tip and compact.base_pos==base
    for starts,counts in [((0,0,0),(5,6,7)),((4,5,3),(7,9,9))]:
        owned=TipPredicate(shape,starts=starts,counts=counts,physical_params=params,**values)
        decomp=SimpleNamespace(global_shape=shape,starts=starts,counts=counts)
        reference,_,_=build_local_tip_mask(decomp,physical_params=params,**values)
        np.testing.assert_array_equal(owned.to_dense(),reference)
    assert compact.nbytes < np.prod(shape)


@pytest.mark.parametrize("threads",[1,2,8])
@pytest.mark.parametrize("shape",[(5,7,9),(19,15,13)])
def test_all_storage_combinations_reuse_exact_dense_kernels(tmp_path,threads,shape):
    rng=np.random.default_rng(719)
    phi=rng.normal(size=shape).astype(np.float32)
    eps=rng.uniform(1,15,tuple(n-1 for n in shape)).astype(np.float32)
    mask=rng.random(shape)<.12
    mask[:,:,0]=True
    dense_dir=tmp_path/"dense"; dense_dir.mkdir()
    expected,_=mg_3d_masked(-1,phi.copy(),mask,eps_r=eps,damping=.8,cpu_threads=threads,
        return_residual=False,plotting_enabled=False,memory_mode="ram_first",
        residual_accumulation="scalar",diagnostic_iterations=11,output_dir=str(dense_dir))
    expected_csv=(dense_dir/"residual_history.csv").read_bytes()
    bank=PlaneEpsilon.from_dense(eps,directory=tmp_path)
    packed=PackedMask.from_dense(mask)
    try:
        for update in ("in_place","buffered"):
            for accumulation in ("scalar","row_array","array"):
                folder=tmp_path/f"{update}_{accumulation}";folder.mkdir()
                current=phi.copy()
                actual,residual=mg_3d_masked(-1,current,packed,eps_r=bank,damping=.8,cpu_threads=threads,
                    return_residual=False,plotting_enabled=False,memory_mode="ram_compact",
                    phi_update_mode=update,residual_accumulation=accumulation,
                    diagnostic_iterations=11,output_dir=str(folder))
                bits_equal(actual,expected)
                assert residual is None
                assert (folder/"residual_history.csv").read_bytes()==expected_csv
                assert MG_TIME["phi_buffer_bytes"]==(phi.nbytes if update=="buffered" else 0)
                assert MG_TIME["residual_workspace_bytes"]==(np.prod([n-2 for n in shape])*4 if accumulation=="array" else 0)
                assert MG_TIME["mask_storage_bytes"]==packed.nbytes
                if update=="in_place":
                    assert actual is current
    finally:
        bank.close()


@pytest.mark.parametrize("nx",[1,7,8,127,128,129,255])
def test_adapter_preserves_mpi_pairwise_order(nx,tmp_path):
    rng=np.random.default_rng(nx)
    shape=(nx+2,7,8)
    phi=rng.normal(size=shape).astype(np.float32)
    eps=rng.uniform(1,15,tuple(n-1 for n in shape)).astype(np.float32)
    mask=rng.random((nx,5,6))<.17
    bank=PlaneEpsilon.from_dense(eps,directory=tmp_path)
    packed=PackedMask.from_dense(mask)
    try:
        for strategy in ("scalar","row_array","array"):
            old_work=np.empty(mask.shape,np.float32); new_work=np.empty_like(old_work)
            expected=ram.residual_cells_stats(phi,eps,mask,(0,0,0),(nx+3,9,10),strategy,old_work,pairwise=True)
            actual=ram.residual_cells_stats(phi,bank.kernel,packed.kernel_view(),(0,0,0),(nx+3,9,10),strategy,new_work,pairwise=True)
            assert actual==expected
    finally:
        bank.close()


def test_validation_and_defaults():
    assert ram.resolve_storage_options("ram_compact")==("in_place","scalar")
    assert ram.resolve_storage_options("ram_first")==("in_place","row_array")
    with pytest.raises(ValueError,match="float32"):
        PlaneEpsilon.from_dense(np.ones((4,4,4),np.float64))
    with pytest.raises(ValueError,match="bool"):
        PackedMask.from_dense(np.zeros((4,4,4),np.float32))
    with pytest.raises(ValueError,match="exceeds"):
        PackedMask((4,4,4)).kernel_view(offset=(1,1,1))


@pytest.mark.parametrize("tip_shape", ["cone", "pyramid"])
def test_shared_hierarchy_gate_precedence_and_cleanup(tmp_path, tip_shape):
    from simulation.main_loop import run_afm_simulation
    common = dict(tip_z=.2,R=.06,r_tip=.4,aspect_ratio=2.,tip_shape=tip_shape,
        blocks=[{"eps_val":12.5,"z_range":[0,.18]},
                {"eps_val":15.,"x_range":[.32,.72],"z_range":[.12,.36]}],
        Vgate=[{"z_range":[0,0]}, {"z_range":[.22,.8],"Vgate_val":-.31}],
        plotting_enabled=False,return_residual=False,verbose=False,cpu_threads=8,
        initial_grid_level=8,eps_reference_resolution=(17,19,21),tol=1e-6,
        mg_max_runtime=None,residual_accumulation="scalar")
    dense_dir=tmp_path/"dense"; dense_dir.mkdir()
    compact_dir=tmp_path/"compact"; compact_dir.mkdir()
    dense=run_afm_simulation(-1,16,15,13,output_dir=str(dense_dir),memory_mode="ram_first",**common)
    compact=run_afm_simulation(-1,16,15,13,output_dir=str(compact_dir),memory_mode="ram_compact",**common)
    bits_equal(compact["phi"],dense["phi"])
    np.testing.assert_array_equal(compact["boundary_mask"].to_dense(),dense["boundary_mask"])
    np.testing.assert_array_equal(compact["tip_mask"].to_dense(),dense["tip_mask"])
    assert (compact_dir/"residual_history.csv").read_bytes()==(dense_dir/"residual_history.csv").read_bytes()
    assert np.all(compact["phi"][dense["tip_mask"]] == -1)
    assert not list(compact_dir.glob("afm_*"))  # only logs, no leaked reference/bank/rebin


def test_failure_releases_owned_storage_but_keeps_caller_bank(tmp_path, monkeypatch):
    from simulation import main_loop
    def failure(*args,**kwargs):
        raise RuntimeError("injected solve failure")
    monkeypatch.setattr(main_loop,"mg_3d_masked",failure)
    common=dict(plotting_enabled=False,return_residual=False,verbose=False,
                memory_mode="ram_compact",initial_grid_level=6,output_dir=str(tmp_path))
    with pytest.raises(RuntimeError,match="injected"):
        main_loop.run_afm_simulation(-1,6,6,6,eps_reference_resolution=9,**common)
    assert list(tmp_path.iterdir())==[]
    bank=PlaneEpsilon.from_dense(np.ones((5,5,5),np.float32),directory=tmp_path)
    try:
        with pytest.raises(RuntimeError,match="injected"):
            main_loop.run_afm_simulation(-1,6,6,6,eps_r=bank,**common)
        assert bank.path.exists() and bank.bank is not None
    finally:
        bank.close()
    assert list(tmp_path.iterdir())==[]


def test_all_fixed_nodes_and_unsupported_modes(tmp_path):
    phi=np.linspace(-.9,-.1,6**3,dtype=np.float32).reshape((6,6,6))
    expected=phi.copy()
    mask=PackedMask.from_dense(np.ones_like(phi,dtype=bool))
    result,_=mg_3d_masked(-1,phi,mask,eps_r=np.ones((5,5,5),np.float32),
        memory_mode="ram_compact",plotting_enabled=False,return_residual=False,
        output_dir=str(tmp_path),cpu_threads=8,tol=1e-6)
    bits_equal(result,expected)
    assert MG_TIME["iterations"]==1 and MG_TIME["residual"]==0 and MG_TIME["residual_max"]==0
    assert MG_TIME["residual_workspace_bytes"]==0 and MG_TIME["phi_buffer_bytes"]==0
    with pytest.raises(ValueError,match="float32"):
        mg_3d_masked(-1,expected.astype(np.float64),mask,
            memory_mode="ram_compact",plotting_enabled=False,return_residual=False)
    with pytest.raises(ValueError,match="plotting disabled"):
        mg_3d_masked(-1,expected,mask,memory_mode="ram_compact",plotting_enabled=True,return_residual=False)


def test_compact_planner_has_unique_plane_fallback_and_actual_selectors():
    from simulation.mpi_domain import rank_layout_table,estimate_rank_peak_bytes
    counts=(11,9,7)
    base=estimate_rank_peak_bytes(counts,memory_mode="ram_compact",cpu_threads=8)
    assert base >= np.prod(tuple(n+1 for n in counts))*4
    assert estimate_rank_peak_bytes(counts,memory_mode="ram_compact",cpu_threads=8,
        residual_accumulation="array") == base + np.prod(counts)*4
    rows=rank_layout_table((21,19,17),(2,1,1),memory_mode="ram_compact",cpu_threads=8,reference_shape=(23,23,23))
    assert rows[0]["estimated_peak_bytes"] > rows[1]["estimated_peak_bytes"]
    for row in rows:
        direct=estimate_rank_peak_bytes(row["counts"],memory_mode="ram_compact",cpu_threads=8)
        assert row["estimated_peak_bytes"] >= direct
