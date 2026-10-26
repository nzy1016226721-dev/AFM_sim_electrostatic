"""Opt-in policy decisions and bounded reductions, with no licensed work."""
import math
import numpy as np
import pytest
from simulation.stopping import StoppingController, resolve_stopping, level_stopping
from simulation.stopping_stats import extra_statistics
from simulation import ram_first as ram


def controller(spec, shape=(7,8,9), context=None, tmp_path=None):
    return StoppingController(spec,1e-6,shape,context,tmp_path)


@pytest.mark.parametrize("spec", [
    {"criterion":"bad"}, {"max_tol":0}, {"confirmation_checks":True},
    {"history":{"mode":"require"}}, {"stagnation":{"mode":"stop"}},
    {"regions":{"cut":{}}}, {"scope":"coarse"}, {"typo":True}])
def test_invalid_configuration_fails(spec):
    with pytest.raises((ValueError,TypeError)):
        resolve_stopping(spec,1e-6)


def test_finest_only_and_legacy_selection():
    assert resolve_stopping(None,1e-6) is None
    assert level_stopping({},(8,8,8),(16,16,16),1e-6) is None
    assert level_stopping({},(16,16,16),(16,16,16),1e-6)["criterion"] == "rms"


@pytest.mark.parametrize("criterion,rms,maximum,reason", [
    ("rms",.5e-6,2e-5,"converged"), ("maximum",.5e-6,2e-5,"running"),
    ("both",.5e-6,2e-5,"running"), ("both",.5e-6,.5e-5,"converged"),
    ("rms",1e-6,1e-6,"running")])
def test_exact_norm_predicates(criterion,rms,maximum,reason):
    stop = controller({"criterion":criterion,"max_tol":1e-5})
    assert stop.observe(1,rms,maximum,10) == reason


def test_short_three_check_confirmation():
    stop = controller({"confirmation_checks":3})
    assert stop.observe(10,.5e-6,.8e-6,10) == "running"
    assert stop.needs_check(11,False)
    assert stop.observe(11,.4e-6,.7e-6,10) == "running"
    assert stop.observe(12,.3e-6,.6e-6,10) == "converged"


def test_confirmation_failure_and_invalid_values_fail_closed():
    stop = controller({"confirmation_checks":3})
    stop.observe(1,.5e-6,.8e-6,10)
    assert stop.observe(2,2e-6,3e-6,10) == "running"
    assert stop.passing == 0
    assert stop.observe(3,float("nan"),3e-6,10) == "invalid_numerics"
    assert controller({}).observe(1,0.,0.,0) == "invalid_region"


def test_required_history_needs_magnitudes_and_unit_gaps():
    stop = controller({"history":{"mode":"require","mean_tol":1e-8,"max_tol":2e-8}})
    for i in (10,20,30):
        assert stop.observe(i,.5e-6,.8e-6,10) == "running"
    assert stop.observe(31,.5e-6,.8e-6,10) == "running"
    assert stop.observe(32,.5e-6,.8e-6,10) == "converged"
    not_small = controller({"history":{"mode":"require","mean_tol":1e-8,"max_tol":2e-8}})
    for i in (1,2,3):
        assert not_small.observe(i,.001,.01,10) == "running"


def test_precision_diagnostics_never_replace_legacy_global_norm():
    stop = controller({"precision":{"mode":"observe"}})
    extra = {"global":{"rms":0.,"maximum":0.,"count":10,"precision":{"rms":0.,"maximum":0.}}}
    assert stop.observe(1,1e-4,1e-3,10,extra) == "running"
    assert stop.last["statistics"]["global"]["rms"] == 1e-4


def test_stagnation_distinct_from_convergence_and_roundoff():
    settings = {"mode":"stop","window":3,"relative_improvement":0.,"motion_tol":0.}
    stop = controller({"stagnation":settings})
    assert stop.observe(1,.001,.01,10) == "running"
    assert stop.observe(2,.001,.01,10) == "running"
    assert stop.observe(3,.001,.01,10) == "stagnated"
    stop = controller({"precision":settings})
    extra = {"global":{"precision":{"rms":0.,"maximum":0.}}}
    for i in (1,2):
        assert stop.observe(i,.001,.01,10,extra) == "running"
    assert stop.observe(3,.001,.01,10,extra) == "precision_limited"


def test_empty_local_region_is_not_global_pass_and_bounds(tmp_path):
    shape=(7,8,9)
    field=np.random.default_rng(12).normal(size=shape).astype(np.float32)
    cells=np.ones(tuple(n-1 for n in shape),np.float32)
    mask=np.zeros(tuple(n-2 for n in shape),bool)
    stop=controller({"regions":{"cut":{"rms_tol":1e-6}},"precision":{"mode":"observe"}},
                    shape,{"cut_slices":[[0,7],[0,8],[0,9]]},tmp_path)
    stats=extra_statistics(field,cells,mask,(1,1,1),shape,stop)
    total,maximum,count=ram.residual_cells_stats(field,cells,mask,(1,1,1),shape,"scalar")
    assert stats["cut"]["count"] == count
    assert stats["cut"]["maximum"] == maximum
    assert stats["cut"]["rms"] == pytest.approx(math.sqrt(total/count),rel=1e-15)
    assert stats["cut"]["precision"]["same_float32_field_and_coefficients"]
    stop=controller({"regions":{"cut":{"rms_tol":1e-6}}},shape,{"cut_slices":[[0,0],[0,0],[0,0]]})
    empty=extra_statistics(field,cells,mask,(1,1,1),shape,stop)
    assert stop.observe(1,0.,0.,count,empty) == "invalid_region"


def test_qd_last_block_precedence_and_packed_dense_equivalence(tmp_path):
    from simulation.compact_storage import PlaneEpsilon,PackedMask
    shape=(7,8,9)
    field=np.random.default_rng(21).normal(size=shape).astype(np.float32)
    cells=np.ones(tuple(n-1 for n in shape),np.float32)
    mask=np.zeros(tuple(n-2 for n in shape),bool)
    blocks=[{"eps_val":15.}]
    spec={"regions":{"qd":{"max_tol":1e-6}}}
    stop=controller(spec,shape,{"blocks":blocks})
    dense=extra_statistics(field,cells,mask,(1,1,1),shape,stop)
    bank=PlaneEpsilon.from_dense(cells,directory=tmp_path)
    try:
        packed=PackedMask.from_dense(np.pad(mask,1)).kernel_view(offset=(1,1,1),shape=mask.shape)
        compact=extra_statistics(field,bank.kernel,packed,(1,1,1),shape,stop)
        assert compact == dense
    finally:
        bank.close()
    blocks.append({"eps_val":1.})
    erased=extra_statistics(field,cells,mask,(1,1,1),shape,controller(spec,shape,{"blocks":blocks}))
    assert erased["qd"]["count"] == 0


def test_new_policy_iteration_limit_is_not_legacy_max_iter():
    stop=controller({"max_iterations":2})
    assert stop.observe(1,.001,.01,10) == "running"
    assert stop.observe(2,.001,.01,10) == "iteration_limit"


def test_three_mpi_face_fields_match_six_shared_views():
    from simulation.mpi_domain import build_local_face_fields
    shape=(7,8,9)
    field=np.random.default_rng(123).normal(size=shape).astype(np.float32)
    cells=np.ones(tuple(n-1 for n in shape),np.float32)
    faces=build_local_face_fields(cells)
    x,y,z=faces
    shifted=(x[1:,:,:],x[:-1,:,:],y[:,1:,:],y[:,:-1,:],z[:,:,1:],z[:,:,:-1])
    mask=np.zeros(tuple(n-2 for n in shape),bool)
    stop=controller({"regions":{"cut":{"rms_tol":1e-6}}},shape,
                    {"cut_slices":[[0,7],[0,8],[0,9]]})
    a=extra_statistics(field,None,mask,(1,1,1),shape,stop,faces=faces)
    b=extra_statistics(field,None,mask,(1,1,1),shape,stop,faces=shifted)
    assert a == b
