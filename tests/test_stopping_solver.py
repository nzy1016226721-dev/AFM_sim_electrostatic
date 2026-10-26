"""Exercise the opt-in seam through the maintained shared numerical paths."""
import numpy as np
import pytest
from simulation.solver import mg_3d_masked, MG_TIME
from simulation.coordinates import normalize_config, physical_domain_nm


def inputs():
    rng=np.random.default_rng(1710)
    phi=np.full((12,10,9),.001,np.float32)
    mask=np.zeros(phi.shape,bool)
    mask[:,:,0]=True; phi[:,:,0]=0.
    mask[:,:,-1]=True; phi[:,:,-1]=-1.
    cells=rng.uniform(1,15,size=(11,9,8)).astype(np.float32)
    return phi,mask,cells


@pytest.mark.parametrize("mode",["standard","ram_first","ram_compact"])
def test_opt_in_rms_reproduces_original_bits_and_csv(tmp_path,mode):
    phi,mask,cells=inputs()
    results=[]; histories=[]
    for name,spec in (("legacy",None),("explicit",{"criterion":"rms"})):
        out=tmp_path/name; out.mkdir()
        result,_=mg_3d_masked(-1,phi.copy(),mask,eps_r=cells,damping=.8,tol=1e-5,
            cpu_threads=2,return_residual=False,plotting_enabled=False,verbose=False,
            memory_mode=mode,output_dir=str(out),stopping=spec)
        assert MG_TIME["reason"] == "converged"
        results.append(result.copy())
        histories.append((out/"residual_history.csv").read_bytes())
    np.testing.assert_array_equal(*results)
    assert histories[0] == histories[1]


@pytest.mark.parametrize("mode",["standard","ram_first","ram_compact"])
def test_new_iteration_limit_and_region_diagnostics_do_not_change_updates(tmp_path,mode):
    phi,mask,cells=inputs()
    options=dict(eps_r=cells,damping=.8,tol=1e-30,cpu_threads=2,return_residual=False,
                 plotting_enabled=False,verbose=False,memory_mode=mode)
    reference_dir=tmp_path/"reference"; reference_dir.mkdir()
    expected,_=mg_3d_masked(-1,phi.copy(),mask,diagnostic_iterations=12,
                            output_dir=str(reference_dir),**options)
    out=tmp_path/"new"; out.mkdir()
    actual,_=mg_3d_masked(-1,phi.copy(),mask,output_dir=str(out),
        stopping={"criterion":"both","max_iterations":12,"regions":{"cut":{"max_tol":1e-30}},
                  "precision":{"mode":"observe"}},
        stopping_context={"cut_slices":[[0,12],[0,10],[0,9]]},**options)
    np.testing.assert_array_equal(actual,expected)
    assert MG_TIME["reason"] == "iteration_limit"
    assert MG_TIME["stopping"]["last"]["statistics"]["cut"]["count"] == 10*8*7
    assert MG_TIME["stopping"]["full_residual_history_bytes"] == 0


@pytest.mark.parametrize("mode",["ram_first","ram_compact"])
def test_timeout_remains_effective_with_active_policy(tmp_path,mode):
    phi,mask,cells=inputs()
    mg_3d_masked(-1,phi,mask,eps_r=cells,damping=.8,tol=1e-30,cpu_threads=2,
                return_residual=False,plotting_enabled=False,verbose=False,memory_mode=mode,
                output_dir=str(tmp_path),mg_max_runtime=0.,stopping={"criterion":"maximum"})
    assert MG_TIME["reason"] == "time_limit"
    # Windows monotonic clocks can report zero elapsed for several tiny passes.
    assert MG_TIME["iterations"] >= 1
    assert MG_TIME["elapsed"] > 0.


def test_explicit_node_interval_extent_without_legacy_default_change():
    cfg={"grid_resolution":{"nx":1024,"ny":1024,"nz":1024},"voxel_nm3":.25}
    assert physical_domain_nm(cfg) == (256.,)*3
    cfg["coordinate_system"]={"extent_convention":"node_intervals"}
    assert normalize_config(cfg)["_physical"]["domain_nm"] == (255.75,)*3


def test_direct_high_level_rejects_nonconverged_final_but_explicit_trial_can_return(tmp_path):
    from simulation.main_loop import run_afm_simulation
    kwargs=dict(Vtip=-1,nx=8,ny=8,nz=8,initial_grid_level=8,cpu_threads=2,
                memory_mode="ram_compact",eps_reference_resolution=8,plotting_enabled=False,
                return_residual=False,verbose=False,tol=1e-30,stopping={"max_iterations":2})
    with pytest.raises(RuntimeError,match="no scientific field"):
        run_afm_simulation(output_dir=str(tmp_path/"reject"),**kwargs)
    result=run_afm_simulation(output_dir=str(tmp_path/"trial"),require_convergence=False,**kwargs)
    assert result["level_reports"][-1]["reason"] == "iteration_limit"
    assert result["phi"].dtype == np.float32


@pytest.mark.parametrize("criterion",["rms","maximum","both"])
def test_all_new_norms_use_drive_relative_scaling(tmp_path,criterion):
    from simulation.main_loop import run_afm_simulation
    kwargs=dict(nx=8,ny=8,nz=8,initial_grid_level=8,cpu_threads=2,
                memory_mode="ram_compact",eps_reference_resolution=8,plotting_enabled=False,
                return_residual=False,verbose=False,tol=1e-4,residual_tolerance_mode="relative_drive",
                tip_z=.2,R=.08,r_tip=.25,stopping={"criterion":criterion,"max_tol":1e-3})
    unit=run_afm_simulation(Vtip=-1,output_dir=str(tmp_path/"unit"),**kwargs)
    scaled=run_afm_simulation(Vtip=-12,output_dir=str(tmp_path/"scaled"),**kwargs)
    np.testing.assert_array_equal(scaled["phi"],np.multiply(unit["phi"],np.float32(12)))
    a=unit["level_reports"][-1]; b=scaled["level_reports"][-1]
    assert a["iterations"] == b["iterations"] and a["residual"] == b["residual"]
    assert b["stopping"]["last"]["drive_scale_volts"] == 12.


def test_relative_drive_zero_and_dominant_declared_gate_contract():
    from simulation.numerics import prepare_drive_voltages
    assert prepare_drive_voltages(0,[],"relative_drive") == (0,[],1.)
    tip,gates,scale=prepare_drive_voltages(0,[{"Vgate_val":-3.}],"relative_drive")
    assert tip == 0 and gates[0]["Vgate_val"] == -1 and scale == 3
    tip,gates,scale=prepare_drive_voltages(-1,[{"Vgate_val":12.,"x_range":[2,3]}],"relative_drive")
    assert scale == 12 and tip == -1/12  # Declared, not surviving-grid, drive scale.
