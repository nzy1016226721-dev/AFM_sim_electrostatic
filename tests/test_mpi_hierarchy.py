"""Planner regression for the actual failed45-rank refinement hierarchy."""
import json
from pathlib import Path
import subprocess
import sys
import pytest


def test_plan_rejects_memory_fitting_layout_with_unreachable_coarse_halo(tmp_path):
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'tests/data/afm_config_mpi_smoke.json').read_text())
    cfg['grid_resolution']={'nx':2048,'ny':2048,'nz':2048}
    cfg['mpi'].update(process_grid=[3,3,5],initial_grid_level=64)
    path=tmp_path/'bad.json';path.write_text(json.dumps(cfg))
    run=subprocess.run([sys.executable,str(root/'run_mpi.py'),str(path),'--plan',
                        '--ranks','45','--ranks-per-node','45','--node-memory-gib','180'],
                       cwd=root,capture_output=True,text=True)
    assert run.returncode!=0, 'Planner admitted the layout that fails128-to-256prolongation'
    assert 'hierarchy' in run.stderr.lower() or 'coarse halo' in run.stderr.lower()


@pytest.mark.parametrize('shape,dims,initial', [
    ((2048,2048,2048),(3,3,5),64),
    ((4096,4096,4096),(6,6,10),64),
    ((2048,2048,2048),(64,1,1),64),
])
def test_hierarchy_rejects_known_transfer_or_initial_ownership_failures(shape,dims,initial):
    from simulation.mpi_domain import validate_refinement_hierarchy
    with pytest.raises(ValueError):
        validate_refinement_hierarchy(shape,dims,initial)


@pytest.mark.parametrize('shape,dims', [
    ((2048,2048,2048),(3,4,4)),
    ((4096,4096,4096),(3,8,16)),
    ((513,257,127),(2,2,1)),
])
def test_supported_hierarchy_includes_every_clipped_level(shape,dims):
    from simulation.mpi_domain import validate_refinement_hierarchy
    report=validate_refinement_hierarchy(shape,dims,64)
    assert report['shapes'][0]==[64,64,64]
    assert report['shapes'][-1]==list(shape)
    assert report['axis_partition_checks']==sum(dims)*(len(report['shapes'])-1)


def test_auto_grid_chooses_supported_layout_over_unsafe_surface_optimum():
    from simulation.mpi_domain import suggest_process_grid,validate_refinement_hierarchy
    original=suggest_process_grid(384,(4096,)*3)
    with pytest.raises(ValueError):
        validate_refinement_hierarchy((4096,)*3,original,64)
    actual=suggest_process_grid(384,(4096,)*3,initial_level=64)
    assert actual[0]*actual[1]*actual[2]==384
    validate_refinement_hierarchy((4096,)*3,actual,64)


def test_auto_grid_reports_no_supported_factorization():
    from simulation.mpi_domain import suggest_process_grid
    with pytest.raises(ValueError,match='refinement hierarchy'):
        suggest_process_grid(45,(2048,)*3,initial_level=64)


def test_runtime_auto_layout_agrees_with_planner_before_any_field_allocation(monkeypatch, tmp_path):
    from simulation.mpi_domain import DomainDecomposition, suggest_process_grid
    from simulation.mpi_main import run_distributed_hierarchy
    class Comm:
        def Get_size(self):return 384
    class ReachedDecomposition(Exception):pass
    expected=suggest_process_grid(384,(4096,)*3,initial_level=64)
    def inspected(cls,comm,shape,process_grid):
        assert tuple(shape)==(64,)*3 and tuple(process_grid)==expected
        raise ReachedDecomposition
    monkeypatch.delenv('AFM_MPI_PROCESS_GRID',raising=False)
    monkeypatch.setattr(DomainDecomposition,'create',classmethod(inspected))
    cfg=dict(grid_resolution=dict(nx=4096,ny=4096,nz=4096),mpi=dict(initial_grid_level=64))
    with pytest.raises(ReachedDecomposition):
        run_distributed_hierarchy(cfg,Vtip=-1.,output_dir=str(tmp_path),case_tag='auto',comm=Comm())
