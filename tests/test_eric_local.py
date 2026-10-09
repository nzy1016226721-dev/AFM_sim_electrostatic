import hashlib
import json
import numpy as np
import pytest
from local_visualization import require_local
from local_visualization.data import PotentialData,select_indices
from local_visualization.bulk_potential_plotter import line_plotter,plane_plotter
from local_visualization.line_profile_plotter import line_profile_plotter
from simulation.output_coordinates import write_coordinate_receipt


@pytest.fixture
def field(tmp_path):
    axes = [np.linspace(-128,128,19),np.linspace(-90,90,17),np.linspace(0,100,21)]
    phi = (axes[0][:,None,None]*.001+axes[1][None,:,None]*.002+axes[2][None,None,:]*.003-1).astype(np.float32)
    selection = (slice(4,15),slice(4,13),slice(4,20))
    path = tmp_path/"afm_phi_1-3.50nm_-1.00V_cut_from_grid19x17x21.npy"
    np.save(path,phi[selection])
    write_coordinate_receipt(path,phi.shape,(-128,128,-90,90,0,100),selection)
    return path


def test_crop_coordinates_endpoints_profiles_and_read_only(field):
    before = hashlib.sha256(field.read_bytes()).hexdigest()
    with PotentialData(field) as data:
        assert isinstance(data.phi,np.memmap)
        assert not data.phi.flags.writeable
        assert data.axes[2][0] == 20
        assert data.axes[2][-1] == 95
        x,y,actual = data.axis_line("z",0,0)
        np.testing.assert_allclose(y,.003*x-1,atol=1e-7)
        start = [data.axes[i][0] for i in range(3)]
        end = [data.axes[i][-1] for i in range(3)]
        distance,profile = data.profile(start,end,103)
        points = np.linspace(start,end,103)
        expected = points[:,0]*.001+points[:,1]*.002+points[:,2]*.003-1
        np.testing.assert_allclose(profile,expected,atol=1e-7)
        assert data.sanity()["nonfinite_count"] == 0
        with pytest.raises(ValueError,match="outside"):
            data.index(2,100)
    assert hashlib.sha256(field.read_bytes()).hexdigest()==before


def test_all_plane_and_axis_options_and_derived_plots(field,tmp_path):
    saved = []
    for plane,position in [("xy",25),("xz",0),("yz",0)]:
        saved += plane_plotter([field],plane,[position],output_dir=tmp_path/"figures",electric_field=True)
    for axis,positions in [("x",([0],[25])),("y",([0],[25])),("z",([0],[0]))]:
        saved += line_plotter([field],axis,positions,output_dir=tmp_path/"figures")
    saved.append(line_profile_plotter(field,(-10,-10,25),(10,10,75),output_dir=tmp_path/"figures"))
    assert len(saved)==7
    assert all(path.stat().st_size>1000 for path in saved)


@pytest.mark.parametrize("variable", ["SLURM_JOB_ID","OMPI_COMM_WORLD_SIZE","PMI_SIZE"])
def test_never_runs_as_a_cluster_or_mpi_task(monkeypatch,variable):
    monkeypatch.setenv(variable,"1")
    with pytest.raises(RuntimeError):
        require_local()


def test_file_selection_supports_eric_ranges_and_stride():
    assert select_indices("0-2,5,7:11:2",12)==[0,1,2,5,7,9,11]
    with pytest.raises(ValueError):
        select_indices("12",12)


def test_plane_pixel_centers_are_true_retained_nodes():
    from local_visualization.bulk_potential_plotter import pixel_extent
    edges = pixel_extent((11,16),(-70,70,20,95))
    centers_x = edges[0]+(np.arange(11)+.5)*(edges[1]-edges[0])/11
    centers_y = edges[2]+(np.arange(16)+.5)*(edges[3]-edges[2])/16
    np.testing.assert_allclose(centers_x,np.linspace(-70,70,11))
    np.testing.assert_allclose(centers_y,np.linspace(20,95,16))


def test_missing_or_wrong_sidecar_is_not_guessed(field):
    receipt = field.with_name(field.name+".coords.json")
    record = json.loads(receipt.read_text())
    record["shape"] = [2,2,2]
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError,match="does not match"):
        PotentialData(field)
    receipt.unlink()
    with pytest.raises(ValueError,match="No coordinate receipt"):
        PotentialData(field)


def test_sanity_scan_closes_each_plane_reader(field,monkeypatch):
    with PotentialData(field) as data:
        load = np.load
        readers = []
        def tracked(*args,**kwargs):
            array = load(*args,**kwargs)
            readers.append(array)
            return array
        monkeypatch.setattr(np,"load",tracked)
        assert data.sanity()["nonfinite_count"] == 0
        assert len(readers) == data.phi.shape[0]
        assert all(reader._mmap.closed for reader in readers)
        assert not data.phi._mmap.closed


def test_eric_optional_conductivity_helpers_are_bounded_and_use_si():
    from local_visualization.eric_conductivity import cell_center_to_node,compute_current_divergence,integrate_power
    x,y,z = np.meshgrid(np.arange(7),np.arange(8),np.arange(9),indexing="ij")
    phi = (x*.001+y*.004+z*.009).astype(np.float32)
    sigma = np.full((6,7,8),.003,np.float32)
    assert cell_center_to_node(sigma).shape==phi.shape
    maximum,rms,full = compute_current_divergence(phi,sigma,1e-9,2e-9,3e-9)
    assert full.shape==phi.shape and np.isfinite(maximum) and np.isfinite(rms)
    actual = integrate_power(phi,sigma,1e-9,2e-9,3e-9)
    expected = sigma.size*.003*(1e6**2+2e6**2+3e6**2)*1e-9*2e-9*3e-9
    np.testing.assert_allclose(actual,expected,rtol=1e-6)
    with pytest.raises(MemoryError):
        integrate_power(phi,sigma,1e-9,2e-9,3e-9,max_workspace_mib=.001)


def test_reconstructs_old_cut_from_exact_configuration(tmp_path):
    from simulation.mpi_config import load_afm_config
    from local_visualization.coordinates import reconstruct_coordinates
    from pathlib import Path
    _,raw = load_afm_config(Path(__file__).resolve().parents[1]/"afm_config_nm.json",normalize=False)
    raw["grid_resolution"] = dict(nx=10,ny=10,nz=10)
    raw["voxel_nm3"] = 1
    raw["movement"] = dict(start_nm=[0,0,2],end_nm=[0,0,2],spacing_nm=1)
    raw["save_cut_box_nm"] = [-2,2,-2,2,-2,3]
    config = tmp_path/"exact_config.json"
    config.write_text(json.dumps(raw),encoding="utf-8")
    path = tmp_path/"afm_phi_1_0nm_-1.00V_cut_from_grid10x10x10.npy"
    np.save(path,np.zeros((4,4,5),np.float32))
    reconstruct_coordinates(path,config,[0,0,2])
    with PotentialData(path) as data:
        np.testing.assert_allclose(data.bounds,[-5+10*3/9,-5+10*6/9,-5+10*3/9,-5+10*6/9,0,10*4/9])
