"""Plotting integration tests for the actual AFM MPI checkout.

In the full MPI tree these exercise its real parser, configuration normalizer,
physical-cut geometry, and plotting functions. Small synthetic arrays only.
"""
import json
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from simulation.output_coordinates import write_coordinate_receipt

from postprocessing.potential_plot_data import load_potential, discover_potentials
from postprocessing.bulk_potential_plotter import (plot_potential_lines, plot_potential_planes)
from postprocessing.line_profile_plotter import sample_segment, InteractiveLineProfile


@pytest.fixture
def sample_case(tmp_path):
    # Physical main domain = 32x24x16 nm, origin = (-16,-12,0) nm.
    # Movement centre = (0,0,4) nm. The saved cut box is 8x12x8 nm.
    cfg = {
        'grid_resolution': {'nx': 16, 'ny': 12, 'nz': 8},
        'voxel_nm3': 2.,
        'coordinate_system': {'origin_fraction':[0.5,0.5,0.0]},
        'movement': {'start_nm':[0.,0.,4.], 'end_nm':[0.,0.,4.], 'spacing_nm':1.},
        'save_cut': True, 'save_full': False,
        'save_cut_box_nm': [-4.,4.,-6.,6.,-2.,6.],
        'blocks_nm': [], 'v_start': -1., 'v_stop': -1., 'v_step': 1.,
    }
    config = tmp_path/'case.json'
    config.write_text(json.dumps(cfg))
    # MPI physical_cut_slices => grid indices [6:10,3:9,1:5];
    # Cut bin bounds differ slightly from the *solver node* coordinates.
    x = -16. + np.arange(6,10)*32./15.
    y = -12. + np.arange(3,9)*24./11.
    z = np.arange(1,5)*16./7.
    field = 2*x[:,None,None] + 3*y[None,:,None] + 4*z[None,None,:]
    cut_path = tmp_path/'afm_phi_1_0nm_-1.00V_cut_from_grid16x12x8.npy'
    np.save(cut_path, field.astype(np.float32))
    write_coordinate_receipt(cut_path,(16,12,8),(-16,16,-12,12,0,16),
                             (slice(6,10),slice(3,9),slice(1,5)))
    return config, cut_path


def test_cut_coordinates_values_and_mmap(sample_case):
    cfg, path = sample_case
    volume = load_potential(path, config_path=cfg)
    assert isinstance(volume.array, np.memmap)
    assert volume.is_cut
    assert volume.bounds_nm == pytest.approx((-4,4,-6,6,2,10))
    assert volume.source_grid == (16,12,8)
    assert volume.centres[0] == pytest.approx(-16. + np.arange(6,10)*32./15.)
    xx, yy = volume.line('x', (-3,5))
    assert xx == pytest.approx(-16. + np.arange(6,10)*32./15.)
    assert yy == pytest.approx(2*xx + 3*volume.centres[1][1]+4*volume.centres[2][1])
    data, extent, axis_names = volume.plane('xy', 5)
    assert data.shape == (6,4)
    assert extent == pytest.approx((-4.266666666666667,4.266666666666667,
                                     -6.545454545454546,6.545454545454546))
    assert axis_names == ('x','y')


def test_potential_plots_and_cut_preference(sample_case):
    cfg, cut = sample_case
    full = cut.with_name('afm_phi_1_0nm_-1.00V.npy')
    np.save(full, np.zeros((16,12,8), dtype=np.float32))
    write_coordinate_receipt(full,(16,12,8),(-16,16,-12,12,0,16))
    assert discover_potentials(cut.parent) == [cut]
    fig = plot_potential_lines([cut], cfg, axis='x', fixed_nm=(-3,5))
    assert fig.axes[0].lines[0].get_xdata().size == 4
    plt.close(fig)
    figs = plot_potential_planes([cut], cfg, plane='xz', at_nm=-3)
    assert len(figs)==1 and figs[0].axes[0].images[0].get_array().shape==(4,4)
    for f in figs: plt.close(f)


def test_wrong_json_cut_shape_is_rejected(sample_case):
    cfg, path=sample_case
    arr_path=path.with_name('afm_phi_1_-2.00V_cut_from_grid16x12x8.npy')
    np.save(arr_path, np.ones((3,3,3), dtype=np.float32))
    write_coordinate_receipt(arr_path,(16,12,8),(-16,16,-12,12,0,16),
                             (slice(6,9),slice(3,6),slice(1,4)))
    with pytest.raises(ValueError, match='Cut shape'):
        load_potential(arr_path, config_path=cfg)


def test_large_grid_requires_cut(tmp_path):
    cfg = {
        'grid_resolution': {'nx':2048,'ny':2,'nz':2},
        'voxel_nm3':1, 'blocks_nm':[],
        'movement': {'start_nm':[0,0,0]},
        'v_start':1,'v_stop':1,'v_step':1,
    }
    conf=tmp_path/'large.json';conf.write_text(json.dumps(cfg))
    full = tmp_path/'afm_phi_1_1.00V.npy'
    fp=np.lib.format.open_memmap(full,mode='w+',dtype=np.float32,shape=(2048,2,2))
    del fp
    write_coordinate_receipt(full,(2048,2,2),(-1024,1024,-1,1,0,2))
    with pytest.raises(ValueError, match='Full-grid potential'):
        load_potential(full, config_path=conf)


def test_segment_sampling_orientation_and_length():
    # imshow array indexed as [row-y, column-x]; 2 nm voxel spacing.
    matrix=np.array([[0.,1.,2.],[10.,11.,12.],[20.,21.,22.]])
    s,v=sample_segment(matrix, (0,6,0,6), (1,1), (5,5), n=5)
    assert s[-1] == pytest.approx(np.sqrt(32))
    assert np.allclose(v,[0,5.5,11,16.5,22])


def test_outside_cut_rejected(sample_case):
    cfg, cut = sample_case
    field=load_potential(cut,config_path=cfg)
    with pytest.raises(ValueError,match='outside'):
        field.line('x',(999,5))


def test_click_profile_event(sample_case):
    cfg, cut=sample_case
    fig=plot_potential_planes([cut],cfg,plane='xy',at_nm=5)[0]
    ax=fig.axes[0]
    handler=InteractiveLineProfile(fig,ax)
    class E:
        def __init__(self,x,y): self.inaxes=ax;self.xdata=x;self.ydata=y
    handler.on_click(E(-3,-5))
    assert handler.last_profile_figure is None
    handler.on_click(E(3,5))
    assert handler.last_profile_figure is not None
    handler.disconnect()
    plt.close(handler.last_profile_figure);plt.close(fig)


def test_negative_movement_cut_coordinates(sample_case, tmp_path):
    cfg, _ = sample_case
    raw = json.loads(cfg.read_text())
    raw['movement']['end_nm'] = [-2.,0.,4.]
    cfg.write_text(json.dumps(raw))
    # Physical x bounds are [-16,16]. Centre moved from x=0 to x=-2.
    # Requested cut x bounds [-6,2] -> voxel bins [5:9], original grid nodes [5:9].
    cut = tmp_path/'afm_phi_1-2nm_-1.00V_cut_from_grid16x12x8.npy'
    np.save(cut, np.zeros((4,6,4),dtype=np.float32))
    write_coordinate_receipt(cut,(16,12,8),(-16,16,-12,12,0,16),
                             (slice(5,9),slice(3,9),slice(1,5)))
    vol=load_potential(cut,config_path=cfg)
    assert vol.bounds_nm == pytest.approx((-6,2,-6,6,2,10))
    assert vol.centres[0] == pytest.approx(-16. + np.arange(5,9)*32./15.)


def test_full_small_grid_has_correct_endpoint_image_extent(sample_case):
    cfg, cut = sample_case
    full = cut.with_name('afm_phi_1_-1.00V.npy')
    np.save(full, np.zeros((16,12,8),dtype=np.float32))
    write_coordinate_receipt(full,(16,12,8),(-16,16,-12,12,0,16))
    field=load_potential(full,config_path=cfg)
    plane,extent,_=field.plane('xy',0)
    assert field.centres[0][0]==-16 and field.centres[0][-1]==16
    assert extent[0] < -16 and extent[1] > 16
