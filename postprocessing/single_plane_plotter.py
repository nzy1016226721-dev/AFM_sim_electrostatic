"""Single downloaded plane; returned metadata never retains a live mmap."""
from local_visualization import require_local
from local_visualization.bulk_potential_plotter import make_plane_figure
from .potential_plot_data import open_local_reader


def single_plane_plotter(npy_path,config_path=None,*,plane="xy",at_nm=0.0,
                         allow_full=False,show=False,bounds_nm=None,center_nm=None):
    require_local()
    import matplotlib.pyplot as plt
    fig,metadata = make_plane_figure(npy_path,plane.lower(),at_nm,
                                     opener=lambda path: open_local_reader(path,config_path=config_path,
                                                                            bounds_nm=bounds_nm,allow_full=allow_full,center_nm=center_nm))
    try:
        if show:
            plt.show()
    except BaseException:
        plt.close(fig)
        raise
    return fig,fig.axes[0],metadata
