"""Compatibility names for canonical local two-click profile sampling."""
from local_visualization import require_local
from local_visualization.line_profile_plotter import sample_segment,InteractiveLineProfile
from .single_plane_plotter import single_plane_plotter


def line_profile_plotter(npy_path=None,config_path=None,*,plane="xy",at_nm=0.0,
                         show=True,bounds_nm=None,center_nm=None,allow_full=False):
    require_local()
    import matplotlib.pyplot as plt
    if npy_path is None:
        npy_path = input("Downloaded potential NPY file: ").strip()
        if config_path is None:
            config_path = input("Matching JSON (optional; validates coordinates): ").strip() or None
    fig,ax,_ = single_plane_plotter(npy_path,config_path,plane=plane,at_nm=at_nm,show=False,
                                    bounds_nm=bounds_nm,center_nm=center_nm,allow_full=allow_full)
    callback = InteractiveLineProfile(fig,ax)
    fig._afm_line_profile_callback = callback
    try:
        if show:
            plt.show()
    except BaseException:
        callback.disconnect()
        plt.close(fig)
        raise
    return fig,callback
