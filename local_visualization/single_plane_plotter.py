"""Eric's single-plane interface, using the local coordinate-aware reader."""
from .bulk_potential_plotter import plane_plotter


def single_plane_plotter(path,plane="xy",position=0,**kwargs):
    return plane_plotter([path],plane,[position],**kwargs)
