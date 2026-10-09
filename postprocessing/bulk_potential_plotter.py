"""Historical plotting names, backed by the current local-only renderers."""
from pathlib import Path
from local_visualization import require_local
from local_visualization.data import select_indices
from local_visualization.bulk_potential_plotter import make_line_figure,make_plane_figure,save_figure
from .potential_plot_data import discover_potentials,open_local_reader


def plot_potential_lines(paths,config_path=None,*,axis,fixed_nm,limits_nm=None,
                         allow_full=False,show=False,bounds_nm=None,center_nm=None):
    require_local()
    import matplotlib.pyplot as plt
    if len(fixed_nm)!=2:
        raise ValueError("Two fixed coordinates are required")
    fig = make_line_figure(paths,axis.lower(),*fixed_nm,limits=limits_nm,
                           opener=lambda path: open_local_reader(path,config_path=config_path,
                                                                  bounds_nm=bounds_nm,allow_full=allow_full,center_nm=center_nm))
    try:
        if show:
            plt.show()
    except BaseException:
        plt.close(fig)
        raise
    return fig


def plot_potential_planes(paths,config_path=None,*,plane,at_nm,allow_full=False,
                          show=False,bounds_nm=None,center_nm=None):
    require_local()
    import matplotlib.pyplot as plt
    paths = list(paths)
    if not paths:
        raise ValueError("At least one AFM NPY file is required")
    figures = []
    try:
        for path in paths:
            fig,_ = make_plane_figure(path,plane.lower(),at_nm,
                                       opener=lambda file: open_local_reader(file,config_path=config_path,
                                                                              bounds_nm=bounds_nm,allow_full=allow_full,center_nm=center_nm))
            figures.append(fig)
        if show:
            plt.show()
        return figures
    except BaseException:
        for fig in figures:
            plt.close(fig)
        raise


def bulk_potential_plots():
    """Interactive local menu, with receipts and collision-free saves."""
    require_local()
    import matplotlib.pyplot as plt
    folder = Path(input("Output directory [outputs]: ").strip() or "outputs").expanduser()
    files = discover_potentials(folder,prefer_cuts=True)
    if not files:
        print(f"No recognizable potential NPY files in {folder}")
        return []
    print("Available arrays (cuts preferred where present):")
    for number,path in enumerate(files):
        print(f"  {number}: {path.name}")
    indices = input("File indices/ranges [0]: ").strip() or "0"
    selected = [files[i] for i in select_indices(indices,len(files))]
    config = input("Matching AFM config JSON (optional; validates saved coordinates): ").strip() or None
    bounds = None
    if any(not Path(str(path)+".coords.json").is_file() for path in selected):
        bounds = [float(value) for value in input("Missing receipt: six FIRST/LAST retained-node bounds in nm: ").replace(","," ").split()]
        if len(bounds)!=6:
            raise ValueError("Six explicit retained-node bounds are required")
    kind = (input("Plot line or plane [line]: ").strip() or "line").lower()
    if kind=="line":
        axis = (input("Line axis [z]: ").strip() or "z").lower()
        if axis not in ("x","y","z"):
            raise ValueError("axis must be x, y or z")
        fixed = tuple(float(input(f"Fixed {name} coordinate (nm): ")) for name in "xyz" if name!=axis)
        figs = [plot_potential_lines(selected,config,axis=axis,fixed_nm=fixed,bounds_nm=bounds)]
    elif kind=="plane":
        plane = (input("Plane (xy/xz/yz) [xy]: ").strip() or "xy").lower()
        if plane not in ("xy","xz","yz"):
            raise ValueError("plane must be xy, xz or yz")
        fixed = next(name for name in "xyz" if name not in plane)
        at_nm = float(input(f"{fixed} plane coordinate (nm): "))
        figs = plot_potential_planes(selected,config,plane=plane,at_nm=at_nm,bounds_nm=bounds)
    else:
        raise ValueError("Expected line or plane")
    try:
        if input("Save figures as PNG? [y/N] ").strip().lower()=="y":
            for number,fig in enumerate(figs):
                saved = save_figure(fig,folder/"saved_plots",f"afm_{kind}_{number:03d}")
                print(f"Saved local figure: {saved}")
        plt.show()
        return figs
    except BaseException:
        for fig in figs:
            plt.close(fig)
        raise
