"""Local, bounded figures adapted from Eric Friesen Waldner's bulk plots.

Both the local CLI and historical plotting functions share these renderers.
Only selected lines/planes are copied; readers close before figures are returned.
"""
from pathlib import Path
import re
from itertools import product
import numpy as np
from . import require_local
from .data import PotentialData, natural_key, select_indices


def npy_file_sorter(directory):
    return sorted(Path(directory).glob("*.npy"),key=natural_key)


def list_to_array(text):
    return select_indices(text,1000000)


def list_to_array_float(text):
    values = [float(value.strip()) for value in text.split(",")]
    if not values or not np.isfinite(values).all():
        raise ValueError("positions must be comma-separated finite nm values")
    return values


def pixel_extent(shape,node_extent):
    """imshow edges that place pixel CENTERS on retained solver nodes."""
    dx = (node_extent[1]-node_extent[0])/(shape[0]-1)
    dy = (node_extent[3]-node_extent[2])/(shape[1]-1)
    return (node_extent[0]-dx/2,node_extent[1]+dx/2,
            node_extent[2]-dy/2,node_extent[3]+dy/2)


def save_figure(fig,directory,name):
    require_local()
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    name = re.sub(r'[^A-Za-z0-9_.-]+',"_",name)[:160]
    path = directory/(name+".png")
    suffix = 1
    while path.exists():
        path = directory/f"{name}_{suffix}.png"
        suffix += 1
    fig.savefig(path,dpi=150,bbox_inches="tight")
    return path


def make_line_figure(paths,axis,first,second,*,opener=None,title=None,limits=None):
    """Caller owns the returned figure; no reader survives this call."""
    require_local()
    import matplotlib.pyplot as plt
    paths = list(map(Path,paths))
    if not paths or axis not in ("x","y","z"):
        raise ValueError("Select files and x/y/z")
    if limits is not None and (len(limits)!=2 or not np.isfinite(limits).all() or limits[1]<=limits[0]):
        raise ValueError("limits must be two finite increasing nm coordinates")
    opener = opener or PotentialData
    fig,ax = plt.subplots()
    colours = plt.cm.plasma(np.linspace(0,1,len(paths)))
    try:
        for number,path in enumerate(paths):
            with opener(path) as data:
                x,y,_ = data.axis_line(axis,first,second)
            if limits is not None:
                selected = (x>=limits[0]) & (x<=limits[1])
                if not selected.any():
                    raise ValueError(f"{path.name}: requested range has no saved samples")
                x,y = x[selected],y[selected]
            ax.plot(x,y,color=colours[number],label=path.stem,lw=1.5)
        ax.set(xlabel=f"{axis} (nm)",ylabel="Potential (V)",
               title=title or f"Potential along {axis}; other coordinates requested ({first:g}, {second:g}) nm")
        if limits is not None:
            ax.set_xlim(*limits)
        handles,labels = ax.get_legend_handles_labels()
        selected = np.linspace(0,len(labels)-1,min(12,len(labels))).astype(int)
        ax.legend([handles[i] for i in selected],[labels[i] for i in selected],fontsize="x-small")
        ax.grid(alpha=.4)
        fig.tight_layout()
        return fig
    except BaseException:
        plt.close(fig)
        raise


def make_plane_figure(path,plane,position,*,opener=None,title=None,electric_field=False):
    """Return a figure and plain metadata, never a live mapped volume."""
    require_local()
    import matplotlib.pyplot as plt
    opener = opener or PotentialData
    with opener(path) as data:
        image,extent,actual = data.plane(plane,position)
        metadata = dict(path=str(data.path),plane=plane,actual_nm=actual,
                        node_bounds_nm=data.bounds.tolist(),shape=list(data.phi.shape),
                        coordinate_receipt=data.coordinate_receipt)
    fig,ax = plt.subplots()
    try:
        im = ax.imshow(image.T,origin="lower",extent=pixel_extent(image.shape,extent),
                       cmap="RdBu_r",aspect="equal",interpolation="none")
        ax.set_xlim(extent[0],extent[1]); ax.set_ylim(extent[2],extent[3])
        ax.set(xlabel=f"{plane[0]} (nm)",ylabel=f"{plane[1]} (nm)",
               title=title or f"{Path(path).stem}\n{plane}, normal coordinate {actual:g} nm")
        fig.colorbar(im,ax=ax,label="Potential (V)")
        if electric_field:
            dx = (extent[1]-extent[0])/(image.shape[0]-1)*1e-9
            dy = (extent[3]-extent[2])/(image.shape[1]-1)*1e-9
            ex,ey = np.gradient(-image.astype(np.float64),dx,dy)
            stride = max(1,max(image.shape)//24)
            x = np.linspace(extent[0],extent[1],image.shape[0])[::stride]
            y = np.linspace(extent[2],extent[3],image.shape[1])[::stride]
            ax.quiver(x,y,ex[::stride,::stride].T,ey[::stride,::stride].T)
        fig.tight_layout()
        return fig,metadata
    except BaseException:
        plt.close(fig)
        raise


def line_plotter(paths,axis,positions,*,bounds_nm=None,title=None,limits=None,
                 output_dir=None,show=False):
    import matplotlib.pyplot as plt
    if len(positions)!=2:
        raise ValueError("Two position lists are required")
    paths,saved = list(paths),[]
    for first,second in product(*positions):
        fig = make_line_figure(paths,axis,first,second,
                               opener=lambda path: PotentialData(path,bounds_nm),
                               title=title,limits=limits)
        try:
            if output_dir is not None:
                saved.append(save_figure(fig,output_dir,f"lines_{axis}_{first:g}_{second:g}"))
            if show:
                plt.show()
        finally:
            plt.close(fig)
    return saved


def plane_plotter(paths,plane,positions,*,bounds_nm=None,title=None,
                  output_dir=None,show=False,electric_field=False):
    import matplotlib.pyplot as plt
    saved = []
    for path in paths:
        for position in positions:
            fig,metadata = make_plane_figure(path,plane,position,
                                             opener=lambda file: PotentialData(file,bounds_nm),
                                             title=title,electric_field=electric_field)
            try:
                if output_dir is not None:
                    saved.append(save_figure(fig,output_dir,
                                             f"{Path(path).stem}_{plane}_{metadata['actual_nm']:g}"))
                if show:
                    plt.show()
            finally:
                plt.close(fig)
    return saved
