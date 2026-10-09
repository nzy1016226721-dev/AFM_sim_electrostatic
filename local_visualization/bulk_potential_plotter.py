"""Local adapter of Eric Friesen Waldner's Sept 23 2026 bulk plots.

Retains natural file ordering, inclusive selection, Cartesian line positions,
plane choices, custom titles, limited axes, colours and trimmed legends.
Coordinates are authoritative retained nodes, not guessed from a legacy name.
"""
from pathlib import Path
import re
from itertools import product
import numpy as np
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
    """imshow edges that place pixel CENTERS on the retained solver nodes."""
    dx = (node_extent[1]-node_extent[0])/(shape[0]-1)
    dy = (node_extent[3]-node_extent[2])/(shape[1]-1)
    return (node_extent[0]-dx/2,node_extent[1]+dx/2,
            node_extent[2]-dy/2,node_extent[3]+dy/2)


def save_figure(fig,directory,name):
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


def line_plotter(paths,axis,positions,*,bounds_nm=None,title=None,limits=None,
                 output_dir=None,show=False):
    import matplotlib.pyplot as plt
    paths = list(map(Path,paths))
    if not paths or axis not in "xyz" or len(positions)!=2:
        raise ValueError("Select files, x/y/z and two position lists")
    colours = plt.cm.plasma(np.linspace(0,1,len(paths)))
    saved = []
    for first,second in product(*positions):
        fig,ax = plt.subplots()
        try:
            for number,path in enumerate(paths):
                with PotentialData(path,bounds_nm) as data:
                    x,y,actual = data.axis_line(axis,first,second)
                ax.plot(x,y,color=colours[number],label=path.stem,lw=1.5)
            ax.set(xlabel=f"{axis} (nm)",ylabel="Potential (V)",
                   title=title or f"Potential along {axis}; other coordinates requested ({first:g}, {second:g}) nm")
            if limits is not None:
                ax.set_xlim(*limits)
            handles,labels = ax.get_legend_handles_labels()
            selected = np.linspace(0,len(labels)-1,min(12,len(labels))).astype(int)
            ax.legend([handles[i] for i in selected],[labels[i] for i in selected],fontsize="x-small")
            ax.grid(alpha=.4)
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
            with PotentialData(path,bounds_nm) as data:
                image,extent,actual = data.plane(plane,position)
            fig,ax = plt.subplots()
            try:
                im = ax.imshow(image.T,origin="lower",extent=pixel_extent(image.shape,extent),cmap="RdBu_r",aspect="equal",interpolation="none")
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
                if output_dir is not None:
                    saved.append(save_figure(fig,output_dir,f"{Path(path).stem}_{plane}_{actual:g}"))
                if show:
                    plt.show()
            finally:
                plt.close(fig)
    return saved
