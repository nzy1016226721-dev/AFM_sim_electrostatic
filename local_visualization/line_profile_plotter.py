"""Eric's drawn-line profile, without a hard IPython/Qt dependency.

Any interactive Matplotlib backend works. Both endpoints are expressed in
actual nm; rectangular and cropped grids use their own axis coordinates.
"""
from pathlib import Path
from .data import PotentialData
from .bulk_potential_plotter import save_figure,pixel_extent


def line_profile_plotter(path,start,end,*,bounds_nm=None,samples=500,title=None,
                         output_dir=None,show=False):
    import matplotlib.pyplot as plt
    with PotentialData(path,bounds_nm) as data:
        distance,values = data.profile(start,end,samples)
    fig,ax = plt.subplots()
    try:
        ax.plot(distance,values)
        ax.set(xlabel="Distance along line (nm)",ylabel="Potential (V)",
               title=title or f"{Path(path).stem}: {start} to {end} nm")
        ax.grid(alpha=.4)
        saved = save_figure(fig,output_dir,"profile_"+Path(path).stem) if output_dir else None
        if show:
            plt.show()
        return saved
    finally:
        plt.close(fig)


def drawn_profile(path,plane,position,*,bounds_nm=None,samples=500,output_dir=None):
    import matplotlib.pyplot as plt
    with PotentialData(path,bounds_nm) as data:
        image,extent,actual = data.plane(plane,position)
    fig,ax = plt.subplots()
    im = ax.imshow(image.T,origin="lower",extent=pixel_extent(image.shape,extent),cmap="RdBu_r",interpolation="none")
    ax.set_xlim(extent[0],extent[1]); ax.set_ylim(extent[2],extent[3])
    fig.colorbar(im,ax=ax,label="Potential (V)")
    ax.set(xlabel=f"{plane[0]} (nm)",ylabel=f"{plane[1]} (nm)",title="Click two endpoints; close to cancel")
    try:
        points = plt.ginput(2,timeout=0)
    finally:
        plt.close(fig)
    if len(points)!=2:
        return None
    normal = next(c for c in "xyz" if c not in plane)
    endpoints = []
    for point in points:
        coordinates = {normal:actual,plane[0]:point[0],plane[1]:point[1]}
        endpoints.append(tuple(coordinates[c] for c in "xyz"))
    return line_profile_plotter(path,*endpoints,bounds_nm=bounds_nm,samples=samples,output_dir=output_dir,show=True)
