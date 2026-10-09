"""Eric's drawn-line profile, without a hard IPython/Qt dependency.

Any interactive Matplotlib backend works. Both endpoints are expressed in
actual nm; rectangular and cropped grids use their own axis coordinates.
"""
from pathlib import Path
import numpy as np
from . import require_local
from .data import PotentialData
from .bulk_potential_plotter import save_figure,pixel_extent


def sample_segment(plane_data,bounds_nm,first_nm,last_nm,*,n=None):
    """Bilinear selected-plane profile, float64 samples, at most100000 points.

    The input matrix is imshow row/column order and bounds are PIXEL EDGES.
    Only four selected-neighbor vectors are read, not a full float64 plane.
    """
    require_local()
    data = np.asarray(plane_data)
    bounds = np.asarray(bounds_nm,dtype=float)
    points = np.asarray((first_nm,last_nm),dtype=float)
    if (data.ndim!=2 or min(data.shape)<2 or bounds.shape!=(4,)
            or points.shape!=(2,2) or not np.isfinite(bounds).all() or not np.isfinite(points).all()):
        raise ValueError("A two-dimensional plane, finite pixel edges and two finite xy points are required")
    left,right,bottom,top = bounds
    rows,cols = data.shape
    if right<=left or top<=bottom:
        raise ValueError("Pixel edges must increase")
    dx,dy = (right-left)/cols,(top-bottom)/rows
    lower = np.array((left+dx/2,bottom+dy/2))
    upper = np.array((right-dx/2,top-dy/2))
    if (points<lower-1e-10).any() or (points>upper+1e-10).any():
        raise ValueError("Line endpoints must be inside saved sample centres")
    points = np.clip(points,lower,upper)
    length = float(np.linalg.norm(points[1]-points[0]))
    if n is None:
        n = max(2,1+int(np.ceil(length/min(dx,dy))))
    if not np.isfinite(n) or n<2 or int(n)!=n:
        raise ValueError("n must be a finite integer >=2")
    n = min(int(n),100000)
    coordinates = (np.linspace(points[0],points[1],n)-lower)/np.array((dx,dy))
    lo = np.floor(coordinates).astype(np.int64)
    lo[:,0] = np.clip(lo[:,0],0,cols-2)
    lo[:,1] = np.clip(lo[:,1],0,rows-2)
    weights = coordinates-lo
    values = np.zeros(n,dtype=np.float64)
    for x in (0,1):
        for y in (0,1):
            weight = (weights[:,0] if x else 1-weights[:,0])*(weights[:,1] if y else 1-weights[:,1])
            values += weight*data[lo[:,1]+y,lo[:,0]+x]
    return np.linspace(0,length,n),values


class InteractiveLineProfile:
    """Two-click adapter for a detached plane; disconnects on figure close."""
    def __init__(self,fig,ax):
        require_local()
        if not ax.images:
            raise ValueError("The given plot has no potential image")
        self.fig,self.ax = fig,ax
        self.image = ax.images[0]
        self.points = []
        self.last_profile_figure = None
        self._cid = fig.canvas.mpl_connect("button_release_event",self.on_click)
        self._close_cid = fig.canvas.mpl_connect("close_event",lambda event: self.disconnect())

    def on_click(self,event):
        if (event.inaxes is not self.ax or event.xdata is None or event.ydata is None
                or getattr(event,"button",1)!=1):
            return
        self.points.append((float(event.xdata),float(event.ydata)))
        if len(self.points)<2:
            return
        first,last = self.points
        self.points.clear()
        try:
            distance,values = sample_segment(np.asarray(self.image.get_array()),self.image.get_extent(),first,last)
        except ValueError as exc:
            self.ax.set_title(str(exc))
            self.fig.canvas.draw_idle()
            return
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        ax.plot(distance,values)
        ax.set(xlabel="Distance along selected line (nm)",ylabel="Potential (V)",title="Potential along selected line")
        ax.grid(alpha=.4)
        fig.tight_layout()
        self.last_profile_figure = fig
        fig.canvas.draw_idle()

    def disconnect(self):
        for attribute in ("_cid","_close_cid"):
            identifier = getattr(self,attribute,None)
            if identifier is not None:
                self.fig.canvas.mpl_disconnect(identifier)
                setattr(self,attribute,None)


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
