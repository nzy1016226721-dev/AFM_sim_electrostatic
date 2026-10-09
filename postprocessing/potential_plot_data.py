"""Compatibility adapter to the one local, coordinate-aware potential reader.

Receipt/node bounds are authoritative. A JSON only validates them; it never
silently reconstructs missing metadata or overrides saved coordinates.
"""
from contextlib import contextmanager
from pathlib import Path

from local_visualization import require_local
from local_visualization.data import PotentialData as LocalData, natural_key
from local_visualization.bulk_potential_plotter import pixel_extent
from postprocessing.npy_utils import parse_phi_filename


class PotentialData:
    """Legacy line/plane interface with explicit close and detached selections.

    bounds_nm retains crop-bin bounds when receipt provenance is present.
    node_bounds_nm always means first/last retained solver nodes.
    """
    def __init__(self,path,*,config_path=None,bounds_nm=None,large_grid_limit=2048,
                 allow_full=False,center_nm=None):
        require_local()
        self._reader = LocalData(Path(path).expanduser(),bounds_nm)
        try:
            self.path = self._reader.path
            meta = parse_phi_filename(self.path.name)
            if meta is None:
                raise ValueError(f"Unrecognized AFM potential filename: {self.path.name}")
            self.is_cut = bool(meta.get("is_cut",False))
            self.source_grid = tuple(meta["source_grid"]) if self.is_cut else self._reader.phi.shape
            if not self.is_cut and max(self.source_grid)>=large_grid_limit and not allow_full:
                raise ValueError(f"Full-grid potential {self.source_grid} requires a saved cut or allow_full=True")
            self.voltage = meta.get("Vtip")
            self.array,self.centres = self._reader.phi,self._reader.axes
            self.node_bounds_nm = tuple(float(v) for v in self._reader.bounds)
            self.bounds_nm = self.node_bounds_nm
            record = self._reader.coordinate_receipt
            if record is not None and "source_shape" in record:
                if tuple(record["source_shape"])!=self.source_grid:
                    raise ValueError("Source grid tag does not match coordinate receipt")
                bin_bounds = []
                for axis,(first,stop) in enumerate(record["source_index_slices"]):
                    lo,hi = record["source_bounds_nm"][2*axis:2*axis+2]
                    n = self.source_grid[axis]
                    bin_bounds.extend((lo+(hi-lo)*first/n,lo+(hi-lo)*stop/n))
                self.bounds_nm = tuple(bin_bounds)
            if config_path is not None:
                from local_visualization.coordinates import validate_coordinates
                validate_coordinates(self.path,config_path,self._reader,center_nm)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()

    def close(self):
        self._reader.close()

    def index(self,axis,coordinate_nm):
        return self._reader.index(axis,coordinate_nm)

    def line(self,axis,at_nm):
        if len(at_nm)!=2:
            raise ValueError("A line requires two fixed coordinates")
        x,y,_ = self._reader.axis_line(axis.lower(),*at_nm)
        return x,y

    def plane(self,plane,at_nm):
        plane = plane.lower()
        image,extent,_ = self._reader.plane(plane,at_nm)
        return image.T,pixel_extent(image.shape,extent),tuple(plane)


def load_potential(path,*,config_path=None,bounds_nm=None,large_grid_limit=2048,
                   allow_full=False,center_nm=None):
    return PotentialData(path,config_path=config_path,bounds_nm=bounds_nm,
                         large_grid_limit=large_grid_limit,allow_full=allow_full,center_nm=center_nm)


@contextmanager
def open_local_reader(path,**kwargs):
    """Use the compatibility checks with the canonical rendering interface."""
    with load_potential(path,**kwargs) as volume:
        yield volume._reader


def discover_potentials(directory,*,prefer_cuts=True):
    require_local()
    files = [p for p in Path(directory).expanduser().glob("*.npy")
             if parse_phi_filename(p.name) is not None]
    if prefer_cuts:
        cut_bases = {p.name.split("_cut_from_grid",1)[0] for p in files if "_cut_from_grid" in p.name}
        files = [p for p in files if "_cut_from_grid" in p.name or p.stem not in cut_bases]
    return sorted(files,key=natural_key)
