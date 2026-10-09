"""Read one downloaded NPY at a time. No simulation imports or full gradients."""
from pathlib import Path
import json
import re
import numpy as np

from . import require_local


def natural_key(path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(path))]


def select_indices(text, count):
    """Eric's inclusive file ranges, extended with start:stop:step selection."""
    if not text.strip():
        return list(range(count))
    indices = []
    for part in text.split(","):
        part = part.strip()
        if re.fullmatch(r"\d+",part):
            values = [int(part)]
        elif re.fullmatch(r"\d+-\d+",part):
            start, stop = map(int,part.split("-"))
            if stop < start:
                raise ValueError("file ranges must be ascending")
            values = range(start,stop+1)
        elif re.fullmatch(r"\d+:\d+:\d+",part):
            start, stop, step = map(int,part.split(":"))
            if step < 1 or stop < start:
                raise ValueError("invalid file interval")
            values = range(start,stop+1,step)
        else:
            raise ValueError("file selection must be 0-5,8 or 0:20:2")
        for value in values:
            if not 0 <= value < count:
                raise ValueError(f"file index {value} outside 0..{count-1}")
            if value not in indices:
                indices.append(value)
    return indices


class PotentialData:
    def __init__(self,path,bounds_nm=None):
        require_local()
        self.path = Path(path).resolve()
        self.phi = np.load(self.path, mmap_mode="r",allow_pickle=False)
        self._closed = False
        self.coordinate_receipt = None
        try:
            if self.phi.ndim != 3 or min(self.phi.shape) < 2 or self.phi.dtype.kind != "f":
                raise ValueError("Expected a three-dimensional real floating-point potential")
            sidecar = Path(str(self.path)+".coords.json")
            if sidecar.is_file():
                record = json.loads(sidecar.read_text(encoding="utf-8-sig"))
                if (not isinstance(record,dict) or record.get("format") != "afm-local-coordinates-v1"
                        or record.get("array_order") != "xyz"
                        or record.get("potential_units") != "V"
                        or record.get("coordinate_units") != "nm"
                        or record.get("shape") != list(self.phi.shape)):
                    raise ValueError("Coordinate receipt does not match this xyz potential")
                recorded_bounds = np.asarray(record["node_bounds_nm"],dtype=np.float64)
                if bounds_nm is not None:
                    supplied = np.asarray(bounds_nm,dtype=np.float64)
                    if supplied.shape != (6,) or recorded_bounds.shape != (6,) or not np.allclose(
                            supplied,recorded_bounds,rtol=0,atol=1e-9):
                        raise ValueError("Explicit bounds does not match the saved coordinate receipt")
                bounds_nm = recorded_bounds
                self.coordinate_receipt = record
            elif bounds_nm is None:
                raise ValueError("No coordinate receipt: supply --bounds with FIRST/LAST RETAINED NODE coordinates in nm; cut filenames alone are insufficient")
            self.bounds = np.asarray(bounds_nm,dtype=np.float64)
            if self.bounds.shape != (6,) or not np.isfinite(self.bounds).all():
                raise ValueError("Six finite node bounds are required")
            if any(self.bounds[2*i+1] <= self.bounds[2*i] for i in range(3)):
                raise ValueError("Node bounds must be strictly increasing")
            if self.coordinate_receipt is not None:
                self._validate_source_provenance()
            self.axes = tuple(np.linspace(self.bounds[2*i], self.bounds[2*i+1], n) for i,n in enumerate(self.phi.shape))
        except BaseException:
            self.close()
            raise

    def _validate_source_provenance(self):
        record = self.coordinate_receipt
        keys = ("source_shape","source_bounds_nm","source_index_slices")
        if not any(key in record for key in keys):
            return  # A minimal explicit-node receipt remains supported.
        if not all(key in record for key in keys):
            raise ValueError("Coordinate receipt source provenance is incomplete")
        shape,indices = record["source_shape"],record["source_index_slices"]
        bounds = np.asarray(record["source_bounds_nm"],dtype=np.float64)
        if (not isinstance(shape,list) or len(shape)!=3 or any(type(n) is not int or n<2 for n in shape)
                or not isinstance(indices,list) or len(indices)!=3 or bounds.shape!=(6,) or not np.isfinite(bounds).all()):
            raise ValueError("Coordinate receipt source provenance is invalid")
        expected = []
        for axis,(n,pair) in enumerate(zip(shape,indices)):
            if (not isinstance(pair,list) or len(pair)!=2 or any(type(v) is not int for v in pair)
                    or not 0<=pair[0]<pair[1]<=n or pair[1]-pair[0]!=self.phi.shape[axis]):
                raise ValueError("Coordinate receipt source indices does not match saved shape")
            lo,hi = bounds[2*axis:2*axis+2]
            if hi<=lo:
                raise ValueError("Coordinate receipt source bounds must increase")
            expected.extend((lo+(hi-lo)*pair[0]/(n-1),lo+(hi-lo)*(pair[1]-1)/(n-1)))
        if not np.allclose(expected,self.bounds,rtol=0,atol=1e-9):
            raise ValueError("Coordinate receipt node bounds does not match source indices")

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()

    def close(self):
        mapping = getattr(self.phi,"_mmap",None)
        if mapping is not None:
            mapping.close()
        self._closed = True

    def index(self,axis,position):
        require_local()
        if self._closed:
            raise ValueError("Potential reader is closed; open a fresh context")
        a = self.axes[axis]
        if not np.isfinite(position) or not a[0] <= position <= a[-1]:
            raise ValueError(f"{position} nm outside {'xyz'[axis]}=[{a[0]}, {a[-1]}] nm")
        return int(np.argmin(np.abs(a-position)))

    def plane(self,plane,position):
        if plane not in ("xy","xz","yz"):
            raise ValueError("plane must be xy, xz or yz")
        normal = next(i for i,c in enumerate("xyz") if c not in plane)
        index = self.index(normal,position)
        selection = [slice(None)]*3
        selection[normal] = index
        data = np.array(self.phi[tuple(selection)],copy=True)
        in_plane = tuple("xyz".index(c) for c in plane)
        extent = tuple(v for i in in_plane for v in (self.axes[i][0],self.axes[i][-1]))
        return data, extent, float(self.axes[normal][index])

    def axis_line(self,axis,first,second):
        along = "xyz".index(axis)
        others = [i for i in range(3) if i != along]
        selection = [slice(None)]*3
        positions = []
        for i,position in zip(others,(first,second)):
            index = self.index(i,position)
            selection[i] = index
            positions.append(float(self.axes[i][index]))
        return self.axes[along].copy(), np.array(self.phi[tuple(selection)],copy=True), positions

    def profile(self,start,end,samples=500):
        start,end = np.asarray(start,float),np.asarray(end,float)
        if start.shape != (3,) or end.shape != (3,) or not 2 <= int(samples) <= 100000:
            raise ValueError("profile requires two xyz points and 2..100000 samples")
        for axis in range(3):
            self.index(axis,start[axis]); self.index(axis,end[axis])
        points = np.linspace(start,end,int(samples))
        fraction = np.empty_like(points)
        for axis in range(3):
            a = self.axes[axis]
            fraction[:,axis] = (points[:,axis]-a[0])/(a[-1]-a[0])*(len(a)-1)
        lo = np.floor(fraction).astype(np.int64)
        for axis,n in enumerate(self.phi.shape):
            lo[:,axis] = np.clip(lo[:,axis],0,n-2)
        weights = fraction-lo
        result = np.zeros(int(samples),dtype=np.float64)
        for i in (0,1):
            for j in (0,1):
                for k in (0,1):
                    weight = ((weights[:,0] if i else 1-weights[:,0])
                              *(weights[:,1] if j else 1-weights[:,1])
                              *(weights[:,2] if k else 1-weights[:,2]))
                    result += weight*self.phi[lo[:,0]+i,lo[:,1]+j,lo[:,2]+k]
        distance = np.linspace(0,float(np.linalg.norm(end-start)),int(samples))
        return distance,result

    def sanity(self):
        require_local()
        if self._closed:
            raise ValueError("Potential reader is closed; open a fresh context")
        count,nonfinite,total,minimum,maximum = 0,0,0.0,float("inf"),-float("inf")
        for i in range(self.phi.shape[0]):
            # Do not fault the entire full-grid file into one long-lived mmap
            # during a sanity scan. Copy one plane and unmap its reader before
            # continuing; the caller's original read-only mapping stays valid.
            reader = np.load(self.path,mmap_mode="r",allow_pickle=False)
            try:
                slab = np.array(reader[i],copy=True)
            finally:
                reader._mmap.close()
            finite = np.isfinite(slab)
            values = slab[finite]
            count += int(values.size)
            nonfinite += int(slab.size-values.size)
            if values.size:
                minimum = min(minimum,float(values.min()))
                maximum = max(maximum,float(values.max()))
                total += float(np.sum(values,dtype=np.float64))
        return dict(file=str(self.path), shape=list(self.phi.shape),dtype=str(self.phi.dtype),
                    node_bounds_nm=self.bounds.tolist(),finite_count=count,nonfinite_count=nonfinite,
                    minimum_V=minimum if count else None, maximum_V=maximum if count else None,
                    mean_V=total/count if count else None,
                    qualification="Read-only data/coordinate sanity, NOT a dielectric residual or convergence certificate")
