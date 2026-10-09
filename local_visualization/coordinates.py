"""Recover actual retained solver-node coordinates from a completed run JSON."""
from pathlib import Path
import numpy as np
from . import require_local


def coordinate_plan(path,config_path,center_nm=None,*,resolve_movement=False):
    """Read-only expected coordinates, shared by validation and reconstruction."""
    require_local()
    from simulation.mpi_config import load_afm_config
    from simulation.mpi_io import physical_cut_slices
    from postprocessing.npy_utils import parse_phi_filename
    _,cfg = load_afm_config(config_path)
    shape = tuple(int(cfg["grid_resolution"][axis]) for axis in ("nx","ny","nz"))
    physical = cfg.get("_physical")
    if physical is None:
        raise ValueError("The exact physical *_nm configuration is required")
    domain = physical["domain_nm"]
    origin = physical["origin_fraction"]
    bounds = tuple(v for i in range(3) for v in (-origin[i]*domain[i],(1-origin[i])*domain[i]))
    info = parse_phi_filename(Path(path).name)
    selection = None
    if info and info.get("is_cut"):
        if tuple(info["source_grid"]) != shape:
            raise ValueError("Cut source-grid tag differs from the supplied configuration; do not guess the domain")
        if center_nm is None:
            movement = cfg.get("movement",{})
            if resolve_movement:
                center_nm = _movement_center(info,cfg)
            elif movement.get("start") != movement.get("end") or "start" not in movement:
                raise ValueError("For a moving run, supply this file's exact --center X Y Z in physical nm")
            else:
                center_nm = tuple((movement["start"][i]-origin[i])*domain[i] for i in range(3))
        selection,_ = physical_cut_slices(shape,center_nm,cfg.get("save_cut_box_nm"),bounds)
        if selection is None:
            raise ValueError("The configured cut does not intersect the original grid")
    if selection is None:
        selection = tuple(slice(0,n) for n in shape)
    node_bounds = tuple(value for axis,s in enumerate(selection) for value in
                        (bounds[2*axis]+(bounds[2*axis+1]-bounds[2*axis])*s.start/(shape[axis]-1),
                         bounds[2*axis]+(bounds[2*axis+1]-bounds[2*axis])*(s.stop-1)/(shape[axis]-1)))
    return dict(source_shape=shape,source_bounds_nm=bounds,slices=selection,
                shape=tuple(s.stop-s.start for s in selection),node_bounds_nm=node_bounds)


def _movement_center(info,cfg):
    """Match the existing signed movement suffix; never infer arbitrary xyz."""
    from simulation.io_utils import movement_suffix_nm
    from simulation.main_loop import compute_block_positions
    movement = cfg.get("movement",{})
    physical = cfg["_physical"]
    domain,origin = physical["domain_nm"],physical["origin_fraction"]
    start = tuple(movement.get("start",(.5,.5,.5)))
    end = tuple(movement.get("end",start))
    if info.get("position") is not None:
        center = info["position"]
    else:
        spacing = physical.get("movement",{}).get("spacing_nm")
        centres = compute_block_positions(start,end,spacing,domain_nm=domain) if spacing is not None else compute_block_positions(start,end,movement.get("spacing",.1))
        offset = info.get("movement_offset_nm")
        if offset is None:
            center = centres[0]
        else:
            matches = []
            for candidate in centres:
                suffix = movement_suffix_nm(candidate,centres[0],domain)
                value = -float(suffix[1:-2]) if suffix.startswith("-") else float(suffix[1:-2])
                if abs(value-float(offset))<1e-10:
                    matches.append(candidate)
            if len(matches)!=1:
                raise ValueError("Cannot identify a unique movement centre; supply exact center_nm")
            center = matches[0]
    return tuple((center[i]-origin[i])*domain[i] for i in range(3))


def validate_coordinates(path,config_path,reader,center_nm=None):
    plan = coordinate_plan(path,config_path,center_nm,resolve_movement=True)
    if reader.phi.shape!=plan["shape"]:
        raise ValueError("Cut shape does not match supplied configuration")
    if not np.allclose(reader.bounds,plan["node_bounds_nm"],rtol=0,atol=1e-9):
        raise ValueError("Supplied configuration does not match saved node coordinates")
    record = reader.coordinate_receipt
    if record is not None and "source_shape" in record:
        if (tuple(record["source_shape"])!=plan["source_shape"]
                or not np.allclose(record["source_bounds_nm"],plan["source_bounds_nm"],rtol=0,atol=1e-9)
                or record["source_index_slices"]!=[[s.start,s.stop] for s in plan["slices"]]):
            raise ValueError("Supplied configuration does not match saved source provenance")


def reconstruct_coordinates(path,config_path,center_nm=None):
    require_local()
    from simulation.output_coordinates import write_coordinate_receipt
    plan = coordinate_plan(path,config_path,center_nm)
    reader = np.load(path,mmap_mode="r",allow_pickle=False)
    try:
        if reader.shape != plan["shape"]:
            raise ValueError("Saved array shape does not match the exact grid/cut configuration")
    finally:
        reader._mmap.close()
    return write_coordinate_receipt(path,plan["source_shape"],plan["source_bounds_nm"],plan["slices"])
