"""Recover actual retained solver-node coordinates from a completed run JSON."""
from pathlib import Path
import numpy as np
from . import require_local


def reconstruct_coordinates(path,config_path,center_nm=None):
    require_local()
    from simulation.mpi_config import load_afm_config
    from simulation.mpi_io import physical_cut_slices
    from simulation.output_coordinates import write_coordinate_receipt
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
            if movement.get("start") != movement.get("end") or "start" not in movement:
                raise ValueError("For a moving run, supply this file's exact --center X Y Z in physical nm")
            center_nm = tuple((movement["start"][i]-origin[i])*domain[i] for i in range(3))
        selection,_ = physical_cut_slices(shape,center_nm,cfg.get("save_cut_box_nm"),bounds)
        if selection is None:
            raise ValueError("The configured cut does not intersect the original grid")
    reader = np.load(path,mmap_mode="r",allow_pickle=False)
    try:
        expected = tuple(s.stop-s.start for s in selection) if selection else shape
        if reader.shape != expected:
            raise ValueError("Saved array shape does not match the exact grid/cut configuration")
    finally:
        reader._mmap.close()
    return write_coordinate_receipt(path,shape,bounds,selection)
