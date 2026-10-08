import numpy as np
import matplotlib.pyplot as plt
import json
try:
    from ..simulation.coordinates import physical_domain_nm, normalize_config
    from ..simulation.materials import generate_eps_level
except ImportError:
    from simulation.coordinates import physical_domain_nm, normalize_config
    from simulation.materials import generate_eps_level
import os
import csv

from .field_calculator import (
    compute_residual_Laplace, check_curl_E, check_dirichlet, check_neumann,
)


def _generate_eps_cell_simple(phi, blocks):
    """Generate epsilon on cells from block definitions (local copy)."""
    Nx, Ny, Nz = phi.shape
    eps_cell = np.ones((Nx-1, Ny-1, Nz-1), dtype=np.float32)
    if blocks is None:
        return eps_cell
    if isinstance(blocks, dict):
        blocks = [blocks]
    for blk in blocks:
        eps_val = blk["eps_val"]
        ix0 = max(0, int(np.floor(blk["x_range"][0] * (Nx-1))))
        ix1 = min(Nx-1, int(np.ceil (blk["x_range"][1] * (Nx-1))))
        jy0 = max(0, int(np.floor(blk["y_range"][0] * (Ny-1))))
        jy1 = min(Ny-1, int(np.ceil (blk["y_range"][1] * (Ny-1))))
        kz0 = max(0, int(np.floor(blk["z_range"][0] * (Nz-1))))
        kz1 = min(Nz-1, int(np.ceil (blk["z_range"][1] * (Nz-1))))
        eps_cell[ix0:ix1, jy0:jy1, kz0:kz1] = eps_val
    return eps_cell


def plot_field_slice(field, Lx, Ly, Lz, plane='xy', coord=0.5, zoom=1.0, title=''):
    """Plot a 2-D electric-field or potential diagnostic slice."""
    Nx, Ny, Nz = field.shape
    if plane=='xy':
        iz=int(coord*(Nz-1)); iz=max(0,min(Nz-1,iz)); data=field[:,:,iz].T; extent=[0,Lx,0,Ly]; xl,yl='x (nm)','y (nm)'; cx,cy=0.5*Lx,0.5*Ly
    elif plane=='xz':
        iy=int(coord*(Ny-1)); iy=max(0,min(Ny-1,iy)); data=field[:,iy,:].T; extent=[0,Lx,0,Lz]; xl,yl='x (nm)','z (nm)'; cx,cy=0.5*Lx,0.5*Lz
    elif plane=='yz':
        ix=int(coord*(Nx-1)); ix=max(0,min(Nx-1,ix)); data=field[ix,:,:].T; extent=[0,Ly,0,Lz]; xl,yl='y (nm)','z (nm)'; cx,cy=0.5*Ly,0.5*Lz
    else: raise ValueError("Invalid plane")
    fig,ax=plt.subplots(figsize=(6,5))
    im=ax.imshow(data, origin='lower', cmap='plasma', aspect='auto', extent=extent)
    if zoom>1.0:
        hw=(extent[1]-extent[0])/(2*zoom); hh=(extent[3]-extent[2])/(2*zoom)
        ax.set_xlim(cx-hw, cx+hw); ax.set_ylim(cy-hh, cy+hh)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
    plt.colorbar(im,ax=ax); plt.tight_layout(); plt.show()


def run_sanity_check(
    phi_file="afm_phi_1_-2.00V.npy",
    config_file="afm_config_nm.json",
    tip_mask_file="",
    check_type="e",
    output_dir=None
):
    """Run sanity checks: Poisson residual, curl E, Dirichlet/Neumann BCs."""
    print("=== AFM Simulation Sanity Check ===\n")
    if os.path.isfile(config_file):
        with open(config_file) as f: cfg = normalize_config(json.load(f))
        Lx_nm, Ly_nm, Lz_nm = physical_domain_nm(cfg)
        Vtip = cfg.get("v_start",5.0)
        gates = cfg.get("Vgate_nm", cfg.get("Vgate", []))
        Vgate = gates[0].get("Vgate_val", 0.0) if gates else 0.0
        blocks = cfg.get("blocks", [])
        if output_dir is None:
            output_dir = cfg.get("output_dir", ".")
    else:
        print("Config not found, using defaults."); Lx_nm=Ly_nm=Lz_nm=512.0; Vtip=5.0; Vgate=0.0; blocks=[]; cfg={}
        if output_dir is None:
            output_dir = "."
    phi_path = os.path.join(output_dir, phi_file) if output_dir else phi_file
    if not os.path.isfile(phi_path):
        print(f"File not found: {phi_path}"); return
    phi = np.load(phi_path); Nx, Ny, Nz = phi.shape
    print(f"Potential shape: {Nx}x{Ny}x{Nz}")
    is_zoom = "zoom" in os.path.basename(phi_file).lower()
    if is_zoom:
        zoom_cfg = cfg.get("zoom_simulation", {})
        cut = zoom_cfg.get("cut", {})
        if cut and all(k in cut for k in ("x_range", "y_range", "z_range")):
            xr, yr, zr = cut["x_range"], cut["y_range"], cut["z_range"]
            Lx0, Ly0, Lz0 = physical_domain_nm(cfg)
            Lx_nm, Ly_nm, Lz_nm = (xr[1]-xr[0])*Lx0, (yr[1]-yr[0])*Ly0, (zr[1]-zr[0])*Lz0
            print(f"  Zoom region: ({Lx_nm:.1f}, {Ly_nm:.1f}, {Lz_nm:.1f}) nm")
    dx = Lx_nm*1e-9/(Nx-1); dy = Ly_nm*1e-9/(Ny-1); dz = Lz_nm*1e-9/(Nz-1)
    tip_mask = None
    if tip_mask_file and os.path.isfile(tip_mask_file):
        tip_mask = np.load(tip_mask_file).astype(bool)
    eps_cell = generate_eps_level(phi.shape, blocks, reference_shape=cfg.get("epsilon_material", {}).get("reference_resolution", 512))
    print(f"  Epsilon range: {eps_cell.min():.3f} - {eps_cell.max():.3f}")
    print("\n--- Performing sanity checks ---")
    print("\n[1] Poisson equation residual (eps)")
    L2_eps, max_eps, resid_full = compute_residual_Laplace(phi, eps_cell, tip_mask)
    print(f"  L2 residual: {L2_eps:.3e}  |  Max residual: {max_eps:.3e}")
    print("  PASSED" if max_eps<1e-4 else "  WARNING: residual larger than expected")
    print("\n[2] Curl of electric field")
    mx,my,mz,mag,curl_mag_full = check_curl_E(phi, dx, dy, dz)
    print(f"  Max |curl_x|: {mx:.3e}  |curl_y|: {my:.3e}  |curl_z|: {mz:.3e}")
    print(f"  Max curl magnitude: {mag:.3e} V/m2")
    print("  PASSED" if mag<1e-3 else "  WARNING: curl not negligible")
    print("\n[3] Dirichlet boundary conditions")
    eb,te,_,_ = check_dirichlet(phi, Vtip, Vgate, tip_mask)
    print(f"  Bottom (z=0) deviation from {Vgate} V: {eb:.3e} V")
    print(f"  Tip voltage deviation: {te:.3e} V")
    print("  PASSED" if eb<1e-3 and te<1e-2 else "  WARNING: boundary condition mismatch")
    print("\n[4] Neumann boundary conditions")
    e0,e1,e2,e3,ez1 = check_neumann(phi, tip_mask)
    print(f"  dphi/dx at x=0: {e0:.3e}  x=1: {e1:.3e}  y=0: {e2:.3e}  y=1: {e3:.3e}  z=1: {ez1:.3e}")
    print("  PASSED" if max(e0,e1,e2,e3,ez1)<1e-3 else "  WARNING: Neumann condition not satisfied")
    print("\n--- Interactive plotting ---")
    while True:
        opts = []
        if resid_full is not None: opts.append(("residual (Laplacian)", resid_full))
        if curl_mag_full is not None: opts.append(("curl magnitude", curl_mag_full))
        if not opts: print("No fields available."); break
        print("\nAvailable fields:")
        for i,(n,_) in enumerate(opts,1): print(f" {i}. {n}")
        ch = input("Select field (number, 'q' to quit): ").strip().lower()
        if ch=='q': break
        try: idx=int(ch)-1; fname,field=opts[idx]
        except: print("Invalid."); continue
        plane = input("Plane (xy, xz, yz) [xy]: ").strip().lower()
        if plane not in ('xy','xz','yz'): plane='xy'
        coord = input(f"Coordinate fraction for {plane} slice (0-1, default 0.5): ").strip()
        coord = float(coord) if coord else 0.5
        zm = input("Zoom factor (default 1 = full box): ").strip()
        zm = float(zm) if zm else 1.0
        plot_field_slice(field, Lx_nm, Ly_nm, Lz_nm, plane=plane, coord=coord, zoom=zm,
                         title=f"{fname} - {plane.upper()} slice at coord={coord:.3f}, zoom={zm:.0f}x")
        more = input("Plot another? (y/n): ").strip().lower()
        if more!='y': break
    print("Sanity check complete.")
