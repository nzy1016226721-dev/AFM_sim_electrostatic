import numpy as np
import matplotlib.pyplot as plt
import json
import os
import csv

from .field_calculator import (
    compute_residual_Laplace, check_curl_E, check_dirichlet,
    check_neumann, compute_current_divergence, integrate_power
)


def _generate_eps_cell_simple(phi, blocks):
    """Generate epsilon on cells from block definitions (local copy).

    Parameters
    ----------
    phi : np.ndarray
        3D potential (for grid dimensions).
    blocks : list of dict or dict
        Dielectric blocks.

    Returns
    -------
    np.ndarray
        Epsilon cell array (Nx-1, Ny-1, Nz-1).
    """
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


def _generate_sigma_cell_simple(phi, blocks):
    """Generate conductivity on cells from block definitions (local copy).

    Parameters
    ----------
    phi : np.ndarray
        3D potential (for grid dimensions).
    blocks : list of dict or dict
        Conductivity blocks.

    Returns
    -------
    np.ndarray
        Conductivity cell array (Nx-1, Ny-1, Nz-1).
    """
    Nx, Ny, Nz = phi.shape
    sigma_cell = np.ones((Nx-1, Ny-1, Nz-1), dtype=np.float32) * 1e-12
    if blocks is None:
        return sigma_cell
    if isinstance(blocks, dict):
        blocks = [blocks]
    for blk in blocks:
        sigma = blk["sigma_val"]
        ix0 = max(0, int(np.floor(blk["x_range"][0] * (Nx-1))))
        ix1 = min(Nx-1, int(np.ceil (blk["x_range"][1] * (Nx-1))))
        jy0 = max(0, int(np.floor(blk["y_range"][0] * (Ny-1))))
        jy1 = min(Ny-1, int(np.ceil (blk["y_range"][1] * (Ny-1))))
        kz0 = max(0, int(np.floor(blk["z_range"][0] * (Nz-1))))
        kz1 = min(Nz-1, int(np.ceil (blk["z_range"][1] * (Nz-1))))
        sigma_cell[ix0:ix1, jy0:jy1, kz0:kz1] = sigma
    return sigma_cell


def plot_field_slice(field, Lx, Ly, Lz, plane='xy', coord=0.5, zoom=1.0, title=''):
    """Plot a 2D slice of a 3D field with optional zoom.

    Parameters
    ----------
    field : np.ndarray
        3D field array.
    Lx, Ly, Lz : float
        Box dimensions in nm.
    plane : str, optional
        'xy', 'xz', or 'yz' (default: 'xy').
    coord : float, optional
        Slice coordinate (fractional, default: 0.5).
    zoom : float, optional
        Zoom factor (default: 1.0).
    title : str, optional
        Plot title (default: '').

    Returns
    -------
    None
    """

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
    config_file="afm_config_nm_frac.json",
    tip_mask_file="",
    check_type="e",
    use_precomputed=True,
    sigma_blocks_file="sigma_blocks.json",
    output_dir=None
):
    """Run a comprehensive suite of sanity checks on an AFM simulation result.

    Tests: Poisson residual (with epsilon), curl of E, Dirichlet/Neumann BCs,
    current conservation (div J = 0), and Joule power consistency.

    Parameters
    ----------
    phi_file : str, optional
        Path to potential .npy (default: "afm_phi_1_-2.00V.npy").
    config_file : str, optional
        Path to config JSON (default: "afm_config_nm_frac.json").
    tip_mask_file : str, optional
        Path to tip mask .npy (default: "").
    check_type : str, optional
        'e' for epsilon checks, 'c' for conductivity checks (default: "e").
    use_precomputed : bool, optional
        Use precomputed NPZ data (default: True).
    sigma_blocks_file : str, optional
        Sigma blocks JSON path (default: "sigma_blocks.json").
    output_dir : str or None, optional
        Output directory (default: None = from config).

    Returns
    -------
    None
    """
    print("=== AFM Simulation Sanity Check ===\n")

    if os.path.isfile(config_file):
        with open(config_file) as f: cfg = json.load(f)
        Lx_nm = cfg.get("Lx_nm",512.0); Ly_nm = cfg.get("Ly_nm",512.0); Lz_nm = cfg.get("Lz_nm",512.0)
        Vtip = cfg.get("v_start",5.0)
        Vgate = cfg.get("Vgate",[{"Vgate_val":0.0}])[0]["Vgate_val"] if cfg.get("Vgate") else 0.0
        blocks = cfg.get("blocks",[])
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
            zoom_Lx = (xr[1] - xr[0]) * Lx_nm
            zoom_Ly = (yr[1] - yr[0]) * Ly_nm
            zoom_Lz = (zr[1] - zr[0]) * Lz_nm
            print(f"  Zoom region: ({zoom_Lx:.1f}, {zoom_Ly:.1f}, {zoom_Lz:.1f}) nm")
            Lx_nm, Ly_nm, Lz_nm = zoom_Lx, zoom_Ly, zoom_Lz
        else:
            print("  Warning: zoom filename but no valid cut - using full-domain")
            is_zoom = False

    dx = Lx_nm*1e-9/(Nx-1); dy = Ly_nm*1e-9/(Ny-1); dz = Lz_nm*1e-9/(Nz-1)

    tip_mask = None
    if tip_mask_file and os.path.isfile(tip_mask_file):
        tip_mask = np.load(tip_mask_file).astype(bool); print(f"Tip mask loaded, shape {tip_mask.shape}")

    eps_cell = sigma_cell = None

    if check_type in ('','e'):
        if is_zoom:
            dx_f = xr[1] - xr[0]; dy_f = yr[1] - yr[0]; dz_f = zr[1] - zr[0]
            zoom_blocks = []
            for b in blocks:
                ix0 = max(b["x_range"][0], xr[0]); ix1 = min(b["x_range"][1], xr[1])
                iy0 = max(b["y_range"][0], yr[0]); iy1 = min(b["y_range"][1], yr[1])
                iz0 = max(b["z_range"][0], zr[0]); iz1 = min(b["z_range"][1], zr[1])
                if ix1>ix0 and iy1>iy0 and iz1>iz0:
                    zoom_blocks.append({
                        "eps_val": b["eps_val"],
                        "x_range": [(ix0-xr[0])/dx_f, (ix1-xr[0])/dx_f],
                        "y_range": [(iy0-yr[0])/dy_f, (iy1-yr[0])/dy_f],
                        "z_range": [(iz0-zr[0])/dz_f, (iz1-zr[0])/dz_f]
                    })
            eps_cell = _generate_eps_cell_simple(phi, zoom_blocks)
            print(f"  Epsilon range (zoom): {eps_cell.min():.3f} - {eps_cell.max():.3f}")
        elif use_precomputed:
            data = np.load("eps_highres.npz")
            bg = data["z_profile_background"]; pillar = data.get("z_profile_pillar"); rect = data.get("pillar_rect")
            nz_high = len(bg); z_high = np.linspace(0,1,nz_high)
            z_edges = np.linspace(0,1,Nz); left = np.searchsorted(z_high,z_edges[:-1]); right = np.searchsorted(z_high,z_edges[1:]); right[-1]=nz_high
            bg_avg = np.array([np.mean(bg[l:r]) for l,r in zip(left,right)], dtype=np.float32)
            if pillar is not None:
                pillar_avg = np.array([np.mean(pillar[l:r]) for l,r in zip(left,right)], dtype=np.float32)
                if rect is not None:
                    x0,x1,y0,y1 = rect; x_ed=np.linspace(0,1,Nx); y_ed=np.linspace(0,1,Ny); dx_f=1.0/(Nx-1); dy_f=1.0/(Ny-1)
                    weights = np.zeros((Nx-1,Ny-1), dtype=np.float32)
                    for i in range(Nx-1):
                        for j in range(Ny-1):
                            ox0=max(x_ed[i],x0); ox1=min(x_ed[i+1],x1)
                            oy0=max(y_ed[j],y0); oy1=min(y_ed[j+1],y1)
                            if ox1>ox0 and oy1>oy0: weights[i,j]=((ox1-ox0)*(oy1-oy0))/(dx_f*dy_f)
                    w = weights[:,:,np.newaxis]
                    eps_cell = ((1-w)*bg_avg[np.newaxis,np.newaxis,:] + w*pillar_avg[np.newaxis,np.newaxis,:]).astype(np.float32)
                else:
                    eps_cell = np.tile(pillar_avg[np.newaxis,np.newaxis,:], (Nx-1,Ny-1,1)).astype(np.float32)
            else:
                eps_cell = np.tile(bg_avg[np.newaxis,np.newaxis,:], (Nx-1,Ny-1,1)).astype(np.float32)
            assert eps_cell.shape==(Nx-1,Ny-1,Nz-1)
            print(f"  Epsilon range: {eps_cell.min():.3f} - {eps_cell.max():.3f}")
            if eps_cell.max()>15.0: print("  WARNING: maximum epsilon > 15 - unphysical for Si")
        else:
            eps_cell = _generate_eps_cell_simple(phi, blocks)
            print(f"  Epsilon range (blocks): {eps_cell.min():.3f} - {eps_cell.max():.3f}")

    elif check_type=='c':
        if is_zoom:
            if not os.path.isfile(sigma_blocks_file):
                print("  sigma_blocks.json not found - cannot generate zoom conductivity"); return
            with open(sigma_blocks_file) as f:
                sblk_full = json.load(f)["sigma_blocks"]
            dx_f = xr[1] - xr[0]; dy_f = yr[1] - yr[0]; dz_f = zr[1] - zr[0]
            zoom_sblk = []
            for b in sblk_full:
                ix0 = max(b["x_range"][0], xr[0]); ix1 = min(b["x_range"][1], xr[1])
                iy0 = max(b["y_range"][0], yr[0]); iy1 = min(b["y_range"][1], yr[1])
                iz0 = max(b["z_range"][0], zr[0]); iz1 = min(b["z_range"][1], zr[1])
                if ix1>ix0 and iy1>iy0 and iz1>iz0:
                    zoom_sblk.append({
                        "sigma_val": b["sigma_val"],
                        "x_range": [(ix0-xr[0])/dx_f, (ix1-xr[0])/dx_f],
                        "y_range": [(iy0-yr[0])/dy_f, (iy1-yr[0])/dy_f],
                        "z_range": [(iz0-zr[0])/dz_f, (iz1-zr[0])/dz_f]
                    })
            sigma_cell = _generate_sigma_cell_simple(phi, zoom_sblk)
            print(f"  Conductivity range (zoom): {sigma_cell.min():.1e} - {sigma_cell.max():.1e} S/m")
        elif use_precomputed:
            data = np.load("sigma_highres.npz")
            bg = data["z_profile_background"]; pillar = data.get("z_profile_pillar"); rect = data.get("pillar_rect")
            nz_high = len(bg); z_high = np.linspace(0,1,nz_high)
            z_edges = np.linspace(0,1,Nz); left = np.searchsorted(z_high,z_edges[:-1]); right = np.searchsorted(z_high,z_edges[1:]); right[-1]=nz_high
            bg_avg = np.array([np.mean(bg[l:r]) for l,r in zip(left,right)], dtype=np.float32)
            if pillar is not None:
                pillar_avg = np.array([np.mean(pillar[l:r]) for l,r in zip(left,right)], dtype=np.float32)
                if rect is not None:
                    x0,x1,y0,y1 = rect; x_ed=np.linspace(0,1,Nx); y_ed=np.linspace(0,1,Ny); dx_f=1.0/(Nx-1); dy_f=1.0/(Ny-1)
                    weights = np.zeros((Nx-1,Ny-1), dtype=np.float32)
                    for i in range(Nx-1):
                        for j in range(Ny-1):
                            ox0=max(x_ed[i],x0); ox1=min(x_ed[i+1],x1)
                            oy0=max(y_ed[j],y0); oy1=min(y_ed[j+1],y1)
                            if ox1>ox0 and oy1>oy0: weights[i,j]=((ox1-ox0)*(oy1-oy0))/(dx_f*dy_f)
                    w = weights[:,:,np.newaxis]
                    sigma_cell = ((1-w)*bg_avg[np.newaxis,np.newaxis,:] + w*pillar_avg[np.newaxis,np.newaxis,:]).astype(np.float32)
                else:
                    sigma_cell = np.tile(pillar_avg[np.newaxis,np.newaxis,:], (Nx-1,Ny-1,1)).astype(np.float32)
            else:
                sigma_cell = np.tile(bg_avg[np.newaxis,np.newaxis,:], (Nx-1,Ny-1,1)).astype(np.float32)
            assert sigma_cell.shape==(Nx-1,Ny-1,Nz-1)
            print(f"  Conductivity range: {sigma_cell.min():.1f} - {sigma_cell.max():.1f} S/m")
        else:
            with open(sigma_blocks_file) as f: sblk = json.load(f)["sigma_blocks"]
            sigma_cell = _generate_sigma_cell_simple(phi, sblk)
            print(f"  Conductivity range (blocks): {sigma_cell.min():.1e} - {sigma_cell.max():.1e} S/m")
    else:
        print("Unknown option."); return

    print("\n--- Performing sanity checks ---")
    resid_full = curl_mag_full = divJ_full = None

    if eps_cell is not None:
        print("\n[1] Poisson equation residual (eps)")
        L2_eps, max_eps, resid_full = compute_residual_Laplace(phi, eps_cell, tip_mask)
        print(f"  L2 residual: {L2_eps:.3e}  |  Max residual: {max_eps:.3e}")
        if max_eps<1e-4: print("  PASSED")
        else: print("  WARNING: residual larger than expected")

    print("\n[2] Curl of electric field")
    mx,my,mz,mag,curl_mag_full = check_curl_E(phi, dx, dy, dz)
    print(f"  Max |curl_x|: {mx:.3e}  |curl_y|: {my:.3e}  |curl_z|: {mz:.3e}")
    print(f"  Max curl magnitude: {mag:.3e} V/m2")
    if mag<1e-3: print("  PASSED")
    else: print("  WARNING: curl not negligible (tip boundary may cause this)")

    print("\n[3] Dirichlet boundary conditions")
    eb,te,_,_ = check_dirichlet(phi, Vtip, Vgate, tip_mask)
    print(f"  Bottom (z=0) deviation from {Vgate} V: {eb:.3e} V")
    print(f"  Tip voltage deviation: {te:.3e} V")
    if eb<1e-3 and te<1e-2: print("  PASSED")
    else: print("  WARNING: boundary condition mismatch")

    print("\n[4] Neumann boundary conditions")
    e0,e1,e2,e3,ez1 = check_neumann(phi, tip_mask)
    print(f"  dphi/dx at x=0: {e0:.3e}  x=1: {e1:.3e}  y=0: {e2:.3e}  y=1: {e3:.3e}  z=1: {ez1:.3e}")
    if max(e0,e1,e2,e3,ez1)<1e-3: print("  PASSED")
    else: print("  WARNING: Neumann condition not satisfied")

    if sigma_cell is not None:
        print("\n[5] Current conservation (divJ = 0)")
        dmax, drms, divJ_full = compute_current_divergence(phi, sigma_cell, dx, dy, dz)
        print(f"  Max |divJ|: {dmax:.3e} A/m3   RMS: {drms:.3e} A/m3")
        if dmax<1e-3: print("  PASSED")
        else: print("  WARNING: current not divergence-free")

        print("\n[6] Total Joule power consistency")
        Pc = integrate_power(phi, sigma_cell, dx, dy, dz)
        print(f"  Calculated total power: {Pc:.6e} W")
        log = "joule_power_total.csv"
        if os.path.isfile(log):
            with open(log) as f:
                reader = csv.reader(f); next(reader)
                for row in reader: P_csv = float(row[2]); break
            print(f"  Logged total power: {P_csv:.6e} W")
            if abs(Pc-P_csv)/max(abs(Pc),1e-30)<0.01: print("  PASSED")
            else: print("  WARNING: discrepancy with logged value")

    print("\n--- Interactive plotting ---")
    while True:
        opts = []
        if resid_full is not None: opts.append(("residual (Laplacian)", resid_full))
        if curl_mag_full is not None: opts.append(("curl magnitude", curl_mag_full))
        if divJ_full is not None: opts.append(("|divJ| (current divergence)", divJ_full))
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
        zoom = input("Zoom factor (default 1 = full box): ").strip()
        zoom = float(zoom) if zoom else 1.0
        plot_field_slice(field, Lx_nm, Ly_nm, Lz_nm, plane=plane, coord=coord, zoom=zoom,
                         title=f"{fname} - {plane.upper()} slice at coord={coord:.3f}, zoom={zoom:.0f}x")
        more = input("Plot another? (y/n): ").strip().lower()
        if more!='y': break

    print("Sanity check complete.")
