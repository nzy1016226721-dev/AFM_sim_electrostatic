"""Eric's additional conductivity helpers, explicitly LOCAL and optional.

Adapted from field_calculator.py at 4db9c60 (Eric Friesen Waldner branch).
These require conductivity in S/m and dx/dy/dz in metres. They are NOT the
AFM dielectric operator, a capacitive-power model, or convergence certificates.
Full derived arrays are intrinsic to this optional API: a conservative RAM
preflight rejects large inputs instead of silently allocating many volumes.
"""
import numpy as np
from . import require_local


def _admit(phi,sigma_cell,max_workspace_mib):
    require_local()
    if phi.ndim!=3 or sigma_cell.shape!=tuple(v-1 for v in phi.shape):
        raise ValueError("conductivity must be on this potential's cell grid")
    if min(phi.shape)<3 or phi.dtype.kind!="f" or not np.isfinite(max_workspace_mib) or max_workspace_mib<=0:
        raise ValueError("invalid field or workspace budget")
    import psutil
    estimated = 28*phi.nbytes+12*sigma_cell.nbytes
    permitted = min(max_workspace_mib*1024**2,max(0,psutil.virtual_memory().available-500000000))
    if estimated>permitted:
        raise MemoryError("Eric's optional full-gradient diagnostic exceeds its local workspace/reserve budget; use a smaller crop")


def cell_center_to_node(sigma_cell,*,max_workspace_mib=256):
    require_local()
    import psutil
    estimated = 12*sigma_cell.nbytes+np.prod([v+1 for v in sigma_cell.shape])*sigma_cell.dtype.itemsize
    permitted = min(max_workspace_mib*1024**2,max(0,psutil.virtual_memory().available-500000000))
    if estimated>permitted:
        raise MemoryError("Cell-to-node diagnostic exceeds the local workspace/reserve budget")
    Nx,Ny,Nz = (v+1 for v in sigma_cell.shape)
    node = np.zeros((Nx,Ny,Nz),dtype=sigma_cell.dtype)
    node[1:-1,1:-1,1:-1] = (
        sigma_cell[:-1,:-1,:-1]+sigma_cell[1:,:-1,:-1]+
        sigma_cell[:-1,1:,:-1]+sigma_cell[1:,1:,:-1]+
        sigma_cell[:-1,:-1,1:]+sigma_cell[1:,:-1,1:]+
        sigma_cell[:-1,1:,1:]+sigma_cell[1:,1:,1:])/8.0
    return node


def compute_current_divergence(phi,sigma_cell,dx,dy,dz,*,max_workspace_mib=256):
    _admit(phi,sigma_cell,max_workspace_mib)
    if min(dx,dy,dz)<=0:
        raise ValueError("spacings must be positive metres")
    Ex,Ey,Ez = np.gradient(-phi,dx,dy,dz,edge_order=2)
    sn = cell_center_to_node(sigma_cell,max_workspace_mib=max_workspace_mib)
    Jx,Jy,Jz = sn*Ex,sn*Ey,sn*Ez
    divJ = ((Jx[2:,1:-1,1:-1]-Jx[:-2,1:-1,1:-1])/(2*dx)
            +(Jy[1:-1,2:,1:-1]-Jy[1:-1,:-2,1:-1])/(2*dy)
            +(Jz[1:-1,1:-1,2:]-Jz[1:-1,1:-1,:-2])/(2*dz))
    full = np.full_like(phi,np.nan)
    full[1:-1,1:-1,1:-1] = divJ
    return np.nanmax(np.abs(divJ)),np.sqrt(np.nanmean(divJ**2)),full


def integrate_power(phi,sigma_cell,dx,dy,dz,*,max_workspace_mib=256):
    _admit(phi,sigma_cell,max_workspace_mib)
    if min(dx,dy,dz)<=0:
        raise ValueError("spacings must be positive metres")
    Ex,Ey,Ez = np.gradient(-phi,dx,dy,dz,edge_order=2)
    # Preserve Eric's eight-node average and arithmetic, not a new power model.
    def centers(e):
        return (e[:-1,:-1,:-1]+e[1:,:-1,:-1]+e[:-1,1:,:-1]+e[1:,1:,:-1]
                +e[:-1,:-1,1:]+e[1:,:-1,1:]+e[:-1,1:,1:]+e[1:,1:,1:])/8.0
    Ex_c,Ey_c,Ez_c = centers(Ex),centers(Ey),centers(Ez)
    density = sigma_cell*(Ex_c**2+Ey_c**2+Ez_c**2)
    return np.sum(density)*(dx*dy*dz)
