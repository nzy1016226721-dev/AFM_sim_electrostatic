import numpy as np
import os


_eps_cache = None
_sigma_cache = None


def _load_eps_cache():
    """Load and cache the high-resolution epsilon NPZ data.

    Returns
    -------
    dict or False
        Dict with keys 'bg', 'pillar', 'rect', or False if file not found.
    """
    global _eps_cache
    if _eps_cache is None:
        try:
            data = np.load("eps_highres.npz")
            _eps_cache = {
                "bg": data["z_profile_background"],
                "pillar": data["z_profile_pillar"] if "z_profile_pillar" in data else None,
                "rect": data["pillar_rect"] if "pillar_rect" in data else None
            }
        except FileNotFoundError:
            _eps_cache = False
    return _eps_cache


def _load_sigma_cache():
    """Load and cache the high-resolution sigma NPZ data.

    Returns
    -------
    dict or False
        Dict with keys 'bg', 'pillar', 'rect', or False if file not found.
    """
    global _sigma_cache
    if _sigma_cache is None:
        try:
            data = np.load("sigma_highres.npz")
            _sigma_cache = {
                "bg": data["z_profile_background"],
                "pillar": data["z_profile_pillar"] if "z_profile_pillar" in data else None,
                "rect": data["pillar_rect"] if "pillar_rect" in data else None
            }
        except FileNotFoundError:
            _sigma_cache = False
    return _sigma_cache


def invalidate_material_cache():
    """Clear both epsilon and sigma material caches.

    Ensures fresh reload from NPZ files on next access.
    """
    global _eps_cache, _sigma_cache
    _eps_cache = None
    _sigma_cache = None


def _generate_eps_cell_simple(phi, blocks):
    """Generate epsilon cell array directly from block definitions (no cache).

    Parameters
    ----------
    phi : np.ndarray
        3D potential array (used for grid dimensions).
    blocks : list of dict or dict
        Dielectric block definitions with 'eps_val' and '*_range'.

    Returns
    -------
    np.ndarray
        Epsilon on cells, shape (Nx-1, Ny-1, Nz-1).
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
    """Generate conductivity cell array directly from block definitions.

    Parameters
    ----------
    phi : np.ndarray
        3D potential array (for grid dimensions).
    blocks : list of dict or dict
        Conductivity block definitions with 'sigma_val' and '*_range'.

    Returns
    -------
    np.ndarray
        Conductivity on cells, shape (Nx-1, Ny-1, Nz-1).
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


def generate_eps_cell(phi, blocks=None, use_precomputed=True):
    """Generate epsilon on cells, optionally using precomputed high-res data.

    Falls back to _generate_eps_cell_simple if precomputed data is unavailable.

    Parameters
    ----------
    phi : np.ndarray
        3D potential array.
    blocks : list of dict or None, optional
        Dielectric blocks (used in fallback, default: None).
    use_precomputed : bool, optional
        If True, use NPZ high-res cache (default: True).

    Returns
    -------
    np.ndarray
        Epsilon cell array (Nx-1, Ny-1, Nz-1).
    """
    if use_precomputed:
        cache = _load_eps_cache()
        if cache is not False:
            Nx, Ny, Nz = phi.shape
            bg_profile = cache["bg"]
            pillar_profile = cache["pillar"]
            rect = cache["rect"]

            nz_high = len(bg_profile)
            z_high = np.linspace(0.0, 1.0, nz_high)
            z_edges = np.linspace(0.0, 1.0, Nz)
            left = np.searchsorted(z_high, z_edges[:-1], side='left')
            right = np.searchsorted(z_high, z_edges[1:], side='right')
            right[-1] = nz_high

            bg_avg = np.array([np.mean(bg_profile[l:r]) for l, r in zip(left, right)], dtype=np.float32)

            if pillar_profile is not None:
                pillar_avg = np.array([np.mean(pillar_profile[l:r]) for l, r in zip(left, right)], dtype=np.float32)
            else:
                return np.tile(bg_avg[np.newaxis, np.newaxis, :], (Nx-1, Ny-1, 1)).astype(np.float32)

            if rect is None:
                weights = np.zeros((Nx-1, Ny-1), dtype=np.float32)
            else:
                x0, x1, y0, y1 = rect
                x_edges_xy = np.linspace(0.0, 1.0, Nx)
                y_edges_xy = np.linspace(0.0, 1.0, Ny)
                dx = x_edges_xy[1] - x_edges_xy[0]
                dy = y_edges_xy[1] - y_edges_xy[0]
                xl = x_edges_xy[:-1, np.newaxis]
                xh = x_edges_xy[1:, np.newaxis]
                yl = y_edges_xy[np.newaxis, :-1]
                yh = y_edges_xy[np.newaxis, 1:]
                ox0 = np.maximum(xl, x0)
                ox1 = np.minimum(xh, x1)
                oy0 = np.maximum(yl, y0)
                oy1 = np.minimum(yh, y1)
                overlap_x = np.maximum(0.0, ox1 - ox0) / dx
                overlap_y = np.maximum(0.0, oy1 - oy0) / dy
                weights = (overlap_x * overlap_y).astype(np.float32)

            w = weights[:, :, np.newaxis]
            bg_3d = bg_avg[np.newaxis, np.newaxis, :]
            pil_3d = pillar_avg[np.newaxis, np.newaxis, :]
            return ((1 - w) * bg_3d + w * pil_3d).astype(np.float32)

    return _generate_eps_cell_simple(phi, blocks)


def generate_sigma_cell(phi, blocks=None, use_precomputed=True):
    """Generate conductivity on cells, optionally using precomputed high-res data.

    Falls back to _generate_sigma_cell_simple if precomputed data is unavailable.
    Clamps values below 1e-6 to 1e-6 to avoid division by zero.

    Parameters
    ----------
    phi : np.ndarray
        3D potential array.
    blocks : list of dict or None, optional
        Conductivity blocks (used in fallback, default: None).
    use_precomputed : bool, optional
        If True, use NPZ high-res cache (default: True).

    Returns
    -------
    np.ndarray
        Conductivity cell array (Nx-1, Ny-1, Nz-1).
    """
    if use_precomputed:
        cache = _load_sigma_cache()
        if cache is not False:
            Nx, Ny, Nz = phi.shape
            bg_profile = cache["bg"]
            pillar_profile = cache["pillar"]
            rect = cache["rect"]

            nz_high = len(bg_profile)
            z_high = np.linspace(0.0, 1.0, nz_high)
            z_edges = np.linspace(0.0, 1.0, Nz)
            left = np.searchsorted(z_high, z_edges[:-1], side='left')
            right = np.searchsorted(z_high, z_edges[1:], side='right')
            right[-1] = nz_high

            bg_avg = np.array([np.mean(bg_profile[l:r]) for l, r in zip(left, right)], dtype=np.float32)
            if pillar_profile is not None:
                pillar_avg = np.array([np.mean(pillar_profile[l:r]) for l, r in zip(left, right)], dtype=np.float32)
            else:
                out = np.tile(bg_avg[np.newaxis, np.newaxis, :], (Nx-1, Ny-1, 1)).astype(np.float32)
                out[out < 1e-9] = 1e-9
                return out

            if rect is None:
                weights = np.zeros((Nx-1, Ny-1), dtype=np.float32)
            else:
                x0, x1, y0, y1 = rect
                x_edges_xy = np.linspace(0.0, 1.0, Nx)
                y_edges_xy = np.linspace(0.0, 1.0, Ny)
                dx = x_edges_xy[1] - x_edges_xy[0]
                dy = y_edges_xy[1] - y_edges_xy[0]
                xl = x_edges_xy[:-1, np.newaxis]
                xh = x_edges_xy[1:, np.newaxis]
                yl = y_edges_xy[np.newaxis, :-1]
                yh = y_edges_xy[np.newaxis, 1:]
                ox0 = np.maximum(xl, x0); ox1 = np.minimum(xh, x1)
                oy0 = np.maximum(yl, y0); oy1 = np.minimum(yh, y1)
                overlap_x = np.maximum(0.0, ox1 - ox0) / dx
                overlap_y = np.maximum(0.0, oy1 - oy0) / dy
                weights = (overlap_x * overlap_y).astype(np.float32)

            w = weights[:, :, np.newaxis]
            bg_3d = bg_avg[np.newaxis, np.newaxis, :]
            pil_3d = pillar_avg[np.newaxis, np.newaxis, :]
            out = ((1 - w) * bg_3d + w * pil_3d).astype(np.float32)
            out[out < 1e-6] = 1e-6
            return out

    return _generate_sigma_cell_simple(phi, blocks)
