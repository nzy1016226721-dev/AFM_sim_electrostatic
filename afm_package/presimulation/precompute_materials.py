import numpy as np
import json
import os
import sys


def build_eps_highres(config_json="afm_config_nm_frac.json", output_npz="eps_highres.npz"):
    """Build high-resolution epsilon z-profiles from a fractional config.

    Separates full-xy (background) and pillar blocks, generates 100k-point
    1D profiles along z, and saves them as a compressed NPZ.

    Parameters
    ----------
    config_json : str, optional
        Fractional config JSON path (default: afm_config_nm_frac.json).
    output_npz : str, optional
        Output NPZ file path (default: eps_highres.npz).

    Returns
    -------
    None
    """
    with open(config_json, 'r') as f:
        cfg = json.load(f)

    blocks = cfg["blocks"]

    full_blocks = []
    pillar_rect = None
    for blk in blocks:
        xr = blk["x_range"]
        yr = blk["y_range"]
        if xr == [0.0, 1.0] and yr == [0.0, 1.0]:
            full_blocks.append(blk)
        else:
            if pillar_rect is None:
                pillar_rect = (xr[0], xr[1], yr[0], yr[1])

    nz = 100000
    z = np.linspace(0.0, 1.0, nz)

    eps_bg = np.ones(nz, dtype=np.float32)
    for blk in full_blocks:
        z0, z1 = blk["z_range"]
        eps_bg[(z >= z0) & (z <= z1)] = blk["eps_val"]

    eps_pillar = None
    if pillar_rect is not None:
        eps_pillar = np.ones(nz, dtype=np.float32)
        for blk in blocks:
            z0, z1 = blk["z_range"]
            eps_pillar[(z >= z0) & (z <= z1)] = blk["eps_val"]

    np.savez_compressed(output_npz,
                        z_profile_background=eps_bg,
                        z_profile_pillar=eps_pillar,
                        pillar_rect=pillar_rect)
    print(f"Saved {output_npz}")


def resolve_sigma_value(block):
    """Resolve conductivity from a block dict, computing from doping if needed.

    Uses 'sigma_val' directly, or computes sigma = e * n * mu from
    carrier_density_cm3 / doping_cm3 / n_cm3 and mobility.

    Parameters
    ----------
    block : dict
        Block dictionary with optional 'sigma_val', 'material',
        'carrier_density_cm3', 'doping_cm3', 'n_cm3', 'mobility_cm2_v_s'.

    Returns
    -------
    float
        Conductivity in S/m.
    """
    E_CHARGE_C = 1.602176634e-19
    AS_DOPED_SI_DEFAULT_N_CM3 = 7.0e20
    AS_DOPED_SI_DEFAULT_MOBILITY_CM2_VS = 30.0

    if "sigma_val" in block:
        return float(block["sigma_val"])
    material = str(block.get("material", "")).replace("_", "").replace("-", "").lower()
    n_cm3 = block.get("carrier_density_cm3", block.get("doping_cm3", block.get("n_cm3")))
    if n_cm3 is None and material in {"sias", "asdopedsi", "si:as"}:
        n_cm3 = AS_DOPED_SI_DEFAULT_N_CM3
    if n_cm3 is None:
        raise KeyError("Conductivity block needs sigma_val or carrier_density_cm3/doping_cm3/n_cm3")
    mobility = block.get("mobility_cm2_v_s", block.get("mobility_cm2_V_s", AS_DOPED_SI_DEFAULT_MOBILITY_CM2_VS))
    n_m3 = float(n_cm3) * 1e6
    mobility_m2_v_s = float(mobility) * 1e-4
    return E_CHARGE_C * n_m3 * mobility_m2_v_s


def build_sigma_highres(sigma_blocks_json="sigma_blocks.json", output_npz="sigma_highres.npz"):
    """Build high-resolution conductivity z-profiles from sigma_blocks JSON.

    Similar to build_eps_highres but for conductivity. Resolves sigma from
    block data via resolve_sigma_value().

    Parameters
    ----------
    sigma_blocks_json : str, optional
        Sigma blocks JSON path (default: sigma_blocks.json).
    output_npz : str, optional
        Output NPZ file path (default: sigma_highres.npz).

    Returns
    -------
    None
    """
    if not os.path.isfile(sigma_blocks_json):
        print(f"Error: file '{sigma_blocks_json}' not found.")
        return

    with open(sigma_blocks_json, 'r') as f:
        data = json.load(f)

    blocks = data.get("sigma_blocks", [])
    if not blocks:
        print("No sigma_blocks found in JSON. Exiting.")
        return

    full_blocks = []
    pillar_rect = None
    for blk in blocks:
        xr = blk.get("x_range", [0.0, 1.0])
        yr = blk.get("y_range", [0.0, 1.0])
        if np.allclose(xr, [0.0, 1.0]) and np.allclose(yr, [0.0, 1.0]):
            full_blocks.append(blk)
        else:
            if pillar_rect is None:
                pillar_rect = (xr[0], xr[1], yr[0], yr[1])

    nz = 100000
    z = np.linspace(0.0, 1.0, nz)

    sigma_bg = np.ones(nz, dtype=np.float32) * 1e-12
    for blk in full_blocks:
        z0, z1 = blk["z_range"]
        mask = (z >= z0) & (z <= z1)
        sigma_bg[mask] = resolve_sigma_value(blk)

    sigma_pillar = None
    if pillar_rect is not None:
        sigma_pillar = np.ones(nz, dtype=np.float32) * 1e-12
        for blk in blocks:
            z0, z1 = blk["z_range"]
            mask = (z >= z0) & (z <= z1)
            sigma_pillar[mask] = resolve_sigma_value(blk)

    np.savez_compressed(output_npz,
                        z_profile_background=sigma_bg,
                        z_profile_pillar=sigma_pillar,
                        pillar_rect=pillar_rect)
    print(f"Saved {output_npz}")
    if pillar_rect is None:
        print("No pillar region detected (all blocks full XY).")


def precompute_grid_arrays(config_or_blocks, output_dir=".", kind="eps", max_grid=512):
    """Precompute downsampled 3D material arrays for multiple grid sizes.

    Reads high-resolution z-profiles, downsamples them to each power-of-two
    grid size N (8..max_grid), applies pillar weighting, and saves as
    eps_N.npy or sigma_N.npy files.

    Parameters
    ----------
    config_or_blocks : str or list
        Path to config JSON or list of block dicts.
    output_dir : str, optional
        Directory for output .npy files (default: ".").
    kind : str, optional
        Material kind "eps" or "sigma" (default: "eps").
    max_grid : int, optional
        Maximum grid size N (default: 512).

    Returns
    -------
    None
    """
    if isinstance(config_or_blocks, str):
        with open(config_or_blocks, 'r') as f:
            cfg = json.load(f)
        blocks = cfg.get("blocks" if kind == "eps" else "sigma_blocks", [])
    else:
        blocks = config_or_blocks

    nz_high = 100000
    z_high = np.linspace(0.0, 1.0, nz_high)

    profile_bg = np.ones(nz_high, dtype=np.float32)
    for blk in blocks:
        xr = blk.get("x_range", [0.0, 1.0])
        yr = blk.get("y_range", [0.0, 1.0])
        if np.allclose(xr, [0.0, 1.0]) and np.allclose(yr, [0.0, 1.0]):
            z0, z1 = blk["z_range"]
            val = blk.get("eps_val" if kind == "eps" else "sigma_val", 1.0)
            profile_bg[(z_high >= z0) & (z_high <= z1)] = val

    pillar_rect = None
    for blk in blocks:
        xr = blk.get("x_range", [0.0, 1.0])
        yr = blk.get("y_range", [0.0, 1.0])
        if not (np.allclose(xr, [0.0, 1.0]) and np.allclose(yr, [0.0, 1.0])):
            if pillar_rect is None:
                pillar_rect = (xr[0], xr[1], yr[0], yr[1])

    profile_pillar = None
    if pillar_rect is not None:
        profile_pillar = np.ones(nz_high, dtype=np.float32)
        for blk in blocks:
            z0, z1 = blk["z_range"]
            val = blk.get("eps_val" if kind == "eps" else "sigma_val", 1.0)
            profile_pillar[(z_high >= z0) & (z_high <= z1)] = val

    for N in range(8, max_grid + 1):
        if N & (N - 1) != 0:
            continue
        z_edges = np.linspace(0.0, 1.0, N)
        left = np.searchsorted(z_high, z_edges[:-1], side='left')
        right = np.searchsorted(z_high, z_edges[1:], side='right')
        right[-1] = nz_high

        bg_avg = np.array([np.mean(profile_bg[l:r]) for l, r in zip(left, right)], dtype=np.float32)

        if profile_pillar is not None:
            pillar_avg = np.array([np.mean(profile_pillar[l:r]) for l, r in zip(left, right)], dtype=np.float32)
            if pillar_rect is not None:
                x0, x1, y0, y1 = pillar_rect
                x_edges_xy = np.linspace(0.0, 1.0, N)
                y_edges_xy = np.linspace(0.0, 1.0, N)
                dx_f = 1.0 / (N - 1)
                dy_f = 1.0 / (N - 1)
                xl = x_edges_xy[:-1, np.newaxis]
                xh = x_edges_xy[1:, np.newaxis]
                yl = y_edges_xy[np.newaxis, :-1]
                yh = y_edges_xy[np.newaxis, 1:]
                ox0 = np.maximum(xl, x0); ox1 = np.minimum(xh, x1)
                oy0 = np.maximum(yl, y0); oy1 = np.minimum(yh, y1)
                overlap_x = np.maximum(0.0, ox1 - ox0) / dx_f
                overlap_y = np.maximum(0.0, oy1 - oy0) / dy_f
                weights = (overlap_x * overlap_y).astype(np.float32)
                w = weights[:, :, np.newaxis]
                bg_3d = bg_avg[np.newaxis, np.newaxis, :]
                pil_3d = pillar_avg[np.newaxis, np.newaxis, :]
                arr = ((1 - w) * bg_3d + w * pil_3d).astype(np.float32)
            else:
                arr = np.tile(pillar_avg[np.newaxis, np.newaxis, :], (N-1, N-1, 1)).astype(np.float32)
        else:
            arr = np.tile(bg_avg[np.newaxis, np.newaxis, :], (N-1, N-1, 1)).astype(np.float32)

        fname = os.path.join(output_dir, f"{kind}_{N}.npy")
        np.save(fname, arr)
        print(f"  Saved {fname} ({arr.shape})")

    print("Grid array precomputation complete.")


def main():
    """CLI entry point for material precomputation steps.

    Menu-driven: builds high-res NPZ and/or precomputes grid arrays.

    Returns
    -------
    None
    """
    print("=== Material Precomputation ===\n")

    print("1. Build eps_highres.npz from config")
    print("2. Build sigma_highres.npz from sigma_blocks.json")
    print("3. Precompute grid arrays (eps_N.npy / sigma_N.npy)")
    print("4. All of the above")

    choice = input("Select option: ").strip()

    if choice in ("1", "4"):
        config = input("Config JSON path (default: afm_config_nm_frac.json): ").strip() or "afm_config_nm_frac.json"
        if os.path.isfile(config):
            build_eps_highres(config)
        else:
            print(f"File not found: {config}")

    if choice in ("2", "4"):
        sigma_json = input("Sigma blocks JSON path (default: sigma_blocks.json): ").strip() or "sigma_blocks.json"
        if os.path.isfile(sigma_json):
            build_sigma_highres(sigma_json)
        else:
            print(f"File not found: {sigma_json}")

    if choice in ("3", "4"):
        kind = input("Kind (eps or sigma, default: eps): ").strip() or "eps"
        config_file = input(f"Config/blocks JSON path (default: afm_config_nm_frac.json): ").strip() or "afm_config_nm_frac.json"
        if os.path.isfile(config_file):
            precompute_grid_arrays(config_file, output_dir=".", kind=kind)
        else:
            print(f"File not found: {config_file}")


if __name__ == "__main__":
    main()
