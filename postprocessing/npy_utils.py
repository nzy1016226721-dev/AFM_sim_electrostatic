import re


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_UNSIGNED_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)"
_LEGACY_POSITION = (
    rf"(?:_cx(?P<cx>{_NUMBER})_cy(?P<cy>{_NUMBER})_cz(?P<cz>{_NUMBER}))?"
)
_NM_MOVEMENT = (
    rf"(?:(?P<movement_sign>[_-])(?P<movement_nm>{_UNSIGNED_NUMBER})nm)?"
)
_CUT_GRID = (
    r"(?:_cut_from_grid(?P<grid_x>\d+)x(?P<grid_y>\d+)x(?P<grid_z>\d+))?"
)
_UNIQUE_SUFFIX = r"(?: \(\d+\))?"


def _metadata(match, *, file_type, include_mag=False):
    """Convert one filename regex match into public metadata."""
    groups = match.groupdict()
    info = {
        "type": file_type,
        "config_idx": int(groups["config_idx"]),
        "Vtip": float(groups["voltage"]),
    }
    if include_mag:
        info["mag"] = int(groups["mag"])
    if groups.get("cx") is not None:
        # Older output used fractional coordinates. Keep this unchanged for
        # scripts that analyze archived result directories.
        info["position"] = (
            float(groups["cx"]),
            float(groups["cy"]),
            float(groups["cz"]),
        )
    elif groups.get("movement_nm") is not None:
        offset_nm = float(groups["movement_nm"])
        if groups.get("movement_sign") == "-":
            offset_nm = -offset_nm
        info["movement_offset_nm"] = offset_nm
    if groups.get("grid_x") is not None:
        info["is_cut"] = True
        info["source_grid"] = (
            int(groups["grid_x"]),
            int(groups["grid_y"]),
            int(groups["grid_z"]),
        )
    else:
        info["is_cut"] = False
    return info


def parse_phi_filename(fname):
    """Extract metadata from a phi .npy filename.

    Handles normal and zoom filenames, with optional physical movement and
    saved-cut source-grid metadata:
      ``afm_phi_<config>_<Vtip>V.npy``
      ``afm_phi_<config>_0nm_<Vtip>V.npy``
      ``afm_phi_<config>-0.001nm_<Vtip>V.npy``
      ``afm_phi_<config>_cx<X>_cy<Y>_cz<Z>_<Vtip>V_cut_from_grid<Nx>x<Ny>x<Nz>.npy``
      ``afm_phi_zoom_<mag>x_<Vtip>V_<config>.npy``

    Parameters
    ----------
    fname : str
        Filename (not path) of the .npy file.

    Returns
    -------
    dict or None
        Keys: ``type`` ("normal" | "zoom"), ``config_idx`` (int),
        ``Vtip`` (float), and ``is_cut`` (bool). New physical movement files
        contain ``movement_offset_nm``; legacy fractional files contain
        ``position``. Cut files contain ``source_grid``. For zoom files the
        result also contains ``mag``.
        Returns None if the filename does not match either pattern.
    """
    patterns = (
        (
            "zoom",
            True,
            rf"afm_phi_zoom_(?P<mag>\d+)x_"
            rf"(?P<voltage>{_NUMBER})V_(?P<config_idx>\d+)"
            rf"{_NM_MOVEMENT}{_CUT_GRID}{_UNIQUE_SUFFIX}\.npy",
        ),
        (
            "normal",
            False,
            rf"afm_phi_(?P<config_idx>\d+){_NM_MOVEMENT}_"
            rf"(?P<voltage>{_NUMBER})V{_CUT_GRID}{_UNIQUE_SUFFIX}\.npy",
        ),
        (
            "zoom",
            True,
            rf"afm_phi_zoom_(?P<mag>\d+)x_"
            rf"(?P<voltage>{_NUMBER})V_(?P<config_idx>\d+)"
            rf"{_LEGACY_POSITION}{_CUT_GRID}{_UNIQUE_SUFFIX}\.npy",
        ),
        (
            "normal",
            False,
            rf"afm_phi_(?P<config_idx>\d+){_LEGACY_POSITION}_"
            rf"(?P<voltage>{_NUMBER})V{_CUT_GRID}{_UNIQUE_SUFFIX}\.npy",
        ),
    )
    for file_type, include_mag, pattern in patterns:
        match = re.fullmatch(pattern, fname)
        if match:
            return _metadata(match, file_type=file_type, include_mag=include_mag)
    return None
