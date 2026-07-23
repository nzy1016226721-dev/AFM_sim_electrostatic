import re


def parse_phi_filename(fname):
    """Extract metadata from a phi .npy filename.

    Handles both normal and zoom filenames:
      ``afm_phi_<config>_<Vtip>V.npy``
      ``afm_phi_zoom_<mag>x_<Vtip>V_<config>.npy``

    Parameters
    ----------
    fname : str
        Filename (not path) of the .npy file.

    Returns
    -------
    dict or None
        Keys: ``type`` ("normal" | "zoom"), ``config_idx`` (int),
        ``Vtip`` (float).  For zoom files also ``mag`` (int).
        Returns None if the filename does not match either pattern.
    """
    m = re.match(r'afm_phi_zoom_(\d+)x_(-?[\d.]+)V_(\d+)\.npy', fname)
    if m:
        return {'type': 'zoom', 'mag': int(m.group(1)),
                'Vtip': float(m.group(2)), 'config_idx': int(m.group(3))}
    m = re.match(r'afm_phi_(\d+)_(-?[\d.]+)V\.npy', fname)
    if m:
        return {'type': 'normal', 'config_idx': int(m.group(1)),
                'Vtip': float(m.group(2))}
    return None
