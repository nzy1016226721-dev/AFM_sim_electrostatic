"""Small coordinate receipts, not plotting code. NPY bytes/names unchanged."""
import json
from pathlib import Path


def write_coordinate_receipt(path, source_shape, source_bounds_nm, slices=None):
    source_shape = tuple(int(v) for v in source_shape)
    if slices is None:
        slices = tuple(slice(0,n) for n in source_shape)
    node_bounds, saved_shape, indices = [], [], []
    for axis, (n, selection) in enumerate(zip(source_shape, slices)):
        lo, hi = map(float, source_bounds_nm[2*axis:2*axis+2])
        if hi < lo:
            lo,hi = hi,lo
        first, stop = int(selection.start), int(selection.stop)
        node_bounds.extend([lo+(hi-lo)*first/(n-1), lo+(hi-lo)*(stop-1)/(n-1)])
        saved_shape.append(stop-first)
        indices.append([first,stop])
    record = dict(format="afm-local-coordinates-v1", array_order="xyz", potential_units="V",
                  coordinate_units="nm", shape=saved_shape, source_shape=list(source_shape),
                  source_bounds_nm=list(source_bounds_nm), source_index_slices=indices,
                  node_bounds_nm=node_bounds,
                  note="Bounds are first/last retained solver nodes, NOT crop-bin edges. No solver geometry changed.")
    receipt = Path(str(path)+".coords.json")
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(record,handle,indent=2)
        handle.write("\n")
    return str(receipt)
