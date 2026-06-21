#!/usr/bin/env python3
"""Export a hyqmom15 campaign case to ParaView (ADC-376).

The campaign writes ``adc.System.write(format="npz")`` snapshots, which ParaView
cannot open. This converts them to a VTK time series on the uniform Cartesian grid
[-0.5, 0.5]^2: one ``<case>_NNNN.vti`` (ImageData, cell data) per snapshot plus a
``<case>.pvd`` collection that ParaView opens as an animated time series. Fields:
density (M00), velocities ux/uy + speed, the potential phi (when present), the
realizability margin lam_min, and the 15 raw moments.

The runs are single-rank (no MPI), so each snapshot is one grid -> a single .vti
per step is correct; parallel pieces (.pvti) would only apply to MPI-distributed
runs. Pure Python (hand-written VTK XML, base64 binary); no VTK/pyvista needed.

Usage:
    python3 hyqmom15/campaigns/to_paraview.py <campaign_dir> [--case NAME] [--out DIR]
"""
from __future__ import annotations

import argparse
import base64
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "plots"))

from snapshots import load_case  # noqa: E402

try:
    from diagnostics import field_realizability  # noqa: E402
    _HAVE_REALIZ = True
except Exception:  # noqa: BLE001
    _HAVE_REALIZ = False


def _b64(arr):
    """Encode a numpy array as a VTK inline-binary payload (UInt32 length + LE f8)."""
    data = np.ascontiguousarray(arr, dtype="<f8").tobytes()
    return base64.b64encode(struct.pack("<I", len(data)) + data).decode("ascii")


def _fields(snap):
    """Return an ordered dict of cell-data field name -> flat (ny*nx,) array."""
    m = snap.moments  # (15, ny, nx)
    names = list(snap.names)
    idx = {n: k for k, n in enumerate(names)}
    rho = m[idx.get("M00", 0)]
    safe = np.where(np.abs(rho) > 1e-30, rho, 1.0)
    ux = m[idx["M10"]] / safe if "M10" in idx else np.zeros_like(rho)
    uy = m[idx["M01"]] / safe if "M01" in idx else np.zeros_like(rho)
    fields = {
        "density": rho, "ux": ux, "uy": uy, "speed": np.hypot(ux, uy),
    }
    if snap.phi is not None:
        fields["phi"] = snap.phi
    if _HAVE_REALIZ:
        try:
            fields["lam_min"] = field_realizability(m)[0]["lam_min"]
        except Exception:  # noqa: BLE001
            pass
    for n in names:
        fields[n] = m[idx[n]]
    return {k: np.asarray(v, dtype=float).ravel() for k, v in fields.items()}


def write_vti(path, ny, nx, fields):
    """Write one ImageData .vti (cell data) on the [-0.5,0.5]^2 uniform grid."""
    dx, dy = 1.0 / nx, 1.0 / ny
    head = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt32">\n'
        '  <ImageData WholeExtent="0 %d 0 %d 0 0" Origin="-0.5 -0.5 0" '
        'Spacing="%.17g %.17g 1">\n'
        '    <Piece Extent="0 %d 0 %d 0 0">\n'
        '      <CellData Scalars="density">\n' % (nx, ny, dx, dy, nx, ny)
    )
    body = "".join(
        '        <DataArray type="Float64" Name="%s" format="binary">%s</DataArray>\n'
        % (name, _b64(arr)) for name, arr in fields.items()
    )
    tail = "      </CellData>\n    </Piece>\n  </ImageData>\n</VTKFile>\n"
    pathlib.Path(path).write_text(head + body + tail, encoding="ascii")


def write_pvd(path, entries):
    """Write a ParaView .pvd collection: (timestep, relative .vti file) pairs."""
    rows = "".join(
        '    <DataSet timestep="%.17g" group="" part="0" file="%s"/>\n' % (t, f)
        for t, f in entries
    )
    pathlib.Path(path).write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">\n'
        "  <Collection>\n%s  </Collection>\n</VTKFile>\n" % rows,
        encoding="ascii")


def export_case(case_dir, out_dir):
    """Convert one case directory to ``<case>.pvd`` + ``<case>_NNNN.vti``; return the .pvd path."""
    snaps = load_case(case_dir)
    if not snaps:
        return None
    case = pathlib.Path(case_dir).name
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for k, s in enumerate(snaps):
        ny, nx = s.density.shape
        fname = "%s_%04d.vti" % (case, k)
        write_vti(out_dir / fname, ny, nx, _fields(s))
        entries.append((float(s.t), fname))
    pvd = out_dir / ("%s.pvd" % case)
    write_pvd(pvd, entries)
    return pvd


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export hyqmom15 campaign cases to ParaView (.vti + .pvd).")
    p.add_argument("campaign_dir", help="root with one snapshot sub-directory per case")
    p.add_argument("--case", help="only this case (default: all)")
    p.add_argument("--out", help="output dir (default: <campaign_dir>/paraview)")
    args = p.parse_args(argv)
    root = pathlib.Path(args.campaign_dir)
    if not root.is_dir():
        print("campaign dir not found: %s" % root, file=sys.stderr)
        return 1
    out_dir = pathlib.Path(args.out) if args.out else root / "paraview"
    case_dirs = [root / args.case] if args.case else sorted(
        d for d in root.iterdir() if d.is_dir() and d.name not in ("figures", "h5", "paraview"))
    n = 0
    for cd in case_dirs:
        pvd = export_case(cd, out_dir)
        if pvd:
            n += 1
            print("  %-20s -> %s" % (cd.name, pvd))
    print("wrote %d ParaView collections to %s" % (n, out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
