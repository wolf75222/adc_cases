#!/usr/bin/env python3
"""Consolidate a hyqmom15 campaign case into a single HDF5 file (ADC-376).

Each case directory holds many ``adc.System.write(format="npz")`` snapshots plus
a ``run_meta.json``; sharing/archiving dozens of npz per case is awkward, so this
packs one case into a single ``<case>.h5``: the time axis, the moment fields, the
potential, the moment names, the per-snapshot realizability summary, and the full
run provenance as HDF5 attributes. Pure Python (h5py + the build-free snapshot
loader); skipped cleanly if h5py is unavailable.

Usage:
    python3 hyqmom15/campaigns/export_h5.py <campaign_dir> [--case NAME] [--out DIR]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))            # diagnostics, matlab_ref
sys.path.insert(0, str(HERE.parent / "plots"))  # snapshots loader

from diagnostics import field_realizability, summarize  # noqa: E402
from snapshots import load_case, load_meta  # noqa: E402


def _flatten_attrs(meta, prefix=""):
    """Flatten the nested run_meta dict into scalar HDF5 attributes."""
    out = {}
    for k, v in meta.items():
        key = "%s%s" % (prefix, k)
        if isinstance(v, dict):
            out.update(_flatten_attrs(v, key + "."))
        elif v is None:
            out[key] = ""
        else:
            out[key] = v
    return out


def export_case(case_dir, out_path):
    """Write one consolidated ``<case>.h5`` for ``case_dir``; return the path or None."""
    import h5py
    snaps = load_case(case_dir)
    if not snaps:
        return None
    meta = load_meta(case_dir)
    t = np.array([s.t for s in snaps], dtype=float)
    moments = np.stack([s.moments for s in snaps], axis=0)          # (nstep, 15, ny, nx)
    has_phi = any(s.phi is not None for s in snaps)
    series = [summarize(*field_realizability(s.moments)) for s in snaps]
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("t", data=t)
        h5.create_dataset("moments", data=moments, compression="gzip")
        h5.create_dataset("names_moments",
                          data=np.array(snaps[0].names, dtype="S8"))
        if has_phi:
            ny, nx = snaps[0].density.shape
            phi = np.stack([s.phi if s.phi is not None else np.zeros((ny, nx))
                            for s in snaps], axis=0)
            h5.create_dataset("phi", data=phi, compression="gzip")
        diag = h5.create_group("realizability")
        for key in series[0]:
            diag.create_dataset(key, data=np.array([r[key] for r in series], dtype=float))
        for k, v in _flatten_attrs(meta).items():
            h5.attrs[k] = v
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Pack hyqmom15 campaign cases into HDF5.")
    p.add_argument("campaign_dir", help="root with one sub-directory of snapshots per case")
    p.add_argument("--case", help="only this case (default: all)")
    p.add_argument("--out", help="output dir for the .h5 files (default: <campaign_dir>/h5)")
    args = p.parse_args(argv)
    try:
        import h5py  # noqa: F401
    except Exception:  # noqa: BLE001
        print("h5py not installed; HDF5 export skipped", file=sys.stderr)
        return 0
    root = pathlib.Path(args.campaign_dir)
    if not root.is_dir():
        print("campaign dir not found: %s" % root, file=sys.stderr)
        return 1
    out_dir = pathlib.Path(args.out) if args.out else root / "h5"
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = [root / args.case] if args.case else sorted(
        d for d in root.iterdir() if d.is_dir() and d.name not in ("figures", "h5", "paraview"))
    n = 0
    for cd in case_dirs:
        path = export_case(cd, out_dir / ("%s.h5" % cd.name))
        if path:
            n += 1
            print("  %-20s -> %s" % (cd.name, path))
    print("wrote %d HDF5 files to %s" % (n, out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
