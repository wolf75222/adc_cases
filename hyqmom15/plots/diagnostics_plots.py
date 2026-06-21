#!/usr/bin/env python3
"""Per-moment, realizability and symmetry figures for hyqmom15 (ADC-384).

Post-processing of a campaign directory (one sub-directory of
``adc.System.write(format="npz")`` snapshots per case, plus a ``run_meta.json``):
beyond the M00 density of ADC-377, this renders every moment, the realizability
maps and time series from ``hyqmom15/diagnostics``, the per-case symmetry
residual, and stamps full run provenance on every figure. No simulation is
re-run, so the figures are versionable report artefacts.

Usage:
    python3 hyqmom15/plots/diagnostics_plots.py <campaign_dir> [--case NAME] [--out DIR]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))            # plots/ (snapshots, provenance)
sys.path.insert(0, str(HERE.parent))     # hyqmom15/ (diagnostics, matlab_ref)

from diagnostics import (  # noqa: E402
    field_realizability,
    summarize,
    symmetry_residual,
)
from matlab_ref import MOMENT_NAMES  # noqa: E402
from provenance import add_footer  # noqa: E402
from snapshots import XMAX, XMIN, load_case, load_meta  # noqa: E402

EXTENT = (XMIN, XMAX, XMIN, XMAX)


def _imshow(ax, data, title, cmap="magma", center=False):
    if center:
        amp = float(np.nanmax(np.abs(data))) or 1.0
        im = ax.imshow(data, origin="lower", extent=EXTENT, vmin=-amp, vmax=amp,
                       cmap="RdBu_r", aspect="equal")
    else:
        im = ax.imshow(data, origin="lower", extent=EXTENT, cmap=cmap, aspect="equal")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_moments(case, snaps, meta, out):
    """3x5 grid of the 15 moments M_pq at the final snapshot."""
    field = snaps[-1].moments
    fig, axes = plt.subplots(3, 5, figsize=(13.5, 8.2))
    for k, ax in enumerate(axes.flat):
        im = _imshow(ax, field[k], MOMENT_NAMES[k])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("%s  moments at t = %.4g" % (case, snaps[-1].t))
    add_footer(fig, meta)
    path = out / ("%s_moments.png" % case)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_velocities(case, snaps, meta, out):
    """Mean velocities u = M10/M00 and v = M01/M00 at the final snapshot."""
    s = snaps[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = s.moment("M10") / s.density
        v = s.moment("M01") / s.density
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, data, name in zip(axes, (u, v), ("u = M10/M00", "v = M01/M00")):
        im = _imshow(ax, data, name, center=True)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("%s  velocities at t = %.4g" % (case, s.t))
    add_footer(fig, meta)
    path = out / ("%s_velocities.png" % case)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_realizability_maps(case, snaps, meta, out):
    """2D maps of lam_min(p2p2), H20, H02 and the non-realizable mask (final snapshot)."""
    maps, realizable = field_realizability(snaps[-1].moments)
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.9))
    _imshow(axes[0], maps["lam_min"], "lam_min(p2p2)", cmap="viridis")
    fig.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.04)
    _imshow(axes[1], maps["H20"], "H20", cmap="viridis")
    fig.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04)
    _imshow(axes[2], maps["H02"], "H02", cmap="viridis")
    fig.colorbar(axes[2].images[0], ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(~realizable["all"], origin="lower", extent=EXTENT, cmap="Reds",
                   vmin=0, vmax=1, aspect="equal")
    axes[3].set_title("non-realizable cells", fontsize=8)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    fig.suptitle("%s  realizability at t = %.4g" % (case, snaps[-1].t))
    add_footer(fig, meta, extra="D = physical init, Dmax = CFL (convention, not a bug)")
    path = out / ("%s_realizability_maps.png" % case)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_realizability_series(case, snaps, meta, out):
    """Time series of the realizability reductions (min lam_min, fractions, min M00)."""
    rows = [summarize(*field_realizability(s.moments)) for s in snaps]
    t = np.array([s.t for s in snaps])
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.4))
    axes[0].plot(t, [r["lam_min"] for r in rows], color="tab:blue")
    axes[0].axhline(0.0, color="0.6", lw=0.8, ls="--")
    axes[0].set_title("min lam_min(p2p2)")
    axes[1].plot(t, [r["frac_nonrealizable"] for r in rows], color="tab:red")
    axes[1].set_title("fraction non-realizable")
    axes[2].plot(t, [r["M00_min"] for r in rows], color="tab:green")
    axes[2].axhline(0.0, color="0.6", lw=0.8, ls="--")
    axes[2].set_title("min M00 (positivity)")
    axes[3].plot(t, [r["frac_negative_M00"] for r in rows], color="tab:purple")
    axes[3].set_title("fraction M00 < 0")
    for ax in axes:
        ax.set_xlabel("t")
    fig.suptitle("%s  realizability over time" % case)
    add_footer(fig, meta)
    path = out / ("%s_realizability_series.png" % case)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_symmetry_series(case, snaps, meta, out):
    """Time series of the per-case symmetry residual (skipped for an unknown case)."""
    try:
        res = [symmetry_residual(s.moments, case) for s in snaps]
    except KeyError:
        print("  %-20s no symmetry residual defined; skipped" % case)
        return None
    t = np.array([s.t for s in snaps])
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(t, res, color="tab:cyan")
    ax.set_xlabel("t")
    ax.set_ylabel("symmetry residual")
    ax.set_title("%s  symmetry residual" % case)
    add_footer(fig, meta)
    path = out / ("%s_symmetry_series.png" % case)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def render_diagnostics(case_dir, out_dir):
    """Render all diagnostic figures for one case directory; returns written paths."""
    case = pathlib.Path(case_dir).name
    snaps = load_case(case_dir)
    if not snaps:
        print("  skip %-20s (no snapshots)" % case)
        return []
    meta = load_meta(case_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        plot_moments(case, snaps, meta, out_dir),
        plot_velocities(case, snaps, meta, out_dir),
        plot_realizability_maps(case, snaps, meta, out_dir),
        plot_realizability_series(case, snaps, meta, out_dir),
    ]
    sym = plot_symmetry_series(case, snaps, meta, out_dir)
    if sym is not None:
        written.append(sym)
    print("  %-20s %d snapshots -> %d diagnostic figures" % (case, len(snaps), len(written)))
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Render hyqmom15 per-moment + diagnostics figures.")
    p.add_argument("campaign_dir", help="root with one sub-directory of snapshots per case")
    p.add_argument("--case", help="render only this case (default: all sub-directories)")
    p.add_argument("--out", help="figure output dir (default: <campaign_dir>/figures)")
    args = p.parse_args(argv)

    root = pathlib.Path(args.campaign_dir)
    if not root.is_dir():
        print("campaign dir not found: %s" % root, file=sys.stderr)
        return 1
    out_dir = pathlib.Path(args.out) if args.out else root / "figures"
    if args.case:
        case_dirs = [root / args.case]
    else:
        case_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name != "figures")

    total = 0
    for cd in case_dirs:
        total += len(render_diagnostics(cd, out_dir))
    print("wrote %d diagnostic figures to %s" % (total, out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
