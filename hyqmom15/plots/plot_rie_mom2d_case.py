#!/usr/bin/env python3
"""Render Matlab-like figures for the RieMOM2D_Electrostatic_periodic cases (ADC-377).

Reads a campaign directory written by the ROMEO campaign (ADC-376) -- one
sub-directory per case, each holding ``adc.System.write(format="npz")``
snapshots plus an optional ``run_meta.json`` -- and produces, per case:

  * ``<case>_density.png`` : M00 density at a few key times, shared colour limits;
  * ``<case>_phi.png``     : the Poisson potential at the same times (waves with
                             Poisson, diocotron), skipped when no field is active;
  * ``<case>_diagnostics.png`` : mass drift, M00 min/max, and dt over time;
  * ``<case>_density.gif`` : a density animation (``--gif``).

Figures are written without re-running any simulation, so they are versionable
report artefacts. The domain is drawn as ``[-0.5, 0.5]^2`` to match the Matlab
axes.

Vocabulary (ADC-378): annotations distinguish a clarified convention from a true
divergence. In particular ``D`` (physical wave init) and ``Dmax`` (CFL max speed)
are a convention, NOT a Matlab bug, so figures never label that split a "bug".

Usage:
    python3 hyqmom15/plots/plot_rie_mom2d_case.py <campaign_dir> [--case NAME]
        [--out DIR] [--gif] [--n-snapshots K]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt           # noqa: E402
from matplotlib import animation          # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from provenance import add_footer  # noqa: E402
from snapshots import XMIN, XMAX, load_case, load_meta, time_series  # noqa: E402

EXTENT = (XMIN, XMAX, XMIN, XMAX)
DENSITY_CMAP = "magma"      # M00 >= 0; a perceptually-uniform stand-in for Matlab "sky"
PHI_CMAP = "RdBu_r"         # phi is signed
CONVENTION_NOTE = "D = physical init, Dmax = CFL max speed (convention, not a bug)"


def _pick(snaps, k):
    """k roughly-evenly-spaced snapshots including the first and last."""
    if len(snaps) <= k:
        return snaps
    idx = np.linspace(0, len(snaps) - 1, k).round().astype(int)
    return [snaps[i] for i in sorted(set(idx))]


def _title(case, meta):
    np_ = meta.get("Np")
    extra = " Np=%s" % np_ if np_ else ""
    return "%s%s" % (case, extra)


def plot_density(case, snaps, meta, out, n=4):
    sel = _pick(snaps, n)
    vmin = min(float(s.density.min()) for s in snaps)
    vmax = max(float(s.density.max()) for s in snaps)
    fig, axes = plt.subplots(1, len(sel), figsize=(3.4 * len(sel), 3.7), squeeze=False)
    im = None
    for ax, s in zip(axes[0], sel):
        im = ax.imshow(s.density, origin="lower", extent=EXTENT, vmin=vmin, vmax=vmax,
                       cmap=DENSITY_CMAP, aspect="equal")
        ax.set_title("t = %.4g" % s.t)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    if im is not None:
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.04, label="M00")
    fig.suptitle("%s  density M00" % _title(case, meta))
    add_footer(fig, meta, extra=CONVENTION_NOTE)
    path = out / ("%s_density.png" % case)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_phi(case, snaps, meta, out, n=4):
    with_phi = [s for s in snaps if s.phi is not None]
    if not with_phi:
        return None
    sel = _pick(with_phi, n)
    amp = max(float(np.abs(s.phi).max()) for s in with_phi) or 1.0
    fig, axes = plt.subplots(1, len(sel), figsize=(3.4 * len(sel), 3.7), squeeze=False)
    im = None
    for ax, s in zip(axes[0], sel):
        im = ax.imshow(s.phi, origin="lower", extent=EXTENT, vmin=-amp, vmax=amp,
                       cmap=PHI_CMAP, aspect="equal")
        ax.set_title("t = %.4g" % s.t)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    if im is not None:
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.04, label="phi")
    fig.suptitle("%s  potential phi" % _title(case, meta))
    add_footer(fig, meta)
    path = out / ("%s_phi.png" % case)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_diagnostics(case, snaps, meta, out):
    ts = time_series(snaps)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    axes[0].plot(ts["t"], ts["mass_rel_drift"], color="tab:blue")
    axes[0].set_title("relative mass drift")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("(m(t) - m0) / m0")
    axes[1].plot(ts["t"], ts["m00_min"], label="min", color="tab:green")
    axes[1].plot(ts["t"], ts["m00_max"], label="max", color="tab:red")
    axes[1].set_title("M00 min / max (positivity)")
    axes[1].set_xlabel("t")
    axes[1].legend()
    axes[2].plot(ts["t"], ts["dt"], color="tab:purple")
    axes[2].set_title("dt over time")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("dt")
    fig.suptitle("%s  diagnostics" % _title(case, meta))
    add_footer(fig, meta)
    path = out / ("%s_diagnostics.png" % case)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def animate_density(case, snaps, meta, out, fps=12):
    vmin = min(float(s.density.min()) for s in snaps)
    vmax = max(float(s.density.max()) for s in snaps)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    im = ax.imshow(snaps[0].density, origin="lower", extent=EXTENT, vmin=vmin, vmax=vmax,
                   cmap=DENSITY_CMAP, aspect="equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="M00")
    ttl = ax.set_title("")

    def update(k):
        s = snaps[k]
        im.set_data(s.density)
        ttl.set_text("%s  M00  t = %.4g" % (_title(case, meta), s.t))
        return [im, ttl]

    anim = animation.FuncAnimation(fig, update, frames=len(snaps), interval=1000 // fps, blit=False)
    path = out / ("%s_density.gif" % case)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def render_case(case_dir, out_dir, make_gif=False, n_snapshots=4):
    case = pathlib.Path(case_dir).name
    snaps = load_case(case_dir)
    if not snaps:
        print("  skip %-20s (no snapshots)" % case)
        return []
    meta = load_meta(case_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [plot_density(case, snaps, meta, out_dir, n_snapshots),
               plot_diagnostics(case, snaps, meta, out_dir)]
    phi_path = plot_phi(case, snaps, meta, out_dir, n_snapshots)
    if phi_path:
        written.append(phi_path)
    if make_gif and len(snaps) > 1:
        written.append(animate_density(case, snaps, meta, out_dir))
    print("  %-20s %d snapshots -> %d figures" % (case, len(snaps), len(written)))
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Render Matlab-like hyqmom15 campaign figures.")
    p.add_argument("campaign_dir", help="root with one sub-directory of snapshots per case")
    p.add_argument("--case", help="render only this case (default: all sub-directories)")
    p.add_argument("--out", help="figure output dir (default: <campaign_dir>/figures)")
    p.add_argument("--gif", action="store_true", help="also write a density animation per case")
    p.add_argument("--n-snapshots", type=int, default=4,
                   help="max 2D snapshots per density/phi panel (default 4)")
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
        total += len(render_case(cd, out_dir, make_gif=args.gif, n_snapshots=args.n_snapshots))
    print("wrote %d figures to %s" % (total, out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
