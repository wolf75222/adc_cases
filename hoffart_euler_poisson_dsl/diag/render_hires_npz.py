#!/usr/bin/env python3
"""Render high-resolution diocotron snapshots from rapatriated sim.write npz (ADC-79).

The high-resolution campaign runs on ROMEO (diag/run_real_npz.py under
slurm/campaign_figures_n512.sbatch) and dumps the real system-schur state as npz at
the snapshot fractions. The rendering does NOT need ROMEO or the `adc` module: this
script reads those npz and draws the same schlieren grid as make_paper_figures.py.

The schlieren mapping is intentionally INLINED (not imported) so this render path stays
free of any `adc` dependency -- make_paper_figures.py imports `adc`/`model`/`run` at load
time (it also runs the simulation), which is unavailable on a plain workstation. The
palette is the ADC-80 one (percentile p99.5 + gamma 1.5, slate exterior, Blues); keep it
in sync with make_paper_figures.py:schlieren_rgba.

Usage:
    python diag/render_hires_npz.py --npz-root <dir> --out <dir> [--modes 3 4 5]
where <dir>/npz_l<M>_n<N>/state_*.npz are the rapatriated dumps. A snapshot fraction
without a npz (e.g. t_f cut by the walltime) is drawn as an annotated placeholder.
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

RW = 16.0  # ring geometry half-width (paper scale 6:8:16), as in make_paper_figures.py
SNAP_FRAC = (0.01, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
FIG_NUM = {3: 1, 4: 2, 5: 3}
SLATE = to_rgba("#3C4358")
BLUES = plt.get_cmap("Blues")


def schlieren_rgba(rho, rmask, n):
    """Inlined from make_paper_figures.py:schlieren_rgba (ADC-80 palette). Keep in sync."""
    h = 2 * RW / n
    gy, gx = np.gradient(rho, h, h, edge_order=2)
    g = np.hypot(gx, gy)
    inside = rmask <= RW
    ref = float(np.percentile(g[inside], 99.5)) if inside.any() else float(g.max())
    img = np.log1p(20.0 * g / max(ref, 1e-30))
    img = np.clip(img / max(float(np.log1p(20.0)), 1e-30), 0.0, 1.0) ** 1.5
    rgba = BLUES(img)
    rgba[rmask > RW] = SLATE
    return rgba


def _rmask(n):
    h = 2 * RW / n
    x = (np.arange(n) + 0.5) * h - RW
    X, Y = np.meshgrid(x, x, indexing="xy")
    return np.hypot(X, Y)


def _load(npz_root, mode):
    files = sorted(glob.glob(os.path.join(npz_root, "npz_l%d_n*" % mode, "state_*.npz")))
    if not files:
        raise FileNotFoundError("no state_*.npz under %s/npz_l%d_n*" % (npz_root, mode))
    frames = [np.asarray(np.load(f)["state_electrons"][0], float) for f in files]  # comp 0 = density
    n = int(np.load(files[0])["nx"])
    return frames, n


def render_mode(npz_root, out, mode):
    frames, n = _load(npz_root, mode)
    rmask = _rmask(n)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9.6))
    fig.patch.set_facecolor("white")
    for k, frac in enumerate(SNAP_FRAC):
        ax = axes.flat[k]
        lbl = "0.01" if abs(frac - 0.01) < 1e-9 else ("%d/8" % round(frac * 8) if frac < 1 else "1")
        if k < len(frames):
            ax.imshow(schlieren_rgba(frames[k], rmask, n), origin="lower", extent=(-RW, RW, -RW, RW))
            ax.set_title(r"(%s) $t=%s\,t_f$" % ("abcdefghi"[k], lbl), fontsize=11, color="#222")
        else:
            ax.text(0.5, 0.5, "t_f frame not reached\n(walltime cut at %s t_f)" % lbl,
                    ha="center", va="center", fontsize=10, color="#a33", transform=ax.transAxes)
            ax.set_title(r"(%s) $t=%s\,t_f$  [timeout]" % ("abcdefghi"[k], lbl), fontsize=11, color="#a33")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(r"Mode $\ell=%d$ -- schlieren density (system-schur, n=%d, ROMEO), Hoffart Fig 5.%d"
                 % (mode, n, FIG_NUM[mode]), fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    p = os.path.join(out, "snapshots_l%d_n%d.png" % (mode, n))
    fig.savefig(p, dpi=160, facecolor="white"); plt.close(fig)
    print("wrote %s (%d/%d frames, n=%d)" % (p, len(frames), len(SNAP_FRAC), n))
    return p


def main():
    ap = argparse.ArgumentParser(description="Render hi-res diocotron snapshots from rapatriated npz.")
    ap.add_argument("--npz-root", required=True, help="dir holding npz_l<M>_n<N>/state_*.npz")
    ap.add_argument("--out", default=".", help="output dir for snapshots_l<M>_n<N>.png")
    ap.add_argument("--modes", type=int, nargs="+", default=[3, 4, 5])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for m in args.modes:
        render_mode(args.npz_root, args.out, m)


if __name__ == "__main__":
    main()
